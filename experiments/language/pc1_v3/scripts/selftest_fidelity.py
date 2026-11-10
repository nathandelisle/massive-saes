"""Check the training and evaluation settings."""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3.common import matching, registry, splice                  # noqa: E402
from v3.common.saes import ARCHS, TopKSAE                          # noqa: E402

FAIL = []


def check(name, cond, extra=""):
    print(f"{'ok  ' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------- 1. layer indexing
class _Blk(nn.Module):
    def __init__(self, d, i):
        super().__init__()
        self.i = i
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return (self.lin(x) + x,)


class _Tiny(nn.Module):
    """Mimics HF: hidden_states[0] = embeddings, hidden_states[i+1] = block i."""

    def __init__(self, d=8, n=6):
        super().__init__()
        self.emb = nn.Embedding(50, d)
        self.layers = nn.ModuleList([_Blk(d, i) for i in range(n)])

    def forward(self, ids, output_hidden_states=False, attention_mask=None):
        h = self.emb(ids)
        hs = [h]
        for b in self.layers:
            h = b(h)[0]
            hs.append(h)
        return type("O", (), {"hidden_states": tuple(hs), "logits": h})()


class _Spec:
    target_layer = 3
    block_path = "layers"
    n_layers = 6
    d_model = 8


m = _Tiny()
spec = _Spec()
ids = torch.randint(0, 50, (2, 5))
out = m(ids, output_hidden_states=True)

buf = {}
li = registry.block_index(spec)
h = m.layers[li].register_forward_hook(
    lambda mod, i, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
m(ids)
h.remove()

check("registry.block_index == target_layer (v2 convention)", li == 3, f"li={li}")
check("registry.hidden_state_index == target_layer + 1",
      registry.hidden_state_index(spec) == 4)
check("hook output == hidden_states[target_layer + 1]  (v2's tensor)",
      torch.allclose(buf["h"], out.hidden_states[4]))
check("hook output != hidden_states[target_layer]  (the old, wrong tensor)",
      not torch.allclose(buf["h"], out.hidden_states[3]))

# ------------------------------------------------------- 2. padding in CE loss
class _CEModel(nn.Module):
    """Emits confident-correct logits on real tokens and garbage on pads, so an
    unmasked CE is huge and a masked CE is ~0. Exactly the failure mode that
    inverted a delta-CE sign in earlier work."""

    def __init__(self, V=20):
        super().__init__()
        self.V = V
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, ids, attention_mask=None):
        B, T = ids.shape
        lg = torch.full((B, T, self.V), -10.0)
        for b in range(B):
            for t in range(T - 1):
                nxt = int(ids[b, t + 1])
                if nxt != 0:                       # 0 == pad: predict it badly
                    lg[b, t, nxt] = 20.0
        return type("O", (), {"logits": lg + self.dummy})()


V = 20
ids = torch.zeros(2, 8, dtype=torch.long)
am = torch.zeros(2, 8, dtype=torch.long)
for b in range(2):
    n = 5 if b == 0 else 4
    ids[b, :n] = torch.randint(1, V, (n,))
    am[b, :n] = 1
cem = _CEModel(V)
ce_masked = splice.ce_loss(cem, ids, chunk=2, am=am)
ce_unmasked = splice.ce_loss(cem, ids, chunk=2, am=None)
check("masked CE excludes padding (near zero here)", ce_masked < 0.5,
      f"{ce_masked:.4f}")
check("unmasked CE is badly inflated by padding", ce_unmasked > 2 * ce_masked + 1,
      f"unmasked={ce_unmasked:.4f} vs masked={ce_masked:.4f}")

# --------------------------------------------------------- 3. signed matching
d, ds = 16, 32
rng = np.random.default_rng(0)
A = rng.normal(size=(ds, d))
A /= np.linalg.norm(A, axis=1, keepdims=True)
B = A.copy()
B[0] = -B[0]                       # feature 0 flipped: NOT the same feature
mc_signed = matching.max_cos(A, B, signed=True)
mc_abs = matching.max_cos(A, B, signed=False)
check("signed matching does NOT match a feature to its negation",
      mc_signed[0] < registry.SHARED_COS_THRESHOLD, f"cos={mc_signed[0]:.3f}")
check("|cos| DOES wrongly match it (the bug)", mc_abs[0] > 0.99,
      f"|cos|={mc_abs[0]:.3f}")
check("signed matching still matches identical features",
      mc_signed[1] > 0.99)
check("max_cos does not restrict to alive_b by default",
      matching.max_cos(A, B).shape == (ds,))

# ------------------------------------------------------------ 4. SAE pre-bias
sae = TopKSAE(d, ds, k=4, seed=0)
with torch.no_grad():
    sae.b_dec.copy_(torch.arange(d, dtype=torch.float32))
x = torch.randn(3, d)
pre_with = sae.pre(x)
pre_without = x @ sae.W_enc.T + sae.b_enc
check("encoder applies the Gao et al. pre-bias (x - b_dec)",
      not torch.allclose(pre_with, pre_without))
check("pre-bias is exactly (x - b_dec) @ W_enc.T + b_enc",
      torch.allclose(pre_with, (x - sae.b_dec) @ sae.W_enc.T + sae.b_enc))

# every architecture must inherit it
for name, cls in ARCHS.items():
    try:
        s = cls(d, ds, seed=0) if name not in ("topk", "batchtopk", "topk_l1") \
            else cls(d, ds, k=4, seed=0)
    except NotImplementedError:
        continue
    with torch.no_grad():
        s.b_dec.copy_(torch.ones(d) * 3.0)
    z = s.encode(x)
    z0 = s.encode(x + 0.0)
    check(f"{name}: encode runs and is deterministic",
          torch.allclose(z, z0) and z.shape == (3, ds))

# --------------------------------------- 5. integrated-gradients path integral
# IG_j = z_j * mean_alpha[ dF/dz'_j at z' = alpha z ]. Under the transform
# z -> alpha*z the captured tensor's grad is alpha * dF/dz' by the chain rule,
# so accumulating z*z.grad integrates an extra alpha. For a LINEAR F the
# completeness axiom pins the answer exactly: sum_j IG_j == F(z) - F(0).
zz = torch.randn(1, 6, dtype=torch.float64)
wv = torch.randn(6, dtype=torch.float64)
S = 8
buggy = torch.zeros(6, dtype=torch.float64)
fixed = torch.zeros(6, dtype=torch.float64)
for s in range(1, S + 1):
    alpha = s / S
    z = zz.clone().requires_grad_(True)
    (z * alpha @ wv).sum().backward()            # F(alpha z) = w . (alpha z)
    buggy += (z.detach() * z.grad).sum(0)        # extra factor of alpha
    fixed += (z.detach() * z.grad / alpha).sum(0)
buggy /= S
fixed /= S
target = (zz[0] * wv)                            # exact IG for a linear F
check("fixed IG satisfies the completeness axiom",
      bool(torch.allclose(fixed, target, atol=1e-9)),
      f"max|err|={float((fixed - target).abs().max()):.2e}")
check("un-corrected IG does NOT (it is alpha-weighted)",
      not bool(torch.allclose(buggy, target, atol=1e-6)),
      f"ratio={float((buggy / target).mean()):.4f} (expected ~0.5625)")

print()
if FAIL:
    print(f"{len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("all v2-fidelity regression checks pass")
