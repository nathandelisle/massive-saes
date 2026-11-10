"""Check the SAE architecture implementations on synthetic inputs."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3.common.saes import (ARCHS, BatchTopKSAE, GatedSAE, JumpReLUSAE,  # noqa: E402
                            batch_topk_mask, load_sae, save_sae, train_sae)

FAIL = []


def check(name, cond, extra=""):
    print(f"{'ok  ' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        FAIL.append(name)


def anisotropic(n, d, seed=0):
    """PC1 carrying ~99% of variance, like a real middle-layer residual stream."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    v = v / v.norm()
    a = torch.randn(n, 1, generator=g) * 10.0
    return a * v + torch.randn(n, d, generator=g) * 0.1


D, DS, N = 32, 128, 4096
X = anisotropic(N, D)
X = X - X.mean(0)


def batches(steps, bs=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        yield X[torch.randint(0, N, (bs,), generator=g)]


# ------------------------------------------------------- BatchTopK: tie safety
z = torch.zeros(4, 10)
z[:, :3] = 1.0                      # 12 identical positives, budget = 4*2 = 8
m = batch_topk_mask(z, 2)
check("BatchTopK keeps exactly B*k entries even under exact ties",
      int((m != 0).sum()) == 8, f"kept={int((m != 0).sum())}")

z2 = torch.zeros(4, 10)             # only 3 positives, fewer than B*k = 8
z2[0, :3] = torch.tensor([5.0, 3.0, 1.0])
m2 = batch_topk_mask(z2, 2)
# keeping all 3 IS correct here; what must never happen is keeping entries that
# were not among the B*k largest (the old `z >= 0` passthrough with thresh==0)
check("BatchTopK keeps only positives when positives < B*k",
      int((m2 != 0).sum()) == 3 and float(m2[m2 != 0].min()) > 0,
      f"kept={int((m2 != 0).sum())}")
z3 = torch.randn(4, 10)             # mixed signs, as a raw pre-activation would be
m3 = batch_topk_mask(F.relu(z3), 2)
check("BatchTopK never keeps a non-positive entry",
      bool((m3[m3 != 0] > 0).all()) and int((m3 != 0).sum()) <= 8,
      f"kept={int((m3 != 0).sum())}")

# ------------------------------------ BatchTopK: inference is batch-independent
bt, _ = train_sae(batches(300), D, DS, arch="batchtopk", k=8, seed=0,
                  device="cpu", verbose=False)
bt.eval()
with torch.no_grad():
    big = bt.encode(X[:512])
    small = torch.cat([bt.encode(X[i:i + 16]) for i in range(0, 512, 16)])
check("BatchTopK inference does not depend on batch composition",
      torch.allclose(big, small, atol=1e-6),
      f"maxdiff={float((big - small).abs().max()):.2e}")
check("BatchTopK learned a positive inference threshold",
      float(bt.threshold) > 0, f"theta={float(bt.threshold):.4g}")

# ------------------------------------------ JumpReLU: the threshold does work
jr, _ = train_sae(batches(400), D, DS, arch="jumprelu", l0_coef=1e-3, seed=0,
                  device="cpu", verbose=False, target_l0=16)
with torch.no_grad():
    pre = F.relu(jr.pre(X[:512]))
    th = jr.log_theta.exp()
    gate_frac = float(((pre > 0) & (pre <= th)).sum() / max((pre > 0).sum(), 1))
    l0 = float((jr.encode(X[:512]) != 0).float().sum(-1).mean())
check("JumpReLU threshold actually gates (not a ReLU in disguise)",
      gate_frac > 0.5, f"gated {gate_frac:.3f} of positive pre-activations")
check("JumpReLU reaches a sparse L0", l0 < DS / 2, f"L0={l0:.1f}")
check("JumpReLU theta was calibrated away from its 1e-3 init",
      float(th.mean()) > 1e-2, f"mean theta={float(th.mean()):.4g}")

# L0 must respond monotonically to l0_coef, or `calibrate` cannot hit a target
l0s = []
for coef in (1e-3, 1e2, 1e4):
    m_ = train_sae(batches(300), D, DS, arch="jumprelu", l0_coef=coef, seed=0,
                   device="cpu", verbose=False, target_l0=16)[0]
    with torch.no_grad():
        l0s.append(float((m_.encode(X[:256]) != 0).float().sum(-1).mean()))
check("JumpReLU L0 decreases as l0_coef rises (calibration is possible)",
      l0s[0] >= l0s[-1] and l0s[0] - l0s[-1] > 0.5, f"L0 across coefs: {l0s}")

# ------------------------- the L0 penalty must not push gradient into the encoder
jr2 = JumpReLUSAE(D, DS, l0_coef=1.0, seed=0, target_l0=16)
jr2.calibrate_from_batch(X[:256])
jr2.zero_grad()
_ = jr2.encode(X[:256])
jr2.sparsity_loss(None, None).backward()
genc = jr2.W_enc.grad
check("L0 term sends ZERO gradient to the encoder (only to theta)",
      genc is None or float(genc.abs().max()) == 0.0,
      f"max|dW_enc|={0.0 if genc is None else float(genc.abs().max()):.2e}")
check("L0 term DOES send gradient to theta",
      jr2.log_theta.grad is not None
      and float(jr2.log_theta.grad.abs().max()) > 0,
      f"max|dtheta|={0.0 if jr2.log_theta.grad is None else float(jr2.log_theta.grad.abs().max()):.2e}")

# an unreachable target must fail loudly rather than silently become a ReLU
try:
    JumpReLUSAE(D, DS, seed=0, target_l0=DS).calibrate_from_batch(X[:256])
    ok_guard = False
except RuntimeError:
    ok_guard = True
check("unreachable target_l0 raises instead of degenerating to a ReLU", ok_guard)

# ------------------------------------------------ Gated: aux loss detaches decoder
gs = GatedSAE(D, DS, l1_coef=1e-3, seed=0)
gs.zero_grad()
_ = gs.encode(X[:128])
gs.aux_loss(X[:128]).backward()
check("Gated aux loss does not update W_dec (paper: detach W_dec and b_dec)",
      gs.W_dec.grad is None or float(gs.W_dec.grad.abs().max()) == 0.0)
# NB: b_dec legitimately still receives gradient here, but via the ENCODER
# pre-bias (x - b_dec), not via the frozen decoder copy. The paper's "do not
# propagate to W_dec or b_dec" governs the aux reconstruction path only, which
# the .detach() calls enforce.
check("Gated aux loss DOES update the gate path",
      gs.W_enc.grad is not None and float(gs.W_enc.grad.abs().max()) > 0)

# ------------------------------------------------------- save/load round trips
import tempfile                                                    # noqa: E402
tmp = tempfile.mkdtemp()
for name in ARCHS:
    kw = dict(seed=0)
    if name in ("topk", "batchtopk", "topk_l1"):
        kw["k"] = 8
    if name == "jumprelu":
        # toy d_sae=128; the production ratio is 64/8192, not 64/128
        kw["target_l0"] = 16
    try:
        s = ARCHS[name](D, DS, **kw)
    except NotImplementedError:
        continue
    for p in s.parameters():
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.01)
    if hasattr(s, "calibrate_from_batch"):
        s.calibrate_from_batch(X[:256])
    if hasattr(s, "threshold"):
        s.threshold.fill_(0.123)
    p_ = os.path.join(tmp, f"{name}.pt")
    save_sae(s, p_)
    s2, _e = load_sae(p_, device="cpu")
    s.eval()
    with torch.no_grad():
        same = torch.allclose(s.encode(X[:64]), s2.encode(X[:64]), atol=1e-6)
    check(f"{name}: save/load round-trips every parameter", same)

print()
if FAIL:
    print(f"{len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("all architecture-fidelity checks pass")
