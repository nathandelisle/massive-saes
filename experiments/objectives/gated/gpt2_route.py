"""Which parameters must a coefficient penalty reach to break the tiling? (GPT-2 Gated, beta = 0.1, pre-registered.)

The full test (gated_magl1) adds beta * sum|z| with z = 1[pi>0] * ReLU(pre * exp(r_mag) + b_enc), pre = (x - b_dec) W_enc^T.
That penalty's gradient reaches W_enc (shared with the gate) and the magnitude-only parameters b_enc and r_mag.
Two route-restricted variants keep the penalty VALUE identical and block one route each:
  gated_mag_rb : gradient only to the magnitude-only parameters (b_enc, r_mag): pre is detached inside the penalty.
  gated_mag_w  : gradient only through pre (W_enc, b_dec): exp(r_mag) and b_enc are detached inside the penalty.
Same seeds / init / data order / schedule / gate L1 5.62397 / auxiliary loss. Evaluated with the ORIGINAL run_cell plus
exact L0, alive, held-out content FVU and the massive-row allocation.
"""
import argparse
import os
import shutil
import time

import numpy as np
import torch

import gi_common as G
from gi_common import io, registry, S

import sys
sys.path.insert(0, G.HERE)
import exp_b1_architecture_law as B1                                     # noqa: E402
import alloc as A                                                       # noqa: E402
import gpt2_gated as GG                                                 # noqa: E402
from v3.common.data import load_model_and_tokenizer, eval_sequences     # noqa: E402
from v3.common.fastabl import tuned_chunk                               # noqa: E402


class _Route(S.GatedSAE):
    def __init__(self, d_model, d_sae, l1_coef=1e-3, beta=0.0, seed=0):
        super().__init__(d_model, d_sae, l1_coef=l1_coef, seed=seed)
        self.beta = float(beta)

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc.T
        self._pre_raw = pre
        return super().encode(x)


class GatedMagRB(_Route):
    arch = "gated_mag_rb"

    def sparsity_loss(self, z, pre):
        g = (self._pi_gate > 0).to(z.dtype)
        zp = g * torch.relu(self._pre_raw.detach() * self.r_mag.exp() + self.b_enc)
        return super().sparsity_loss(z, pre) + self.beta * zp.sum(-1).mean()


class GatedMagW(_Route):
    arch = "gated_mag_w"

    def sparsity_loss(self, z, pre):
        g = (self._pi_gate > 0).to(z.dtype)
        zp = g * torch.relu(self._pre_raw * self.r_mag.exp().detach() + self.b_enc.detach())
        return super().sparsity_loss(z, pre) + self.beta * zp.sum(-1).mean()


S.ARCHS["gated_mag_rb"] = GatedMagRB
S.ARCHS["gated_mag_w"] = GatedMagW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["gated_mag_rb", "gated_mag_w"], required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    a = ap.parse_args()
    spec = GG.GPT2
    t0 = time.time()
    st = G.store(spec)
    X = G.harvest_split(spec, "test", 200_000)
    args = argparse.Namespace(
        models=["gpt2"], archs=[a.arch], eval_seqs=registry.EVAL_SEQS_IMPORTANCE, topk=50, chunk=0, screen=1500,
        knob_min=1e-6, knob_max=1e5, recal_passes=0, target_l0=64.0, l0_tol=0.25, cal_steps=4000, force=True,
        seeds=[42, 137, 0], allow_uncalibrated=False, device="cuda", progress=500)
    model, tok = load_model_and_tokenizer(spec, device="cuda")
    seqs, am = eval_sequences(tok, n_seqs=args.eval_seqs)
    args.chunk = tuned_chunk(spec, vocab_size=int(model.config.vocab_size))
    akw = dict(l1_coef=GG.GATED["gpt2"], beta=a.beta)
    cell = B1.run_cell(model, spec, seqs, st["adir"], st["evecs"], a.arch, akw, "standard", args, am=am)
    npz = io.path("results", "gated_investigation", f"feat_gpt2_{a.arch}_beta{a.beta:g}.npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    shutil.copy(io.path("results", B1.EXP, f"feat_gpt2_{a.arch}_standard.npz"), npz)
    comp, top = GG.band_comp(npz)
    x = np.asarray(st["acts"][:G.N_EXACT], dtype=np.float32) - st["mean"].astype(np.float32)
    nrm = np.linalg.norm(x, axis=1)
    massive = nrm > 10 * np.median(nrm)
    del x
    per_seed = []
    for s in args.seeds:
        sae, _v, _m, _a, extra = io.get_sae(spec, s, arch=a.arch, condition="standard", device="cuda", **akw)
        fid, rate = G.store_fidelity(sae, st)
        r = dict(seed=s, **fid, **G.heldout_fidelity(sae, X, st["mean"], st["v1"]), train_log_tail=G.log_tail(extra))
        al, spec_mask, _rm, _ro = A.alloc(sae, st, massive, st["v1"])
        r["massive_rows"] = al
        if s == 42:
            r["top50_frac_massive_specific"] = float(spec_mask[top].mean())
        per_seed.append(r)
        del sae
        torch.cuda.empty_cache()
    G.dump(dict(role="route", arch=a.arch, beta=a.beta, l1_coef=GG.GATED["gpt2"], seeds=args.seeds, cell=cell, composition=comp,
                per_seed=per_seed, minutes=(time.time() - t0) / 60), f"gpt2_{a.arch}_beta{a.beta:g}.json")
    print(f"[{G.now()}] {a.arch} beta {a.beta:g}: top-50 {cell['top50_mean']:.3f} dict-wide {cell['dictwide_mean']:.3f} "
          f"L0 exact {np.mean([p['l0_exact'] for p in per_seed]):.2f} alive {np.mean([p['n_alive'] for p in per_seed]):.0f} "
          f"held-out content {np.mean([p['heldout_fvu_content'] for p in per_seed]):.4f} | massive atoms for 90% PC1 "
          f"{per_seed[0]['massive_rows']['massive_atoms_for_90pct_pc1_median']}", flush=True)


if __name__ == "__main__":
    main()
