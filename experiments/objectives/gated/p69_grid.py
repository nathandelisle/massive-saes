"""Pythia-6.9B Gated / standard: parallel full-length calibration grid plus the cell, one (coef, seed) per box.

Why: the archived calibration stopped on its flatness gate (l1 11.67 -> L0 41.7, 10.34 -> 43.8: 4.9% over a 12.9%
coefficient change). A run at l1 4.0 on 2026-09-18 was still at train-log L0 231 at step 12k, so the target lies
between 4 and ~10. Each box trains one (coefficient, seed) at full length (20,000 steps, original trainer).
The seed-42 box of each coefficient then waits for its sibling seeds (137, 0) from S3 and runs the ORIGINAL
exp_b1_architecture_law.run_cell, so every coefficient yields a complete cell. The coefficient used for the grid
is chosen afterwards on the exact L0 of seed 42 only (sparsity only, target 64); all cells are kept.

usage: python scripts/p69_grid.py --coef 7.0 --seed 42 [--cell --sibling-prefix p69g-c7-s]
"""
import argparse
import os
import shutil
import time

import numpy as np
import torch

import gi_common as G
from gi_common import io, registry

import sys
sys.path.insert(0, G.HERE)
import exp_b1_architecture_law as B1                                     # noqa: E402
from v3.common.data import load_model_and_tokenizer, eval_sequences     # noqa: E402
from v3.common.fastabl import tuned_chunk                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coef", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cell", action="store_true")
    ap.add_argument("--sibling-prefix", default="")
    ap.add_argument("--wait-min", type=float, default=150)
    a = ap.parse_args()
    spec = registry.MODELS["pythia-6.9b"]
    tag = f"c{a.coef:g}"
    t0 = time.time()
    st = G.store(spec)
    print(f"[{G.now()}] store ready ({(time.time()-t0)/60:.1f}m); w1 {st['w1']:.4f}", flush=True)
    sae, _v1, _m, _adir, extra = io.get_sae(spec, a.seed, arch="gated", condition="standard", device="cuda", l1_coef=a.coef)
    fid, _ = G.store_fidelity(sae, st)
    arch_l0, arch_alive = G.archived_l0(sae, st)
    path = io.sae_cache_path(spec, a.seed, arch="gated", condition="standard", l1_coef=a.coef)
    rec = dict(model="pythia-6.9b", arch="gated", condition="standard", l1_coef=a.coef, seed=a.seed,
               checkpoint=os.path.basename(path), checkpoint_sha256=G.sha256(path), train_log_tail=G.log_tail(extra),
               l0_archived_counter=arch_l0, n_alive_archived_counter=arch_alive, **fid,
               minutes_to_here=(time.time() - t0) / 60)
    G.dump(rec, f"p69_{tag}_s{a.seed}.json")
    print(f"[{G.now()}] coef {a.coef:g} seed {a.seed}: L0 exact {fid['l0_exact']:.2f} (archived counter {arch_l0:.2f}) "
          f"alive {fid['n_alive']} fvu {fid['fvu_total']:.5f} content {fid['fvu_content']:.4f}", flush=True)
    del sae
    torch.cuda.empty_cache()
    if not a.cell:
        return

    # ---- gather sibling seeds (fall back to training them here if they never arrive)
    need = {s: G.cache_name(spec, s, "gated", l1_coef=a.coef) for s in (137, 0)}
    deadline = time.time() + 60 * a.wait_min
    while True:
        missing = [s for s, h in need.items() if not os.path.exists(os.path.join(io.ROOT, "saes", h))]
        for s in list(missing):
            key = f"{a.sibling_prefix}{s}/saes/{need[s]}"
            if G.s3_exists(key):
                G.fetch(key, os.path.join(io.ROOT, "saes", need[s]))
                print(f"[{G.now()}] fetched sibling seed {s}", flush=True)
        missing = [s for s, h in need.items() if not os.path.exists(os.path.join(io.ROOT, "saes", h))]
        if not missing or time.time() > deadline:
            break
        time.sleep(60)
    if missing:
        print(f"[{G.now()}] siblings {missing} did not arrive; run_cell will train them here", flush=True)

    args = __import__("argparse").Namespace(
        models=["pythia-6.9b"], archs=["gated"], eval_seqs=registry.EVAL_SEQS_IMPORTANCE, topk=50, chunk=0, screen=1500,
        knob_min=1e-6, knob_max=1e5, recal_passes=0, target_l0=64.0, l0_tol=0.25, cal_steps=4000, force=True,
        seeds=[42, 137, 0], allow_uncalibrated=False, device="cuda", progress=500)
    model, tok = load_model_and_tokenizer(spec, device="cuda")
    seqs, am = eval_sequences(tok, n_seqs=args.eval_seqs)
    args.chunk = tuned_chunk(spec, vocab_size=int(model.config.vocab_size))
    akw = dict(l1_coef=a.coef)
    cell = B1.run_cell(model, spec, seqs, st["adir"], st["evecs"], "gated", akw, "standard", args, am=am)
    src = io.path("results", B1.EXP, f"feat_{spec.key}_gated_standard.npz")
    shutil.copy(src, io.path("results", "gated_investigation", f"feat_pythia-6.9b_gated_standard_{tag}.npz"))
    per_seed = []
    for s in args.seeds:
        sae, _v1, _m, _a, ex = io.get_sae(spec, s, arch="gated", condition="standard", device="cuda", **akw)
        f, _ = G.store_fidelity(sae, st)
        per_seed.append(dict(seed=s, **f, train_log_tail=G.log_tail(ex)))
        del sae
        torch.cuda.empty_cache()
    G.dump(dict(model="pythia-6.9b", arch="gated", condition="standard", l1_coef=a.coef, seeds=args.seeds,
                cell=cell, per_seed_exact=per_seed, siblings_trained_here=missing), f"p69_{tag}_cell.json")
    print(f"[{G.now()}] cell coef {a.coef:g}: top-50 {cell['top50_mean']:.3f} dict-wide {cell['dictwide_mean']:.3f} "
          f"L0 exact {np.mean([p['l0_exact'] for p in per_seed]):.2f}", flush=True)


if __name__ == "__main__":
    main()
