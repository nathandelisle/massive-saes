"""Is the seed-0 Gated degradation on GPT-2 / Pythia-1.4B the same optimizer spike as Qwen's, and does the Qwen fix
(Adam lr 1e-4) change the Gated cell on these models?

1. 'gated_diag' seed 0: the ORIGINAL Gated trainer (lr 3e-4, bit-identical), logging loss terms and per-parameter
   gradient norms every 100 steps (same logger as qwen_fix.py).
2. 'gated_lr' seeds 42/137/0 at the given lr, otherwise the original objective, coefficient, init, data order and
   schedule; ORIGINAL run_cell; exact L0, alive, held-out content FVU, massive-row allocation, band composition.

usage: lr_cell.py --model gpt2|pythia-1.4b [--lr 1e-4]
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
import qwen_fix as QF                                                   # noqa: E402  (patches S.train_sae)
import exp_b1_architecture_law as B1                                     # noqa: E402
import alloc as A                                                       # noqa: E402
import gpt2_gated as GG                                                 # noqa: E402
import seed_swap as SW                                                  # noqa: E402
from v3.common.data import load_model_and_tokenizer, eval_sequences     # noqa: E402
from v3.common.fastabl import tuned_chunk                               # noqa: E402

COEF = {"gpt2": 5.62397, "pythia-1.4b": 3.32599}


def slim(diag):
    return [{k: v for k, v in r.items()} for r in (diag or [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()
    QF.LR_FIX = a.lr
    spec = registry.MODELS[a.model]
    coef = COEF[a.model]
    t0 = time.time()
    st = G.store(spec)
    X = G.harvest_split(spec, "test", 200_000)
    _geo, massive = SW.massive_geometry(st)
    name = f"lrcell_{a.model}_lr{a.lr:g}.json"
    out = dict(model=a.model, l1_coef=coef, lr=a.lr, runs=[])

    def measure(sae, label, extra):
        fid, _ = G.store_fidelity(sae, st)
        r = dict(label=label, **fid, **G.heldout_fidelity(sae, X, st["mean"], st["v1"]), train_log_tail=G.log_tail(extra))
        r["massive_rows"] = A.alloc(sae, st, massive, st["v1"])[0]
        _pt, tot = G.gate_stats(sae, st)
        r["gate_totals"] = tot
        print(f"[{G.now()}] {label}: L0 {fid['l0_exact']:.2f} alive {fid['n_alive']} content {fid['fvu_content']:.4f} held-out "
              f"{r['heldout_fvu_content']:.4f} aux/token {tot.get('aux_mse_per_token', float('nan')):.1f} massive atoms 90% "
              f"{r['massive_rows']['massive_atoms_for_90pct_pc1_median']:.0f}", flush=True)
        return r

    # 1. original trainer, seed 0, with the per-100-step trace
    sae, _v, _m, _a, extra = io.get_sae(spec, 0, arch="gated_diag", condition="standard", device="cuda", l1_coef=coef)
    r = measure(sae, "gated_diag s0 (original lr 3e-4)", extra)
    r["diag_every_100_steps"] = slim(QF.DIAG.get(("gated_diag", 0), {}).get("diag"))
    out["runs"].append(dict(arch="gated_diag", seed=0, lr=3e-4, **r))
    del sae
    torch.cuda.empty_cache()
    G.dump(out, name)

    # 2. the cell at the lower lr
    args = argparse.Namespace(
        models=[a.model], archs=["gated_lr"], eval_seqs=registry.EVAL_SEQS_IMPORTANCE, topk=50, chunk=0, screen=1500,
        knob_min=1e-6, knob_max=1e5, recal_passes=0, target_l0=64.0, l0_tol=0.25, cal_steps=4000, force=True,
        seeds=[42, 137, 0], allow_uncalibrated=False, device="cuda", progress=500)
    model, tok = load_model_and_tokenizer(spec, device="cuda")
    seqs, am = eval_sequences(tok, n_seqs=args.eval_seqs)
    args.chunk = tuned_chunk(spec, vocab_size=int(model.config.vocab_size))
    cell = B1.run_cell(model, spec, seqs, st["adir"], st["evecs"], "gated_lr", dict(l1_coef=coef), "standard", args, am=am)
    del model
    torch.cuda.empty_cache()
    npz = io.path("results", "gated_investigation", f"feat_{a.model}_gated_lr{a.lr:g}_standard.npz")
    shutil.copy(io.path("results", B1.EXP, f"feat_{spec.key}_gated_lr_standard.npz"), npz)
    comp, top = GG.band_comp(npz)
    out["cell"], out["composition"] = cell, comp
    for s in args.seeds:
        sae, _v, _m, _a, extra = io.get_sae(spec, s, arch="gated_lr", condition="standard", device="cuda", l1_coef=coef)
        r = measure(sae, f"gated_lr s{s} (lr {a.lr:g})", extra)
        r["diag_every_100_steps"] = slim(QF.DIAG.get(("gated_lr", s), {}).get("diag"))
        out["runs"].append(dict(arch="gated_lr", seed=s, lr=a.lr, **r))
        del sae
        torch.cuda.empty_cache()
    out["minutes"] = (time.time() - t0) / 60
    G.dump(out, name)
    print(f"[{G.now()}] {a.model} gated lr {a.lr:g}: top-50 {cell['top50_mean']:.3f} +/- {cell['top50_std']:.3f} dict-wide "
          f"{cell['dictwide_mean']:.3f} | {comp}", flush=True)


if __name__ == "__main__":
    main()
