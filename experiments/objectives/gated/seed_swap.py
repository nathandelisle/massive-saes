"""Seed 0 is the weak Gated seed in GPT-2, Pythia-1.4B and Qwen-2.5-7B. Does the damage follow the SAE initialization
or the data order? The ORIGINAL trainer (v3 saes.train_sae) seeds both from one integer; here the two are split:
init seed A (torch.manual_seed + decoder/encoder init) and data seed B (activation_batches order). Same objective,
coefficient, schedule and store as the archived cell.

usage: seed_swap.py --model gpt2|pythia-1.4b|qwen-2.5-7b --pairs 0:42 42:0 0:0
Also records the geometry of the massive-activation rows of the store (for the GPT-2 vs Pythia-1.4B comparison).
"""
import argparse
import os
import time

import numpy as np
import torch

import gi_common as G
from gi_common import io, registry, S

import sys
sys.path.insert(0, G.HERE)
import alloc as A                                                       # noqa: E402
from v3.common.data import activation_batches                           # noqa: E402

COEF = {"gpt2": 5.62397, "pythia-1.4b": 3.32599, "qwen-2.5-7b": 5.64369}
PREFIX = {"gpt2": "b1v2-gpt2", "pythia-1.4b": "b1v2-p14b"}


@torch.no_grad()
def massive_geometry(st):
    x = np.asarray(st["acts"][:G.N_EXACT], dtype=np.float32) - st["mean"].astype(np.float32)
    nrm = np.linalg.norm(x, axis=1)
    med = float(np.median(nrm))
    mk = nrm > 10 * med
    M = x[mk]
    mbar = M.mean(0)
    cos_mean = (M @ mbar) / (np.linalg.norm(M, axis=1) * np.linalg.norm(mbar) + 1e-30)
    s_unc = np.linalg.svd(M, compute_uv=False) ** 2
    s_cen = np.linalg.svd(M - mbar, compute_uv=False) ** 2
    pc1 = M @ st["v1"]
    out = dict(n_rows=int(len(x)), n_massive=int(mk.sum()), massive_fraction=float(mk.mean()), median_row_norm=med,
               massive_norm_median=float(np.median(nrm[mk])), massive_norm_over_median=float(np.median(nrm[mk]) / med),
               cos_to_massive_mean_median=float(np.median(cos_mean)), cos_to_massive_mean_p10=float(np.percentile(cos_mean, 10)),
               uncentered_energy_top1=float(s_unc[0] / s_unc.sum()), uncentered_energy_top5=float(s_unc[:5].sum() / s_unc.sum()),
               centered_var_top1=float(s_cen[0] / s_cen.sum()), centered_var_top5=float(s_cen[:5].sum() / s_cen.sum()),
               centered_participation_ratio=float(s_cen.sum() ** 2 / (s_cen ** 2).sum()),
               within_massive_var_over_mean_sq=float(s_cen.sum() / len(M) / (np.linalg.norm(mbar) ** 2)),
               pc1_coef_massive_median=float(np.median(pc1)), pc1_coef_massive_cv=float(np.std(pc1) / (abs(np.mean(pc1)) + 1e-30)),
               pc1_coef_over_rms_norm=float(abs(np.median(pc1)) / np.sqrt((nrm ** 2).mean())))
    del x, M
    return out, mk


def measure(sae, st, X, massive, label, extra=None):
    fid, _ = G.store_fidelity(sae, st)
    r = dict(label=label, **fid, **G.heldout_fidelity(sae, X, st["mean"], st["v1"]))
    al, _spec, _rm, _ro = A.alloc(sae, st, massive, st["v1"])
    r["massive_rows"] = al
    r["init_similarity"] = None
    if extra is not None:
        r["train_log_tail"] = G.log_tail(extra)
    print(f"[{G.now()}] {label}: L0 {fid['l0_exact']:.2f} alive {fid['n_alive']} content {fid['fvu_content']:.4f} "
          f"held-out content {r['heldout_fvu_content']:.4f} massive atoms for 90% PC1 {al['massive_atoms_for_90pct_pc1_median']:.0f} "
          f"massive L1 {al['massive_total_l1_mean']:.0f}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", nargs="+", default=["0:42", "42:0", "0:0"])
    ap.add_argument("--no-archived", action="store_true")
    a = ap.parse_args()
    spec = registry.MODELS[a.model]
    coef = COEF[a.model]
    t0 = time.time()
    st = G.store(spec)
    X = G.harvest_split(spec, "test", 200_000)
    geo, massive = massive_geometry(st)
    print(f"[{G.now()}] massive geometry: {geo}", flush=True)
    out = dict(model=a.model, l1_coef=coef, massive_geometry=geo, runs=[])
    tag = "_".join(p.replace(":", "d") for p in a.pairs)
    name = f"seedswap_{a.model}_{tag}.json"
    if not a.no_archived and a.model in PREFIX:
        for s in (0, 42):
            key = f"{PREFIX[a.model]}/saes/{G.cache_name(spec, s, 'gated', l1_coef=coef)}"
            p = G.fetch(key, os.path.join(io.ROOT, "arch_saes", os.path.basename(key)))
            sae, extra, _cfg = G.load_ckpt(p)
            out["runs"].append(dict(init_seed=s, data_seed=s, source="archived", **measure(sae, st, X, massive, f"archived i{s} d{s}", extra)))
            del sae
            torch.cuda.empty_cache()
            G.dump(out, name)
    for pr in a.pairs:
        i, d = (int(v) for v in pr.split(":"))
        path = io.path("saes", f"seedswap_{a.model}_gated_i{i}_d{d}.pt")
        if not os.path.exists(path):
            batches = activation_batches(st["adir"], registry.BATCH_SIZE, registry.STEPS, d, device="cuda")
            sae, log = S.train_sae(batches, spec.d_model, spec.d_sae, arch="gated", seed=i, device="cuda",
                                   tag=f"{spec.key}/gated/i{i}/d{d}", total_steps=registry.STEPS, l1_coef=coef)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            S.save_sae(sae, path + ".tmp", extra=dict(cfg=dict(model=spec.key, init_seed=i, data_seed=d, arch="gated"), log=log[-3:]))
            os.replace(path + ".tmp", path)
        sae, extra = S.load_sae(path, device="cuda")
        out["runs"].append(dict(init_seed=i, data_seed=d, source="trained_here", **measure(sae, st, X, massive, f"trained i{i} d{d}", extra)))
        del sae
        torch.cuda.empty_cache()
        G.dump(out, name)
    out["minutes"] = (time.time() - t0) / 60
    G.dump(out, name)


if __name__ == "__main__":
    main()
