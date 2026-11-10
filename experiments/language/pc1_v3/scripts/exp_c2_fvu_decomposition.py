#!/usr/bin/env python
"""Measure total reconstruction error and error outside PC1."""

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from v3.common import registry, io_utils as io                      # noqa: E402
from v3.common.data import open_activations                          # noqa: E402
from v3.common.pca import estimate_pcs, pc1_stats                    # noqa: E402

EXP = "c2_fvu_decomposition"


@torch.no_grad()
def decompose(sae, adir, v1, condition, device, n=100_000, batch=8192,
              slice_="tail"):
    acts, mean = open_activations(adir)
    total = acts.shape[0]
    n = int(min(n, total))
    start = total - n if slice_ == "tail" else 0
    mu = torch.tensor(np.asarray(mean), dtype=torch.float32, device=device)
    v1_t = torch.tensor(np.asarray(v1), dtype=torch.float32, device=device)
    v1d = v1_t.double()
    err = err1 = tgt = tgt1 = expl = expl1 = l0 = 0.0
    seen = 0
    for i in range(start, start + n, batch):
        x = torch.tensor(np.asarray(acts[i:min(i + batch, start + n)],
                                    dtype=np.float32), device=device) - mu
        coef = x @ v1_t
        if condition == "bypass":
            x_in = x - coef[:, None] * v1_t[None, :]
            recon, z = sae(x_in)
            recon = recon + coef[:, None] * v1_t[None, :]
            # what the DICTIONARY itself explains (PC1 is handed to it, so it
            # must not be credited for PC1): dictionary target is x_perp.
            dict_err, dict_tgt = (recon - x).double(), x_in.double()
        else:
            recon, z = sae(x)
            dict_err, dict_tgt = (recon - x).double(), x.double()
        e, xd = (recon - x).double(), x.double()
        err += e.pow(2).sum().item()
        err1 += (e @ v1d).pow(2).sum().item()
        tgt += xd.pow(2).sum().item()
        tgt1 += (xd @ v1d).pow(2).sum().item()
        # explained variance attributable to the dictionary output
        expl += (dict_tgt.pow(2).sum() - dict_err.pow(2).sum()).item()
        expl1 += ((dict_tgt @ v1d).pow(2).sum()
                  - (dict_err @ v1d).pow(2).sum()).item()
        l0 += (z != 0).float().sum(-1).sum().item()
        seen += x.shape[0]

    tgt_c, err_c = tgt - tgt1, err - err1
    w1 = tgt1 / max(tgt, 1e-12)
    fvu_total = err / max(tgt, 1e-12)
    fvu_pc1 = err1 / max(tgt1, 1e-12)
    fvu_content = err_c / max(tgt_c, 1e-12)
    out = dict(
        n_eval_tokens=seen, eval_slice=slice_, l0=l0 / max(seen, 1),
        pc1_var_weight=w1,
        fvu_total=fvu_total, fvu_pc1=fvu_pc1, fvu_content=fvu_content,
        fve_total=1 - fvu_total, fve_pc1=1 - fvu_pc1,
        fve_content=1 - fvu_content,
        # additive shares of TOTAL variance (these sum to fve_total exactly)
        fve_from_pc1=(tgt1 - err1) / max(tgt, 1e-12),
        fve_from_content=(tgt_c - err_c) / max(tgt, 1e-12),
        # share of explained variance living in v1.  For bypass the PC1
        # pass-through is excluded (dictionary credit only) -- otherwise the
        # bypass row trivially reads ~w1 and inverts the paper's message.
        # Undefined (nan) if the dictionary explains no variance at all, which
        # only happens for an untrained/degenerate SAE -- do NOT silently
        # divide by ~0.
        pc1_share_of_explained=(expl1 / expl if expl > 1e-6 * tgt
                                else float("nan")),
        dict_explained_var_frac=expl / max(tgt, 1e-12),
        pc1_share_of_error=err1 / max(err, 1e-12),
        # naive, NOT comparable across conditions (denominator differs) --
        # this is what train_sae logs and what a naive report would quote.
        fvu_own_input=(err / max(tgt_c, 1e-12) if condition == "bypass"
                       else fvu_total),
    )
    assert abs(w1 * fvu_pc1 + (1 - w1) * fvu_content - fvu_total) < \
        1e-6 * max(fvu_total, 1e-12) + 1e-9
    assert abs(out["fve_from_pc1"] + out["fve_from_content"]
               - out["fve_total"]) < 1e-9
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--include-70b", action="store_true")
    p.add_argument("--seeds", nargs="*", type=int,
                   default=[registry.REFERENCE_SEED],
                   help="A5 caches 5 seeds; pass them for error bars")
    p.add_argument("--eval-tokens", type=int, default=100_000)
    p.add_argument("--eval-slice", default="tail", choices=["tail", "head"],
                   help="tail = disjoint from the first-50k PCA sample")
    p.add_argument("--k-sweep", nargs="*", type=int, default=None,
                   help="standard-condition TopK sweep (e.g. 16 32 64 128): "
                        "gives the naive-FVU vs content-FVU RANKING table, "
                        "which is the claim that does not rest on the "
                        "cross-condition denominator. Trains missing SAEs.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    # The 70B guard used to live here because io.get_sae dropped device_map and
    # a 141 GB model landed on one card.  get_sae now forwards it, and C2 itself
    # is activation-space only -- with the store and the SAE already cached it
    # never loads the LM at all.  device_map only matters on a cold cache.

    rows = {}
    for spec in registry.iter_models(args.models,
                                     include_70b=args.include_70b):
        row = dict(seeds=args.seeds, conditions={})
        for condition in ["standard", "bypass"]:
            per_seed = {}
            for s in args.seeds:
                sae, v1, mean, adir, _ = io.get_sae(
                    spec, s, arch="topk", condition=condition,
                    device=args.device, k=registry.TOPK_K,
                    device_map=("auto" if spec.min_gpu_gb > 100 else None))
                per_seed[str(s)] = decompose(sae, adir, v1, condition,
                                             args.device, n=args.eval_tokens,
                                             slice_=args.eval_slice)
                if condition == "standard" and "R" not in row:
                    evals, evecs, _ = estimate_pcs(adir)
                    stats, _ = pc1_stats(evals, evecs, mean)
                    row["R"] = stats["R"]
                    row["pc1_var"] = stats["pc1_var_explained"]
            keys = [k for k, v in next(iter(per_seed.values())).items()
                    if isinstance(v, (int, float))]
            agg = {k: float(np.mean([per_seed[str(s)][k] for s in args.seeds]))
                   for k in keys}
            agg.update({k + "_std": float(np.std([per_seed[str(s)][k]
                                                  for s in args.seeds]))
                        for k in ("fvu_total", "fvu_content")})
            row["conditions"][condition] = dict(mean=agg, per_seed=per_seed)
        if args.k_sweep:
            sweep = {}
            for kk in args.k_sweep:
                sae, v1, mean, adir, _ = io.get_sae(
                    spec, args.seeds[0], arch="topk", condition="standard",
                    device=args.device, k=kk)
                sweep[str(kk)] = decompose(sae, adir, v1, "standard",
                                           args.device, n=args.eval_tokens,
                                           slice_=args.eval_slice)
            row["k_sweep_standard"] = sweep
        rows[spec.key] = row
        st = row["conditions"]["standard"]["mean"]
        by = row["conditions"]["bypass"]["mean"]
        print(f"[{spec.key}] R={row['R']:.1f} w1={st['pc1_var_weight']:.3f} | "
              f"std: FVE {st['fve_total']:.3f} content {st['fve_content']:.3f} "
              f"pc1-share-of-expl {st['pc1_share_of_explained']:.2f} | "
              f"byp: FVE {by['fve_total']:.3f} content {by['fve_content']:.3f} "
              f"(naive own-input FVU {by['fvu_own_input']:.3f} vs "
              f"{st['fvu_total']:.3f})", flush=True)
    io.write_json(EXP, "fve_table", rows)


if __name__ == "__main__":
    main()
