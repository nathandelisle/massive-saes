"""How each objective represents the massive-activation rows (GPT-2 and Pythia-1.4B, store rows 0..99,999).

Massive rows: ||x - mean|| > 10 x the median row norm (on GPT-2 these are the first-position rows that carry ~98% of
PC1 variance). For every checkpoint:
  * per atom: firing rate on massive vs ordinary rows, mean code on each; "massive-specific" atoms fire on >= 50% of
    massive rows and <= 0.1% of ordinary rows;
  * per massive row: how many atoms are active by PC1 class, the L1 cost (sum of codes) by class, and the fraction of
    the row's reconstructed PC1 coefficient each class carries; atoms needed for 90% of that coefficient;
  * cross-seed sharing (signed max cosine >= 0.7) of the massive-specific atoms;
  * for seed 42 with an archived reference ranking: what fraction of the top 50 is massive-specific.
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

import gi_common as G
from gi_common import registry

GPT2, P14 = registry.MODELS["gpt2"], registry.MODELS["pythia-1.4b"]
PREFIX = {"gpt2": "b1v2-gpt2", "pythia-1.4b": "b1v2-p14b"}


@torch.no_grad()
def alloc(sae, st, massive, v1):
    v = torch.tensor(v1, dtype=torch.float32, device="cuda")
    dv = (sae.W_dec.float() @ v)
    cos2 = (dv / sae.W_dec.float().norm(dim=1)) ** 2
    cls = torch.where(cos2 >= 0.3, 0, torch.where(cos2 >= 0.01, 1, 2))           # canonical, band, content
    d = sae.d_sae
    fm = torch.zeros(d, device="cuda", dtype=torch.float64); fo = torch.zeros_like(fm)
    zm = torch.zeros_like(fm); zo = torch.zeros_like(fm)
    per_row = []
    nm = no = 0
    start = 0
    for x in G._rows(st):
        n = x.shape[0]
        mk = torch.tensor(massive[start:start + n], device="cuda")
        start += n
        _hat, z = sae(x)
        act = (z != 0)
        fm += act[mk].double().sum(0); fo += act[~mk].double().sum(0)
        zm += z[mk].double().sum(0); zo += z[~mk].double().sum(0)
        nm += int(mk.sum()); no += int((~mk).sum())
        if mk.any():
            zz = z[mk]
            contrib = zz * dv[None, :]                                   # PC1 coefficient carried by each atom
            tot = contrib.sum(1, keepdim=True)
            for c in range(3):
                sel = (cls == c)
                per_row.append(torch.stack([torch.full((zz.shape[0],), float(c), device="cuda"),
                                            (zz[:, sel] != 0).float().sum(1), zz[:, sel].sum(1),
                                            contrib[:, sel].sum(1) / tot.squeeze(1)], 1).cpu())
            # atoms needed for 90% of the row's reconstructed PC1 coefficient (largest contributions first)
            sc = torch.sort(contrib * torch.sign(tot), dim=1, descending=True).values
            cum = sc.cumsum(1) / (tot.abs() + 1e-30)
            need = (cum < 0.9).sum(1) + 1
            per_row.append(torch.stack([torch.full((zz.shape[0],), 9.0, device="cuda"), need.float(),
                                        zz.sum(1), tot.squeeze(1)], 1).cpu())
    pr = torch.cat(per_row).numpy()
    out = dict(n_massive_rows=nm, n_ordinary_rows=no)
    names = {0: "canonical", 1: "band", 2: "content"}
    for c in range(3):
        m = pr[:, 0] == c
        out[f"massive_{names[c]}_active_atoms_mean"] = float(pr[m, 1].mean())
        out[f"massive_{names[c]}_l1_mean"] = float(pr[m, 2].mean())
        out[f"massive_{names[c]}_pc1_fraction_mean"] = float(np.nanmean(pr[m, 3]))
    m9 = pr[:, 0] == 9
    out["massive_atoms_for_90pct_pc1_median"] = float(np.median(pr[m9, 1]))
    out["massive_total_l1_mean"] = float(pr[m9, 2].mean())
    out["massive_reconstructed_pc1_coef_mean"] = float(pr[m9, 3].mean())
    rate_m, rate_o = (fm / max(nm, 1)).cpu().numpy(), (fo / max(no, 1)).cpu().numpy()
    spec_mask = (rate_m >= 0.5) & (rate_o <= 1e-3)
    out["n_massive_specific_atoms"] = int(spec_mask.sum())
    c2 = cos2.cpu().numpy()
    out["massive_specific_by_class"] = {names[c]: int((spec_mask & (G.classes(c2) == names[c])).sum()) for c in range(3)}
    return out, spec_mask, rate_m, rate_o


def run_model(spec, items, out):
    st = G.store(spec)
    x = np.asarray(st["acts"][:G.N_EXACT], dtype=np.float32) - st["mean"].astype(np.float32)
    nrm = np.linalg.norm(x, axis=1)
    massive = nrm > 10 * np.median(nrm)
    res = dict(massive_threshold=float(10 * np.median(nrm)), massive_row_fraction=float(massive.mean()),
               massive_share_of_pc1_energy=float(((x[massive] @ st["v1"]) ** 2).sum() / ((x @ st["v1"]) ** 2).sum()),
               models={})
    del x
    for label, (key_fn, seeds, npz) in items.items():
        decs, masks, recs = [], [], []
        for s in seeds:
            key = key_fn(s)
            if not G.s3_exists(key):
                print(f"[{G.now()}] {label} s{s}: checkpoint not in S3 ({key}); skipped", flush=True)
                continue
            p = G.fetch(key, os.path.join(G.io.ROOT, "alloc_saes", os.path.basename(key)))
            sae, extra, cfg = G.load_ckpt(p)
            r, spec_mask, rm, ro = alloc(sae, st, massive, st["v1"])
            r["seed"] = s
            decs.append(G.dec_np(sae)); masks.append(spec_mask); recs.append(r)
            del sae
            torch.cuda.empty_cache()
        if not recs:
            continue
        seeds = [r["seed"] for r in recs]
        # cross-seed sharing of massive-specific atoms and of all atoms
        shr = []
        for i, j in itertools.permutations(range(len(seeds)), 2):
            mc = G.max_cos(decs[i], decs[j])
            shr.append(dict(source=seeds[i], target=seeds[j],
                            massive_specific_sharing=float((mc[masks[i]] >= G.THR).mean()) if masks[i].any() else None))
        entry = dict(per_seed=recs, massive_specific_pairwise=shr)
        if npz and npz.startswith("s3:"):
            key = npz[3:]
            npz = G.fetch(key, os.path.join(G.io.ROOT, "alloc_npz", os.path.basename(key))) if G.s3_exists(key) else None
        if npz:
            z = np.load(os.path.join(G.ASSETS, npz)) if not os.path.isabs(npz) else np.load(npz)
            imp = z["imp_zero"].astype(float); imp[~z["alive"]] = -np.inf
            top = np.argsort(imp)[::-1][:50]
            if seeds[0] == 42:
                entry["top50_frac_massive_specific_s42"] = float(masks[0][top].mean())
        res["models"][label] = entry
        print(f"[{G.now()}] {spec.key} {label}: " + json.dumps({k: v for k, v in recs[0].items() if not isinstance(v, dict)})[:400], flush=True)
    out[spec.key] = res
    G.dump(out, "alloc.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", nargs="*", type=float, default=[0.03, 0.1, 0.3, 1.0])
    a = ap.parse_args()
    out = {}
    nm = lambda spec, arch, **kw: (lambda s: f"{PREFIX[spec.key]}/saes/{G.cache_name(spec, s, arch, **kw)}")
    items = {"gated": (nm(GPT2, "gated", l1_coef=5.62397), [42, 137, 0], "feat_gpt2_gated_standard.npz"),
             "relu_l1": (nm(GPT2, "relu_l1", l1_coef=5.06247), [42, 137, 0], "feat_gpt2_relu_l1_standard.npz"),
             "topk": (nm(GPT2, "topk", k=64), [42, 137, 0], "feat_gpt2_topk_standard.npz")}
    for b in a.betas:
        items[f"gated_magl1_beta{b:g}"] = ((lambda s, b=b: f"g2beta-{b:g}/saes/{G.cache_name(GPT2, s, 'gated_magl1', l1_coef=5.62397, beta=b)}"),
                                            [42, 137, 0], f"s3:g2beta-{b:g}/results/gated_investigation/feat_gpt2_gated_magl1_beta{b:g}.npz")
    run_model(GPT2, items, out)
    items = {"gated": (nm(P14, "gated", l1_coef=3.32599), [42, 137, 0], "feat_pythia-1.4b_gated_standard.npz"),
             "relu_l1": (nm(P14, "relu_l1", l1_coef=3.18648), [42, 137, 0], "feat_pythia-1.4b_relu_l1_standard.npz")}
    run_model(P14, items, out)


if __name__ == "__main__":
    main()
