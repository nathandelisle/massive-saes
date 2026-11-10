"""Compute ranking, importance, and behavioral summaries from per-feature arrays.
Usage: python claim_numbers.py RANKS FEATURES LANGUAGE_RESULTS OUTPUT"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FULL, FEAT, PACK, OUT = map(Path, sys.argv[1:5])
RAW = PACK / "raw"
MODELS = ["gpt2", "pythia-70m-deduped", "pythia-160m", "pythia-1.4b", "pythia-6.9b", "gemma-2-2b", "gemma-2-9b",
          "llama-3.1-8b", "qwen-2.5-7b"]
res = {}


def top(score, ok, k):
    ix = np.flatnonzero(ok & np.isfinite(score))
    return ix[np.lexsort((ix, -score[ix]))][:k]


rows = []
for m in MODELS:
    for c in ("standard", "bypass"):
        z = np.load(FULL / ("%s_%s" % (m, c)) / "rank_zero.npz")
        s = z["score"]
        P = np.load(FEAT / m / ("shared_%s_s42.npz" % c))
        sh, al = P["shared"], P["alive"].astype(bool)
        ok = al & np.isfinite(s)
        base = sh[al].mean()
        # L119: rank correlation between importance and reproducibility over all alive features
        rho = spearmanr(s[ok], sh[ok]).correlation
        # L121: smallest N at which top-N sharing reaches the dictionary-wide baseline
        order = top(s, al, int(ok.sum()))
        cum = np.cumsum(sh[order]) / np.arange(1, len(order) + 1)
        cross = int(np.argmax(cum >= base) + 1) if (cum >= base).any() else None
        r = dict(model=m, condition=c, spearman_importance_vs_sharing=float(rho), N_first_at_or_above_base=cross,
                 top50=float(sh[order[:50]].mean()), dict_base=float(base),
                 unmatched_of_top50=float(50 * (1 - sh[order[:50]].mean())), unmatched_of_random50=float(50 * (1 - base)),
                 max_single_feature_dCE_nats=float(np.nanmax(s[ok])),
                 max_single_feature_ppl_increase_pct=float(100 * (np.exp(np.nanmax(s[ok])) - 1)),
                 base_ce=float(z["base_ce"]), mean_dCE_top50=float(s[order[:50]].mean()))
        mech = list(RAW.glob("rx-wsx-%s-%s-s42*/wsx/%s/%s_s42/mechanism.npz" % (m, c, m, c)))
        if mech and c == "standard":
            cos2 = np.load(mech[0])["pc1_cos2"]
            ex = al & (cos2 >= 0.01)
            r["posthoc_excluded_frac_alive"] = float(ex.sum() / al.sum())
            ph = top(np.where(ex, np.nan, s), al, 50)
            r["posthoc_top50_shared"] = float(sh[ph].mean())
            r["posthoc_top50_mean_dCE"] = float(s[ph].mean())
        rows.append(r)
pd.DataFrame(rows).to_csv(OUT / "claim_numbers_ranking.csv", index=False)

# behavioural (SVA, primary endpoint) means over available seeds
pp = pd.read_csv(OUT / "patching_primary.csv")
q = pp[(pp.cohort == "eligible") & pp.pool.isin(["standard", "posthoc", "bypass"])]
beh = q.groupby(["task", "model", "pool"]).agg(n_seeds=("seed", "nunique"), joint_mean=("joint_success", "mean"),
                                              joint_min=("joint_success", "min"), joint_max=("joint_success", "max")).reset_index()
s42 = q[q.seed == 42][["task", "model", "pool", "joint_success"]].rename(columns={"joint_success": "joint_seed42"})
beh = beh.merge(s42, on=["task", "model", "pool"], how="left")
beh.to_csv(OUT / "claim_numbers_behavioral.csv", index=False)

# objective grid at the canonical site (A4), 5-seed mean over sources, with corrected L0
t = pd.read_csv(OUT / "train_l0_corrected.csv")
g = t[t.kind == "a4"].pivot_table(index="model", columns=["arch", "condition"], values="top50_shared")
g.to_csv(OUT / "claim_numbers_objective_grid_top50.csv")
d = t[t.kind == "a4"].pivot_table(index="model", columns=["arch", "condition"], values="dictwide")
d.to_csv(OUT / "claim_numbers_objective_grid_dictwide.csv")
print(pd.DataFrame(rows)[["model", "condition", "spearman_importance_vs_sharing", "N_first_at_or_above_base",
                          "unmatched_of_top50", "unmatched_of_random50", "max_single_feature_ppl_increase_pct",
                          "mean_dCE_top50", "posthoc_excluded_frac_alive", "posthoc_top50_mean_dCE"]].round(3).to_string(index=False))
print(beh[beh.task == "sva"].round(3).to_string(index=False))
print((g * 100).round(1).to_string())
