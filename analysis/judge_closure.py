"""Summarize automated feature detection and bootstrap confidence intervals."""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACK, OUT = Path(sys.argv[1]), Path(sys.argv[2])
j = pd.read_csv(PACK / "analysis" / "judge_per_feature.csv")
j["model"] = j["dict"].str.split(":").str[0]
j["condition"] = j["dict"].str.split(":").str[1]
j["orientation"] = j["explainer"] + "_explains/" + j["judge"] + "_judges"
assert set(j["membership"].unique()) <= {"top", "random", "top,random"}, j["membership"].unique()
j["in_top"] = j["membership"].str.split(",").apply(lambda x: "top" in x)
j["in_random"] = j["membership"].str.split(",").apply(lambda x: "random" in x)
rng = np.random.default_rng(20261010)
NB = 2000
NEGS = ("random", "hard")


def ci(v):
    return [round(float(x), 4) for x in np.quantile(v, [0.025, 0.975])]


def boot_cell(g):
    """-> {(membership, neg): (point, replicates)}; stratified, dependence-preserving resampling."""
    strata = [g[g.in_top & ~g.in_random], g[~g.in_top & g.in_random], g[g.in_top & g.in_random]]
    out = {}
    for neg in NEGS:
        col = "auroc_%s_neg" % neg
        vals = [s[col].to_numpy(float) for s in strata]
        top_sum, rnd_sum = np.zeros(NB), np.zeros(NB)
        top_n, rnd_n = len(vals[0]) + len(vals[2]), len(vals[1]) + len(vals[2])
        for k, v in enumerate(vals):
            if len(v) == 0:
                continue
            draw = v[rng.integers(0, len(v), (NB, len(v)))].sum(1)
            if k in (0, 2):
                top_sum += draw
            if k in (1, 2):
                rnd_sum += draw
        out[("top", neg)] = (float(g[g.in_top][col].mean()), top_sum / top_n)
        out[("random", neg)] = (float(g[g.in_random][col].mean()), rnd_sum / rnd_n)
    return out


cells, boots = [], {}
for (m, c, o), g0 in j.groupby(["model", "condition", "orientation"]):
    for mem in ("top", "random"):
        n = int(g0["in_%s" % mem].sum())
        assert n == 100, (m, c, o, mem, n)
    for excl in (False, True):
        g = g0[g0.parse_ok] if excl else g0
        b = boot_cell(g)
        if not excl:
            boots[(m, c, o)] = b
        r = dict(model=m, condition=c, orientation=o, excluding_parse_failures=excl,
                 n_dual_membership=int((g.in_top & g.in_random).sum()))
        for mem in ("top", "random"):
            x = g[g["in_%s" % mem]]
            x0 = g0[g0["in_%s" % mem]]
            r["n_%s" % mem] = len(x)
            r["n_%s_parse_fail" % mem] = int((~x0.parse_ok).sum())
            r["n_%s_no_pattern" % mem] = int((x0.no_pattern == True).sum())  # noqa: E712
            r["n_%s_dense_flag_gt50pct_firing" % mem] = int(x.dense_flag.sum())
            for neg in NEGS:
                pt, bs = b[(mem, neg)]
                r["auroc_%s_neg_%s" % (neg, mem)] = pt
                r["auroc_%s_neg_%s_ci95" % (neg, mem)] = ci(bs)
        for neg in NEGS:
            d = b[("top", neg)][1] - b[("random", neg)][1]
            r["top_minus_random_%s_neg" % neg] = b[("top", neg)][0] - b[("random", neg)][0]
            r["top_minus_random_%s_neg_ci95" % neg] = ci(d)
        cells.append(r)
cdf = pd.DataFrame(cells)

# the corrected means must reproduce the original judge_summary.csv (all features, no exclusion)
s = pd.read_csv(PACK / "analysis" / "judge_summary.csv")
s["orientation"] = s["explainer"] + "_explains/" + s["explainer"].map({"qwen": "mistral", "mistral": "qwen"}) + "_judges"
full = cdf[~cdf.excluding_parse_failures]
maxdiff = 0.0
for _, r in s.iterrows():
    row = full[(full.model == r.model) & (full.condition == r.condition) & (full.orientation == r.orientation)].iloc[0]
    assert int(r.n) == row["n_%s" % r.membership], (r.model, r.condition, r.membership)
    for neg in NEGS:
        maxdiff = max(maxdiff, abs(row["auroc_%s_neg_%s" % (neg, r.membership)] - r["auroc_%s_mean" % neg]))
assert maxdiff < 1e-9, maxdiff
print("matches judge_summary.csv: %d rows, max |diff| = %.2e" % (len(s), maxdiff))

contrasts = []
for (m, o) in sorted({(k[0], k[2]) for k in boots}):
    S, Bp = boots.get((m, "standard", o)), boots.get((m, "bypass", o))
    if S is None or Bp is None:
        continue
    for neg in NEGS:
        for mem in ("top", "random"):
            d = Bp[(mem, neg)][1] - S[(mem, neg)][1]
            contrasts.append(dict(model=m, orientation=o, negatives=neg, quantity="%s AUROC" % mem,
                                  standard=S[(mem, neg)][0], bypass=Bp[(mem, neg)][0],
                                  bypass_minus_standard=Bp[(mem, neg)][0] - S[(mem, neg)][0], ci95=ci(d)))
        gs = S[("top", neg)][1] - S[("random", neg)][1]
        gb = Bp[("top", neg)][1] - Bp[("random", neg)][1]
        contrasts.append(dict(model=m, orientation=o, negatives=neg, quantity="top-minus-random gap",
                              standard=S[("top", neg)][0] - S[("random", neg)][0],
                              bypass=Bp[("top", neg)][0] - Bp[("random", neg)][0],
                              bypass_minus_standard=(Bp[("top", neg)][0] - Bp[("random", neg)][0])
                              - (S[("top", neg)][0] - S[("random", neg)][0]), ci95=ci(gb - gs)))


def write(name, rows):
    rows = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else rows
    keys = list(dict.fromkeys(k for x in rows for k in x))
    tmp = OUT / (name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(OUT / name)


write("judge_cells_ci.csv", cdf)
write("judge_condition_contrasts.csv", contrasts)
print(len(cdf), "cells", len(contrasts), "contrasts")
