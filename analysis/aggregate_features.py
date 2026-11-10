"""Aggregate recalculate_features.py pair outputs (cohort v3core_rxzero).

Primary convention (paper Table 1 caption): source = reference seed 42, targets 137/0/1/2; report the
mean over the 4 pairs and the min-max range. These 4 pairs share the seed-42 dictionary, so they are
NOT independent replicates: no CI is computed over pairs. Sensitivity: mean over all 20 ordered pairs.
Rules kept distinct: posthoc = drop a_j >= 0.01 (cos^2 with PC1), posthoc_5x = drop cos^2 > 5x alive
median. Dense budget/subspace use the 5x-median rule. Importance = signed RX zero-ablation dCE.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

F = Path(__file__).resolve().parent / "features"
MODELS = ["gpt2", "pythia-70m-deduped", "pythia-160m", "pythia-1.4b", "pythia-6.9b", "gemma-2-2b", "gemma-2-9b",
          "llama-3.1-8b", "qwen-2.5-7b"]


FINAL = "--final" in sys.argv     # release mode: every required pair must exist and parse
EXPECT_REF, EXPECT_ALL = 4, 20     # per model x condition (5 seeds, ordered pairs)


def load(model, cond, src=None):
    out = []
    for p in sorted((F / model / "pairs").glob("%s_s*_vs_s*.json" % cond)):
        a = int(p.stem.split("_s")[1].split("_vs")[0])
        if src is None or a == src:
            try:
                out.append(json.loads(p.read_text()))
            except ValueError:
                if FINAL:
                    raise SystemExit("FINAL: malformed pair file %s" % p)
                print("WARNING: skipping malformed pair file (in-progress mode only)", p)
    if FINAL:
        need = EXPECT_REF if src is not None else EXPECT_ALL
        if len(out) != need:
            raise SystemExit("FINAL: %s %s src=%s has %d pairs, expected %d" % (model, cond, src, len(out), need))
    return out


def stat(rows, fn):
    v = [fn(r) for r in rows]
    v = [x for x in v if x is not None]
    if not v:
        return None, None, None
    return float(np.mean(v)), float(np.min(v)), float(np.max(v))


def g(r, *keys):
    for k in keys:
        if r is None or k not in r:
            return None
        r = r[k]
    return r


def main():
    rows = []
    for m in MODELS:
        for c in ("standard", "bypass"):
            ref, allp = load(m, c, 42), load(m, c)
            if not ref:
                continue
            row = dict(model=m, condition=c, n_ref_pairs=len(ref), n_all_pairs=len(allp),
                       exhaustive=all(r["exhaustive_alive_ranking"] for r in ref), n_alive_s42=ref[0]["n_alive"])
            for name, fn in [
                ("dict_sharing", lambda r: r["dictionary_sharing"]),
                ("top10", lambda r: g(r, "top", "10", "sharing")),
                ("top50", lambda r: g(r, "top", "50", "sharing")),
                ("top10_enrich", lambda r: g(r, "top", "10", "enrichment_over_alive_baseline")),
                ("top50_enrich", lambda r: g(r, "top", "50", "enrichment_over_alive_baseline")),
                ("posthoc_a001_top10", lambda r: g(r, "posthoc", "10", "sharing")),
                ("posthoc_a001_top50", lambda r: g(r, "posthoc", "50", "sharing")),
                ("posthoc_a001_top10_enrich", lambda r: g(r, "posthoc", "10", "enrichment_over_alive_baseline")),
                ("posthoc_5x_top50", lambda r: g(r, "posthoc_5x_median", "50", "sharing")),
                ("dense5x_n", lambda r: g(r, "dense_budget_5x_median", "n_selected")),
                ("dense5x_frac_alive", lambda r: g(r, "dense_budget_5x_median", "fraction_alive")),
                ("dense5x_signed_share", lambda r: g(r, "dense_budget_5x_median", "signed_importance_share")),
                ("dense5x_positive_share", lambda r: g(r, "dense_budget_5x_median", "positive_part_importance_share")),
                ("dense5x_signed_enrich", lambda r: g(r, "dense_budget_5x_median", "signed_share_over_dictionary_fraction")),
                ("dense5x_random_exceed", lambda r: g(r, "dense_budget_5x_median", "random_unit_direction_exceedance_at_observed_cutoff")),
                ("subspace_rank_a", lambda r: g(r, "subspace_5x_median", "rank_a")),
                ("subspace_mean_cos2", lambda r: g(r, "subspace_5x_median", "mean_cos2")),
                ("subspace_haar_null", lambda r: g(r, "subspace_5x_median", "haar_random_mean_cos2_expectation")),
                ("subspace_excess", lambda r: g(r, "subspace_5x_median", "excess_above_random_expectation")),
            ]:
                mu, lo, hi = stat(ref, fn)
                row[name] = mu
                if name in ("top10", "top50", "dict_sharing", "posthoc_a001_top50", "top10_enrich", "posthoc_a001_top10_enrich"):
                    row[name + "_min"], row[name + "_max"] = lo, hi
                    row[name + "_allpairs"] = stat(allp, fn)[0]
            rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(F.parent / "features_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(F.parent / "features_summary.json", "w"), indent=1)
    fmt = lambda x: "-" if x is None else ("%.3f" % x if isinstance(x, float) else str(x))
    cols = ["model", "condition", "n_alive_s42", "dict_sharing", "top10", "top50", "top10_enrich", "posthoc_a001_top50",
            "posthoc_5x_top50", "dense5x_n", "dense5x_frac_alive", "dense5x_signed_share", "dense5x_positive_share",
            "subspace_mean_cos2", "subspace_haar_null"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(fmt(r.get(c)) for c in cols))


if __name__ == "__main__":
    main()
