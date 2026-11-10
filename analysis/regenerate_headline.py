"""Recompute seed-42 top-feature and dictionary sharing from results/feature_rankings. Compare with results/tables/features_summary.csv. Uses NumPy; no downloads."""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RANK = ROOT / "results" / "feature_rankings"
REF = ROOT / "results" / "tables" / "features_summary.csv"


def top(score, ok, k):
    ix = np.flatnonzero(ok & np.isfinite(score))
    return ix[np.lexsort((ix, -score[ix]))][:k]


ref = {(r["model"], r["condition"]): r for r in csv.DictReader(open(REF))}
fails = 0
print("%-20s %-9s %8s %8s %8s %8s %8s  %s" % ("model", "cond", "top10", "top50", "dict", "ref50", "refdict", "status"))
for f in sorted(RANK.glob("*_s42.npz")):
    model, cond = f.stem[:-4].rsplit("_", 1)
    z = np.load(f)
    s, al, sh = z["zero_ablation_score"], z["alive"].astype(bool), z["shared"]
    t = top(s, al, 50)
    t50, t10, base = sh[t].mean(), sh[t[:10]].mean(), sh[al].mean()
    r = ref[(model, cond)]
    ok = abs(t50 - float(r["top50"])) < 1e-9 and abs(base - float(r["dict_sharing"])) < 1e-9 and abs(t10 - float(r["top10"])) < 1e-9
    fails += not ok
    print("%-20s %-9s %8.3f %8.3f %8.3f %8.3f %8.3f  %s" % (model, cond, t10, t50, base, float(r["top50"]),
                                                          float(r["dict_sharing"]), "OK" if ok else "MISMATCH"))
print("\n%d mismatches" % fails)
sys.exit(1 if fails else 0)
