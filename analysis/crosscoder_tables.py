"""Regenerate the Appendix M numbers (Tables 62, 63, 64) from results/tables/C_runs.csv and C_seed_pair_sharing.csv.
Usage: python analysis/crosscoder_tables.py [tables_dir]"""
import collections
import csv
import statistics as st
import sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "additional" / "crosscoders")
runs = list(csv.DictReader(open(D / "C_runs.csv")))
pairs = list(csv.DictReader(open(D / "C_seed_pair_sharing.csv")))
NAME = dict(C0="Identical", C1="PC1 doubled", C3a="Layers 12 and 16", C3b="Step 71,000 and final")


def mean_by(rows, key, col, scale=1.0):
    g = collections.defaultdict(list)
    for r in rows:
        g[key(r)].append(float(r[col]) * scale)
    return {k: (st.mean(v), len(v)) for k, v in g.items()}


top = mean_by(pairs, lambda r: (r["group"], r["arm"]), "native_top50_reconstruction_ablation_shared_signed", 100)
dic = mean_by(pairs, lambda r: (r["group"], r["arm"]), "native_dictionary_shared_signed", 100)
fvu = mean_by([r for r in runs if r["arm"].startswith("X")], lambda r: (r["condition"], r["arm"]), "content_fvu")
print("Table 62: top-50 sharing (%), native (original-unit) decoder cosine > 0.7, mean over 6 directed seed pairs")
print("pair | X0 X1 X2 X3 X4")
for c in NAME:
    print(NAME[c], "|", " ".join("%.1f" % top[(c, a)][0] for a in ("X0", "X1", "X2", "X3", "X4")))
print("\nTable 63: dictionary sharing (%) and content FVU (mean of 3 seeds)")
print("pair | Dict X0 | Dict X2 | FVU X0 | FVU X2")
for c in NAME:
    print(NAME[c], "| %.1f | %.1f | %.4f | %.4f" % (dic[(c, "X0")][0], dic[(c, "X2")][0], fvu[(c, "X0")][0], fvu[(c, "X2")][0]),
          "  (unrounded FVU: %.6f, %.6f)" % (fvu[(c, "X0")][0], fvu[(c, "X2")][0]))
print("\nTable 64: ReLU-L1 crosscoders, seed 42 (A-only/B-only % are block_sum)")
idx = {(r["condition"], r["arm"]): r for r in runs}
for c in ("C3a", "C3b"):
    for l1 in ("0.001", "0.003", "0.01"):
        b, j = idx[(c, "L1block_sum_" + l1)], idx[(c, "L1joint_" + l1)]
        print(NAME[c], l1, "| L0 block_sum %.1f | L0 joint %.1f | A-only %.2f | B-only %.2f" % (
            float(b["mean_l0"]), float(j["mean_l0"]), 100 * float(b["frac_A_only"]), 100 * float(b["frac_B_only"])))
print("\nExclusive-feature check (paper: joint/TopK none; ReLU none on C0/C1):")
for r in runs:
    a, bb = float(r["frac_A_only"] or 0), float(r["frac_B_only"] or 0)
    if a or bb:
        print("  nonzero:", r["condition"], r["arm"], r["seed"], a, bb)
