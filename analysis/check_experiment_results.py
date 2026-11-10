"""Check experiment measurements against the saved paper tables."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# Recompute correlations from the actual feature arrays.
expected = rows(ROOT / "results/tables/claim_numbers_ranking.csv")
standard = []
for row in expected:
    path = ROOT / "results/feature_rankings" / f"{row['model']}_{row['condition']}_s42.npz"
    with np.load(path) as data:
        score, shared = data["zero_ablation_score"], data["shared"]
        keep = data["alive"].astype(bool) & np.isfinite(score)
        rho = pd.Series(score[keep]).rank().corr(pd.Series(shared[keep]).rank())
    assert abs(rho - float(row["spearman_importance_vs_sharing"])) < 1e-12, path
    if row["condition"] == "standard":
        standard.append(rho)
print(f"Importance/sharing correlations: {len(expected)} rows match; standard range {min(standard):+.4f} to {max(standard):+.4f}.")

# Verify the recorded image-selection rule.
manifest = json.loads((ROOT / "experiments/vision/images.json").read_text())
assert (manifest["dataset"], manifest["split"], manifest["selection_seed"]) == ("uoft-cs/cifar100", "train", 0)
selected = sorted(np.random.default_rng(0).choice(50000, size=640, replace=False).tolist())
assert selected == [item["index"] for item in manifest["images"]]
print("Vision image list: all 640 CIFAR-100 indices match the recorded selection rule.")

# Rebuild the magnitude summary from its bundled intermediates.
with tempfile.TemporaryDirectory() as tmp:
    result = subprocess.run([sys.executable, str(ROOT / "experiments/objectives/magnitude/assemble.py")],
                            cwd=ROOT, env={**os.environ, "SEC5_OUT": tmp}, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    actual = (Path(tmp) / "sec5_sweeps.json").read_text()
    saved = (ROOT / "results/additional/objectives/five_seed_beta/sec5_sweeps.json").read_text()
    assert actual == saved, "Magnitude summary differs beyond line endings."
print("Magnitude summary: identical after normalizing Windows/Linux line endings.")

# Check the underlying magnitude measurements without downloading calibration arrays.
inputs = ROOT / "results/additional/objectives/five_seed_beta/inputs"
bridge = json.loads((inputs / "BRIDGE_RESULTS.json").read_text())
sweep = json.loads((inputs / "gpt2_b1_sweep.json").read_text())
assert bridge["protocol_id"] == sweep["sources"]["protocol_id"]
assert bridge["complete"] and not bridge["missing_runs"]
fidelity = {(x["condition"], float(x["beta"]), x["seed"], x["split"]): x for x in bridge["fidelity"]}
ranked = {(x["run"], x["split"]): x for x in bridge["per_seed"]
          if x["baseline"] == "model" and x["metric"] == "dce_sum" and x["top_n"] == 50 and x["threshold"] == 0.7}
assert len(fidelity) == len(ranked) == 80
assert len(sweep["per_seed"]) == 40
for row in sweep["per_seed"]:
    for split in ("test", "selection"):
        f = fidelity[row["condition"], row["beta"], row["seed"], split]
        p = ranked[row["run"], split]
        expected = dict(n_tokens=f["n_tokens"], top50_pairwise_sharing=p["sharing"]["pairwise"],
                        top50_any_other_seed=p["sharing"]["any_other_seed"],
                        top50_no_other_seed_count=p["sharing"]["no_other_seed_count"],
                        dictwide_pairwise_sharing=p["dictwide"]["pairwise"], l0=f["l0"],
                        fvu_total=f["fvu_total"], fvu_content=f["fvu_perp"],
                        fvu_pc1=f["fvu_pc1"], w1=f["w1"], splice_dce=f["splice_dce"])
        assert row[split] == expected, (row["run"], split)
print("Magnitude source results: all 880 per-run values match across 40 runs and two splits.")

# Rebuild crosscoder table rows from the run and seed-pair summaries.
result = subprocess.run([sys.executable, str(ROOT / "analysis/crosscoder_tables.py")],
                        cwd=ROOT, check=True, capture_output=True, text=True)
saved = (ROOT / "results/additional/crosscoders/appM_tables_regenerated.txt").read_text()
assert result.stdout == saved, "Crosscoder table rows differ."
print("Crosscoder tables: all regenerated rows match the saved summary.")

# Rebuild all ten released-SAE outputs, including bootstrap intervals.
base = ROOT / "results/additional/released_saes"
with tempfile.TemporaryDirectory() as tmp:
    result = subprocess.run([sys.executable, str(ROOT / "experiments/released_saes/collect_v2.py"),
                             "--out", tmp], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    checked = 0
    for path in sorted((base / "sec7_tables_v2").iterdir()):
        actual = Path(tmp) / path.name
        if path.suffix == ".json":
            assert json.loads(actual.read_text()) == json.loads(path.read_text()), path.name
        elif path.suffix == ".csv":
            expected_rows, actual_rows = rows(path), rows(actual)
            assert len(expected_rows) == len(actual_rows), path.name
            for expected, new in zip(expected_rows, actual_rows):
                assert expected.keys() == new.keys(), path.name
                for key in expected:
                    if key not in {"source", "source_file"}:
                        assert expected[key] == new[key], (path.name, key, expected[key], new[key])
        else:
            continue
        checked += 1
    assert checked == 10
print("Released-SAE tables: all ten outputs match, including intervals; local source-path columns are ignored.")

# Rebuild the Gated tables from per-run JSONs.
base = ROOT / "results/additional/objectives/gated"
with tempfile.TemporaryDirectory() as tmp:
    result = subprocess.run([sys.executable, str(ROOT / "experiments/objectives/gated/assemble.py")],
                            cwd=ROOT, env={**os.environ, "GATED_RAW": str(base / "raw"),
                                           "GATED_OUT": tmp, "GATED_FETCH": "0"},
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    saved_files = sorted(base.glob("*.csv"))
    assert len(saved_files) == 17
    assert {p.name for p in saved_files} == {p.name for p in Path(tmp).glob("*.csv")}
    for path in saved_files:
        expected, actual = rows(path), rows(Path(tmp) / path.name)
        assert len(expected) == len(actual), path.name
        for old, new in zip(expected, actual):
            assert old.keys() == new.keys(), path.name
            for key in old:
                a, b = old[key], new[key]
                if key == "source":
                    a, b = a.replace(chr(92), "/"), b.replace(chr(92), "/")
                assert a == b, (path.name, key, a, b)
print("Gated tables: all 17 match; Windows/Linux path separators are normalized.")
