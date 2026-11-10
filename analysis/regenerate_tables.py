"""Recompute six summary CSVs from results/language and compare with results/tables. Uses NumPy and pandas; no downloads."""
import filecmp
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "results" / "language"
REF = ROOT / "results" / "tables"
CODE = ROOT / "analysis"
jobs = [("judge_closure.py", ["judge_cells_ci.csv", "judge_condition_contrasts.csv"]),
        ("mechanism_closure.py", ["mechanism_individual_vs_joint.csv", "mechanism_invalid_flags.csv"]),
        ("l0_closure.py", ["train_l0_corrected.csv"]),
        ("gradient_production_table.py", ["gradient_validation_production.csv"])]
bad = 0
with tempfile.TemporaryDirectory() as tmp:
    for script, outs in jobs:
        r = subprocess.run([sys.executable, str(CODE / script), str(PACK), tmp], capture_output=True, text=True)
        if r.returncode:
            print("FAIL", script, r.stderr[-500:]); bad += 1; continue
        for o in outs:
            same = filecmp.cmp(Path(tmp) / o, REF / o, shallow=False)
            bad += not same
            print("%-40s %s" % (o, "identical" if same else "DIFFERS"))
print("\n%d problems" % bad)
sys.exit(1 if bad else 0)
