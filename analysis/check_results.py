"""Run the offline result checks from any working directory."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
scripts = ["regenerate_headline.py", "regenerate_tables.py", "llama70b_numbers.py", "sec4_tile_band_stats.py", "check_experiment_results.py", "check_crosscoder_raw.py"]
failed = []
for name in scripts:
    print(f"\n{name}", flush=True)
    result = subprocess.run([sys.executable, str(HERE / name)], cwd=HERE.parent)
    if result.returncode:
        failed.append(name)
if failed:
    raise SystemExit("Failed: " + ", ".join(failed))
print("\nAll checks passed.")
