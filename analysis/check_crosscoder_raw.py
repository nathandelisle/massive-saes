"""Compare the crosscoder run and pair summaries with their saved JSON inputs."""
import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "results/additional/crosscoders"


def rows(name):
    with (BASE / name).open(newline="") as f:
        return list(csv.DictReader(f))


def check(saved, measured, identity):
    for key, value in measured.items():
        actual = saved[key]
        if isinstance(value, (int, float)):
            assert float(actual) == value, (identity, key, actual, value)
        else:
            assert actual == str(value), (identity, key, actual, value)


metrics = {}
for path in sorted((BASE / "raw").glob("*/results/extC/*/*/*/test_best_metrics.json")):
    model, condition, run = path.parts[-4:-1]
    arm, seed = run.rsplit("_s", 1)
    identity = model, condition, arm, seed
    assert identity not in metrics, identity
    metrics[identity] = json.loads(path.read_text())

saved_runs = rows("C_runs.csv")
assert len(metrics) == len(saved_runs) == 84
for saved in saved_runs:
    identity = tuple(saved[k] for k in ("model", "condition", "arm", "seed"))
    m = metrics[identity]
    measured = {k: m["config"][k] for k in ("variant", "activation", "penalty", "l1", "scale_mode")}
    for column, key in {
        "selected_step": "selected_step", "test_tokens": "n_tokens",
        "total_mse": "total_mse_native", "content_mse": "content_mse_native",
        "total_fvu": "total_fvu_native", "content_fvu": "content_fvu_native",
        "mean_l0": "mean_l0", "alive": "alive_features", "n_alive": "alive_features",
        "dense_pc_mse_rel": "dense_branch_pc_mse_over_train_centered_energy",
        "pc_tail_cluster_active_fraction": "pc_tail_cluster_active_fraction",
    }.items():
        measured[column] = m[key]
    for block in m["block_metrics"]:
        measured[f"stream{block['block']}_fvu"] = block["fvu"]
        measured[f"stream{block['block']}_content_fvu"] = block["content_fvu"]
    check(saved, measured, identity)

pairs = {}
for path in sorted((BASE / "raw").glob("*/results/extC/*/*/compare/*_s*_vs_s*.json")):
    model, condition = path.parts[-4:-2]
    arm, rest = path.stem.split("_s", 1)
    seed_a, seed_b = rest.split("_vs_s")
    identity = model, condition, arm, seed_a, seed_b
    assert identity not in pairs, identity
    pairs[identity] = json.loads(path.read_text())

saved_pairs = rows("C_seed_pair_sharing.csv")
assert len(pairs) == len(saved_pairs) == 120
for saved in saved_pairs:
    identity = tuple(saved[k] for k in ("model", "group", "arm", "seed_a", "seed_b"))
    measured = {f"{space}_{key}": value for space in ("objective", "native")
                for key, value in pairs[identity][space].items()}
    check(saved, measured, identity)

print("Crosscoder raw results: 84 run summaries and 120 seed-pair comparisons match.")
print("Exclusivity inputs: separate feature arrays.")
