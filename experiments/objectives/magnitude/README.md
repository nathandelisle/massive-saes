# Magnitude penalties

| Scripts | Purpose |
|---|---|
| `train/cloud/e4_real.py`, `train/experiments/` | Train the five-seed GPT-2 magnitude sweep. |
| `evaluate/pc1next/` | Measure importance, sharing, and reconstruction. |
| `build_gpt2.py`, `build_gpt2_geometry.py` | Build intermediate summaries. |
| `assemble.py` | Combine GPT-2 and Pythia measurements. |
| `download_calibration.py` | Download the 40 calibration arrays (415 MB). |
| `../../ten_seed_gpt2/pc1night/` | Train the native reference runs. |

From the repository root, build tables from the bundled summaries:

```bash
python experiments/objectives/magnitude/assemble.py
```

Inputs are in `results/additional/objectives/five_seed_beta/inputs/`; outputs go to `generated/magnitude/`. Run settings are in `training_config.json` and `evaluate/protocol.json`.

To rebuild the intermediate summaries:

```bash
python experiments/objectives/magnitude/download_calibration.py
python experiments/objectives/magnitude/build_gpt2.py
python experiments/objectives/magnitude/build_gpt2_geometry.py
```

The download uses the Hugging Face revision and file hashes in `calibration_files.json`.
