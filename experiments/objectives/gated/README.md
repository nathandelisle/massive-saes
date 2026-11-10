# Gated SAEs

| Scripts | Purpose |
|---|---|
| `gpt2_gated.py`, `gpt2_route.py`, `alloc.py` | GPT-2 magnitude and feature-allocation measurements. |
| `p69_grid.py`, `p69_beta.py` | Pythia-6.9B calibration and magnitude penalties. |
| `qwen_diag.py`, `qwen_fix.py`, `train_one.py` | Qwen measurements and retraining. |
| `lr_cell.py`, `seed_swap.py` | Learning-rate and initialization comparisons. |
| `gi_common.py`, `exp_b1_architecture_law.py`, `v3/` | Training and checkpoint helpers. |
| `variants/` | Script variants for the Pythia and Qwen runs. |
| `assemble.py` | Build the 17 result tables. |

From the repository root:

```bash
python experiments/objectives/gated/assemble.py
```

Reads `results/additional/objectives/gated/raw/` and writes `generated/gated/`. Set `GATED_RAW` and `GATED_OUT` to use other folders.
