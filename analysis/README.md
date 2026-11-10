# Analysis

Run from the repository root:

```bash
python analysis/check_results.py
```

| Script | Checks or outputs |
|---|---|
| `regenerate_headline.py` | Feature-sharing summary. |
| `regenerate_tables.py` | Six summary CSVs. |
| `llama70b_numbers.py` | Two-seed Llama-70B comparison. |
| `sec4_tile_band_stats.py` | PC1-alignment counts. |
| `check_experiment_results.py` | Correlations, image selection, magnitude, Gated, released-SAE, and crosscoder tables. |
| `check_crosscoder_raw.py` | Crosscoder reconstruction and sharing summaries against per-run measurements. |
| `judge_closure.py` | Feature-detection means and intervals. |
| `mechanism_closure.py` | Individual and joint component ablations. |
| `l0_closure.py` | Mean active-feature counts. |
| `gradient_production_table.py` | Gradient and integration checks. |
| `importance_closure.py` | Sharing under each importance measure. |
| `patching_closure.py` | Behavioral comparisons on shared test examples. |
| `probe_closure.py` | Sparse-probe comparisons. |
| `claim_numbers.py` | Ranking and behavioral summaries. |
| `crosscoder_tables.py` | Crosscoder table rows. |

The first six scripts run together in `check_results.py`. Table scripts accept the inputs listed at the top of each file. For example:

```bash
python analysis/judge_closure.py results/language generated
```

`export_inputs.py` prepares token inputs; `export_features.py` exports decoders and importance scores from checkpoints. `precompute_shared.py` and `aggregate_features.py` process these arrays. `rx_analysis.py` and `rx_summaries.py` assemble result tables.

`check_saes.py` downloads three checkpoints and compares their outputs with saved fixtures. Install PyTorch and `huggingface_hub` to run it.
