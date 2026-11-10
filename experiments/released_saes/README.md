# Released SAEs

| Scripts | Purpose |
|---|---|
| `prepare_corpus.py`, `harvest.py` | Prepare text and collect activations. |
| `adapters.py`, `audit.py`, `worker.py` | Load SAEs and measure reconstruction. |
| `native_check.py` | Compare outputs with the release's implementation. |
| `qwen_filter.py` | Evaluate Qwen norm thresholds. |
| `pythia_release_style.py` | Evaluate Pythia on EOS-packed contexts. |
| `collect_v2.py` | Build the paper's ten tables. |
| `v1/`, `variants/` | Code for the first evaluation and model-specific checks. |

From the repository root:

```bash
python experiments/released_saes/collect_v2.py --out generated/released_saes
```

Inputs are in `results/additional/released_saes/v1_inputs/` and `round2_results/`.

To rebuild the first evaluation, download its statistics and run:

```bash
python -m pip install huggingface_hub
python experiments/released_saes/download_statistics.py --all-rounds
cd results/additional/released_saes
python ../../../experiments/released_saes/v1/collect.py --results round1_results --out ../../../generated/released_saes_v1
```
