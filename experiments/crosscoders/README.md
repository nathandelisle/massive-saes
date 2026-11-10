# Crosscoders

Four Pythia-1.4B comparisons: identical activations; PC1 doubled in one input; layers 12 and 16; and layer 12 at step 71,000 and the final checkpoint.

| Script | Purpose |
|---|---|
| `run.py` | Run the comparisons. |
| `ext/cache.py` | Prepare paired activations. |
| `ext/cached_coders.py` | Train, evaluate, and compare crosscoders. |
| `ext/coders_job.py` | Run each configuration and seed. |
| `configs/` | The 84 training configurations. |

From the repository root, train on a GPU:

```bash
python experiments/crosscoders/run.py --model pythia-1.4b --conds C0,C1,C3a,C3b --seeds 42,137,0
```

Print table rows from `results/additional/crosscoders/`:

```bash
python analysis/crosscoder_tables.py
```
