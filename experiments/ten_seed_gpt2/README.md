# Ten-seed GPT-2 comparison

`pc1night/` contains the trainer and evaluator; `configs/` contains run settings. `cloud/fastrows.py` and `experiments/controls/collect_ten_seed.py` collect results.

```bash
python experiments/ten_seed_gpt2/overnight.py --help
```

Commands: `prepare`, `train`, `rank`, `causal`, `collect`. Each accepts `--help`. Saved results are in `results/additional/objectives/ten_seed_native/`.
