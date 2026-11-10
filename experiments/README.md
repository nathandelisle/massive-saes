# Experiments

Scripts and results by paper section. Script paths are relative to this folder; result paths start at the repository root.

| Paper section | Scripts | Saved results |
|---|---|---|
| A: training and activation extraction | [language/pc1_v3/v3/common/](language/pc1_v3/v3/common/), [language/rx/prep.py](language/rx/prep.py) | `results/language/prep/` |
| B: importance and sharing | [language/rx/evaljob.py](language/rx/evaljob.py), [language/rx/engine.py](language/rx/engine.py) | `results/feature_rankings/`, `results/tables/features_summary.csv` |
| B: original PC1-alignment ranges | [language/pc1_v3/scripts/exp_a5_metric_battery.py](language/pc1_v3/scripts/exp_a5_metric_battery.py), [language/pc1_v3/scripts/exp_e1_alignment_cliff.py](language/pc1_v3/scripts/exp_e1_alignment_cliff.py) | `results/examples/sec4_tiles/` |
| B: resampling and projected matching | [controls/track_g.py](controls/track_g.py), [controls/track_f2.py](controls/track_f2.py), [language/rx/engine.py](language/rx/engine.py) | `results/additional/reproducibility/track_g/`, `results/additional/objectives/track_f2/`, `results/tables/resampling_closure.csv` |
| C: component ablations and cancellation | [language/rx/mechanism.py](language/rx/mechanism.py) | `results/tables/mechanism_individual_vs_joint.csv`, `results/language/raw/` |
| C: reinitializing the PC1-aligned features | [language/rx/trainjob.py](language/rx/trainjob.py) | `results/language/raw/` |
| D: architecture comparison | [language/pc1_v3/scripts/exp_b1_architecture_law.py](language/pc1_v3/scripts/exp_b1_architecture_law.py), [language/rx/trainjob.py](language/rx/trainjob.py) | `results/additional/objectives/architecture_grid/`, `results/language/analysis/train.csv` |
| D: magnitude penalties and Gated controls | [objectives/magnitude/](objectives/magnitude/README.md), [objectives/gated/](objectives/gated/README.md), [language/pc1_v3/scripts/exp_b2_l1_phase.py](language/pc1_v3/scripts/exp_b2_l1_phase.py) | `results/additional/objectives/five_seed_beta/`, `results/additional/objectives/gated/` |
| D: learned JumpReLU | [controls/track_c.py](controls/track_c.py) | `results/additional/objectives/track_c/` |
| D: ten-seed GPT-2 comparison | [ten_seed_gpt2/](ten_seed_gpt2/README.md), [controls/collect_ten_seed.py](controls/collect_ten_seed.py) | `results/additional/objectives/ten_seed_native/` |
| E: reconstruction, PCA, and matched coefficients | [reconstruction/sec6_fidelity.py](reconstruction/sec6_fidelity.py), [language/pc1_v3/scripts/exp_c3_linear_baseline.py](language/pc1_v3/scripts/exp_c3_linear_baseline.py), [controls/track_b.py](controls/track_b.py) | `results/additional/reconstruction/` |
| F: original agreement edits and transfer | [behavior/experiment_a2.py](behavior/experiment_a2.py), [behavior/rank_paper.py](behavior/rank_paper.py), [behavior/tasks2.py](behavior/tasks2.py) | `results/additional/behavior/v2/` |
| F: expanded agreement and indirect-object edits | [language/rx/patching.py](language/rx/patching.py), [language/rx/tasks_sva.py](language/rx/tasks_sva.py), [language/rx/tasks_ioi.py](language/rx/tasks_ioi.py) | `results/tables/patching_primary.csv`, `results/language/raw/` |
| G: automated interpretation | [language/rx/contexts.py](language/rx/contexts.py), [language/rx/judge.py](language/rx/judge.py) | `results/language/analysis/judge_per_feature.csv`, `results/tables/judge_condition_contrasts.csv` |
| H: sparse probes | [language/rx/probing.py](language/rx/probing.py), [language/rx/probejob.py](language/rx/probejob.py) | `results/language/analysis/probe.csv`, `results/tables/probe_paired.csv` |
| I: PC1, norm, and token frequency | [geometry/collect_pythia.py](geometry/collect_pythia.py), [geometry/measure_pc1.py](geometry/measure_pc1.py) | `results/geometry/E20_pc1_associations.json` |
| I: prediction, deletion, and MLP interventions | [language/rx/auxjob.py](language/rx/auxjob.py) | `results/language/analysis/aux_a1.csv`, `results/language/analysis/aux_a3.csv` |
| J: geometry across layers | [geometry/collect_pythia.py](geometry/collect_pythia.py) | `results/geometry/E19_summary.json` |
| J: SAEs at additional layers | [language/rx/trainjob.py](language/rx/trainjob.py), [layers/](layers/README.md) | `results/language/analysis/train.csv`, `results/additional/layers/layer_sweep_raw.json` |
| J: removing more PCs | [earlier/pca_sweep.py](earlier/pca_sweep.py) | Original two-seed experiment; see script. |
| J: first positions and special tokens | [controls/track_k.py](controls/track_k.py) | `results/additional/layers/track_k/K_geometry.csv` |
| K: released SAEs | [released_saes/](released_saes/README.md) | `results/additional/released_saes/sec7_tables_v2/` |
| K: normalization, filtering, and whitening | [controls/track_b.py](controls/track_b.py), [controls/xforms.py](controls/xforms.py) | `results/additional/released_saes/track_b_raw/` |
| L: vision models | [vision/run_arm.py](vision/run_arm.py), [vision/jiang.py](vision/jiang.py), [vision/vitlib.py](vision/vitlib.py) | `results/additional/vision/` |
| M: crosscoders | [crosscoders/](crosscoders/README.md) | `results/additional/crosscoders/` |
| Main text: two-seed Llama-70B comparison | [earlier/llama70b.py](earlier/llama70b.py) | `results/examples/llama70b/` |
| Main text: Llama BOS features | [earlier/llama_anatomy.py](earlier/llama_anatomy.py) | Original experiment; see script. |

## Language-model runs

`language/run.py` dispatches the following jobs:

| Job | Script | Purpose |
|---|---|---|
| `prep` | `rx/prep.py` | Prepare activations and calibration data. |
| `eval` | `rx/evaljob.py` | Measure feature importance. |
| `wsx` | `rx/wsxjob.py` | Collect feature contexts and run component or behavioral edits. |
| `train` | `rx/trainjob.py` | Compare objectives and additional layers. |
| `trainseed` | `rx/trainseed.py` | Train one SAE. |
| `probe` | `rx/probejob.py` | Fit sparse probes. |
| `aux` | `rx/auxjob.py` | Predict model quantities and run PC1 interventions. |
| `judge` | `rx/judge.py` | Generate and evaluate feature descriptions. |
| `gradcheck`, `nativecheck`, `qwennan` | Corresponding files in `rx/` | Numerical checks. |

For example, from the repository root:

```bash
python experiments/language/run.py --help
python experiments/language/run.py eval --help
```

Set `PC1_V3_ROOT` to the folder containing activations and checkpoints. For S3 inputs, set `RX_LOCAL=0` and `S3_BUCKET`.

The scripts in `earlier/` use Modal and its stored volumes.
