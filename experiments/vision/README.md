# Vision models

| Script | Purpose |
|---|---|
| `run_arm.py` | Train and evaluate each condition. |
| `vitlib.py` | Load images, extract activations, and train SAEs. |
| `jiang.py` | Apply test-time registers. |

`images.json` lists 640 images from CIFAR-100's training split: 512 for SAE training and 128 for evaluation. `run_arm.py` copies this list into its working folder on the first run.

Saved measurements are in `results/additional/vision/`.
