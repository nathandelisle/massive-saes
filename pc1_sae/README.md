# Loading an SAE

Install PyTorch and `huggingface_hub`. Run Python from the repository root so `pc1_sae` and `experiments` are importable.

```python
import json
from pc1_sae import load_release_sae, load_preprocessing, Splice

core = json.load(open("checkpoints/repositories.json"))["core"]
sae, config = load_release_sae(
    core["repo"], "gpt2/bypass/seed42", revision=core["revision"]
)
mean, pc1 = load_preprocessing(
    core["repo"], "gpt2", revision=core["revision"]
)
reconstruct = Splice(sae, "bypass", mean, pc1)
# h contains residual-stream activations at the checkpoint's layer.
# h_hat = reconstruct(h)
```

See [checkpoints](../checkpoints/README.md) for the available repositories.

For the standard SAE, reconstruction is `SAE(h - mean) + mean`. With bypass, the PC1 component is removed before encoding and added back after decoding. Use the mean and PC1 saved for that checkpoint's layer.

Layer numbers start at zero and refer to transformer block outputs. In Hugging Face hidden states, block `layer` is usually `hidden_states[layer + 1]`; the last output must be checked for final normalization. The model registry specifies the locations used in these experiments.
