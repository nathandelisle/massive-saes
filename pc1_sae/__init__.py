"""Load and apply the released PC1 / massive-activation SAEs.

    from pc1_sae import load_release_sae, load_preprocessing, Splice
    sae, cfg = load_release_sae("nathandelisle/pc1-saes-core", "gpt2/bypass/seed42", revision=REV)
    mean, v1 = load_preprocessing("nathandelisle/pc1-saes-core", "gpt2", revision=REV)
    sp = Splice(sae, cfg["condition"], mean, v1)
    x_hat = sp(h)            # h: [..., d_model] residual-stream activations (output of block `layer`)

Conventions (identical to the training/evaluation code, rx/modelio.Dictionary):
  hook      the OUTPUT of transformer block index `layer` (0-based), i.e. HF `hidden_states[layer + 1]`.
  standard  x_hat = SAE(h - mu) + mu
  bypass    c = <h - mu, v1>;  x_hat = SAE((h - mu) - c v1) + c v1 + mu
            (the SAE never sees the PC1 component; it is added back exactly, from the input)
  mu, v1    the per-model training mean and unit first principal component shipped in
            <model>/preprocessing/{mean.npy, pca.npz}. A bypass SAE is only meaningful with these assets.
  arithmetic float32; cast to the model dtype once at the end when splicing into a model.
The SAE classes are shared with experiments/language/pc1_v3/v3/common/saes.py.
"""
import json

import numpy as np
import torch

from experiments.language.pc1_v3.v3.common.saes import ARCHS, load_sae  # noqa: F401  (re-exported)

__all__ = ["load_sae_file", "load_release_sae", "load_preprocessing", "Splice", "hook_layer"]


def load_sae_file(path, device="cpu"):
    """Load a released .pt checkpoint (plain tensors/dicts; safe with weights_only=True)."""
    blob = torch.load(path, map_location=device, weights_only=True)
    cfg = dict(blob["cfg"])
    arch = cfg.pop("arch")
    d_model, d_sae = cfg.pop("d_model"), cfg.pop("d_sae")
    sae = ARCHS[arch](d_model, d_sae, **cfg).to(device)
    sae.load_state_dict(blob["state"])
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae, blob.get("extra", {})


def load_release_sae(repo_id, subdir, revision, device="cpu", cache_dir=None):
    """Download ONE checkpoint (sae.pt + config.json) from a revision-pinned Hugging Face repo."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id, "%s/sae.pt" % subdir, revision=revision, cache_dir=cache_dir)
    c = hf_hub_download(repo_id, "%s/config.json" % subdir, revision=revision, cache_dir=cache_dir)
    cfg = json.load(open(c))
    import hashlib
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != cfg["sha256"]:
        raise RuntimeError("sha256 mismatch for %s: %s != %s" % (subdir, h, cfg["sha256"]))
    sae, _ = load_sae_file(p, device)
    return sae, cfg


def load_preprocessing(repo_id, model, revision, cache_dir=None):
    """Training mean (d,) and unit PC1 direction v1 (d,) for `model` at its released layer."""
    from huggingface_hub import hf_hub_download
    m = hf_hub_download(repo_id, "%s/preprocessing/mean.npy" % model, revision=revision, cache_dir=cache_dir)
    q = hf_hub_download(repo_id, "%s/preprocessing/pca.npz" % model, revision=revision, cache_dir=cache_dir)
    return np.load(m), np.load(q)["v1"]


class Splice:
    """The standard / bypass reconstruction used throughout the paper (float32)."""

    def __init__(self, sae, condition, mean, v1, device="cpu"):
        assert condition in ("standard", "bypass"), condition
        self.sae, self.condition = sae, condition
        self.mu = torch.as_tensor(np.asarray(mean), dtype=torch.float32, device=device)
        v = torch.as_tensor(np.asarray(v1), dtype=torch.float32, device=device)
        self.v1 = v / v.norm()

    @torch.no_grad()
    def encode(self, h):
        """-> (z, c): latents and the bypass PC1 coefficient (None for standard)."""
        x = h.float() - self.mu
        c = None
        if self.condition == "bypass":
            c = x @ self.v1
            x = x - c[..., None] * self.v1
        return self.sae.encode(x), c

    @torch.no_grad()
    def decode(self, z, c=None):
        r = z @ self.sae.W_dec.float() + self.sae.b_dec.float() + self.mu
        if c is not None:
            r = r + c[..., None] * self.v1
        return r

    def __call__(self, h):
        z, c = self.encode(h)
        return self.decode(z, c)


def hook_layer(cfg):
    """HF `output_hidden_states` index for this SAE's site: hidden_states[layer + 1]."""
    return int(cfg["layer"]) + 1
