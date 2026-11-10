"""HF block-output adapter and the supplied v3 SAE checkpoint format.

The bypass forward exactly follows the provided runtime: project the encoder input,
decode, restore the original scalar PC1 coefficient and mean. No silent projection
of decoder outputs, precision fallback, or training occurs here.
"""
import sys
from pathlib import Path
import numpy as np
import torch


class Backend:
    def __init__(self, cfg):
        from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
        self.cfg = cfg; self.device = cfg.get("device", "cuda")
        if cfg.get("device_map"): raise ValueError("Use one device per model in this runner")
        torch.set_grad_enabled(False)
        if cfg.get("toy_model"):
            torch.manual_seed(cfg.get("model_seed", 991))
            self.model = GPT2LMHeadModel(GPT2Config(**cfg["toy_model"])).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(cfg["model_path"], revision=cfg.get("revision"),
                torch_dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
                local_files_only=cfg.get("local_files_only", True),
                attn_implementation=cfg.get("attn_implementation", "eager")).to(self.device)
        self.model.eval()
        self.block = self.model
        for attr in cfg["block_path"].split("."): self.block = getattr(self.block, attr)
        self.block = self.block[int(cfg["block_index"])]
        self.mean = torch.as_tensor(np.load(cfg["mean_path"]), dtype=torch.float32, device=self.device)
        self.v = torch.as_tensor(np.load(cfg["pc1_path"]), dtype=torch.float32, device=self.device)
        if self.v.ndim != 1 or self.mean.shape != self.v.shape or not torch.isfinite(self.v).all() or self.v.norm() <= 0:
            raise ValueError("Invalid mean or PC1")
        self.v = self.v / self.v.norm()

    def load_sae(self, run):
        blob = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
        c = dict(blob["cfg"]); arch = c.pop("arch"); dm = c.pop("d_model"); ds = c.pop("d_sae"); c.pop("seed", None)
        if self.cfg.get("toy_model"):
            from .toy import ToySAE
            sae = ToySAE(dm, ds, **c)
        else:
            path = Path(self.cfg["v3_path"]).resolve()
            if not (path / "v3/common/saes.py").is_file(): raise FileNotFoundError("v3_path must contain v3/common/saes.py")
            sys.path.insert(0, str(path))
            from v3.common.saes import ARCHS
            sae = ARCHS[arch](dm, ds, **c)
        sae.load_state_dict(blob["state"], strict=True); sae.to(self.device).float().eval()
        if tuple(sae.W_dec.shape) != (ds, dm): raise ValueError("Expected W_dec rows [features,channels]")
        if dm != len(self.v): raise ValueError("Checkpoint and hook dimensionality differ")
        return sae

    def encode(self, sae, x, run):
        xc = x.float() - self.mean; coef = xc @ self.v
        if run["condition"] == "bypass": xc = xc - coef[...,None] * self.v
        elif run["condition"] != "standard": raise ValueError("Unknown condition")
        z = sae.encode(xc); hat = sae.decode(z)
        if run["condition"] == "bypass": hat = hat + coef[...,None] * self.v
        if (z < -1e-6).any(): raise ValueError("These interventions require nonnegative SAE coefficients")
        return z, hat + self.mean

    def forward(self, ids, mask, replacement=None):
        captured = {}
        def hook(_m, _args, output):
            x = output[0] if isinstance(output, tuple) else output
            captured["x"] = x.detach()
            if not torch.isfinite(x).all(): raise FloatingPointError("Nonfinite target-block output BEFORE edit")
            if replacement is None: return output
            y = replacement(x) if callable(replacement) else replacement
            if y.shape != x.shape or not torch.isfinite(y).all(): raise FloatingPointError("Invalid edit before dtype cast")
            y = y.to(x.dtype)
            if not torch.isfinite(y).all(): raise FloatingPointError("Edit overflowed at target-block dtype cast")
            return (y,) + output[1:] if isinstance(output, tuple) else y
        h = self.block.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
        finally: h.remove()
        if "x" not in captured: raise RuntimeError("Target block hook never fired")
        if not torch.isfinite(logits).all(): raise FloatingPointError("Nonfinite downstream logits; keep this cell failed, diagnose precision")
        return logits, captured["x"]

    def tensors(self, data, ix):
        return (torch.as_tensor(data["ids"][ix], device=self.device),
                torch.as_tensor(data["mask"][ix], device=self.device))


def scores(ref, other, ids, mask):
    """Last dimension vocabulary. Extra leading intervention dimension broadcasts."""
    p = torch.log_softmax(ref[..., :-1, :], -1)
    q = torch.log_softmax(other[..., :-1, :], -1)
    valid = (mask[..., :-1] & mask[..., 1:]).float()
    target = ids[..., 1:, None]
    if q.ndim > target.ndim: target = target.unsqueeze(0).expand(q.shape[:-1] + (1,))
    true_p = p.gather(-1, ids[...,1:,None]).squeeze(-1)
    true_q = q.gather(-1, target).squeeze(-1)
    kl = (p.exp() * (p - q)).sum(-1)
    return ((true_p - true_q) * valid).sum(-1), (kl * valid).sum(-1), valid.sum(-1)


def absolute_ce(logits, ids, mask):
    p = torch.log_softmax(logits[:,:-1], -1)
    valid = mask[:,:-1] & mask[:,1:]
    return (-p.gather(-1, ids[:,1:,None]).squeeze(-1) * valid).sum(-1)


def valid_predictors(mask):
    m = torch.zeros_like(mask, dtype=torch.bool); m[:,:-1] = mask[:,:-1] & mask[:,1:]
    return m
