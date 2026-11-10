"""Jiang et al., "Vision Transformers Don't Need Trained Registers" (arXiv:2506.08010), for HF DINOv2.

Two stages, following the authors' code (github.com/nickjiang2378/test-time-registers):

  1. DISCOVERY (`shared/algorithms.py:find_register_neurons`). Run the frozen model on calibration images.
     At a detection layer, find high-norm patch tokens. At those positions, take the post-nonlinearity MLP
     activations of every layer, average |activation| over the outlier positions, then over images, and
     select the top-N (layer, neuron) pairs from the flattened score matrix. These are ordinary MLP hidden
     units — not SAE features and not residual-stream coordinates.

  2. INTERVENTION (`shared/hook_fn.py:activate_on_registers`). Append R zero-valued tokens. At each layer
     holding selected neurons, hook the MLP activation module (post-nonlinearity, pre-`fc2`) and, for each
     selected neuron: write its sign-preserving max over the original token positions into the appended
     token, and zero it at the image-patch positions. CLS is left alone, as in the authors' code. No weights
     are trained and no PC1 is subtracted.

VERIFIED AT THE CALL SITE (not inferred from the helper alone): `hook_manager.finalize()` registers one hook
per layer with `neuron_indices=register_neurons[layer]`, a LIST, and `configs/dinov2_large.yaml` uses
`top_k: 50` neurons spread over layers <= `highest_layer`, so a layer generally holds several neurons.
`output[0]` is batch element 0 (their pipeline runs one image at a time). Hence in the released code
`sign_max(output[0, :, neuron_indices])` reduces over BOTH positions and all of that layer's selected neurons
and writes ONE scalar into every register slot. The paper describes a per-neuron maximum. Both variants are
implemented here ("literal" and "per_neuron"), compared on the same calibration batch before any SAE is
trained, and the arm is labelled an adaptation if they differ materially.
"""
import argparse
import json

import numpy as np
import torch

from vitlib import DEV, token_layout


def mlp_act_module(layer):
    """The module whose OUTPUT is the post-nonlinearity MLP activation (their `blocks[l].mlp.act`)."""
    mlp = layer.mlp
    for name in ("activation", "act", "act_fn", "intermediate_act_fn", "gelu"):
        if hasattr(mlp, name) and isinstance(getattr(mlp, name), torch.nn.Module):
            return getattr(mlp, name)
    raise RuntimeError(f"cannot find the MLP nonlinearity module on {type(mlp).__name__}")


def layers_of(model):
    return model.encoder.layer


class TTR:
    """Test-time registers on an HF Dinov2Model."""

    def __init__(self, model, n_registers=1, variant="per_neuron"):
        assert variant in ("per_neuron", "literal")
        self.model = model
        self.R = n_registers
        self.variant = variant
        self.per_neuron_max = (variant == "per_neuron")
        self.selected = {}                  # layer -> LongTensor of neuron indices
        self.enabled = False
        self._handles = []
        self._log = None

    # ---------------------------------------------------------------- forward
    def _run(self, pixel_values, n_extra, output_hidden_states=True):
        """Manual encoder loop so extra tokens can be appended; mirrors Dinov2Model.forward."""
        m = self.model
        h = m.embeddings(pixel_values)
        if n_extra:
            h = torch.cat([h, h.new_zeros(h.shape[0], n_extra, h.shape[2])], dim=1)
        # Match HF exactly: Dinov2Model returns hidden_states = (embeddings, after block 0, ..., after
        # block N-1) and applies `layernorm` only to the value it returns as last_hidden_state, NOT to the
        # hidden_states tuple. Applying it here made the wrapper differ from HF by 193 in max abs value at
        # the final entry, which the validation gate caught before any SAE was trained on these activations.
        hidden = [h]
        for layer in layers_of(m):
            out = layer(h)
            h = out[0] if isinstance(out, (tuple, list)) else out
            hidden.append(h)
        return hidden

    def baseline_hidden(self, pixel_values):
        return self._run(pixel_values, 0)

    def intervened_hidden(self, pixel_values):
        if not self.selected:
            raise RuntimeError("no register neurons selected")
        self._install(pixel_values)
        try:
            return self._run(pixel_values, self.R)
        finally:
            self._remove()

    # ---------------------------------------------------------------- hooks
    def _install(self, pixel_values):
        n_prefix, grid, _ = token_layout(self.model, pixel_values)
        R, per_neuron = self.R, self.per_neuron_max

        def make(idx):
            def hook(module, inp, out):
                o = out
                sel = idx.to(o.device)
                block = o[:, :, sel]                       # [B, T, n_sel]
                orig = block[:, :-R, :]                    # original positions (CLS + patches)
                if per_neuron:
                    pos = orig.max(dim=1).values
                    neg = orig.min(dim=1).values
                    val = torch.where(pos.abs() >= neg.abs(), pos, neg)      # sign-preserving max per neuron
                else:
                    flat = orig.reshape(orig.shape[0], -1)
                    pos = flat.max(dim=1).values
                    neg = flat.min(dim=1).values
                    val = torch.where(pos.abs() >= neg.abs(), pos, neg).unsqueeze(-1).expand_as(orig[:, 0, :])
                o[:, -R:, sel] = val.unsqueeze(1).expand(-1, R, -1)
                o[:, n_prefix:-R, sel] = 0                 # zero at image patches; CLS untouched
                return o
            return hook

        for L, idx in self.selected.items():
            self._handles.append(mlp_act_module(layers_of(self.model)[L]).register_forward_hook(make(idx)))

    def _remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # ---------------------------------------------------------------- discovery
    @torch.no_grad()
    def discover(self, processor, images, detect_layer=-2, norm_mult=10.0, batch=8, highest_layer=None, log=print):
        """Authors' procedure, with the outlier threshold expressed as a multiple of the median patch norm."""
        m = self.model
        n_layers = len(layers_of(m))
        acts = {}
        handles = [mlp_act_module(layers_of(m)[L]).register_forward_hook(
            (lambda L: (lambda mod, i, o: acts.__setitem__(L, o.detach())))(L)) for L in range(n_layers)]
        scores = None
        n_used = 0
        n_out_total = n_tok_total = 0
        try:
            for i in range(0, len(images), batch):
                px = processor(images=images[i:i + batch], return_tensors="pt")["pixel_values"].to(DEV)
                acts.clear()
                hidden = self._run(px, 0)
                n_prefix, grid, _ = token_layout(m, px)
                h = hidden[detect_layer].float()
                patch = h[:, n_prefix:]
                nrm = patch.norm(dim=-1)                                  # [B, grid]
                med = nrm.median()
                mask = nrm > norm_mult * med
                n_out_total += int(mask.sum()); n_tok_total += int(mask.numel())
                if not bool(mask.any()):
                    continue
                if scores is None:
                    scores = torch.zeros(n_layers, acts[0].shape[-1], dtype=torch.float64, device=DEV)
                for L in range(n_layers):
                    a = acts[L].float()[:, n_prefix:]                     # [B, grid, neurons]
                    scores[L] += a[mask].abs().double().mean(0)           # mean over outlier positions
                n_used += 1
        finally:
            for h_ in handles:
                h_.remove()
        if scores is None:
            raise RuntimeError("no outlier patches found; lower norm_mult or change detect_layer")
        scores /= max(n_used, 1)
        if highest_layer is not None:                      # their configs/*.yaml `highest_layer`
            scores[highest_layer + 1:] = -float("inf")
        self.scores = scores.cpu()
        log(f"[discover] batches with outliers {n_used}; outlier patch fraction "
            f"{n_out_total / max(n_tok_total,1):.4f}; detect_layer {detect_layer}; threshold {norm_mult}x median")
        return self.scores

    def select_top(self, n_neurons):
        flat = self.scores.flatten()
        order = torch.argsort(flat, descending=True)[:n_neurons]
        nper = self.scores.shape[1]
        pairs = [(int(i) // nper, int(i) % nper) for i in order]
        sel = {}
        for L, nidx in pairs:
            sel.setdefault(L, []).append(nidx)
        self.selected = {L: torch.tensor(sorted(v), dtype=torch.long) for L, v in sel.items()}
        return dict(n_neurons=n_neurons, layers={str(L): [int(x) for x in v] for L, v in self.selected.items()},
                    scores_top=[float(flat[i]) for i in order[:min(8, len(order))]])


# ---------------------------------------------------------------------- gates
@torch.no_grad()
def validate(ttr, processor, images, detect_layer=-1, norm_mult=10.0, log=print):
    """Pre-registered checks, run before any SAE sees these activations."""
    m = ttr.model
    px = processor(images=images, return_tensors="pt")["pixel_values"].to(DEV)
    n_prefix, grid, n_reg = token_layout(m, px)

    ref = m(pixel_values=px, output_hidden_states=True).hidden_states
    mine = ttr.baseline_hidden(px)
    wrapper_err = max(float((a.float() - b.float()).abs().max()) for a, b in zip(ref, mine))

    base = mine[detect_layer].float()
    inter = ttr.intervened_hidden(px)[detect_layer].float()
    bp = base[:, n_prefix:]
    ip = inter[:, n_prefix:inter.shape[1] - ttr.R]
    med = float(bp.norm(dim=-1).median())
    out_b = float((bp.norm(dim=-1) > norm_mult * med).float().mean())
    out_i = float((ip.norm(dim=-1) > norm_mult * med).float().mean())
    reg_norm = float(inter[:, -ttr.R:].float().norm(dim=-1).mean())
    patch_max_b = float(bp.norm(dim=-1).max())
    patch_max_i = float(ip.norm(dim=-1).max())

    # does the intervened model still produce a sensible representation? (CLS agreement, lightweight)
    cls_b = base[:, 0]
    cls_i = inter[:, 0]
    cls_cos = float(torch.nn.functional.cosine_similarity(cls_b, cls_i, dim=-1).mean())
    rep = dict(
        cls_cosine_baseline_vs_intervened=cls_cos,
        wrapper_matches_hf_max_abs_diff=wrapper_err,
        patch_tokens_baseline=int(bp.shape[1]), patch_tokens_intervened=int(ip.shape[1]),
        token_counts_match=bool(bp.shape[1] == ip.shape[1] == grid),
        outlier_frac_baseline=out_b, outlier_frac_intervened=out_i,
        outlier_frac_ratio=(out_i / out_b if out_b > 0 else None),
        max_patch_norm_baseline=patch_max_b, max_patch_norm_intervened=patch_max_i,
        added_register_mean_norm=reg_norm,
        register_carries_more_than_max_patch=bool(reg_norm > patch_max_i),
        median_patch_norm_baseline=med, detect_layer=detect_layer, norm_mult=norm_mult,
        variant=ttr.variant, per_neuron_max=ttr.per_neuron_max, n_registers=ttr.R,
        selected_layers={str(L): len(v) for L, v in ttr.selected.items()},
    )
    rep["passes"] = bool(rep["wrapper_matches_hf_max_abs_diff"] < 1e-3 and rep["token_counts_match"]
                         and out_i < out_b and rep["register_carries_more_than_max_patch"]
                         and cls_cos > 0.5)
    log("[validate] " + json.dumps(rep))
    return rep
