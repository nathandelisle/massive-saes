"""Steering: inject a decoder direction into the residual stream at the target
layer during a forward pass; measure KL between next-token distributions
(v2 Section 6 protocol). Kept minimal — Section 6 remains canonical in the
existing repo's causal/; this module serves exp_a1 and figure sanity checks."""

import contextlib

import torch
import torch.nn.functional as F

from .data import get_blocks
from .registry import block_index


@contextlib.contextmanager
def steer(model, spec, direction, alpha, layer=None):
    blocks = get_blocks(model, spec.block_path)
    li = block_index(spec, layer)
    dev = next(model.parameters()).device
    d = torch.tensor(direction, dtype=torch.float32, device=dev)
    d = d / d.norm()

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        new = (h.float() + alpha * d).to(h.dtype)
        return (new,) + output[1:] if is_tuple else new

    handle = blocks[li].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def next_token_dist(model, tok, prompt, spec=None, direction=None, alpha=0.0,
                    layer=None):
    dev = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    ctx = (steer(model, spec, direction, alpha, layer=layer)
           if direction is not None else contextlib.nullcontext())
    with ctx:
        logits = model(ids).logits[0, -1].float()
    return F.log_softmax(logits, dim=-1)


def kl(logp, logq):
    return float((logp.exp() * (logp - logq)).sum())
