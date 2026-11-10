"""Splice an SAE into a model's residual stream at the target layer.

modes:
  "none"     — model untouched (baseline CE)
  "standard" — h -> SAE(h - mu) + mu
  "bypass"   — h -> SAE(P_perp(h - mu)) + <h - mu, v1> v1 + mu   (v2 Eq. 1)
  "delete"   — h -> P_perp(h - mu) + mu            (delete-vs-bypass, exp_r5)
  "delete_k" — h -> h with top-n_pcs projections removed (All-but-the-Top)

transform: optional callable z -> z applied before decoding (zero-ablate,
mean-ablate, resample, group-ablate, alpha-scale for IG...). capture=True keeps
the (possibly grad-requiring) latent tensor for gradient metrics.
"""

import contextlib

import torch
import torch.nn.functional as F

from .data import get_blocks
from .registry import block_index


class SpliceState:
    def __init__(self):
        self.z = None            # last captured latents [B, T, d_sae]
        self.pc1_coef = None


def hook_device(model, spec, layer=None):
    """Device the TARGET BLOCK's output lives on.

    Not `next(model.parameters()).device`. Under device_map="auto" the model is
    sharded across GPUs and that expression returns the FIRST shard -- the
    embedding -- while the block we hook is on some later one. Every tensor the
    hook combines with the block output (mu, v1, the SAE's own weights) has to
    live where that output does, or the very first hook call dies with a
    cross-device RuntimeError. This is the only thing standing between the 70B
    and zero hours of useful work, and it is invisible on every smaller model
    because there the two expressions are equal.
    """
    blocks = get_blocks(model, spec.block_path)
    return next(blocks[block_index(spec, layer)].parameters()).device


@contextlib.contextmanager
def splice(model, spec, sae=None, mu=None, v1=None, mode="standard",
           transform=None, capture=False, pcs=None, n_pcs=1, layer=None):
    """Context manager installing the splice hook; yields a SpliceState."""
    state = SpliceState()
    if mode == "none":
        yield state
        return
    blocks = get_blocks(model, spec.block_path)
    li = block_index(spec, layer)
    dev = hook_device(model, spec, layer)
    # the SAE is loaded on args.device (cuda:0) but must run where the block
    # output is; .to() is in-place for parameters, so this costs one move on the
    # first splice and nothing thereafter
    if sae is not None:
        p0 = next(sae.parameters(), None)
        if p0 is not None and p0.device != dev:
            sae = sae.to(dev)
    mu_t = mu.to(dev) if torch.is_tensor(mu) else torch.tensor(
        mu, dtype=torch.float32, device=dev)
    v1_t = None
    if v1 is not None:
        v1_t = v1.to(dev) if torch.is_tensor(v1) else torch.tensor(
            v1, dtype=torch.float32, device=dev)
    pcs_t = None
    if pcs is not None:
        pcs_t = torch.tensor(pcs[:, :n_pcs], dtype=torch.float32, device=dev)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        orig_dtype = h.dtype
        x = h.float() - mu_t
        if mode == "delete":
            coef = x @ v1_t
            new = x - coef[..., None] * v1_t + mu_t
        elif mode == "delete_k":
            coef = x @ pcs_t                              # [B,T,n_pcs]
            new = x - coef @ pcs_t.T + mu_t
        else:
            if mode == "bypass":
                coef = x @ v1_t
                state.pc1_coef = coef
                x_in = x - coef[..., None] * v1_t
            else:
                x_in = x
            B, T, d = x_in.shape
            z = sae.encode(x_in.reshape(-1, d)).reshape(B, T, -1)
            if capture:
                z.requires_grad_(True)
                z.retain_grad()
                state.z = z
            if transform is not None:
                z = transform(z)
            recon = sae.decode(z.reshape(-1, z.shape[-1])).reshape(B, T, d)
            new = recon + mu_t
            if mode == "bypass":
                new = new + state.pc1_coef[..., None] * v1_t
        new = new.to(orig_dtype)
        if is_tuple:
            return (new,) + output[1:]
        return new

    handle = blocks[li].register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def ce_loss(model, seqs, chunk=8, grad=False, am=None):
    """Mean next-token cross-entropy over a [N, T] LongTensor of sequences.

    PADDING MUST BE MASKED. v2 set `labels[attention_mask == 0] = -100`. With
    one document per sequence, 30-50% of positions can be padding, and letting
    them into the average both inflates CE by orders of magnitude and can flip
    the sign of a delta-CE. The attention mask is also passed to the model, so
    the baseline itself is not computed on a corrupted forward pass.
    """
    dev = next(model.parameters()).device
    total, count = 0.0, 0
    ctx = contextlib.nullcontext() if grad else torch.no_grad()
    with ctx:
        for i in range(0, seqs.shape[0], chunk):
            batch = seqs[i:i + chunk].to(dev)
            mb = None if am is None else am[i:i + chunk].to(dev)
            logits = model(batch, attention_mask=mb).logits.float()
            tgt = batch[:, 1:].reshape(-1).clone()
            if mb is not None:
                tgt[mb[:, 1:].reshape(-1) == 0] = -100
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), tgt,
                reduction="sum", ignore_index=-100)
            n = int((tgt != -100).sum())
            if grad and n:
                (loss / n).backward()
            total += loss.item()
            count += n
    return total / max(count, 1)


def spliced_ce(model, spec, seqs, sae=None, mu=None, v1=None, mode="standard",
               transform=None, layer=None, chunk=8, pcs=None, n_pcs=1, am=None):
    with splice(model, spec, sae=sae, mu=mu, v1=v1, mode=mode,
                transform=transform, layer=layer, pcs=pcs, n_pcs=n_pcs):
        return ce_loss(model, seqs, chunk=chunk, am=am)


# ------------------------- latent transforms ---------------------------------

def zero_feature(j):
    def t(z):
        z = z.clone()
        z[..., j] = 0.0
        return z
    return t


def set_feature_scalar(j, value):
    """Mean-ablation: replace feature j's activation with its dataset mean
    (computed including zeros) at every position."""
    def t(z):
        z = z.clone()
        z[..., j] = value
        return z
    return t


def resample_feature(j, bank, rng):
    """Resample ablation: wherever feature j fires, replace
    its activation with a draw from its own empirical nonzero distribution, so
    the reconstruction stays in-distribution. Positions where it does not fire
    stay zero (support preserved; magnitudes typicalized)."""
    def t(z):
        z = z.clone()
        mask = z[..., j] != 0
        n = int(mask.sum().item())
        if n > 0 and len(bank) > 0:
            idx = rng.integers(0, len(bank), size=n)
            vals = torch.tensor(bank[idx], dtype=z.dtype, device=z.device)
            zj = z[..., j]
            zj[mask] = vals
            z[..., j] = zj
        return z
    return t


def zero_group(mask_bool):
    idx = torch.tensor([i for i, m in enumerate(mask_bool) if m])
    def t(z):
        z = z.clone()
        z[..., idx] = 0.0
        return z
    return t


def scale_all(alpha):
    def t(z):
        return z * alpha
    return t
