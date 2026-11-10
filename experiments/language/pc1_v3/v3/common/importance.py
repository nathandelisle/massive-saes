"""Feature-importance metrics.

Ablation family (one full forward per feature — the expensive path):
  zero      : z_j -> 0
  mean      : z_j -> dataset mean of z_j (incl. zeros), all positions
  resample  : z_j -> draw from its own nonzero distribution at firing positions

Gradient family (ALL features in one forward/backward per eval chunk — cheap):
  gradxinput: |sum_t z_j * dCE/dz_j|
  intgrad   : |z_j * mean_alpha dCE/dz_j(alpha * z)|, 8-step path from zero
              baseline (linear decoder => latent path = reconstruction path)
"""

import numpy as np
import torch

from .splice import (splice, ce_loss, spliced_ce, zero_feature,
                     set_feature_scalar, resample_feature, scale_all)


@torch.no_grad()
def latent_stats(model, spec, seqs, sae, mu, v1=None, mode="standard",
                 layer=None, chunk=8, bank_cap=1024):
    """One pass over eval seqs collecting per-feature: mean activation
    (incl. zeros), firing count, and a nonzero value bank for resampling."""
    d_sae = sae.d_sae
    dev = next(model.parameters()).device
    sums = torch.zeros(d_sae, device=dev)
    fires = torch.zeros(d_sae, device=dev)
    banks = [[] for _ in range(d_sae)]
    n_tok = 0
    with splice(model, spec, sae=sae, mu=mu, v1=v1, mode=mode,
                capture=True, layer=layer) as st:
        for i in range(0, seqs.shape[0], chunk):
            model(seqs[i:i + chunk].to(dev))
            z = st.z.detach()
            n_tok += z.shape[0] * z.shape[1]
            sums += z.sum((0, 1))
            nz = z != 0
            fires += nz.sum((0, 1)).float()
            # value banks (cap per feature to bound memory)
            zi = z.reshape(-1, d_sae)
            for j in torch.nonzero(nz.any((0, 1))).flatten().tolist():
                vals = zi[:, j][zi[:, j] != 0]
                if len(banks[j]) < bank_cap:
                    banks[j].extend(vals[: bank_cap - len(banks[j])].cpu().tolist())
    means = (sums / max(n_tok, 1)).cpu().numpy()
    rates = (fires / max(n_tok, 1)).cpu().numpy()
    banks = [np.asarray(b, dtype=np.float32) for b in banks]
    return means, rates, banks


def ablation_importance(model, spec, seqs, sae, mu, v1=None, mode="standard",
                        metric="zero", features=None, layer=None, chunk=8,
                        means=None, banks=None, seed=0, progress=None):
    """Returns dict feature_id -> delta CE (spliced-ablated minus spliced-baseline)."""
    base = spliced_ce(model, spec, seqs, sae=sae, mu=mu, v1=v1,
                      mode=mode, layer=layer, chunk=chunk)
    rng = np.random.default_rng(seed)
    features = features if features is not None else range(sae.d_sae)
    out = {}
    for n, j in enumerate(features):
        if metric == "zero":
            t = zero_feature(j)
        elif metric == "mean":
            t = set_feature_scalar(j, float(means[j]))
        elif metric == "resample":
            t = resample_feature(j, banks[j], rng)
        else:
            raise ValueError(metric)
        ce = spliced_ce(model, spec, seqs, sae=sae, mu=mu, v1=v1,
                        mode=mode, transform=t, layer=layer, chunk=chunk)
        out[int(j)] = ce - base
        if progress and n % progress == 0:
            print(f"    [{metric}] {n}/{len(list(features)) if hasattr(features,'__len__') else '?'}", flush=True)
    return out, base


def gradient_importance(model, spec, seqs, sae, mu, v1=None, mode="standard",
                        layer=None, chunk=4, ig_steps=8):
    """gradxinput and integrated-gradients for ALL features simultaneously.
    Gradients flow through the frozen model into the captured latents only."""
    dev = next(model.parameters()).device
    d_sae = sae.d_sae
    for p in model.parameters():
        p.requires_grad_(False)
    for p in sae.parameters():
        p.requires_grad_(False)

    gxi = torch.zeros(d_sae, device=dev)
    ig = torch.zeros(d_sae, device=dev)
    n_chunks = 0
    for i in range(0, seqs.shape[0], chunk):
        batch = seqs[i:i + chunk].to(dev)
        # grad x input at alpha = 1
        with splice(model, spec, sae=sae, mu=mu, v1=v1, mode=mode,
                    capture=True, layer=layer) as st:
            logits = model(batch).logits.float()
            import torch.nn.functional as F
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   batch[:, 1:].reshape(-1))
            loss.backward()
            z, g = st.z.detach(), st.z.grad.detach()
            gxi += (z * g).sum((0, 1)).abs()
        # integrated gradients: rescale captured z by alpha inside the transform
        acc = torch.zeros(d_sae, device=dev)
        for s in range(1, ig_steps + 1):
            alpha = s / ig_steps
            with splice(model, spec, sae=sae, mu=mu, v1=v1, mode=mode,
                        capture=True, transform=scale_all(alpha),
                        layer=layer) as st:
                logits = model(batch).logits.float()
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    batch[:, 1:].reshape(-1))
                loss.backward()
                # dCE/d(alpha*z) * z = (grad wrt captured z) / alpha * ... ;
                # st.z.grad is dCE/dz_captured with transform z->alpha z, i.e.
                # alpha * dCE/d(z_eff). Path integral uses dCE/d(z_eff) at z_eff
                # = alpha*z times dz = z dalpha  ->  accumulate grad(z_captured).
                acc += (st.z.detach() * st.z.grad.detach()).sum((0, 1))
        ig += (acc / ig_steps).abs()
        n_chunks += 1
    return (gxi / n_chunks).cpu().numpy(), (ig / n_chunks).cpu().numpy()


def rank_top(imp_dict_or_array, k=50, alive_mask=None):
    if isinstance(imp_dict_or_array, dict):
        arr = np.full(max(imp_dict_or_array) + 1, -np.inf)
        for j, v in imp_dict_or_array.items():
            arr[j] = v
    else:
        arr = np.asarray(imp_dict_or_array, dtype=np.float64).copy()
        if alive_mask is not None:
            arr[~alive_mask] = -np.inf
    return np.argsort(arr)[::-1][:k]
