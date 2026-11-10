"""Fast, exactness-checked feature-importance measurement.

`importance.ablation_importance` runs one full forward over the whole eval set
per feature. At d_sae = 16384 on a 9B model that is ~6 GPU-hours per
(metric, condition, seed) cell, so the full battery is ~700 GPU-h rather than
the 60-90 the README assumes. This module makes it affordable WITHOUT changing
what is measured, via three levers:

1. `tuned_chunk` — the default chunk of 8 sequences is 1024 tokens, far too
   small to saturate an L40S. Batch to the largest chunk whose logits fit.

2. VARIANCE SCREEN. Ablating latent j removes the displacement z_j * d_j from
   the residual stream, so to second order

       Delta_CE(j) ~= (1/2) * kappa_j * E[z_j^2],       kappa_j = d_j^T F d_j

   with F the Fisher of the output distribution w.r.t. the residual stream.
   The directional curvature kappa varies over a narrow range across decoder
   directions (measured independently on Qwen 0.5-3B: per-unit kappa of PC1 is
   1.05x the random-direction median, massive-activation dims 1.11x), while
   E[z_j^2] spans many orders of magnitude. So E[z_j^2] is a strong screen:
   rank by it, ablate only the top M exactly, and the true top-k survives.

   This is a claim about the data, not an assumption we get to make, so
   `screen_report` measures the screen's recall against a full exhaustive
   ablation. Gate 0 runs the exhaustive version on pythia-1.4b and refuses to
   enable the screen elsewhere unless recall of the true top-50 is 1.0.

3. Vectorised latent statistics. `importance.latent_stats` loops in Python over
   thousands of features per chunk, each doing a GPU boolean-index plus a
   `.cpu().tolist()` — thousands of device syncs. `z_stats` below does the same
   work in a handful of fused ops.

Everything here is measurement-equivalent to `importance.py` on the features it
evaluates; the screen only changes WHICH features get evaluated.
"""

import numpy as np
import torch
import torch.nn.functional as F

import torch.nn.functional as F

from .splice import (splice, spliced_ce, zero_feature, set_feature_scalar,
                     resample_feature, scale_all)


def tuned_chunk(spec, vocab_size=None, budget_gb=6.0):
    """Largest number of 128-token sequences whose fp32 logits fit in budget_gb.

    Logits dominate: chunk * seq_len * vocab * 4 bytes. Keeping this under a few
    GB leaves room for the model and the SAE on a 48 GB card.
    """
    v = vocab_size or 152064            # pessimistic (Qwen-class vocab)
    per_seq = 128 * v * 4 / 1e9
    c = max(1, int(budget_gb / max(per_seq, 1e-9)))
    return min(c, 64)


@torch.no_grad()
def z_stats(model, spec, seqs, sae, mu, v1=None, mode="standard", layer=None,
            chunk=8, bank_cap=512, max_bank_tokens=200_000, am=None):
    """One pass over the eval sequences returning, per feature:
        mean  E[z_j]   (including non-firing positions, for mean-ablation)
        ez2   E[z_j^2] (including zeros — the screen statistic)
        rate  firing rate
        banks list of np arrays of observed nonzero values (for resampling)

    Vectorised: no per-feature Python loop.
    """
    d_sae = sae.d_sae
    dev = next(model.parameters()).device
    s1 = torch.zeros(d_sae, device=dev, dtype=torch.float64)
    s2 = torch.zeros(d_sae, device=dev, dtype=torch.float64)
    fires = torch.zeros(d_sae, device=dev, dtype=torch.float64)
    n_tok = 0
    feat_idx, feat_val = [], []
    banked = 0
    with splice(model, spec, sae=sae, mu=mu, v1=v1, mode=mode,
                capture=True, layer=layer) as st:
        for i in range(0, seqs.shape[0], chunk):
            mb = None if am is None else am[i:i + chunk].to(dev)
            model(seqs[i:i + chunk].to(dev), attention_mask=mb)
            z = st.z.detach()
            zf = z.reshape(-1, d_sae)
            if mb is not None:                 # padding must not enter E[z^2]
                zf = zf[mb.reshape(-1).bool()]
            n_tok += zf.shape[0]
            s1 += zf.sum(0).double()
            s2 += zf.pow(2).sum(0).double()
            nz = zf != 0
            fires += nz.sum(0).double()
            if banked < max_bank_tokens:
                # one fused nonzero() instead of d_sae boolean indexes
                rows, cols = torch.nonzero(nz, as_tuple=True)
                feat_idx.append(cols.cpu().numpy())
                feat_val.append(zf[rows, cols].float().cpu().numpy())
                banked += int(cols.numel())
    mean = (s1 / max(n_tok, 1)).cpu().numpy()
    ez2 = (s2 / max(n_tok, 1)).cpu().numpy()
    rate = (fires / max(n_tok, 1)).cpu().numpy()

    banks = [np.empty(0, dtype=np.float32)] * d_sae
    if feat_idx:
        fi = np.concatenate(feat_idx)
        fv = np.concatenate(feat_val)
        order = np.argsort(fi, kind="stable")
        fi, fv = fi[order], fv[order]
        bounds = np.searchsorted(fi, np.arange(d_sae + 1))
        banks = [fv[bounds[j]:min(bounds[j + 1], bounds[j] + bank_cap)]
                 for j in range(d_sae)]
    return dict(mean=mean, ez2=ez2, rate=rate, banks=banks, n_tok=n_tok)


def screen_candidates(ez2, alive_mask, m):
    """Indices of the top-m alive features by E[z^2] (the screen)."""
    s = np.asarray(ez2, dtype=np.float64).copy()
    s[~np.asarray(alive_mask, dtype=bool)] = -np.inf
    m = int(min(m, int(np.isfinite(s).sum())))
    return np.argsort(s)[::-1][:m].copy()


def ablation_over(model, spec, seqs, sae, mu, features, v1=None,
                  mode="standard", metric="zero", layer=None, chunk=8,
                  stats=None, seed=0, on_partial=None, partial_every=250,
                  am=None):
    """Delta-CE for an explicit feature list. Same measurement as
    `importance.ablation_importance`, but with a caller-chosen feature set,
    tuned chunking, and partial-result callbacks so a spot eviction costs at
    most `partial_every` features of work."""
    base = spliced_ce(model, spec, seqs, sae=sae, mu=mu, v1=v1,
                      mode=mode, layer=layer, chunk=chunk, am=am)
    rng = np.random.default_rng(seed)
    out = {}
    for n, j in enumerate(features):
        j = int(j)
        if metric == "zero":
            t = zero_feature(j)
        elif metric == "mean":
            t = set_feature_scalar(j, float(stats["mean"][j]))
        elif metric == "resample":
            t = resample_feature(j, stats["banks"][j], rng)
        else:
            raise ValueError(metric)
        ce = spliced_ce(model, spec, seqs, sae=sae, mu=mu, v1=v1, mode=mode,
                        transform=t, layer=layer, chunk=chunk, am=am)
        out[j] = float(ce - base)
        if on_partial and (n + 1) % partial_every == 0:
            on_partial(n + 1, len(features), out, float(base))
    return out, float(base)


def gradient_importance(model, spec, seqs, sae, mu, v1=None, mode="standard",
                        layer=None, chunk=4, ig_steps=8, am=None):
    """gradxinput and integrated gradients, padding-masked, for ALL features.

    Two corrections relative to `importance.gradient_importance`:

    1. PADDING. The loss must be averaged over real tokens only, and latents at
       padded positions must not contribute to the accumulators. Otherwise both
       metrics are diluted by however much of each sequence is padding (12-50%
       here), unevenly across features.

    2. THE IG PATH INTEGRAL. IG for latent j is
           z_j * mean_alpha [ dCE/dz'_j evaluated at z' = alpha*z ].
       With the transform z -> alpha*z, autograd populates `z.grad` with
       alpha * dCE/dz'|_{alpha z} by the chain rule. Accumulating
       `z * z.grad` therefore integrates an extra factor of alpha, giving
       sum_s (s/S) * z * dCE/dz' — a magnitude-weighted variant, not IG.
       Dividing by alpha removes it and recovers the Riemann sum.
    """
    dev = next(model.parameters()).device
    d_sae = sae.d_sae
    for p in model.parameters():
        p.requires_grad_(False)
    for p in sae.parameters():
        p.requires_grad_(False)

    gxi = torch.zeros(d_sae, device=dev, dtype=torch.float64)
    ig = torch.zeros(d_sae, device=dev, dtype=torch.float64)
    n_chunks = 0

    def _loss(logits, batch, mb):
        tgt = batch[:, 1:].reshape(-1).clone()
        if mb is not None:
            tgt[mb[:, 1:].reshape(-1) == 0] = -100
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               tgt, ignore_index=-100)

    for i in range(0, seqs.shape[0], chunk):
        batch = seqs[i:i + chunk].to(dev)
        mb = None if am is None else am[i:i + chunk].to(dev)
        keep = None if mb is None else mb.bool()

        with splice(model, spec, sae=sae, mu=mu, v1=v1, mode=mode,
                    capture=True, layer=layer) as st:
            _loss(model(batch, attention_mask=mb).logits.float(), batch,
                  mb).backward()
            z, g = st.z.detach(), st.z.grad.detach()
            prod = z * g
            if keep is not None:
                prod = prod * keep.unsqueeze(-1)
            gxi += prod.sum((0, 1)).abs().double()

        acc = torch.zeros(d_sae, device=dev, dtype=torch.float64)
        for s in range(1, ig_steps + 1):
            alpha = s / ig_steps
            with splice(model, spec, sae=sae, mu=mu, v1=v1, mode=mode,
                        capture=True, transform=scale_all(alpha),
                        layer=layer) as st:
                _loss(model(batch, attention_mask=mb).logits.float(), batch,
                      mb).backward()
                # /alpha undoes the chain-rule factor introduced by z -> alpha*z
                prod = st.z.detach() * st.z.grad.detach() / alpha
                if keep is not None:
                    prod = prod * keep.unsqueeze(-1)
                acc += prod.sum((0, 1)).double()
        ig += (acc / ig_steps).abs()
        n_chunks += 1

    n = max(n_chunks, 1)
    return (gxi / n).float().cpu().numpy(), (ig / n).float().cpu().numpy()


def to_array(d_sae, imp: dict):
    """Sparse {feature: dCE} -> dense array with -inf for unevaluated features,
    so ranking never silently promotes something we did not measure."""
    a = np.full(d_sae, -np.inf, dtype=np.float64)
    for j, v in imp.items():
        a[int(j)] = v
    return a


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def screen_report(full_imp: dict, ez2, alive_mask, ks=(20, 50, 100),
                  ms=(200, 500, 1000, 2000)):
    """Validate the variance screen against an EXHAUSTIVE ablation.

    For each screen size m and each top-k of interest, what fraction of the
    true top-k (by measured Delta-CE over the whole dictionary) is contained in
    the top-m by E[z^2]? Recall 1.0 at some m licenses the screen; anything
    less means m must grow or the screen is invalid for this model.
    """
    d = len(ez2)
    imp = to_array(d, full_imp)
    alive = np.asarray(alive_mask, dtype=bool)
    imp_masked = imp.copy()
    imp_masked[~alive] = -np.inf
    rep = {"n_alive": int(alive.sum()), "n_evaluated": int(len(full_imp)),
           "spearman_logez2_vs_dce": spearman(
               np.log(np.maximum(np.asarray(ez2)[alive], 1e-30)),
               np.log(np.maximum(imp_masked[alive], 1e-30))),
           "recall": {}}
    for m in ms:
        cand = set(screen_candidates(ez2, alive, m).tolist())
        for k in ks:
            true_top = np.argsort(imp_masked)[::-1][:k]
            hit = sum(int(j) in cand for j in true_top)
            rep["recall"][f"m{m}_top{k}"] = hit / max(k, 1)
    return rep


def dense_share(imp, dmask, alive, min_denom_frac=0.05):
    """Share of ablation importance carried by the dense (PC1-aligned) cluster.

    Two estimators, because the signed one is ill-conditioned BY CONSTRUCTION.
    dCE can be negative -- ablating a feature sometimes lowers loss -- so the
    signed denominator sum(dCE | alive) is a difference of large opposite-sign
    terms and can land near, or below, zero. When it does the ratio stops being
    a share of anything. Measured overnight, all from `max(denom, 1e-12)`
    failing to guard a NEGATIVE denominator:

        a5 pythia-160m/bypass/mean   denom = -2.14   -> share -6.2e10
        a5 pythia-160m/standard/mean denom =  1.29   -> share  3.30
        b3 gemmascope L12 l0_82      denom ~ 0       -> share -5.3e11
        b1 (every cell)              -inf contamination from unscreened feats

    `positive` uses the positive part of dCE. It lies in [0, 1] by
    construction, it needs no guard, and it agreed with the signed estimator in
    every cell where the signed estimator was well conditioned. It is the one
    to quote. `signed` is emitted only when the denominator carries at least
    `min_denom_frac` of the total absolute mass, and is None otherwise with the
    reason recorded, so a broken cell reads as missing rather than as a number.

    Non-finite entries (fastabl.to_array writes -inf for features the variance
    screen never evaluated) are imputed to 0: the screen is validated at recall
    1.000, so an unevaluated feature is one the screen certified as immaterial.
    """
    imp = np.asarray(imp, dtype=np.float64)
    dmask = np.asarray(dmask, dtype=bool)
    alive = np.asarray(alive, dtype=bool)
    fin = np.isfinite(imp)
    sgn = np.where(fin, imp, 0.0)
    pos = np.where(fin & (imp > 0), imp, 0.0)

    den_signed = float(sgn[alive].sum())
    den_abs = float(np.abs(sgn[alive]).sum())
    den_pos = float(pos[alive].sum())

    ok = den_abs > 0 and den_signed > min_denom_frac * den_abs
    ratio = float(sgn[dmask].sum() / den_signed) if ok else None
    # A "share" outside [0, 1] is not a share, even when the denominator passed
    # the magnitude test: it means content features carry enough NEGATIVE dCE
    # to make the dense subset exceed the whole. Measured overnight at 3.30
    # (a5 pythia-160m/standard/mean) and 1.18 (.../resample). Suppress those
    # too rather than let a >100% "budget" reach a table.
    if ratio is not None and not (-0.01 <= ratio <= 1.01):
        ok, ratio = False, None
    out = dict(
        positive=(float(pos[dmask].sum() / den_pos) if den_pos > 0 else None),
        signed=ratio,
        denominator_signed=den_signed,
        denominator_abs=den_abs,
        n_evaluated=int(fin[alive].sum()),
        n_alive=int(alive.sum()),
    )
    if not ok:
        out["signed_suppressed"] = (
            f"signed estimator unusable (denominator {den_signed:.4g} vs "
            f"absolute mass {den_abs:.4g}; needs a positive denominator above "
            f"{min_denom_frac:g}x that, and a ratio in [0,1]); "
            f"quote `positive` instead")
    return out


def json_safe(obj):
    """Recursively replace NaN/inf with None. json.dump emits bare NaN/Infinity,
    which is not valid JSON -- strict parsers (and every JS consumer) reject the
    file, so a single empty-slice mean poisons a whole results file."""
    import math
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return obj
