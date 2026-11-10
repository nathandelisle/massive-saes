"""PC1 estimation, exactly per arXiv-v2 Appendix A.

First 50,000 centered activations, cast to float64; covariance eigendecomposition
via np.linalg.eigh. Returns eigenvalues (descending), v1, and variance-explained
spectrum. Also provides bypass projection and dense-cluster identification.
"""

import os

import numpy as np

from .registry import PCA_SAMPLE, DENSE_COS2_MULT


def estimate_pcs(act_dir, n_sample=PCA_SAMPLE, cache=True):
    cache_path = os.path.join(act_dir, f"pca_{n_sample}.npz")
    if cache and os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["evals"], z["evecs"], z["mean"]
    acts = np.load(os.path.join(act_dir, "acts.npy"), mmap_mode="r")
    mean = np.load(os.path.join(act_dir, "mean.npy"))
    X = np.asarray(acts[:n_sample], dtype=np.float64) - mean[None, :]
    C = (X.T @ X) / (X.shape[0] - 1)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]   # columns are PCs, descending
    if cache:
        np.savez(cache_path, evals=evals, evecs=evecs, mean=mean)
    return evals, evecs, mean


def pc1_stats(evals, evecs, mean):
    v1 = evecs[:, 0]
    total = evals.sum()
    mu = mean / (np.linalg.norm(mean) + 1e-12)
    return dict(
        pc1_var_explained=float(evals[0] / total),
        pc2_var_explained=float(evals[1] / total),
        spectral_gap=float(evals[0] / max(evals[1], 1e-30)),
        cos_pc1_mean=float(abs(v1 @ mu)),
        R=float(evals[0] / max(total - evals[0], 1e-30)),
    ), v1


def project_out(X, v1):
    """x_bypass = (x - mu) - ((x - mu)^T v1) v1 ; expects already-centered X."""
    coef = X @ v1
    return X - np.outer(coef, v1), coef


def project_out_torch(x, v1_t):
    coef = x @ v1_t
    return x - coef[:, None] * v1_t[None, :], coef


def dense_cluster(decoder, v1, alive_mask, mult=DENSE_COS2_MULT):
    """Dense/PC1-aligned features: decoder cos^2 with v1 > mult x median over
    alive features (v2 Sections 3.2, App D). decoder: [d_sae, d_model] unit rows.
    Returns (bool mask over full dict, cos2 array, threshold)."""
    cos = decoder @ v1
    cos2 = cos ** 2
    med = np.median(cos2[alive_mask]) if alive_mask.any() else np.median(cos2)
    thr = mult * med
    mask = (cos2 > thr) & alive_mask
    return mask, cos2, float(thr)


def layer_profile(spec, harvest_fn, out_json, corpus="wikitext",
                  n_tokens=500_000, device="cuda"):
    """PC1/PC2 variance explained at every layer (v2 Table 6 protocol)."""
    import json
    rows = []
    for layer in range(1, spec.n_layers + 1):
        d = harvest_fn(spec, layer=layer, n_tokens=n_tokens,
                       corpus=corpus, device=device)
        evals, evecs, mean = estimate_pcs(d)
        s, _ = pc1_stats(evals, evecs, mean)
        rows.append(dict(layer=layer, **s))
    json.dump(rows, open(out_json, "w"), indent=2)
    return rows

def pc1_alignment_spectrum(decoder, v1, alive_mask=None,
                           cos_thresholds=(0.75, 0.5, 0.3, 0.1)):
    """Count PC1-aligned atoms at SEVERAL criteria, not just ours.

    WHY THIS EXISTS. Sun, Stolfo et al., "Dense SAE Latents Are Features, Not
    Bugs" (NeurIPS 2025, arXiv:2506.15679) report that on Gemma-2-2B -- the same
    model and layer we audit, using the same Gemma Scope dictionaries -- only
    ONE ANTIPODAL PAIR per layer aligns with PC1 at cosine > 0.75, and that
    other dense latents carry little top-PC mass. Our dense-cluster criterion
    (cos^2 > 5x the median over alive atoms) admits atoms at cos ~ 0.1-0.3, so
    it can return many atoms while their criterion returns two. Those are not
    contradictory measurements, but reporting only ours would look like a
    contradiction, and reporting only theirs would hide the effect we care
    about. So report BOTH, always, from one function.

    Returns counts at each cosine threshold, the signed-cosine extremes (their
    "antipodal pair" is a +v1 atom and a -v1 atom), and our 5x-median cluster
    size, so a reader can see exactly which criterion produces which number.
    """
    import numpy as _np
    cos = decoder @ v1
    cos2 = cos ** 2
    alive = (_np.ones(len(cos), bool) if alive_mask is None
             else _np.asarray(alive_mask, bool))
    med = float(_np.median(cos2[alive])) if alive.any() else float(_np.median(cos2))
    out = {
        "n_alive": int(alive.sum()),
        "median_cos2_alive": med,
        "ours_5x_median_n": int(((cos2 > DENSE_COS2_MULT * med) & alive).sum()),
        "max_cos_signed": float(cos[alive].max()) if alive.any() else float("nan"),
        "min_cos_signed": float(cos[alive].min()) if alive.any() else float("nan"),
        "antipodal_pair_present": bool(
            alive.any() and cos[alive].max() > 0.75 and cos[alive].min() < -0.75),
    }
    for t in cos_thresholds:
        out[f"n_abscos_gt_{t}"] = int(((_np.abs(cos) > t) & alive).sum())
    # how much of the dictionary's total PC1 mass the top-2 atoms carry: if it
    # is most of it, Sun et al.'s "one pair" reading is the right one and ours
    # is picking up a long low-cosine tail
    order = _np.argsort(-cos2 * alive)
    tot = float(cos2[alive].sum()) if alive.any() else float("nan")
    out["top2_share_of_pc1_mass"] = (float(cos2[order[:2]].sum() / tot)
                                     if tot and tot > 0 else float("nan"))
    out["top10_share_of_pc1_mass"] = (float(cos2[order[:10]].sum() / tot)
                                      if tot and tot > 0 else float("nan"))
    # ABSOLUTE PC1 mass in the top atoms. Unlike every fraction above, this does
    # not renormalise by the dictionary's own total, so it separates standard
    # from bypass by orders of magnitude instead of by a few percent (measured:
    # gpt2 1.45 vs 0.106, pythia-1.4b 1.48 vs 0.0016, gemma-2-2b 1.91 vs 0.0037).
    out["top10_pc1_mass"] = float(cos2[order[:10]].sum())
    out["top2_pc1_mass"] = float(cos2[order[:2]].sum())
    # THE CHANCE FLOOR of our own 5x-median rule. For randomly oriented unit
    # decoders in R^d, cos^2 ~ Beta(1/2, (d-1)/2), and P(cos^2 > 5 x median) is
    # ~0.13 for every d of interest. So "13% of atoms are dense" is what PURE
    # NOISE produces, and a fraction near that number is not evidence of a
    # cluster. This is not hypothetical: the D2 case study compared 0.185
    # (their ReLU+L1) against 0.160 (our TopK) and returned a null, when both
    # sit on a 0.131 floor and no atom in either exceeded cos^2 = 0.13.
    d_model = int(decoder.shape[1])
    out["chance_5x_median_frac"] = _chance_5x_median(d_model)
    out["excess_over_chance"] = (out["ours_5x_median_n"] / max(int(alive.sum()), 1)
                                 - out["chance_5x_median_frac"])
    return out


def _chance_5x_median(d_model):
    """P(cos^2 > 5 x median) for a unit vector uniform on the sphere in R^d.

    cos^2 against a fixed axis is Beta(1/2, (d-1)/2). Falls back to a Monte
    Carlo estimate if scipy is unavailable, because this number gates whether a
    reported cluster is real and must never silently go missing.
    """
    import numpy as _np
    try:
        from scipy.stats import beta as _beta
        B = _beta(0.5, (d_model - 1) / 2.0)
        return float(1.0 - B.cdf(5.0 * B.ppf(0.5)))
    except Exception:
        rng = _np.random.default_rng(0)
        W = rng.standard_normal((20000, d_model))
        c2 = (W[:, 0] ** 2) / (W ** 2).sum(1)
        return float((c2 > 5 * _np.median(c2)).mean())
