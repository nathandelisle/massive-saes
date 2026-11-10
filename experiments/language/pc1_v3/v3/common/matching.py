"""Cross-seed decoder matching and recovery statistics (v2 Section 4, App N)."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .registry import SHARED_COS_THRESHOLD


def cos_matrix(dec_a, dec_b, block=2048, signed=True):
    """Cosine similarity between unit decoder rows of two SAEs, blockwise.

    SIGNED by default, matching v2 and the paper ("maximum cosine similarity").
    Feature activations are non-negative, so -d is not the same feature as +d;
    taking |cos| matches a feature to its own negation and inflates sharing.
    """
    out = np.empty((dec_a.shape[0], dec_b.shape[0]), dtype=np.float32)
    for i in range(0, dec_a.shape[0], block):
        c = dec_a[i:i + block] @ dec_b.T
        out[i:i + block] = c if signed else np.abs(c)
    return out


def max_cos(dec_a, dec_b, alive_b=None, signed=True):
    """Max cosine of each row of A against ALL of B (v2 / paper §4: 'all
    decoder directions'). `alive_b` restricts the pool and is off by default."""
    C = cos_matrix(dec_a, dec_b, signed=signed)
    if alive_b is not None:
        C[:, ~alive_b] = -1.0
    return C.max(axis=1)


def mnn_shared(dec_a, dec_b, thr=SHARED_COS_THRESHOLD):
    C = cos_matrix(dec_a, dec_b)
    best_ab = C.argmax(1)
    best_ba = C.argmax(0)
    mutual = best_ba[best_ab] == np.arange(dec_a.shape[0])
    return mutual & (C.max(1) >= thr)


def hungarian_shared(dec_a, dec_b, idx_a, thr=SHARED_COS_THRESHOLD):
    """One-to-one matching restricted to a candidate feature set idx_a of A
    against all of B (full bipartite on 8k x 8k is fine at these sizes but
    slow; restrict to rows of interest for top-k stats)."""
    C = cos_matrix(dec_a[idx_a], dec_b)
    r, c = linear_sum_assignment(-C)
    return C[r, c] >= thr


def sharing_stats(dec_a, dec_b, alive_a, alive_b, importance_a, k=50,
                  thr=SHARED_COS_THRESHOLD, restrict_to_alive_b=False,
                  signed=True):
    """The paper's core statistics for one ordered seed pair (A = reference)."""
    mc = max_cos(dec_a, dec_b, alive_b=alive_b if restrict_to_alive_b else None,
                 signed=signed)
    shared = mc >= thr
    dict_wide = float(shared[alive_a].mean()) if alive_a.any() else 0.0
    imp = np.asarray(importance_a, dtype=np.float64).copy()
    imp[~alive_a] = -np.inf
    top = np.argsort(imp)[::-1][:k]
    return dict(
        dict_wide=dict_wide,
        top_k=float(shared[top].mean()),
        top_idx=top.tolist(),
        top_shared=shared[top].tolist(),
        max_cos_top=mc[top].tolist(),
    )


def pairwise_over_seeds(decoders, alives, importances, ref_seed_pos=0, k=50,
                        thr=SHARED_COS_THRESHOLD):
    """Mean +/- std over the seed pairs involving the reference seed (v2 Table 1)."""
    ref_d, ref_a, ref_i = (decoders[ref_seed_pos], alives[ref_seed_pos],
                           importances[ref_seed_pos])
    tops, dws = [], []
    per_pair = []
    for p in range(len(decoders)):
        if p == ref_seed_pos:
            continue
        s = sharing_stats(ref_d, decoders[p], ref_a, alives[p], ref_i,
                          k=k, thr=thr)
        tops.append(s["top_k"]); dws.append(s["dict_wide"]); per_pair.append(s)
    return dict(
        top_k_mean=float(np.mean(tops)), top_k_std=float(np.std(tops)),
        dict_wide_mean=float(np.mean(dws)), dict_wide_std=float(np.std(dws)),
        pairs=per_pair,
    )
