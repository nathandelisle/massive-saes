"""Workstream A4 (per dictionary): component interventions and cancellation geometry.

Features studied, all selected without any downstream outcome:
  top50      top 50 by the independent zero-ablation ranking (rank partition)
  dense      the PC1-aligned cluster: alive, cos^2(d_j, v1) > 5 x median over alive (v3 rule)
  controls   for the dense cluster: count-matched random alive features, firing-rate-matched
             and activation-energy-matched features (nearest neighbours outside the cluster),
             each a DISTINCT control

Interventions (rank partition, suffix engine, same-batch baselines):
  full  remove z_j d_j
  par   remove only its PC1 component       z_j (d_j . v1) v1
  perp  remove only its perpendicular part  z_j (d_j - (d_j . v1) v1)
The rest of the splice -- including the routed PC1 in bypass -- is untouched. Effects need not
add in CE, and both reconstruction error and CE are recorded for every item. The same three are
run for whole groups (dense cluster and each control group).

Cancellation (cal partition), with BOTH terms reported, since a ratio can blow up merely
because its denominator approaches zero:
  N2 = sum_ij ||q_j(i)||^2     S2 = sum_i ||sum_j q_j(i)||^2     q_j = z_j d_j_perp
also co-firing, signed coefficient correlations, support size, aggregate effective rank and
conditional norm distributions.
"""
import time

import numpy as np
import torch

from . import common, engine as E, modelio as M
from .common import log


def groups_for(dic, alive, zero_score, rate, cs_ez2, seed=0):
    a = np.asarray(alive, bool)
    Wn = dic.W_dec / dic.W_dec.norm(dim=1, keepdim=True)
    cos2 = (Wn @ dic.v1).pow(2).cpu().numpy()
    med = np.median(cos2[a])
    dense = np.nonzero(a & (cos2 > 5 * med))[0]
    s = np.where(a & np.isfinite(zero_score), zero_score, -np.inf)
    top = np.argsort(-s)[:50]
    rng = np.random.default_rng(seed)
    outside = np.nonzero(a & ~np.isin(np.arange(len(a)), dense))[0]
    n = len(dense)
    g = dict(top50=top, dense=dense)
    if n and len(outside) >= n:
        g["ctl_count"] = rng.choice(outside, size=n, replace=False)

        def matched(stat):
            used, out = set(), []
            for j in dense[np.argsort(-stat[dense])]:
                cand = outside[np.argsort(np.abs(stat[outside] - stat[j]))]
                for c in cand:
                    if c not in used:
                        used.add(int(c)); out.append(int(c)); break
            return np.asarray(out)
        g["ctl_rate"] = matched(np.asarray(rate))
        energy = np.asarray(cs_ez2) * (dic.W_dec - (dic.W_dec @ dic.v1)[:, None] * dic.v1).pow(2).sum(1).cpu().numpy()
        g["ctl_energy"] = matched(energy)
    return g, cos2, float(med)


def directions(dic):
    par = (dic.W_dec @ dic.v1)[:, None] * dic.v1[None, :]
    return dict(full=dic.W_dec, par=par, perp=dic.W_dec - par)


@torch.no_grad()
def cancellation(cache, feats, dic):
    """Closed-form cancellation statistics for a feature set, over the cached partition."""
    f = torch.as_tensor(np.asarray(feats), device=dic.device)
    if len(f) == 0:
        return dict(n=0)
    Dp = dic.W_dec[f] - (dic.W_dec[f] @ dic.v1)[:, None] * dic.v1
    dn2 = Dp.pow(2).sum(1)
    N, T, k = cache.Zidx.shape
    N2 = S2 = L1 = S1 = 0.0
    support = []
    agg_norm = []
    S_sum = torch.zeros(Dp.shape[1], device=dic.device, dtype=torch.float64)
    S_out = torch.zeros(Dp.shape[1], Dp.shape[1], device=dic.device, dtype=torch.float64)
    n_rows = 0
    zcols = []
    for i in range(N):
        L = int(cache.part.lengths[i])
        idx = cache.Zidx[i, :L].long()
        val = cache.Zval[i, :L]
        Z = torch.zeros(L, len(f), device=dic.device)
        for c in range(len(f)):
            Z[:, c] = (val * (idx == f[c])).sum(1)
        q = Z @ Dp                                       # sum_j q_j per position
        N2 += float((Z.pow(2) * dn2).sum())
        L1 += float((Z.abs() * dn2.sqrt()).sum())
        rn = q.norm(dim=1)
        S2 += float(rn.pow(2).sum())
        S1 += float(rn.sum())
        sup = (Z > 0).sum(1)
        support.append(sup.cpu().numpy())
        agg_norm.append(rn[sup > 0].cpu().numpy())
        S_sum += q.double().sum(0)
        S_out += q.double().T @ q.double()
        n_rows += L
        zcols.append(Z.cpu().numpy())
    Zall = np.concatenate(zcols)
    m = S_sum / max(n_rows, 1)
    C = S_out / max(n_rows, 1) - torch.outer(m, m)
    ev = torch.linalg.eigvalsh(C).clamp_min(0)
    pr = float(ev.sum() ** 2 / max(float((ev ** 2).sum()), 1e-30))
    fires = Zall > 0
    co = fires.T.astype(np.float64) @ fires.astype(np.float64)
    nf = fires.sum(0)
    jac = co / np.maximum(nf[:, None] + nf[None, :] - co, 1)
    off = ~np.eye(len(f), dtype=bool)
    with np.errstate(invalid="ignore"):
        cc = np.corrcoef(Zall.T) if len(f) > 1 else np.ones((1, 1))
    sup = np.concatenate(support)
    an = np.concatenate(agg_norm) if agg_norm else np.zeros(1)
    return dict(n=int(len(f)), N2=N2, S2=S2, L1=L1, S1=S1,
                ratio_energy=N2 / S2 if S2 > 0 else None,
                ratio_norm=S1 / L1 if L1 > 0 else None,
                aggregate_participation_ratio=pr,
                mean_pairwise_cofiring_jaccard=float(jac[off].mean()) if off.any() else None,
                mean_pairwise_coef_corr=float(np.nanmean(cc[off])) if off.any() else None,
                support_mean=float(sup.mean()), support_p95=float(np.quantile(sup, .95)),
                frac_rows_any_active=float((sup > 0).mean()),
                agg_norm_given_active_median=float(np.median(an)),
                agg_norm_given_active_p95=float(np.quantile(an, .95)))


@torch.no_grad()
def recon_delta(cache, H_err, F, W):
    """Change in summed squared reconstruction error when z_j d_j (direction W) is removed:
    sum_t ||e_t + z_tj W_j||^2 - ||e_t||^2, e = h - r. One value per feature."""
    out = np.zeros(len(F))
    for n, j in enumerate(F):
        s = cache.seqs_firing(int(j))
        tot = 0.0
        for i in s:
            L = int(cache.part.lengths[i])
            z = ((cache.Zidx[i, :L] == int(j)) * cache.Zval[i, :L]).sum(1)
            d = z[:, None] * W[int(j)][None, :]
            e = H_err[i, :L]
            tot += float((2 * (e * d).sum(1) + d.pow(2).sum(1)).sum())
        out[n] = tot
    return out


def run(model, spec, dic, cache, H_err, alive, zero_score, cs, out):
    """Everything above for one dictionary. Writes <out>/mechanism.npz + mechanism.json."""
    t0 = time.time()
    g, cos2, med = groups_for(dic, alive, zero_score, cs.rate, cs.ez2, seed=1)
    dirs = directions(dic)
    res = dict(groups={k: [int(x) for x in v] for k, v in g.items()}, median_cos2=med,
               n_dense=int(len(g["dense"])))
    # cancellation on the rank partition cache, every group
    res["cancellation"] = {k: cancellation(cache, v, dic) for k, v in g.items()}
    # per-feature component ablations: top50 + dense (capped) + count control (capped)
    feats = np.unique(np.concatenate([g["top50"], g["dense"][:50],
                                      g.get("ctl_count", np.zeros(0, int))[:50]])).astype(np.int64)
    arrays = dict(features=feats)
    zero_fn = lambda s, f, old, am: torch.zeros_like(old)
    for comp, W in dirs.items():
        S, F = E.items_firing(cache, feats)
        dl, ntg = cache.run(S, F, zero_fn, max_tokens=8192, W=W)
        arrays["dce_" + comp] = np.array([dl[F == j].sum() for j in feats]) / cache.n_targets
        arrays["drecon_" + comp] = recon_delta(cache, H_err, feats, W)
    # whole-group ablations
    grp = {}
    for gname, gf in g.items():
        if len(gf) == 0:
            continue
        for comp, W in dirs.items():
            grp["%s_%s" % (gname, comp)] = cache.run_group(np.asarray(gf), W) / cache.n_targets
    res["group_dce"] = grp
    res["seconds"] = time.time() - t0
    common.save_npz(out / "mechanism.npz", pc1_cos2=cos2, **arrays)
    common.write_json(out / "mechanism.json", res)
    log("mech", "%d features, %d groups, %.0fs" % (len(feats), len(g), time.time() - t0))
    return res
