"""Per-model aux job: A1 (what PC1 tracks) and A3 (functional necessity + writer blocks).

A1 -- five prediction targets per valid position:
  entropy     of the native next-token distribution (float32)
  surprisal   -log p(actual next token)
  logfreq     log(1 + count) of the CURRENT token in the 5M-token training store, a corpus
              separate from every evaluation partition (counted in prep)
  logpos      log(1 + position)
  sink        mean over heads of block b+1's attention from this query to position 0, i.e. the
              DOWNSTREAM attention a post-block vector could cause (same-block attention has
              already happened); undefined at position 0 itself and reported as such
Feature sets (identical document splits; standardisation fitted on train only; ridge alpha by
5-fold inner CV; five inner/data seeds; fixed outer test = `a1test`):
  pc1 scalar | residual with PC1 removed | raw activation | 32 random raw 1-D projections |
  32 random 1-D projections orthogonal to PC1 | pc2 scalar | norm scalar | position only
Raw random projections can themselves contain PC1, which is why both random families appear.
Primary metric: out-of-sample R^2 (negative values kept) and MAE, with an article-level
bootstrap interval. A scalar spline (knots chosen in inner CV) is reported beside the linear
scalar probes, for PC1 and for the equally flexible scalar controls; the linear result stays
primary. Strata: all valid / first position / non-first / actual BOS token / massive
(norm > 10 x calibration median). Robustness: a second corpus (FineWeb-Edu) with the fitted PCA
held FIXED, and separately with PCA re-fitted.

A3 -- on the `rank` partition, CE of distinct rows:
  native | centred PC1 deletion | uncentred-direction deletion | per-position training-mean
  coefficient | All-but-the-Top (D = d/100; mean kept, and strict: mean removed too) |
  random-direction deletions (5) | PC2 / PC5 / PC10 deletions | standard and bypass splices
Writer blocks (causal test of the early-write account): candidate blocks before the site with
the largest |write| along the site's PC1 (from prep's calibration profile). Their write along v1
is suppressed at held-out tokens; measured: site PC1 coefficient, downstream sink mass, CE.
Controls: suppress a random direction of matched write variance; suppress only at position 0 /
only at positions > 0. Restoration: add the removed component back right after the writer
(algebra check) or restore the native PC1 coefficient at the SITE (propagation; may include
mediated effects).
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import common, modelio as M, partitions as P
from .common import log
from .evaljob import fetch_prep, fetch_sae

N_RAND = 32
ALPHAS = np.logspace(-3, 4, 15)


# ---------------------------------------------------------------------------------------
# A1 data collection
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def _tok_budget(spec, base):
    return base if spec.d_model < 3000 else max(256, base // 4)


def collect(model, tok, spec, part, unigram, max_tokens=4096):
    max_tokens = _tok_budget(spec, max_tokens)
    """Per valid position: site residual (fp32), and the five targets."""
    b = M.site_index(spec)
    blk_next = M.blocks(model, spec)[b + 1] if b + 1 < len(M.blocks(model, spec)) else None
    H, ent, sur, lf, lp, sink, pos_all, doc_all, bos = [], [], [], [], [], [], [], [], []
    bos_id = tok.bos_token_id
    for sel, ids, am in part.batches(max_tokens=max_tokens):
        o = model(ids.to(M.input_device(model)), attention_mask=am.to(M.input_device(model)),
                  output_hidden_states=True, output_attentions=True)
        h = o.hidden_states[b + 1].float()
        lg = o.logits.float()
        lsm = torch.log_softmax(lg, -1)
        e = -(lsm.exp() * lsm).nan_to_num(0.0).sum(-1)   # 0*log0 = 0 (else NaN where p underflows)
        nxt = torch.cat([ids[:, 1:], ids[:, -1:]], 1).to(lsm.device)
        s = -lsm.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
        att = o.attentions[b + 1] if (o.attentions and b + 1 < len(o.attentions)) else None
        snk = att[:, :, :, 0].mean(1).float() if att is not None else torch.full_like(e, float("nan"))
        B, T = ids.shape
        valid = am.bool().clone()
        valid[:, -1] = False
        for i in range(B):
            L = int(am[i].sum())
            v = torch.arange(T) < (L - 1)                                   # has a next token
            idx = torch.nonzero(v).squeeze(1)
            H.append(h[i, idx].cpu())
            ent.append(e[i, idx].cpu()); sur.append(s[i, idx].cpu())
            tid = ids[i, idx]
            lf.append(torch.log1p(torch.as_tensor(unigram[np.minimum(tid.numpy(), len(unigram) - 1)], dtype=torch.float32)))
            lp.append(torch.log1p(idx.float()))
            sk = snk[i, idx].cpu().clone()
            sk[idx == 0] = float("nan")
            sink.append(sk)
            pos_all.append(idx.numpy())
            doc_all.append(np.full(len(idx), part.doc[sel[i]]))
            bos.append((tid == bos_id).numpy() if bos_id is not None else np.zeros(len(idx), bool))
    cat = lambda xs: torch.cat(xs).numpy()
    return dict(H=cat(H), targets=dict(entropy=cat(ent), surprisal=cat(sur), logfreq=cat(lf),
                                       logpos=cat(lp), sink=cat(sink)),
                pos=np.concatenate(pos_all), doc=np.concatenate(doc_all), bos=np.concatenate(bos))


def _ridge_cv(Xtr, ytr, seed, folds=5):
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    kf = KFold(folds, shuffle=True, random_state=seed)
    sc = np.zeros(len(ALPHAS))
    for tr, va in kf.split(Xtr):
        mu, sd = Xtr[tr].mean(0), Xtr[tr].std(0) + 1e-8
        for k, a in enumerate(ALPHAS):
            m = Ridge(alpha=a).fit((Xtr[tr] - mu) / sd, ytr[tr])
            p = m.predict((Xtr[va] - mu) / sd)
            sc[k] += 1 - ((ytr[va] - p) ** 2).sum() / max(((ytr[va] - ytr[va].mean()) ** 2).sum(), 1e-12)
    a = ALPHAS[int(np.argmax(sc))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return Ridge(alpha=a).fit((Xtr - mu) / sd, ytr), mu, sd, float(a)


def _ridge_cv_gpu(Xtr, ytr, Xte, seed, folds=5, alphas=np.logspace(-3, 6, 19)):
    """Same estimator as _ridge_cv (standardised ridge, alpha by K-fold CV), solved through one
    eigendecomposition of the Gram matrix per fold so every alpha is free. For d in the
    thousands, where refitting sklearn per alpha would take hours."""
    from sklearn.model_selection import KFold
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.as_tensor(Xtr, dtype=torch.float32, device=dev)
    y = torch.as_tensor(ytr, dtype=torch.float64, device=dev)

    def fit(idx):
        Xa = X[idx]; mu = Xa.mean(0); sd = Xa.std(0) + 1e-6
        Xa = (Xa - mu) / sd
        ym = y[idx].mean()
        G = (Xa.T @ Xa).double()
        b = Xa.double().T @ (y[idx] - ym)
        w, U = torch.linalg.eigh(G)
        return mu, sd, ym, w.clamp_min(0), U, U.T @ b

    def coef(st, a):
        mu, sd, ym, w, U, Ub = st
        return U @ (Ub / (w + a))

    def pred(st, c, Xq):
        mu, sd, ym = st[:3]
        return ((Xq - mu) / sd).double() @ c + ym
    sc = np.zeros(len(alphas))
    for tr, va in KFold(folds, shuffle=True, random_state=seed).split(Xtr):
        st = fit(torch.as_tensor(tr, device=dev))
        yv = y[torch.as_tensor(va, device=dev)]
        Xv = X[torch.as_tensor(va, device=dev)]
        for k, a in enumerate(alphas):
            p = pred(st, coef(st, a), Xv)
            sc[k] += float(1 - ((yv - p) ** 2).sum() / ((yv - yv.mean()) ** 2).sum().clamp_min(1e-12))
    a = float(alphas[int(np.argmax(sc))])
    st = fit(torch.arange(len(ytr), device=dev))
    p = pred(st, coef(st, a), torch.as_tensor(Xte, dtype=torch.float32, device=dev))
    return p.cpu().numpy(), a


def _spline_cv(xtr, ytr, seed, folds=5, knots=(3, 5, 8, 12, 20)):
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import SplineTransformer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    kf = KFold(folds, shuffle=True, random_state=seed)
    best, bk = -np.inf, knots[0]
    for k in knots:
        s = 0.0
        for tr, va in kf.split(xtr):
            m = make_pipeline(SplineTransformer(n_knots=k, degree=3, extrapolation="linear"),
                              Ridge(alpha=1e-3)).fit(xtr[tr], ytr[tr])
            p = m.predict(xtr[va])
            s += 1 - ((ytr[va] - p) ** 2).sum() / max(((ytr[va] - ytr[va].mean()) ** 2).sum(), 1e-12)
        if s > best:
            best, bk = s, k
    m = make_pipeline(SplineTransformer(n_knots=bk, degree=3, extrapolation="linear"),
                      Ridge(alpha=1e-3)).fit(xtr, ytr)
    return m, bk


def _r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))


def _boot(y, p, doc, n=500, seed=0):
    rng = np.random.default_rng(seed)
    ud = np.unique(doc)
    idx = {d: np.nonzero(doc == d)[0] for d in ud}
    out = []
    for _ in range(n):
        s = np.concatenate([idx[d] for d in rng.choice(ud, len(ud))])
        out.append(_r2(y[s], p[s]))
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def feature_sets(H, mu, v1, v2, R_raw, R_orth, pos):
    X = H - mu
    c1 = X @ v1
    return {"pc1": c1[:, None], "pc1_removed": X - np.outer(c1, v1), "raw": H,
            "pc2": (X @ v2)[:, None], "norm": np.linalg.norm(H, axis=1)[:, None],
            "position": np.log1p(pos)[:, None],
            **{"rand_raw_%02d" % i: (X @ R_raw[i])[:, None] for i in range(N_RAND)},
            **{"rand_orth_%02d" % i: (X @ R_orth[i])[:, None] for i in range(N_RAND)}}


def a1(model, tok, spec, mean, pca, unigram, out, seeds=(0, 1, 2, 3, 4), max_seqs=None):
    f = out / "a1.json"
    if common.fresh(f.parent, f.name, "aux"):
        return json.loads(f.read_text())
    t0 = time.time()
    tr = collect(model, tok, spec, P.Partition("cal", tok, max_seqs=max_seqs), unigram)
    te = collect(model, tok, spec, P.Partition("a1test", tok, max_seqs=max_seqs), unigram)
    v1, v2 = pca["v1"].astype(np.float64), pca["evecs"][:, 1].astype(np.float64)
    d = len(v1)
    rng = np.random.default_rng(20261005)
    R_raw = rng.standard_normal((N_RAND, d)); R_raw /= np.linalg.norm(R_raw, axis=1, keepdims=True)
    R_orth = R_raw - np.outer(R_raw @ v1, v1); R_orth /= np.linalg.norm(R_orth, axis=1, keepdims=True)
    Ftr = feature_sets(tr["H"], mean, v1, v2, R_raw, R_orth, tr["pos"])
    Fte = feature_sets(te["H"], mean, v1, v2, R_raw, R_orth, te["pos"])
    med = np.median(np.linalg.norm(tr["H"], axis=1))
    strata_te = dict(all=np.ones(len(te["pos"]), bool), first=te["pos"] == 0, nonfirst=te["pos"] > 0,
                     bos=te["bos"], massive=np.linalg.norm(te["H"], axis=1) > 10 * med)
    res = dict(n_train=int(len(tr["pos"])), n_test=int(len(te["pos"])),
               strata_counts={k: int(v.sum()) for k, v in strata_te.items()}, targets={},
               nonfinite_targets={k: dict(train=int((~np.isfinite(tr["targets"][k])).sum()),
                                          test=int((~np.isfinite(te["targets"][k])).sum()))
                                  for k in tr["targets"]})
    for tname in tr["targets"]:
        ytr, yte = tr["targets"][tname], te["targets"][tname]
        mtr, mte = np.isfinite(ytr), np.isfinite(yte)
        tres = {}
        for fname in Ftr:
            per_seed = []
            hi_d = Ftr[fname].shape[1] > 1
            # the estimator is deterministic: seeds change only the CV fold assignment. For the
            # two d-dimensional sets each seed costs 6 eigendecompositions, so 2 seeds there.
            for s in (seeds[:2] if hi_d else seeds):
                if hi_d:
                    p, a = _ridge_cv_gpu(Ftr[fname][mtr], ytr[mtr], Fte[fname][mte], s)
                else:
                    m, mu_, sd_, a = _ridge_cv(Ftr[fname][mtr], ytr[mtr], s)
                    p = m.predict((Fte[fname][mte] - mu_) / sd_)
                per_seed.append(dict(seed=s, alpha=a, r2=_r2(yte[mte], p),
                                     mae=float(np.abs(yte[mte] - p).mean())))
                if s == seeds[0]:
                    p0 = p
            row = dict(r2_mean=float(np.mean([x["r2"] for x in per_seed])),
                       r2_per_seed=[x["r2"] for x in per_seed],
                       mae_mean=float(np.mean([x["mae"] for x in per_seed])),
                       r2_ci_article_boot=_boot(yte[mte], p0, te["doc"][mte]))
            st = {}
            for sname, smask in strata_te.items():
                mm = smask[mte]
                st[sname] = (_r2(yte[mte][mm], p0[mm]) if mm.sum() > 20 and np.var(yte[mte][mm]) > 0
                             else "unavailable (n=%d)" % int(mm.sum()))
            row["r2_by_stratum"] = st
            tres[fname] = row
        for fname in ("pc1", "pc2", "norm", "rand_raw_00", "rand_orth_00"):
            m, k = _spline_cv(Ftr[fname][mtr], ytr[mtr], seeds[0])
            tres[fname]["spline_r2"] = _r2(yte[mte], m.predict(Fte[fname][mte]))
            tres[fname]["spline_knots"] = k
        rr = [tres["rand_raw_%02d" % i]["r2_mean"] for i in range(N_RAND)]
        ro = [tres["rand_orth_%02d" % i]["r2_mean"] for i in range(N_RAND)]
        tres["random_raw_distribution"] = dict(mean=float(np.mean(rr)), q05=float(np.quantile(rr, .05)),
                                               q95=float(np.quantile(rr, .95)), max=float(np.max(rr)))
        tres["random_orth_distribution"] = dict(mean=float(np.mean(ro)), q05=float(np.quantile(ro, .05)),
                                                q95=float(np.quantile(ro, .95)), max=float(np.max(ro)))
        res["targets"][tname] = tres
    # second corpus: PCA held fixed, and re-fitted
    res["second_corpus"] = second_corpus(model, tok, spec, mean, v1, unigram, Ftr, tr, seeds[0])
    res["seconds"] = time.time() - t0
    common.write_json(f, res)
    common.pin(f.parent, f.name, "aux")
    log("a1", "done %.0fs" % (time.time() - t0))
    return res


def second_corpus(model, tok, spec, mean, v1, unigram, Ftr, tr, seed, n_docs=400):
    from datasets import load_dataset
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    except Exception as e:
        return dict(status="unavailable", why=repr(e)[:200])

    class _P:
        pass
    p = _P()
    texts = []
    for i, r in enumerate(ds):
        if len(texts) >= n_docs:
            break
        texts.append(r["text"])
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids, am = [], []
    for t in texts:
        x = tok(t, truncation=True, max_length=P.SEQ_LEN)["input_ids"]
        ids.append(x + [pad] * (P.SEQ_LEN - len(x))); am.append([1] * len(x) + [0] * (P.SEQ_LEN - len(x)))
    p.ids, p.am = torch.tensor(ids), torch.tensor(am)
    p.lengths = p.am.sum(1).numpy(); p.doc = np.arange(len(ids)); p.row = np.arange(len(ids))

    def batches(max_tokens=4096, max_seqs=64):
        o = np.argsort(-p.lengths, kind="stable"); i = 0
        while i < len(o):
            T = int(p.lengths[o[i]]); n = max(1, min(max_seqs, max_tokens // T)); s = o[i:i + n]
            yield s, p.ids[s, :T], p.am[s, :T]; i += n
    p.batches = batches
    te = collect(model, tok, spec, p, unigram)
    out = dict(status="ok", n_tokens=int(len(te["pos"])))
    X = te["H"] - mean
    for cond, vv in (("pca_fixed", v1),):
        c = (X @ vv)[:, None]
        for tname in ("entropy", "surprisal", "logfreq", "logpos"):
            yt, ye = tr["targets"][tname], te["targets"][tname]
            ft, fe = np.isfinite(yt), np.isfinite(ye)
            m, mu_, sd_, a = _ridge_cv(Ftr["pc1"][ft], yt[ft], seed)
            out["%s_%s_r2" % (cond, tname)] = _r2(ye[fe], m.predict((c[fe] - mu_) / sd_))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xc = torch.as_tensor(te["H"][:50000], dtype=torch.float32, device=dev)
    Xc = Xc - Xc.mean(0)
    ev, evec = torch.linalg.eigh((Xc.T @ Xc).double() / len(Xc))
    ev, vr = ev.cpu().numpy(), evec[:, -1].cpu().numpy()
    Xf = torch.as_tensor(X[:50000], dtype=torch.float32, device=dev)
    fixed_share = float((Xf @ torch.as_tensor(v1, dtype=torch.float32, device=dev)).var() /
                        (Xf - Xf.mean(0)).pow(2).sum(1).mean())
    del Xc, Xf
    cr = ((te["H"] - te["H"].mean(0)) @ vr)[:, None]
    half = te["doc"] < np.median(np.unique(te["doc"]))
    for tname in ("entropy", "surprisal", "logfreq", "logpos"):
        y = te["targets"][tname]
        tr_m, te_m = half & np.isfinite(y), ~half & np.isfinite(y)
        m, mu_, sd_, a = _ridge_cv(cr[tr_m], y[tr_m], seed)
        out["pca_refit_%s_r2" % tname] = _r2(y[te_m], m.predict((cr[te_m] - mu_) / sd_))
    out["refit_pc1_cos_with_fixed"] = float(abs(vr @ v1))
    out["refit_pc1_share"] = float(ev[-1] / ev.sum())
    out["fixed_pc1_share_on_this_corpus"] = fixed_share
    return out


# ---------------------------------------------------------------------------------------
# A3: functional necessity
# ---------------------------------------------------------------------------------------
def _ce_with_site(model, spec, part, fn, max_tokens=4096):
    """CE (mean over valid targets) with the site residual replaced by fn(h, pos)."""
    b = M.site_index(spec)
    blk = M.blocks(model, spec)[b]
    tot, n = 0.0, 0
    state = {}

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        hn = fn(h.float(), state["pos"].to(h.device)).to(h.dtype)
        return (hn,) + tuple(o[1:]) if isinstance(o, tuple) else hn
    hd = blk.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for sel, ids, am in part.batches(max_tokens=max_tokens):
                state["pos"] = torch.arange(ids.shape[1])[None, :].expand_as(ids)
                lg = model(ids.to(M.input_device(model)), attention_mask=am.to(M.input_device(model))).logits
                nll, val = M.token_nll(lg, ids, am)
                tot += float(nll[val].sum()); n += int(val.sum())
    finally:
        hd.remove()
    return tot / max(n, 1)


def a3(model, tok, spec, mean, pca, prep, dics, out, max_seqs=None):
    f = out / "a3.json"
    if common.fresh(f.parent, f.name, "aux"):
        return json.loads(f.read_text())
    t0 = time.time()
    part = P.Partition("rank", tok, max_seqs=max_seqs)
    dev = M.site_device(model, spec)
    mu = torch.as_tensor(mean, dtype=torch.float32, device=dev)
    E = torch.as_tensor(pca["evecs"], dtype=torch.float32, device=dev)
    v1 = E[:, 0]
    d = len(mean)
    D = max(1, d // 100)
    cal = P.Partition("cal", tok, max_seqs=max_seqs)
    # per-position training (calibration) mean of the centred coefficient
    sums, cnts = torch.zeros(P.SEQ_LEN, device=dev), torch.zeros(P.SEQ_LEN, device=dev)
    for sel, ids, am in cal.batches(max_tokens=4096):
        h = M.prefix(model, spec, ids, am).float().to(dev)
        c = (h - mu) @ v1
        m = am.to(dev).float()
        sums[:c.shape[1]] += (c * m).sum(0); cnts[:c.shape[1]] += m.sum(0)
    posmean = sums / cnts.clamp_min(1)
    rng = np.random.default_rng(3)
    rows = {}
    rows["native"] = _ce_with_site(model, spec, part, lambda h, p: h)
    rows["pc1_delete_centred"] = _ce_with_site(model, spec, part, lambda h, p: h - ((h - mu) @ v1)[..., None] * v1)
    rows["pc1_delete_uncentred_direction"] = _ce_with_site(model, spec, part, lambda h, p: h - (h @ v1)[..., None] * v1)
    rows["pc1_coef_to_training_mean_zero"] = rows["pc1_delete_centred"]
    rows["pc1_coef_to_position_mean"] = _ce_with_site(
        model, spec, part, lambda h, p: h - (((h - mu) @ v1) - posmean[p.clamp_max(P.SEQ_LEN - 1)])[..., None] * v1)
    U = E[:, :D]
    rows["abtt_D%d_mean_kept" % D] = _ce_with_site(model, spec, part, lambda h, p: h - ((h - mu) @ U) @ U.T)
    rows["abtt_D%d_strict_mean_removed" % D] = _ce_with_site(model, spec, part, lambda h, p: (h - mu) - ((h - mu) @ U) @ U.T)
    for k in (1, 4, 9):
        vk = E[:, k]
        rows["pc%d_delete_centred" % (k + 1)] = _ce_with_site(model, spec, part, lambda h, p, vk=vk: h - ((h - mu) @ vk)[..., None] * vk)
    for i in range(5):
        r = torch.as_tensor(rng.standard_normal(d), dtype=torch.float32, device=dev)
        r = r / r.norm()
        rows["random_dir_delete_%d" % i] = _ce_with_site(model, spec, part, lambda h, p, r=r: h - ((h - mu) @ r)[..., None] * r)
    for name, dic in dics.items():
        def sp(h, p, dic=dic):
            B, T, dd = h.shape
            z, coef = dic.encode(h.reshape(-1, dd))
            return dic.splice_fp32(z, coef).reshape(B, T, dd)
        rows["splice_" + name] = _ce_with_site(model, spec, part, sp)
    res = dict(rows=rows, D_abtt=D, n_targets=part.n_targets,
               note=("pc1_coef_to_training_mean_zero equals centred deletion by construction: the "
                     "training mean of the centred coefficient is 0. The position-conditioned mean "
                     "replacement is the informative variant. An exact projection-and-restore "
                     "identity is not an SAE reconstruction result."))
    res["writers"] = writers(model, tok, spec, mean, pca, prep, part)
    res["seconds"] = time.time() - t0
    common.write_json(f, res)
    common.pin(f.parent, f.name, "aux")
    log("a3", "done %.0fs" % (time.time() - t0))
    return res


def _sub(block, family):
    from .prep import _sublayers
    return _sublayers(block, family)


def writers(model, tok, spec, mean, pca, prep, part, n_cand=2):
    """Suppress the candidate writer blocks' MLP write along v1; measure site PC1 coefficient,
    downstream sink mass, CE; controls and restorations."""
    prof_f = common.WORK / "prep" / spec.key / "layer_profile.json"
    common.s3_get("%s/%s/prep/layer_profile.json" % (common.SHARED, spec.key), prof_f)
    prof = json.loads(prof_f.read_text())
    b = M.site_index(spec)
    cands = sorted([p for p in prof if p["block"] < b],
                   key=lambda p: -abs(p["mlp_write_on_site_pc1_absmean"]))[:n_cand]
    blks = M.blocks(model, spec)
    dev = M.site_device(model, spec)
    mu = torch.as_tensor(mean, dtype=torch.float32, device=dev)
    v1 = torch.as_tensor(pca["v1"], dtype=torch.float32, device=dev)
    rng = np.random.default_rng(5)
    out = dict(candidates=[c["block"] for c in cands], conditions={})

    batches = list(part.batches(max_tokens=_tok_budget(spec, 2048)))
    native_coefs = []
    with torch.no_grad():
        for sel, ids, am in batches:
            hh = M.prefix(model, spec, ids, am).float()
            native_coefs.append((hh - mu.to(hh.device)) @ v1.to(hh.device))

    def measure(edit_block, edit_fn, restore_site=False):
        hooks, st = [], {}
        if edit_block is not None:
            _, mlp = _sub(blks[edit_block], spec.family)

            def mh(m, i, o):
                w = o[0] if isinstance(o, tuple) else o
                wn = edit_fn(w.float(), st["pos"].to(w.device)).to(w.dtype)
                return (wn,) + tuple(o[1:]) if isinstance(o, tuple) else wn
            hooks.append(mlp.register_forward_hook(mh))

        def sh(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            hf = h.float()
            c = (hf - mu.to(h.device)) @ v1.to(h.device)
            st.setdefault("coef", []).append(c[st["am"].to(h.device)].abs().mean().item())
            if restore_site:
                hf = hf + (st["native_coef"].to(h.device) - c)[..., None] * v1.to(h.device)
                hn = hf.to(h.dtype)
                return (hn,) + tuple(o[1:]) if isinstance(o, tuple) else hn
            return o
        hooks.append(blks[b].register_forward_hook(sh))
        tot, n, sink = 0.0, 0, []
        try:
            with torch.no_grad():
                for bi, (sel, ids, am) in enumerate(batches):
                    st["pos"] = torch.arange(ids.shape[1])[None, :].expand_as(ids)
                    st["am"] = am.bool()
                    st["native_coef"] = native_coefs[bi]
                    o = model(ids.to(M.input_device(model)), attention_mask=am.to(M.input_device(model)),
                              output_attentions=True)
                    nll, val = M.token_nll(o.logits, ids, am)
                    tot += float(nll[val].sum()); n += int(val.sum())
                    if o.attentions and b + 1 < len(o.attentions):
                        a = o.attentions[b + 1][:, :, 1:, 0].mean(1)
                        sink.append(a[am[:, 1:].bool().to(a.device)].float().mean().item())
        finally:
            for hd in hooks:
                hd.remove()
        return dict(ce=tot / max(n, 1), site_pc1_abs_coef=float(np.mean(st.get("coef", [0]))),
                    sink_mass_block_b1=float(np.mean(sink)) if sink else None)

    out["conditions"]["native"] = measure(None, None)
    for c in cands:
        blk = c["block"]
        kill = lambda w, p: w - (w @ v1.to(w.device))[..., None] * v1.to(w.device)
        out["conditions"]["suppress_b%d" % blk] = measure(blk, kill)
        out["conditions"]["suppress_b%d_pos0_only" % blk] = measure(
            blk, lambda w, p: torch.where((p == 0)[..., None], kill(w, p), w))
        out["conditions"]["suppress_b%d_nonfirst_only" % blk] = measure(
            blk, lambda w, p: torch.where((p > 0)[..., None], kill(w, p), w))
        r = torch.as_tensor(rng.standard_normal(len(v1)), dtype=torch.float32, device=dev)
        r = r - (r @ v1) * v1; r = r / r.norm()
        out["conditions"]["suppress_b%d_random_direction" % blk] = measure(
            blk, lambda w, p, r=r: w - (w @ r.to(w.device))[..., None] * r.to(w.device))
        out["conditions"]["suppress_b%d_restore_same_site" % blk] = measure(
            blk, lambda w, p: kill(w, p) + (w @ v1.to(w.device))[..., None] * v1.to(w.device))
        out["conditions"]["suppress_b%d_restore_at_site" % blk] = measure(blk, kill, restore_site=True)
    out["note"] = ("restore_same_site = suppression followed immediately by adding the removed "
                   "component back at the writer's output (an algebra check: must equal native). "
                   "restore_at_site = native PC1 coefficient restored at the SAE site; tests "
                   "propagation and can include mediated effects. If the sink link is only "
                   "correlational here, that is what it is reported as.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--max-seqs", type=int, default=None)
    a = ap.parse_args(argv)
    spec = M.spec_of(a.model)
    out = common.RESULTS / "aux" / a.model
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prep, mean, pca, alive_npz = fetch_prep(a.model)
    uf = common.WORK / "prep" / a.model / "unigram_counts.npy"
    common.s3_get("%s/%s/prep/unigram_counts.npy" % (common.SHARED, a.model), uf)
    unigram = np.load(uf)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = getattr(torch, a.dtype) if a.dtype else getattr(torch, spec.dtype)
    if spec.dtype == "float16" and spec.family != "gpt2" and not a.dtype:
        # EAGER attention overflows in fp16 for these models (measured: 81% NaN attention weights on
        # Pythia-160M; Qwen-2.5-7B surprisal non-finite on 58k/66k rows); the default sdpa kernel
        # used by every other job does not. Eager is needed for attention weights, so use a
        # full-range dtype. GPT-2 (fp16, measured finite) and Gemma (bf16) are unaffected.
        dt = torch.float32 if spec.d_model <= 2048 else torch.bfloat16
    kw = dict(torch_dtype=dt, attn_implementation="eager")   # attention weights are a measured target
    if a.device_map:
        mm = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.8)
              for i in range(torch.cuda.device_count())}
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, device_map="auto", max_memory=mm, **kw).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kw).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    common.write_json(out / "meta.json", dict(model=a.model, dtype=str(dt), attn="eager", env=common.env_info()))
    r1 = a1(model, tok, spec, mean, pca, unigram, out, max_seqs=a.max_seqs)
    dics = {}
    for cond in ("standard", "bypass"):
        pth, _ = fetch_sae(a.model, cond, 42)
        dics["%s_s42" % cond], _ = M.load_dictionary(pth, cond, mean, pca["v1"], M.site_device(model, spec))
    r3 = a3(model, tok, spec, mean, pca, prep, dics, out, max_seqs=a.max_seqs)
    return dict(a1=bool(r1), a3=bool(r3))
