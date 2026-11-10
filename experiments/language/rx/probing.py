"""Workstream D: sparse probing.

Two protocols, kept separate and labelled:

HISTORICAL (exact reproduction of exp_r3_sparse_probing.py): first 2,000 examples per class
from each task's TRAIN split; 5 random stratified 75/25 splits; features = SAE latents
mean-pooled over real tokens at max_len 64; MI on binarised firing, fit on the train rows of
each split; LogisticRegression(max_iter=2000), default C=1, NO standardisation.

CORRECTED (primary for new claims): train pool = stratified random 2,000/class from TRAIN
(fixed seed, selection ids saved), exact-duplicate texts removed; OUTER TEST = the task's
official labelled held-out split (SST-2: the public validation split, declared as the external
test; all tuning on train folds only). Inside 5 inner folds: MI selection, standardisation with
train-fold statistics, and C chosen from a fixed grid; then refit on the whole train pool and
score the untouched outer test once. Exactly k nonconstant features are used where possible;
when fewer exist, the shortfall is recorded, never silently absorbed.

Baselines on the identical splits: dense raw-activation probe, dense PC1-removed residual,
scalar PC1 coefficient, random alive subsets of size k (5 draws per k), and -- for bypass
dictionaries -- the bypass latents plus the routed PC1 scalar (k+1 coordinates, counted).

Pooling: mean over real tokens at max_len 64 (primary, historical); a predeclared alternative
of last-token pooling at max_len 128 reports what truncation and pooling change.
"""
import json
import os
import zlib
import time

import numpy as np
import torch

from . import common, modelio as M
from .common import log

TASKS = {
    "sst2": dict(path=("nyu-mll/glue", "sst2"), text="sentence", label="label",
                 test_split="validation"),
    "ag_news": dict(path=("fancyzhx/ag_news",), text="text", label="label", test_split="test"),
    "amazon": dict(path=("fancyzhx/amazon_polarity",), text="content", label="label",
                   test_split="test"),
    "emotion": dict(path=("dair-ai/emotion",), text="text", label="label", test_split="test"),
}
KS_PRIMARY = [1, 20, 100]
KS_ALL = [1, 2, 5, 10, 20, 50, 100]
N_PER_CLASS = 2000
TEST_PER_CLASS_CAP = 500           # outer test: stratified, at most 500 per class (published ids)
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
SEED = 20261005


def _load(cfg, split):
    from datasets import load_dataset
    return load_dataset(*cfg["path"], split=split)


def historical_selection(cfg):
    """exp_r3 load_task: first N_PER_CLASS per class, in TRAIN order."""
    ds = _load(cfg, "train")
    n_classes = getattr(ds.features[cfg["label"]], "num_classes", None)
    texts, labels, idx, counts = [], [], [], {}
    for i, row in enumerate(ds):
        lab = int(row[cfg["label"]])
        if counts.get(lab, 0) >= N_PER_CLASS:
            continue
        counts[lab] = counts.get(lab, 0) + 1
        texts.append(row[cfg["text"]]); labels.append(lab); idx.append(i)
        if (n_classes is not None and len(counts) == n_classes
                and all(v >= N_PER_CLASS for v in counts.values())):
            break
    return texts, np.asarray(labels), np.asarray(idx)


def corrected_selection(cfg):
    rng = np.random.default_rng(SEED)
    tr = _load(cfg, "train")
    lab = np.asarray(tr[cfg["label"]])
    txt = tr[cfg["text"]]
    seen, keep = set(), []
    for c in np.unique(lab):
        ids = np.nonzero(lab == c)[0]
        rng.shuffle(ids)
        got = 0
        for i in ids:
            t = txt[int(i)]
            if t in seen:
                continue
            seen.add(t)
            keep.append(int(i))
            got += 1
            if got >= N_PER_CLASS:
                break
    keep = np.asarray(sorted(keep))
    te = _load(cfg, cfg["test_split"])
    tl = np.asarray(te[cfg["label"]])
    tt = te[cfg["text"]]
    tkeep = []
    for c in np.unique(tl):
        ids = np.nonzero(tl == c)[0]
        rng.shuffle(ids)
        for i in ids[:TEST_PER_CLASS_CAP]:
            if tt[int(i)] not in seen:          # no train/test exact duplicates
                tkeep.append(int(i))
    tkeep = np.asarray(sorted(tkeep))
    return ([txt[int(i)] for i in keep], lab[keep], keep,
            [tt[int(i)] for i in tkeep], tl[tkeep], tkeep)


@torch.no_grad()
def pooled(model, tok, spec, texts, dics, max_len, pool, batch=32):
    """Returns raw pooled activations [N,d], PC1 coefficient [N], {name: sparse pooled latents},
    and truncation statistics. `dics`: name -> Dictionary."""
    import scipy.sparse as sp
    raw, coef = [], []
    lat = {k: [] for k in dics}
    n_trunc, n_tot = 0, 0
    tok.padding_side = "right"
    v1 = next(iter(dics.values())).v1
    mu = next(iter(dics.values())).mu
    for i in range(0, len(texts), batch):
        enc = tok(list(texts[i:i + batch]), return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        full = tok(list(texts[i:i + batch]), truncation=False)["input_ids"]
        n_trunc += sum(len(f) > max_len for f in full)
        n_tot += len(full)
        ids, am = enc["input_ids"], enc["attention_mask"]
        h = M.prefix(model, spec, ids, am).float()
        m = am.to(h.device).float().unsqueeze(-1)
        if pool == "mean":
            den = m.sum(1).clamp_min(1)
            raw.append(((h * m).sum(1) / den).cpu())
        else:
            last = am.sum(1).to(h.device) - 1
            raw.append(h[torch.arange(h.shape[0], device=h.device), last].cpu())
        B, T, d = h.shape
        c = ((h - mu) @ v1)
        coef.append(((c * m[..., 0]).sum(1) / m[..., 0].sum(1).clamp_min(1)).cpu() if pool == "mean"
                    else c[torch.arange(B, device=h.device), last].cpu())
        for name, dic in dics.items():
            z, _ = dic.encode(h.reshape(-1, d))
            z = z.reshape(B, T, -1)
            if pool == "mean":
                zz = (z * m).sum(1) / m.sum(1).clamp_min(1)
            else:
                zz = z[torch.arange(B, device=z.device), last]
            lat[name].append(sp.csr_matrix(zz.cpu().numpy().astype(np.float32)))
    return (torch.cat(raw).numpy(), torch.cat(coef).numpy(),
            {k: sp.vstack(v).tocsr() for k, v in lat.items()},
            dict(truncated=int(n_trunc), total=int(n_tot), rate=n_trunc / max(n_tot, 1)))


def mi_binary(Xb, y):
    """Mutual information (nats) between each binary feature column and a discrete label.
    Identical to sklearn mutual_info_classif(discrete_features=True) on 0/1 columns."""
    import scipy.sparse as sp
    classes = np.unique(y)
    Y = np.stack([(y == c) for c in classes], 1).astype(np.float64)       # [n, C]
    n = len(y)
    n1c = np.asarray((Xb.T @ Y) if sp.issparse(Xb) else Xb.T @ Y)         # [F, C]  x=1
    nc = Y.sum(0)[None, :]
    n0c = nc - n1c
    n1 = n1c.sum(1, keepdims=True)
    n0 = n - n1
    mi = np.zeros(n1c.shape[0])
    for nxy, nx in ((n1c, n1), (n0c, n0)):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (nxy / n) * np.log((nxy * n) / (nx * nc))
        mi += np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).sum(1)
    return mi


def _cols(X, idx):
    import scipy.sparse as sp
    out = X[:, idx]
    return out.toarray() if sp.issparse(out) else np.asarray(out)


def _select(Xtr, ytr, k, alive_cols):
    Xb = (Xtr[:, alive_cols] != 0)
    if hasattr(Xb, "astype"):
        Xb = Xb.astype(np.float64)
    mi = mi_binary(Xb, ytr)
    order = alive_cols[np.argsort(-mi, kind="stable")]
    sub = _cols(Xtr, order[: max(k * 4, k)])
    nonconst = order[: max(k * 4, k)][sub.std(0) > 0]
    chosen = nonconst[:k]
    return chosen, int(k - len(chosen))


def _fit(Xtr, ytr, Xte, C, standardize):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import warnings
    if standardize:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clf = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr)
        conv = not any("converge" in str(x.message).lower() for x in w)
    return clf.predict(Xte), conv


def historical_probe(Z, y, alive_cols, ks=KS_PRIMARY, n_splits=5):
    """exp_r3 exactly: 5 x stratified 75/25, MI on train rows, LR C=1 unstandardised."""
    from sklearn.model_selection import train_test_split
    out = {}
    idx = np.arange(len(y))
    splits = [train_test_split(idx, test_size=0.25, random_state=s, stratify=y)
              for s in range(n_splits)]
    for k in ks:
        accs = []
        for tr, te in splits:
            ch, _ = _select(Z[tr], y[tr], k, alive_cols)
            pred, _ = _fit(_cols(Z[tr], ch), y[tr], _cols(Z[te], ch), 1.0, False)
            accs.append(float((pred == y[te]).mean()))
        out[str(k)] = dict(mean=float(np.mean(accs)), std=float(np.std(accs)), per_split=accs)
    return out


def corrected_probe(Ztr, ytr, Zte, yte, cols_fn, ks=KS_ALL, inner=5):
    """Inner-CV selection + standardisation + C tuning, refit, one outer-test evaluation.
    cols_fn(Xtr, ytr, k) -> (chosen columns, shortfall)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    res = {}
    skf = StratifiedKFold(n_splits=inner, shuffle=True, random_state=SEED)
    for k in ks:
        scores = np.zeros(len(C_GRID))
        for tri, vai in skf.split(np.zeros(len(ytr)), ytr):
            ch, _ = cols_fn(Ztr[tri], ytr[tri], k)
            A, Bv = _cols(Ztr[tri], ch), _cols(Ztr[vai], ch)
            for ci, C in enumerate(C_GRID):
                p, _ = _fit(A, ytr[tri], Bv, C, True)
                scores[ci] += (p == ytr[vai]).mean() / inner
        C = C_GRID[int(np.argmax(scores))]
        ch, short = cols_fn(Ztr, ytr, k)
        pred, conv = _fit(_cols(Ztr, ch), ytr, _cols(Zte, ch), C, True)
        res[str(k)] = dict(acc=float((pred == yte).mean()),
                           balanced_acc=float(balanced_accuracy_score(yte, pred)),
                           C=C, inner_cv=scores.tolist(), n_features=int(len(ch)),
                           shortfall=short, converged=conv, chosen=[int(c) for c in ch],
                           pred=pred.astype(np.int16).tolist())
    return res


def dense_probe(Xtr, ytr, Xte, yte, inner=5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    skf = StratifiedKFold(n_splits=inner, shuffle=True, random_state=SEED)
    scores = np.zeros(len(C_GRID))
    for tri, vai in skf.split(Xtr, ytr):
        for ci, C in enumerate(C_GRID):
            p, _ = _fit(Xtr[tri], ytr[tri], Xtr[vai], C, True)
            scores[ci] += (p == ytr[vai]).mean() / inner
    C = C_GRID[int(np.argmax(scores))]
    pred, conv = _fit(Xtr, ytr, Xte, C, True)
    return dict(acc=float((pred == yte).mean()),
                balanced_acc=float(balanced_accuracy_score(yte, pred)), C=C, converged=conv,
                pred=pred.astype(np.int16).tolist())


def _fit_one(name, alive, Ltr, Lte, Ctr, Cte, tr_y, te_y, mode, n_random):
    al = np.nonzero(alive[name])[0]
    sel = lambda X, y, k, al=al: _select(X, y, k, al)
    d = dict(mi=corrected_probe(Ltr[name], tr_y, Lte[name], te_y, sel))
    rng = np.random.default_rng(SEED + zlib.crc32(name.encode()) % 100000)
    rnd = {}
    for k in KS_PRIMARY:
        accs = []
        for r in range(n_random):
            cols = rng.choice(al, size=min(k, len(al)), replace=False)
            rr = corrected_probe(Ltr[name], tr_y, Lte[name], te_y,
                                 lambda X, y, kk, cols=cols: (cols, 0), ks=[k])
            accs.append(rr[str(k)]["acc"])
        rnd[str(k)] = dict(mean=float(np.mean(accs)), per_draw=accs)
    d["random_alive"] = rnd
    if mode == "bypass":
        import scipy.sparse as sp
        Xa = sp.hstack([Ltr[name], sp.csr_matrix(Ctr[:, None])]).tocsr()
        Xb = sp.hstack([Lte[name], sp.csr_matrix(Cte[:, None])]).tocsr()
        extra_col = Xa.shape[1] - 1

        def sel_pc(X, y, k, al=al, ec=extra_col):
            ch, sh = _select(X, y, k, al)
            return np.concatenate([ch, [ec]]), sh
        d["mi_plus_pc1_scalar"] = corrected_probe(
            Xa, tr_y, Xb, te_y, sel_pc, ks=KS_PRIMARY)
    return name, d


class PoolCache:
    """Pooled features saved once (GPU stage) and reloaded for fitting (CPU stage). Keyed by
    task/split/pooling; stored under <out>/cache/ and mirrored to S3 by the job."""

    def __init__(self, cdir, model=None, tok=None, spec=None, dics=None, names=None):
        self.cdir, self.model, self.tok, self.spec, self.dics = cdir, model, tok, spec, dics
        self.names = names if names is not None else sorted(dics or {})
        cdir.mkdir(parents=True, exist_ok=True)

    def __call__(self, key, texts, max_len, pool):
        import scipy.sparse as sp
        f = self.cdir / ("%s.npz" % key)
        if f.exists():
            z = np.load(f, allow_pickle=False)
            lat = {n: sp.csr_matrix((z["lat_%s_data" % n], z["lat_%s_indices" % n], z["lat_%s_indptr" % n]),
                                    shape=tuple(z["lat_%s_shape" % n])) for n in self.names}
            return z["raw"], z["coef"], lat, json.loads(str(z["trunc"]))
        if self.model is None:
            raise FileNotFoundError("fit stage: pooled cache missing for %s" % key)
        raw, coef, lat, tr = pooled(self.model, self.tok, self.spec, texts, self.dics, max_len, pool)
        arrs = dict(raw=raw.astype(np.float32), coef=coef.astype(np.float32), trunc=json.dumps(tr))
        for n, m in lat.items():
            arrs.update({"lat_%s_data" % n: m.data, "lat_%s_indices" % n: m.indices,
                         "lat_%s_indptr" % n: m.indptr, "lat_%s_shape" % n: np.asarray(m.shape)})
        common.save_npz(f, **arrs)
        return raw, coef, lat, tr


def run(model, tok, spec, dics, alive, out, conditions, n_random=5, stage="all", names=None,
        v1=None, mode=None):
    """dics: name -> Dictionary (all seeds of one condition); alive: name -> bool mask.
    stage: 'cache' = pooled features only (needs the model); 'fit' = probes from the cache only
    (no model, no GPU); 'all' = both."""
    out.mkdir(parents=True, exist_ok=True)
    names = names if names is not None else sorted(dics)
    PC = PoolCache(out / "cache", model, tok, spec, dics, names)
    if v1 is None:
        v1 = next(iter(dics.values())).v1.cpu().numpy()
    if mode is None:
        mode = next(iter(dics.values())).mode
    summary = {}
    for tname, cfg in TASKS.items():
        f = out / ("%s.json" % tname)
        if stage == "cache":
            ht, hy, hidx = historical_selection(cfg)
            PC("%s_hist_mean64" % tname, ht, 64, "mean")
            tr_t, tr_y, tr_i, te_t, te_y, te_i = corrected_selection(cfg)
            for pool, ml in (("mean", 64), ("last", 128)):
                PC("%s_tr_%s%d" % (tname, pool, ml), tr_t, ml, pool)
                PC("%s_te_%s%d" % (tname, pool, ml), te_t, ml, pool)
            log("probe", "%s pooled features cached" % tname)
            continue
        if common.fresh(f.parent, f.name, "probe"):
            summary[tname] = json.loads(f.read_text())
            continue
        t0 = time.time()
        res = dict(task=tname, dictionaries=list(names), conditions=conditions)
        # historical
        ht, hy, hidx = historical_selection(cfg)
        raw_h, coef_h, lat_h, tr_h = PC("%s_hist_mean64" % tname, ht, 64, "mean")
        res["historical"] = dict(selection="first %d/class, train split" % N_PER_CLASS,
                                 n=len(hy), class_counts=np.bincount(hy).tolist(),
                                 truncation_64=tr_h, per_dict={})
        for name in names:
            res["historical"]["per_dict"][name] = historical_probe(
                lat_h[name], hy, np.nonzero(alive[name])[0])
        res["historical"]["raw_dense_C1_unstd"] = historical_probe(
            raw_h, hy, np.arange(raw_h.shape[1]), ks=[raw_h.shape[1]])
        # corrected
        tr_t, tr_y, tr_i, te_t, te_y, te_i = corrected_selection(cfg)
        res["corrected"] = dict(train_ids=tr_i.tolist(), test_ids=te_i.tolist(),
                                test_split=cfg["test_split"],
                                train_class_counts=np.bincount(tr_y).tolist(),
                                test_class_counts=np.bincount(te_y).tolist(), variants={})
        for pool, ml in (("mean", 64), ("last", 128)):
            Rtr, Ctr, Ltr, trn = PC("%s_tr_%s%d" % (tname, pool, ml), tr_t, ml, pool)
            Rte, Cte, Lte, tte = PC("%s_te_%s%d" % (tname, pool, ml), te_t, ml, pool)
            v = dict(max_len=ml, pool=pool, truncation_train=trn, truncation_test=tte,
                     baselines={}, per_dict={})
            v["baselines"]["dense_raw"] = dense_probe(Rtr, tr_y, Rte, te_y)
            perp = lambda X: X - np.outer(X @ v1, v1)
            v["baselines"]["dense_pc1_removed"] = dense_probe(perp(Rtr), tr_y, perp(Rte), te_y)
            v["baselines"]["scalar_pc1"] = dense_probe(Ctr[:, None], tr_y, Cte[:, None], te_y)
            from joblib import Parallel, delayed
            nj = max(1, min(len(names), (os.cpu_count() or 2) - 1))
            outs = Parallel(n_jobs=nj)(delayed(_fit_one)(
                name, alive, Ltr, Lte, Ctr, Cte, tr_y, te_y, mode, n_random) for name in names)
            for name, d in outs:
                v["per_dict"][name] = d
            res["corrected"]["variants"]["%s_%d" % (pool, ml)] = v
        res["seconds"] = time.time() - t0
        common.write_json(f, res)
        common.pin(f.parent, f.name, "probe")
        summary[tname] = res
        log("probe", "%s done in %.0fs" % (tname, time.time() - t0))
    return summary
