"""Numerical primitives. CPU core needs numpy; scipy only for rank statistics."""
from pathlib import Path
import datetime, hashlib, json, os
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def slices(n, batch):
    if n <= 0 or batch <= 0:
        raise ValueError('n and batch must be positive')
    for i in range(0, n, batch):
        yield slice(i, min(i + batch, n))


def clean(obj):
    if isinstance(obj, dict): return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray): return clean(obj.tolist())
    if isinstance(obj, np.generic): return clean(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj): return None
    return obj


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def start_run(out, config, inputs=()):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier experiment.
    meta = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                config=config, inputs=[dict(path=str(Path(p).resolve()), sha256=sha256(p)) for p in inputs],
                code_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                numpy_version=np.__version__, status='running')
    write_json(out / 'manifest.json', meta)
    return out, meta


def finish_run(out, meta):
    meta['status'] = 'complete'
    meta['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(Path(out) / 'manifest.json', meta)


def unit_rows(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    return x / np.maximum(norms[:, None], eps), norms > eps


def signed_matches(a, b, block=512):
    """Exact signed maximum cosine; blocks both axes, deterministic first tie."""
    a, av = unit_rows(a); b, bv = unit_rows(b)
    if not bv.any(): raise ValueError('No nonzero candidate directions')
    vals = np.full(len(a), -np.inf); ids = np.full(len(a), -1, int)
    for s in slices(len(a), block):
        best = np.full(len(a[s]), -np.inf); bid = np.full(len(a[s]), -1, int)
        for t in slices(len(b), block):
            sim = a[s] @ b[t].T
            sim[:, ~bv[t]] = -np.inf
            loc = sim.argmax(axis=1); v = sim[np.arange(len(loc)), loc]
            take = v > best
            best[take], bid[take] = v[take], t.start + loc[take]
        vals[s], ids[s] = best, bid
    vals[~av] = np.nan; ids[~av] = -1
    return vals, ids


def sharing_rules(max_cos, ids=None, threshold=0.7):
    """Rows are comparison seeds; columns are reference features."""
    c = np.asarray(max_cos, float)
    if c.ndim != 2 or not np.isfinite(c).all(): raise ValueError('finite [seed,feature] matrix required')
    if ids is not None: c = c[:, ids]
    hit = c > threshold
    return dict(n_features=c.shape[1], n_comparisons=c.shape[0],
                pairwise_sharing=float(hit.mean()),
                mean_cosine_sharing=float((c.mean(0) > threshold).mean()),
                any_seed_sharing=float(hit.any(0).mean()),
                every_seed_sharing=float(hit.all(0).mean()),
                no_other_seed_count=int((~hit.any(0)).sum()))


def fvu_decomposition(x, xhat, v, mean, denominator='evaluation_mean'):
    """Direct float64 projection; common denominator and same examples throughout.
    mean is the training mean. evaluation_mean gives true evaluation variance.
    training_mean gives squared distance about that fixed center, labelled as such.
    """
    x, xhat = np.asarray(x, np.float64), np.asarray(xhat, np.float64)
    v = np.asarray(v, np.float64); v = v / np.linalg.norm(v)
    if x.shape != xhat.shape or x.ndim != 2: raise ValueError('shape mismatch')
    center = x.mean(0) if denominator == 'evaluation_mean' else np.asarray(mean)
    if denominator not in ('evaluation_mean', 'training_mean'): raise ValueError(denominator)
    a = x - center; e = x - xhat
    a1, e1 = a @ v, e @ v
    ap, ep = a - a1[:, None] * v, e - e1[:, None] * v
    t1, tp = np.square(a1).sum(), np.square(ap).sum()
    n1, np_ = np.square(e1).sum(), np.square(ep).sum()
    if min(t1, tp) <= 0: raise ValueError('Degenerate component variance')
    w = t1 / (t1 + tp); f1, fp = n1 / t1, np_ / tp
    ft = np.square(e).sum() / (t1 + tp)
    eta = w * f1 / ft if ft > 0 else np.nan
    return dict(n=len(x), denominator=denominator, w1=w, fvu_total=ft, fvu_pc1=f1,
                fvu_perp=fp, ratio=fp / ft if ft else np.nan, asymptotic_ratio=1/(1-w),
                eta_pc1_error=eta, split_error=ft-w*f1-(1-w)*fp,
                variance_sum=t1+tp, mse_sum=np.square(e).sum())


def group_geometry(z, d, v):
    """z:[tokens,group], d:[group,d_model]; biases explicitly excluded."""
    z, d = np.asarray(z, float), np.asarray(d, float)
    v = np.asarray(v, float); v /= np.linalg.norm(v)
    y = z @ d; yp = y - (y @ v)[:, None] * v
    dp = d - (d @ v)[:, None] * v
    indiv = (np.square(z).sum(0) * np.square(dp).sum(1)).sum()
    group = np.square(yp).sum()
    zc = z-z.mean(0)
    cz = zc.T @ zc
    evz, uz = np.linalg.eigh(cz)
    root = (uz * np.sqrt(np.maximum(evz, 0))) @ uz.T
    ev = np.maximum(np.linalg.eigvalsh(root @ (d @ d.T) @ root), 0)[::-1]
    total = np.square(y).sum()
    return dict(pc1_energy_fraction=np.square(y @ v).sum()/total if total else None,
                centered_effective_rank=ev.sum()**2/(ev@ev) if ev@ev else None,
                centered_top_eigenvalue_share=ev[0]/ev.sum() if ev.sum() else None,
                offaxis_individual_energy=indiv, offaxis_group_energy=group,
                cancellation_ratio=indiv/group if group > 1e-25 else None,
                cancellation_fraction=1-group/indiv if indiv > 1e-25 else None,
                offaxis_cross_term=group-indiv, n_tokens=len(z), n_atoms=len(d))


def rank_corr(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x), np.asarray(y)
    good = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[good], y[good]).statistic) if good.sum() > 2 else np.nan


def partial_rank(x, y, controls):
    from scipy.stats import rankdata
    x,y=np.asarray(x),np.asarray(y); c=np.asarray(controls)
    if c.ndim == 1: c=c[:,None]
    good=np.isfinite(x)&np.isfinite(y)&np.isfinite(c).all(1)
    x,y,c=x[good],y[good],c[good]
    if len(x) <= c.shape[1]+3: return np.nan
    a=np.column_stack([np.ones(len(x))]+[rankdata(col, method='average') for col in c.T])
    rx,ry=rankdata(x,method='average'),rankdata(y,method='average')
    rx-=a@np.linalg.lstsq(a,rx,rcond=None)[0];ry-=a@np.linalg.lstsq(a,ry,rcond=None)[0]
    return float(np.corrcoef(rx,ry)[0,1]) if min(np.std(rx),np.std(ry))>0 else np.nan


def efficient_stiffness(fisher, w, tol=1e-10):
    f=np.asarray(fisher,float); w=np.asarray(w,float)
    ev,u=np.linalg.eigh((f+f.T)/2)
    if ev.min() < -tol: raise ValueError('Fisher must be positive semidefinite')
    coords=u.T@w; pos=ev>tol*max(1,ev.max())
    if np.square(coords[~pos]).sum() > tol * (w@w): return 0.0
    compliance=(coords[pos]**2/ev[pos]).sum()
    return float(1/compliance) if compliance>0 else np.inf
