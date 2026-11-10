"""Float64 PC1 error decomposition; no training or matching assumptions."""
import numpy as np


def fit_pc1(x, max_iter=100, tol=1e-8, seed=1729):
    """Deterministic power iteration; returns calibration mean, axis, diagnostics."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('Need a finite calibration matrix with >=2 rows')
    mu = a.mean(0)
    a = a - mu
    v = np.random.default_rng(seed).normal(size=a.shape[1])
    v /= np.linalg.norm(v)
    for step in range(max_iter):
        cv = a.T @ (a @ v) / len(a)
        size = np.linalg.norm(cv)
        if size == 0:
            raise ValueError('Zero calibration variance')
        w = cv / size
        change = min(np.linalg.norm(w-v), np.linalg.norm(w+v))
        v = w
        if change < tol:
            break
    v *= 1 if v[np.argmax(np.abs(v))] >= 0 else -1
    cv = a.T @ (a @ v) / len(a)
    eigen = float(v @ cv)
    residual = float(np.linalg.norm(cv-eigen*v) / max(eigen, 1e-30))
    return mu, v, dict(iterations=step+1, relative_eigen_residual=residual,
                      w1_calibration=eigen / float(np.mean(np.sum(a*a, axis=1))),
                      converged=residual < 1e-5)


def sufficient_stats(x, xhat, origin, axis):
    """Stats permit exact pooled evaluation centering and article resampling.

    Compute off-axis vectors directly to avoid subtracting nearly equal energies.
    origin is a fixed calibration mean, NOT a per-batch evaluation mean.
    """
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(xhat, dtype=np.float64)
    v = np.asarray(axis, dtype=np.float64)
    if x.shape != h.shape or x.ndim != 2 or len(x) == 0:
        raise ValueError('Input and reconstruction must have equal nonempty shapes')
    if not (np.isfinite(x).all() and np.isfinite(h).all()):
        raise ValueError('Nonfinite input or reconstruction')
    if not np.isclose(np.linalg.norm(v), 1., atol=1e-7):
        raise ValueError('PC1 axis must have unit norm')
    c = x - np.asarray(origin, dtype=np.float64)
    p = c @ v
    u = c - p[:, None]*v
    r = x-h
    rp = r @ v
    ru = r-rp[:, None]*v
    # Central moments + mean permit stable Chan merging even after mean shifts.
    mean = c.mean(0)
    pc = p-p.mean()
    uc = u-u.mean(0)
    return np.r_[len(x), mean, pc@pc, np.sum(uc*uc), rp@rp, np.sum(ru*ru)]


def merge_stats(groups, axis, counts=None):
    g = np.asarray(groups, dtype=np.float64)
    v = np.asarray(axis, dtype=np.float64)
    counts = np.ones(len(g)) if counts is None else np.asarray(counts,dtype=np.float64)
    weights = counts*g[:,0]
    n = weights.sum()
    if n <= 0:raise ValueError('No rows to merge')
    means = g[:,1:1+len(v)]
    # Subtract a nearby reference before averaging to preserve small offsets.
    mean = means[0]+weights@(means-means[0])/n
    delta = means-mean
    dp = delta@v
    du = delta-dp[:,None]*v
    vp = counts@g[:,-4]+weights@(dp*dp)
    vu = counts@g[:,-3]+weights@np.sum(du*du,axis=1)
    return np.r_[n,mean,vp,vu,counts@g[:,-2],counts@g[:,-1]]


def summarize(stats, axis):
    s = np.asarray(stats, dtype=np.float64)
    v = np.asarray(axis, dtype=np.float64)
    n = s[0]
    if n < 2:
        raise ValueError('Insufficient evaluation rows')
    vp, vu, erp, eru = s[-4:]
    if vp < -1e-8 or vu <= 0:
        raise ValueError('Degenerate/corrupt evaluation variance')
    vp = max(0., vp)
    total_v, total_e = vp+vu, erp+eru
    ft, fc = total_e/total_v, eru/vu
    fp = erp/vp if vp > 0 else None
    ratio = fc/ft if ft > 0 else None
    eta = erp/total_e if total_e > 0 else None
    w1 = vp/total_v
    predicted = (1-eta)/(1-w1) if eta is not None else None
    return dict(n_rows=int(n), fvu_total=float(ft), fvu_pc1=None if fp is None else float(fp),
                fvu_content=float(fc), content_over_total=None if ratio is None else float(ratio),
                w1_eval=float(w1), residual_energy_on_pc1=None if eta is None else float(eta),
                identity_predicted=predicted,
                identity_error=None if ratio is None else float(abs(ratio-predicted)),
                variance_pc1=float(vp/n), variance_content=float(vu/n))


def article_bootstrap(group_stats, axis, draws=2000, seed=20260920):
    """Whole-document percentile intervals, conditional on fixed axis/checkpoint."""
    g = np.asarray(group_stats, dtype=np.float64)
    if len(g) < 2:
        return {'status':'insufficient_documents', 'n_documents':len(g)}
    rng = np.random.default_rng(seed)
    fields = ['fvu_total','fvu_content','content_over_total','w1_eval']
    samples = {k:[] for k in fields}
    # Re-express group sums around the full-data mean before batched resampling.
    origin=merge_stats(g,axis)[1:1+len(axis)]
    dm=g[:,1:1+len(axis)]-origin
    dp=dm@axis;du=dm-dp[:,None]*axis
    raw=np.column_stack([g[:,0],g[:,0,None]*dm,
        g[:,-4]+g[:,0]*dp*dp,g[:,-3]+g[:,0]*np.sum(du*du,axis=1),g[:,-2:]])
    for start in range(0, draws, 128):
        counts = rng.multinomial(len(g), np.full(len(g),1/len(g)),
                                 size=min(128,draws-start))
        for s in counts @ raw:
            n=s[0];sx=s[1:1+len(axis)];sp=sx@axis;su=sx-sp*axis
            central=np.r_[n,origin+sx/n,s[-4]-sp*sp/n,s[-3]-su@su/n,s[-2:]]
            m = summarize(central, axis)
            for k in fields:
                if m[k] is not None:
                    samples[k].append(m[k])
    return {'status':'ok','n_documents':len(g),'draws':draws,
            'conditioning':'fixed checkpoint and calibration PC1; whole documents',
            **{k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items() if v}}
