"""Numerical utilities. No cloud access, job submission, or checkpoint loading."""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def provenance(paths=()):
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                files=[dict(path=str(Path(p).resolve()), sha256=sha256(p)) for p in paths])


def write_json(path, obj):
    """Atomic, strict JSON; NaN/Inf are errors, never silently serialized."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    s = json.dumps(obj, indent=2, allow_nan=False, default=convert) + '\n'
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(s, encoding='utf-8')
    temp.replace(p)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v), 1, atol=1e-5):
        raise ValueError('PC1 must be a finite, unit-length vector (no silent normalization).')
    return v / np.linalg.norm(v)  # only remove serialization round-off


class Moments:
    """Chan/Welford centered cross-products; avoids SS/n - mean*mean cancellation."""
    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError('Expected a finite [tokens, hidden] array.')
        if not len(x):
            return
        n = len(x)
        mu = x.mean(0)
        xc = x - mu
        m2 = xc.T @ xc
        if not self.n:
            self.n, self.mean, self.m2 = n, mu, m2
            return
        if mu.shape != self.mean.shape:
            raise ValueError('Hidden dimension changed.')
        delta = mu - self.mean
        nt = self.n + n
        self.m2 += m2 + np.outer(delta, delta) * (self.n * n / nt)
        self.mean += delta * (n / nt)
        self.n = nt

    def pca(self):
        if self.n < 2:
            raise ValueError('Need at least two tokens.')
        c = (self.m2 + self.m2.T) * (0.5 / (self.n - 1))
        ev, q = np.linalg.eigh(c)
        scale = max(float(abs(ev).max()), 1e-300)
        if ev.min() < -1e-10 * scale or ev.sum() <= 0:
            raise ValueError('Invalid covariance spectrum.')
        ev = np.maximum(ev[::-1], 0)
        q = q[:, ::-1]
        # Sign convention is cosmetic; all reported alignments are absolute.
        if q[:, 0] @ self.mean < 0:
            q[:, 0] *= -1
        return ev, q


def safe_corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    a, b = x - x.mean(), y - y.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return None if den == 0 else float(np.clip(a @ b / den, -1, 1))


def quantile_interval(values, total):
    finite = np.asarray([x for x in values if x is not None and np.isfinite(x)])
    return dict(lo=float(np.quantile(finite, .025)) if len(finite) else None,
                hi=float(np.quantile(finite, .975)) if len(finite) else None,
                valid_draws=int(len(finite)), requested_draws=total)


def describe_geometry(acc):
    ev, q = acc.pca()
    nm = np.linalg.norm(acc.mean)
    return dict(n_tokens=acc.n, pc1_variance_share=float(ev[0] / ev.sum()),
                pc2_variance_share=float(ev[1] / ev.sum()) if len(ev) > 1 else None,
                spectral_gap=float(ev[0] / ev[1]) if len(ev) > 1 and ev[1] > 0 else None,
                abs_cos_pc1_mean=float(abs(q[:, 0] @ acc.mean) / nm) if nm else None,
                total_centered_variance=float(ev.sum())), ev, q
