"""Activation preprocessing transforms, with exact inverses.

Every transform is fitted on TRAINING rows only and frozen. Each carries its inverse, so
an SAE trained in the transformed space can be spliced back into the raw residual stream
and compared with every other arm on the SAME raw-space population.

Two kinds:

  per-token scalar   (unitnorm, gao) -- the inverse needs scalars derived from the token
                     being reconstructed. During an intervention those scalars stay those
                     of the ORIGINAL RECIPIENT activation; they are never recomputed from
                     the edited vector. They are extra dense coefficients and are counted
                     in the budget table.

  fixed linear       (whiten) -- the inverse is a fixed matrix, so a decoder direction maps
                     back into raw coordinates as d_j @ W_inv.

`raw` and `normfilter` are identity maps; `normfilter` differs only in which TRAINING rows
are used. The language model always runs on full sequences: no token is ever deleted from
the model context.
"""
import numpy as np
import torch


class Xf:
    name = "raw"
    per_token_scalars = 0
    train_row_filter = False

    def fit(self, X, mean):
        return self

    def fwd(self, x):
        """x: centered raw [N, d] -> (t, aux)"""
        return x, None

    def inv(self, t, aux):
        return t

    def raw_direction(self, W_dec):
        """Decoder rows expressed in raw centered coordinates (up to token scale)."""
        return W_dec

    def train_mask(self, X):
        return None

    def meta(self):
        return dict(name=self.name, per_token_scalars=self.per_token_scalars)


class UnitNorm(Xf):
    """Row-normalize AFTER global centering. The coordinate mean is NOT removed, so this
    is deliberately not Gao's recipe -- it is the arm the old appendix table used."""
    name = "unitnorm"
    per_token_scalars = 1

    def fwd(self, x):
        n = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return x / n, n

    def inv(self, t, aux):
        return t * aux

    def meta(self):
        return dict(name=self.name, per_token_scalars=1,
                    centering="global mean only; coordinate mean NOT removed",
                    carried="original row L2 norm")


class Gao(Xf):
    """Gao et al. 2406.04093 §2.1: subtract each token vector's mean ACROSS COORDINATES,
    then divide by its norm. Two per-token scalars, both carried back at reconstruction."""
    name = "gao"
    per_token_scalars = 2

    def fwd(self, x):
        m = x.mean(dim=-1, keepdim=True)
        c = x - m
        n = c.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return c / n, (m, n)

    def inv(self, t, aux):
        m, n = aux
        return t * n + m

    def meta(self):
        return dict(name=self.name, per_token_scalars=2,
                    definition="subtract per-token coordinate mean, divide by L2 norm",
                    carried="coordinate mean and row norm",
                    caveat="a normalization comparison, not a reproduction of OpenAI's "
                           "full released training recipe")


class NormFilter(Xf):
    """Identity map; the arm differs only in which TRAINING rows are kept. The cutoff is
    fitted on training rows as a multiple of their median L2 norm and frozen."""
    name = "normfilter10x"
    train_row_filter = True

    def __init__(self, mult=10.0):
        self.mult = mult
        self.cutoff = None
        self.train_median = None

    def fit(self, X, mean):
        n = np.linalg.norm(np.asarray(X, dtype=np.float32) - mean[None, :], axis=1)
        self.train_median = float(np.median(n))
        self.cutoff = self.mult * self.train_median
        self.frac_removed = float((n > self.cutoff).mean())
        return self

    def train_mask(self, X):
        return X.norm(dim=-1) <= self.cutoff

    def meta(self):
        return dict(name=self.name, multiplier=self.mult, cutoff=self.cutoff,
                    train_median_norm=self.train_median,
                    frac_training_rows_removed=getattr(self, "frac_removed", None),
                    scope="Filters SAE training activations; language-model inputs use full sequences",
                    caveat="Fixed calibration threshold; the release leaves its multiplier and batch definition unspecified.")


class Whiten(Xf):
    """Stabilized centered PCA whitening. Named as such, not as any published recipe.

    t = x V diag((lam + eps)^-1/2);  inverse t diag((lam + eps)^1/2) V^T.
    No truncation and no pseudoinverse: every direction is retained, so the inverse is
    exact up to the floor. Whitening reweights the reconstruction loss across directions,
    which is reported plainly rather than treated as a free improvement.
    """
    name = "whiten"

    def __init__(self, eps_frac=1e-4):
        self.eps_frac = eps_frac

    def fit(self, X, mean):
        Xc = np.asarray(X, dtype=np.float64) - mean[None, :]
        C = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
        lam, V = np.linalg.eigh(C)
        order = np.argsort(lam)[::-1]
        lam, V = lam[order], V[:, order]
        self.eps = float(self.eps_frac * lam[0])
        self.lam, self.V = lam, V
        self.n_fit = int(Xc.shape[0])
        s = 1.0 / np.sqrt(lam + self.eps)
        self.W = (V * s[None, :]).astype(np.float32)          # [d, d]
        self.Winv = (np.diag(np.sqrt(lam + self.eps)) @ V.T).astype(np.float32)
        self.cond = float(np.sqrt((lam[0] + self.eps) / (lam[-1] + self.eps)))
        return self

    def to(self, device):
        self.Wt = torch.tensor(self.W, device=device)
        self.Winvt = torch.tensor(self.Winv, device=device)
        return self

    def fwd(self, x):
        return x @ self.Wt, None

    def inv(self, t, aux):
        return t @ self.Winvt

    def raw_direction(self, W_dec):
        return W_dec @ self.Winvt

    def meta(self):
        return dict(name=self.name, eigenvalue_floor_frac=self.eps_frac,
                    eigenvalue_floor=float(self.eps), n_fit_rows=self.n_fit,
                    condition_number=self.cond, truncation="none",
                    variant="stabilized centered PCA whitening (not ZCA, and not a "
                            "reproduction of any published whitening recipe)",
                    caveat="whitening changes the loss weighting across directions; in a "
                           "whitened space the top eigenvalues are nearly tied, so 'its "
                           "PC1' is unstable and its own-space FVU is a diagnostic only")




class ScalarNorm(Xf):
    """A single dataset-level scalar s with E||x/s||^2 = 1, fitted on training rows.

    This is the normalization Rajamanoharan et al. (2407.14435 App. I) and the Gemma Scope
    report (App. A) apply before using theta_init = 1e-3: without it a hardcoded threshold
    sits orders of magnitude below the pre-activation scale and the architecture silently
    degenerates. It is ONE fixed scalar, not a per-token statistic, so the inverse is exact
    and nothing is carried per token.
    """
    name = "scalarnorm"
    per_token_scalars = 0

    def fit(self, X, mean):
        Xc = np.asarray(X, dtype=np.float64) - mean[None, :]
        self.s = float(np.sqrt((Xc ** 2).sum(1).mean()))
        self.n_fit = int(Xc.shape[0])
        return self

    def fwd(self, x):
        return x / self.s, None

    def inv(self, t, aux):
        return t * self.s

    def raw_direction(self, W_dec):
        return W_dec

    def meta(self):
        return dict(name=self.name, scalar=getattr(self, "s", None),
                    n_fit_rows=getattr(self, "n_fit", None), per_token_scalars=0,
                    definition="single dataset-level scalar giving E||x/s||^2 = 1",
                    source="Rajamanoharan et al. 2407.14435 App. I; Gemma Scope App. A")


BUILD = {"raw": lambda: Xf(), "unitnorm": lambda: UnitNorm(), "gao": lambda: Gao(),
         "normfilter10x": lambda: NormFilter(10.0), "whiten": lambda: Whiten(),
         "scalarnorm": lambda: ScalarNorm()}
