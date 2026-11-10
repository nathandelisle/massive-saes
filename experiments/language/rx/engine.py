"""Exhaustive importance engine: every alive feature, five estimands, per-document effects.

Importance measures, all relative to the UNMODIFIED splice on the same tokens and all
divided once by the partition's total number of valid next-token targets:

  zero        z_j := 0                                     wherever it fires
  mean        z_j := E_cal[z_j] (zeros included)           at every valid position
  sp_resample z_j := donor from j's CONDITIONAL NONZERO calibration distribution,
                     only where j currently fires           (support-preserving)
  mg_resample z_j := donor from j's FULL calibration distribution INCLUDING zeros,
                     at every valid position                (marginal)
  st_resample support-preserving, donor drawn from the recipient's POSITION STRATUM with a
                     declared backoff (< 32 donors -> pooled) (position-stratified)
  gradxinput  sum_{t} |z_tj * dL_sum/dz_tj| / N            primary magnitude; also the signed
                     sum and |signed sum| as separate sensitivity estimands
  ig          global all-latents path from the zero LATENT baseline (SAE bias, dataset mean
                     and routed PC1 held fixed), gradient w.r.t. the PATH latent, midpoint rule,
                     steps doubled until completeness and top-50 stability tolerances are met

EXACTNESS / EFFICIENCY
  * The site residual is cached once; every intervention runs only the suffix (gate 7).
  * An intervention whose latent change is zero everywhere in a sequence is skipped and
    contributes EXACT zero (the activation is unchanged). Zero ablation and
    support-preserving resampling therefore touch only sequences where the feature fires.
  * Positions before the first changed position are causally unaffected and contribute exact
    zero (this also removes batch-composition float noise from them).
  * Donor draws are keyed by (experiment seed, feature, article, row, position, draw id), not by
    batch or shard, so results do not depend on batching or resume order (gate 8).
  * Features alive by the archived rule but never firing in a partition are recorded as
    MEASURED_ZERO for deletion/support-preserving metrics; dead features are NOT_EVALUATED (NaN).
"""
import json
import pathlib
import time

import hashlib
import zlib

import numpy as np
import torch

from . import common, modelio
from .common import log

STATUS_NOT_EVALUATED, STATUS_MEASURED, STATUS_NOOP_ZERO = 0, 1, 2
STRATA = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 1 << 30))


def _stratum(pos):
    s = np.zeros_like(pos)
    for i, (lo, hi) in enumerate(STRATA):
        s[(pos >= lo) & (pos < hi)] = i
    return s


def splitmix64(x):
    """Vectorised splitmix64 on uint64 numpy arrays (wraparound arithmetic is defined here)."""
    x = (x + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
    return x ^ (x >> np.uint64(31))


def draw_keys(seed, feat, doc, row, pos, draw):
    """One independent 64-bit key per (seed, feature, article, row, position, draw)."""
    with np.errstate(over="ignore"):
        k = splitmix64(np.uint64(seed) ^ splitmix64(np.asarray(feat, np.uint64)))
        k = splitmix64(k ^ np.asarray(doc, np.uint64))
        k = splitmix64(k ^ np.asarray(row, np.uint64))
        k = splitmix64(k ^ np.asarray(pos, np.uint64))
        k = splitmix64(k ^ np.asarray(draw, np.uint64))
    return k


class Cache:
    """Site residual, latents and baseline NLL for one (dictionary, partition)."""

    @torch.no_grad()
    def __init__(self, model, spec, suf, dic, part, max_tokens=8192, dtype=None, keep_err=False):
        self.model, self.spec, self.suf, self.dic, self.part = model, spec, suf, dic, part
        dev = dic.device
        N, T = part.ids.shape
        d, k = spec.d_model, None
        self.dtype = dtype or modelio.blocks(model, spec)[0].parameters().__next__().dtype
        self.R = torch.zeros(N, T, d, dtype=torch.float32, device=dev)   # spliced residual, fp32
        self.nll0 = torch.zeros(N, T, dtype=torch.float32, device=dev)
        self.valid = torch.zeros(N, T, dtype=torch.bool, device=dev)
        self.E = torch.zeros(N, T, d, dtype=torch.float32, device=dev) if keep_err else None
        zi, zv, coefs = [], [], torch.zeros(N, T, dtype=torch.float32, device=dev)
        l0_sum, l0_rows = 0.0, 0
        for sel, ids, am in part.batches(max_tokens=max_tokens):
            Tb = ids.shape[1]
            h = modelio.prefix(model, spec, ids, am)
            B = h.shape[0]
            z, coef = dic.encode(h.reshape(-1, d))
            r = dic.splice_fp32(z, coef).reshape(B, Tb, d)
            nll, val = modelio.suffix_nll(suf, r.to(h.dtype), ids, am)
            s = torch.as_tensor(sel, device=dev)
            self.R[s, :Tb] = r
            if self.E is not None:
                self.E[s, :Tb] = (h.float() - r) * am.to(r.device)[..., None].float()
            self.nll0[s, :Tb] = nll.to(dev)
            self.valid[s, :Tb] = val.to(dev)
            if coef is not None:
                coefs[s, :Tb] = coef.reshape(B, Tb)
            m = am.bool().to(z.device).reshape(-1)
            nz = (z != 0)
            l0_sum += float(nz[m].sum())
            l0_rows += int(m.sum())
            kk = int(nz.sum(1).max())
            k = max(k or 0, kk)
            vals, idx = torch.topk(z, kk, dim=1)
            vals = torch.where(m[:, None], vals, torch.zeros_like(vals))
            zi.append((s, idx.reshape(B, Tb, kk), vals.reshape(B, Tb, kk)))
        self.k = k
        self.Zidx = torch.full((N, T, k), -1, dtype=torch.int32, device=dev)
        self.Zval = torch.zeros(N, T, k, dtype=torch.float32, device=dev)
        for s, idx, vals in zi:
            Tb, kk = idx.shape[1], idx.shape[2]
            self.Zidx[s, :Tb, :kk] = torch.where(vals > 0, idx.int(), torch.full_like(idx.int(), -1))
            self.Zval[s, :Tb, :kk] = vals
        self.coef = coefs
        self.lengths = torch.as_tensor(part.lengths, device=dev)
        self.am = part.am.to(dev)
        self.ids = part.ids.to(dev)
        self.n_targets = int(self.valid.sum())
        self.base_ce = float(self.nll0[self.valid].mean())
        self.l0 = l0_sum / max(l0_rows, 1)
        self._index()

    def _index(self):
        """CSR: feature -> sequences in which it fires."""
        N, T, k = self.Zidx.shape
        idx = self.Zidx.reshape(N, -1).long()
        rows = torch.arange(N, device=idx.device)[:, None].expand_as(idx)
        m = idx >= 0
        f, r = idx[m], rows[m]
        key = f * N + r
        key = torch.unique(key)
        f_u, r_u = key // N, key % N
        d_sae = self.dic.d_sae
        counts = torch.bincount(f_u, minlength=d_sae)
        self.fire_ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=idx.device),
                                   counts.cumsum(0)]).cpu().numpy()
        self.fire_seq = r_u.cpu().numpy()
        self.fire_nseq = counts.cpu().numpy()

    def seqs_firing(self, j):
        return self.fire_seq[self.fire_ptr[j]:self.fire_ptr[j + 1]]

    def oldz(self, S, F, Tb):
        """Current value of feature F[b] in sequence S[b], positions [:Tb] -> [B, Tb]."""
        idx = self.Zidx[S, :Tb]
        return ((idx == F[:, None, None].int()) * self.Zval[S, :Tb]).sum(-1)

    def _batches(self, S, F, idx, max_tokens):
        L = self.part.lengths[S[idx]]
        order = idx[np.lexsort((F[idx], S[idx], -L))]
        Lo = self.part.lengths[S[order]]
        i = 0
        while i < len(order):
            Tb = int(Lo[i])
            nb = max(1, max_tokens // max(Tb, 1))
            yield order[i:i + nb], Tb
            i += nb

    @torch.no_grad()
    def run(self, S, F, newz_fn, max_tokens=8192, progress=None, prefilter=False, W=None):
        """Evaluate items (sequence S[i], feature F[i]); newz_fn(S, F, oldz, am) -> new values.

        Returns per-item summed delta-NLL over valid targets at or after the first changed
        position, and the number of such targets. An item whose latent change is zero at every
        valid position is a provable no-op and returns exactly 0 without a forward pass.
        `prefilter=True` finds those items first (cheap: no suffix), so the suffix batches that
        remain are full -- this matters for the marginal operator, where most (rare feature,
        sequence) items draw nothing.
        """
        S = np.asarray(S, dtype=np.int64)
        F = np.asarray(F, dtype=np.int64)
        n = len(S)
        out = np.zeros(n, dtype=np.float64)
        ntg = np.zeros(n, dtype=np.int64)
        if n == 0:
            return out, ntg
        Wd = self.dic.W_dec if W is None else W      # W: e.g. the PC1 or perpendicular part
        dev = self.dic.device
        eff_idx = np.arange(n)
        if prefilter:
            keep = np.zeros(n, dtype=bool)
            for sel, Tb in self._batches(S, F, eff_idx, max_tokens * 8):
                Sb = torch.as_tensor(S[sel], device=dev)
                Fb = torch.as_tensor(F[sel], device=dev)
                old = self.oldz(Sb, Fb, Tb)
                am = self.am[Sb, :Tb]
                new = newz_fn(S[sel], F[sel], old, am)
                keep[sel] = (((new - old) * am.float()) != 0).any(1).cpu().numpy()
            eff_idx = np.nonzero(keep)[0]
        done, t0 = 0, time.time()
        for sel, Tb in self._batches(S, F, eff_idx, max_tokens):
            Sb = torch.as_tensor(S[sel], device=dev)
            Fb = torch.as_tensor(F[sel], device=dev)
            old = self.oldz(Sb, Fb, Tb)
            am = self.am[Sb, :Tb]
            new = newz_fn(S[sel], F[sel], old, am)
            dz = (new - old) * am.float()
            changed = dz != 0
            has = changed.any(1)
            done += len(sel)
            if not bool(has.any()):
                continue
            keep = has.nonzero().squeeze(1)
            Sb, Fb, dz, changed, am = Sb[keep], Fb[keep], dz[keep], changed[keep], am[keep]
            selk = sel[keep.cpu().numpy()]
            r = self.R[Sb, :Tb] + dz[..., None] * Wd[Fb][:, None, :]
            # SAME-BATCH BASELINE. In half precision the suffix's rounding depends on the batch
            # shape (cuBLAS picks a different kernel for a different M). A baseline computed in
            # another batch would add that noise to every delta, and summed over hundreds of
            # targets it swamps small effects. Rows of one forward call are computed
            # independently, so an UNMODIFIED copy of each distinct sequence in this very call
            # gives a baseline whose rounding matches the modified rows exactly.
            us, inv = torch.unique(Sb, return_inverse=True)
            R_all = torch.cat([r, self.R[us, :Tb]], 0)
            am_all = torch.cat([am, self.am[us, :Tb]], 0)
            ids = torch.cat([self.ids[Sb, :Tb], self.ids[us, :Tb]], 0)
            nll, val = modelio.suffix_nll(self.suf, R_all.to(self.dtype), ids, am_all)
            nll, val = nll.to(dev), val.to(dev)
            nb = r.shape[0]
            base = nll[nb:][inv]
            nll, val = nll[:nb], val[:nb]
            first = changed.float().argmax(1)                       # first changed position
            pos = torch.arange(Tb, device=dev)[None, :]
            m = val & (pos >= first[:, None])
            dl = ((nll - base) * m).sum(1)
            out[selk] = dl.double().cpu().numpy()
            ntg[selk] = m.sum(1).cpu().numpy()
            if progress and (done // 50000) != ((done - len(sel)) // 50000):
                progress(done, len(eff_idx), time.time() - t0)
        return out, ntg


    @torch.no_grad()
    def run_group(self, feats, W=None, max_tokens=8192):
        """Remove sum_{j in feats} z_j W_j at every position at once; summed delta-NLL over the
        partition (same-batch baselines, causal first-change masking as in run())."""
        Wd = self.dic.W_dec if W is None else W
        dev = self.dic.device
        f = torch.as_tensor(np.asarray(feats), device=dev)
        lookup = torch.full((self.dic.d_sae,), -1, dtype=torch.long, device=dev)
        lookup[f] = torch.arange(len(f), device=dev)
        Wg = Wd[f]
        seqs = np.unique(np.concatenate([self.seqs_firing(int(j)) for j in feats])) \
            if len(feats) else np.zeros(0, np.int64)
        total = 0.0
        L = self.part.lengths[seqs]
        order = seqs[np.argsort(-L, kind="stable")]
        i = 0
        while i < len(order):
            Tb = int(self.part.lengths[order[i]])
            nb = max(1, (max_tokens // 2) // max(Tb, 1))
            sel = torch.as_tensor(order[i:i + nb], device=dev)
            i += nb
            idx = self.Zidx[sel, :Tb].long()
            loc = lookup[idx.clamp_min(0)]
            loc = torch.where(idx >= 0, loc, torch.full_like(loc, -1))
            coef = torch.where(loc >= 0, self.Zval[sel, :Tb], torch.zeros_like(self.Zval[sel, :Tb]))
            Zg = torch.zeros(len(sel), Tb, len(f), device=dev)
            Zg.scatter_add_(2, loc.clamp_min(0), coef)
            delta = -(Zg @ Wg)                                      # [B,Tb,d]
            changed = (Zg != 0).any(-1)
            r = self.R[sel, :Tb] + delta
            am = self.am[sel, :Tb]
            ids = self.ids[sel, :Tb]
            nll, val = modelio.suffix_nll(self.suf, torch.cat([r, self.R[sel, :Tb]]).to(self.dtype),
                                          torch.cat([ids, ids]), torch.cat([am, am]))
            nll, val = nll.to(dev), val.to(dev)
            b = len(sel)
            first = changed.float().argmax(1)
            pos = torch.arange(Tb, device=dev)[None, :]
            m = val[:b] & (pos >= first[:, None]) & changed.any(1)[:, None]
            total += float(((nll[:b] - nll[b:]) * m).sum())
        return total


# ---------------------------------------------------------------------------------------
# calibration statistics: means, firing rates, donor reservoirs (separate documents)
# ---------------------------------------------------------------------------------------
class CalStats:
    """From the `cal` partition only: E[z_j] (zeros included), firing rate, and per-feature
    uniform random donor samples (equivalent to a reservoir: every firing has equal
    probability), optionally position-stratified."""

    @torch.no_grad()
    def __init__(self, model, spec, dic, cal, cap=2048, seed=1234, max_tokens=8192):
        d_sae = dic.d_sae
        s1 = torch.zeros(d_sae, dtype=torch.float64, device=dic.device)
        s2 = torch.zeros(d_sae, dtype=torch.float64, device=dic.device)
        fires = torch.zeros(d_sae, dtype=torch.float64, device=dic.device)
        n = 0
        F, V, P = [], [], []
        for sel, ids, am in cal.batches(max_tokens=max_tokens):
            h = modelio.prefix(model, spec, ids, am)
            B, T, d = h.shape
            z, _ = dic.encode(h.reshape(-1, d))
            m = am.bool().to(z.device).reshape(-1)
            zv = z[m]
            s1 += zv.sum(0).double()
            s2 += zv.pow(2).sum(0).double()
            fires += (zv != 0).sum(0).double()
            n += zv.shape[0]
            pos = torch.arange(T, device=z.device).repeat(B)[m]
            r, c = torch.nonzero(zv, as_tuple=True)
            F.append(c.cpu().numpy())
            V.append(zv[r, c].float().cpu().numpy())
            P.append(pos[r].cpu().numpy())
        self.mean = (s1 / max(n, 1)).float()
        self.ez2 = (s2 / max(n, 1)).cpu().numpy()
        self.rate = (fires / max(n, 1)).cpu().numpy()
        self.n_tokens = n
        F, V, P = np.concatenate(F), np.concatenate(V), np.concatenate(P)
        rng = np.random.default_rng(seed)
        key = rng.random(len(F))
        o = np.lexsort((key, F))                   # by feature, then random key
        F, V, P = F[o], V[o], P[o]
        bounds = np.searchsorted(F, np.arange(d_sae + 1))
        self.n_firings = np.diff(bounds)
        take = np.minimum(self.n_firings, cap)
        self.ptr = np.concatenate([[0], np.cumsum(take)])
        sel = np.concatenate([np.arange(bounds[j], bounds[j] + take[j]) for j in range(d_sae)])
        self.don_v = V[sel].astype(np.float32)
        self.don_p = P[sel].astype(np.int64)
        self.don_s = _stratum(self.don_p)
        # per-feature, per-stratum donor index lists for the stratified operator
        self.cap = cap
        self.device = dic.device
        self.don_v_t = torch.as_tensor(self.don_v, device=dic.device)

    def donors(self, j):
        return self.don_v[self.ptr[j]:self.ptr[j + 1]]

    def report(self, alive):
        a = np.asarray(alive, bool)
        nf = self.n_firings[a]
        return dict(n_tokens=int(self.n_tokens), donors_cap=int(self.cap),
                    alive_with_zero_cal_firings=int((nf == 0).sum()),
                    donors_median=float(np.median(np.minimum(nf, self.cap))) if a.any() else 0,
                    donors_min=int(np.minimum(nf, self.cap).min()) if a.any() else 0,
                    frac_alive_saturated=float((nf > self.cap).mean()) if a.any() else 0)


def _donor_values(cs, F, keys, n_per):
    """Pick donor values for features F (one per key) from each feature's donor pool."""
    lo = cs.ptr[F]
    cnt = cs.ptr[F + 1] - lo
    safe = np.maximum(cnt, 1)
    off = (keys % safe.astype(np.uint64)).astype(np.int64)
    v = cs.don_v[lo + off]
    v[cnt == 0] = 0.0
    return v


def _stratified_values(cs, F, pos, keys, min_donors=32):
    """Donor from the recipient's position stratum; < min_donors there -> pooled backoff."""
    out = np.zeros(len(F), dtype=np.float32)
    strat = _stratum(pos)
    for j in np.unique(F):
        m = F == j
        lo, hi = cs.ptr[j], cs.ptr[j + 1]
        if hi == lo:
            continue
        ds, dv = cs.don_s[lo:hi], cs.don_v[lo:hi]
        for s in np.unique(strat[m]):
            mm = m & (strat == s)
            pool = dv[ds == s]
            if len(pool) < min_donors:
                pool = dv
            out[mm] = pool[(keys[mm] % np.uint64(len(pool))).astype(np.int64)]
    return out


# ---------------------------------------------------------------------------------------
# metric drivers
# ---------------------------------------------------------------------------------------
def _accumulate(cache, S, F, dl, ntg, d_sae):
    n_docs = len(cache.part.docs)
    doc = cache.part.doc_index[S]
    eff = np.zeros((d_sae, n_docs), dtype=np.float64)
    np.add.at(eff, (F, doc), dl)
    tg = np.zeros((d_sae, n_docs), dtype=np.int64)
    np.add.at(tg, (F, doc), ntg)
    return eff, tg


def items_firing(cache, feats):
    S, F = [], []
    for j in feats:
        s = cache.seqs_firing(int(j))
        S.append(s)
        F.append(np.full(len(s), j, dtype=np.int64))
    return (np.concatenate(S) if S else np.zeros(0, np.int64),
            np.concatenate(F) if F else np.zeros(0, np.int64))


def items_all(cache, feats):
    N = cache.part.ids.shape[0]
    S = np.tile(np.arange(N), len(feats))
    F = np.repeat(np.asarray(feats, np.int64), N)
    return S, F


# Parallel sharding (set by evaljob). SHARD = (index, count) or None; SHARED = S3 prefix where
# shard boxes publish chunks. Every chunk carries a fingerprint of the exact (sequence, feature)
# items it covers; a chunk whose fingerprint differs from this box's item list is never used.
SHARD = None
SHARED = None
SHARD_MODE = False        # True on helper boxes: compute own chunks, publish, never assemble


def _sfh(S, F):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(S, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(F, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _load_chunk(f, n, sfh):
    if f is None or not f.exists():
        return None
    try:
        z = np.load(f)
    except Exception:
        return None
    if int(z["n"]) != n or ("sfh" in z.files and str(z["sfh"]) != sfh):
        return None
    return z["dl"], z["ntg"]


def _fetch_shared(f, n, sfh):
    if SHARED is None or f is None:
        return None
    common.s3_get("%s/%s/%s" % (SHARED, f.parent.name, f.name), f, required=False)
    return _load_chunk(f, n, sfh)


def _chunked(cache, S, F, fn, max_tokens, progress, chunk_dir, tag, chunk_items, prefilter):
    """Run items in fixed chunks of the (deterministic) item order, saving each chunk; a
    restarted box resumes at the first missing chunk, so a reclaim costs at most one chunk.
    With SHARD=(i, n): first pass computes only chunks k % n == i (publishing them to SHARED);
    the assembling box then takes every other chunk from SHARED when present, computing any
    that are missing itself, so a dead helper only costs time."""
    n = len(S)
    dl = np.zeros(n, dtype=np.float64)
    ntg = np.zeros(n, dtype=np.int64)
    starts = list(range(0, n, chunk_items))
    done = set()

    def path(c0):
        return None if chunk_dir is None else pathlib.Path(chunk_dir) / ("%s_%09d.npz" % (tag, c0))

    def compute(k, c0):
        c1 = min(n, c0 + chunk_items)
        sfh = _sfh(S[c0:c1], F[c0:c1])
        x, y = cache.run(S[c0:c1], F[c0:c1], fn, max_tokens, progress, prefilter=prefilter)
        dl[c0:c1], ntg[c0:c1] = x, y
        f = path(c0)
        if f is not None:
            common.save_npz(f, dl=x, ntg=y, n=np.int64(c1 - c0), sfh=np.array(sfh))
            if SHARED is not None:
                common.s3_put(f, "%s/%s/%s" % (SHARED, f.parent.name, f.name))
        done.add(k)

    # pass 0: anything already on this box (own earlier work or a restored sync)
    for k, c0 in enumerate(starts):
        c1 = min(n, c0 + chunk_items)
        r = _load_chunk(path(c0), c1 - c0, _sfh(S[c0:c1], F[c0:c1]))
        if r is not None:
            dl[c0:c1], ntg[c0:c1] = r
            done.add(k)
    # pass 1: this box's own residue class
    for k, c0 in enumerate(starts):
        # residue of (chunk index + tag hash): spreads one-chunk-per-draw stages across boxes too
        if k in done or (SHARD is not None and (k + zlib.crc32(tag.encode())) % SHARD[1] != SHARD[0]):
            continue
        if SHARD is not None:
            c1 = min(n, c0 + chunk_items)
            r = _fetch_shared(path(c0), c1 - c0, _sfh(S[c0:c1], F[c0:c1]))
            if r is not None:
                dl[c0:c1], ntg[c0:c1] = r
                done.add(k)
                continue
        compute(k, c0)
    if SHARD_MODE:
        return dl, ntg            # helper: partial arrays, never saved as a metric
    # pass 2 (assembler): every remaining chunk from SHARED if a helper made it, else compute;
    # walk from the END so the assembler and still-running helpers rarely duplicate a chunk
    for k, c0 in reversed(list(enumerate(starts))):
        if k in done:
            continue
        c1 = min(n, c0 + chunk_items)
        r = _fetch_shared(path(c0), c1 - c0, _sfh(S[c0:c1], F[c0:c1]))
        if r is not None:
            dl[c0:c1], ntg[c0:c1] = r
            done.add(k)
        else:
            compute(k, c0)
    return dl, ntg


def run_metric(cache, cs, metric, feats, alive, draws=1, exp_seed=20261005,
               max_tokens=8192, progress=None, chunk_dir=None, chunk_items=200_000):
    """Returns dict(eff[d_sae, n_docs (, draws)], targets, status, n_items)."""
    d_sae = cache.dic.d_sae
    dev = cache.dic.device
    part = cache.part
    status = np.zeros(d_sae, dtype=np.int8)
    status[np.asarray(alive, bool)] = STATUS_MEASURED
    if metric in ("zero", "sp_resample", "st_resample"):
        never = np.asarray(alive, bool) & (cache.fire_nseq == 0)
        status[never] = STATUS_NOOP_ZERO
    feats = np.asarray(feats, dtype=np.int64)
    outside = np.ones(d_sae, bool)
    outside[feats] = False
    status[outside & (status != 0)] = STATUS_NOT_EVALUATED   # alive but outside this universe

    if metric == "zero":
        S, F = items_firing(cache, feats)
        fn = lambda s, f, old, am: torch.zeros_like(old)
        dl, ntg = _chunked(cache, S, F, fn, max_tokens, progress, chunk_dir, "zero",
                           chunk_items, False)
        eff, tg = _accumulate(cache, S, F, dl, ntg, d_sae)
        return dict(eff=eff, targets=tg, status=status, n_items=len(S))

    if metric == "mean":
        mean = cs.mean.to(dev)
        fz = np.asarray([j for j in feats if float(cs.mean[j]) != 0.0], dtype=np.int64)
        S, F = items_all(cache, fz)
        fn = lambda s, f, old, am: mean[torch.as_tensor(f, device=dev)][:, None].expand_as(old).clone()
        dl, ntg = _chunked(cache, S, F, fn, max_tokens, progress, chunk_dir, "mean",
                           chunk_items, False)
        eff, tg = _accumulate(cache, S, F, dl, ntg, d_sae)
        return dict(eff=eff, targets=tg, status=status, n_items=len(S))

    salt = {"sp_resample": 1, "mg_resample": 2, "st_resample": 3}[metric]
    effs, tgs, n_items = [], None, 0
    for dr in range(draws):
        if metric in ("sp_resample", "st_resample"):
            S, F = items_firing(cache, feats)
        else:
            S, F = items_all(cache, feats)

        def fn(s, f, old, am, dr=dr):
            B, Tb = old.shape
            pos = np.tile(np.arange(Tb), B)
            ss = np.repeat(s, Tb)
            ff = np.repeat(f, Tb)
            keys = draw_keys(exp_seed + salt, ff, part.doc[ss], part.row[ss], pos, dr)
            if metric == "mg_resample":
                u = (keys >> np.uint64(11)).astype(np.float64) / float(1 << 53)
                fire = u < cs.rate[ff]
                with np.errstate(over="ignore"):
                    k2 = splitmix64(keys ^ np.uint64(0xD1B54A32D192ED03))
                v = _donor_values(cs, ff, k2, 1)
                v = np.where(fire, v, 0.0).astype(np.float32)
                return torch.as_tensor(v.reshape(B, Tb), device=old.device)
            if metric == "sp_resample":
                v = _donor_values(cs, ff, keys, 1).reshape(B, Tb)
            else:
                v = _stratified_values(cs, ff, pos, keys).reshape(B, Tb)
            v = torch.as_tensor(v, device=old.device)
            return torch.where(old > 0, v, old)

        dl, ntg = _chunked(cache, S, F, fn, max_tokens, progress, chunk_dir,
                           "%s_d%02d" % (metric, dr), chunk_items, metric == "mg_resample")
        e, tg = _accumulate(cache, S, F, dl, ntg, d_sae)
        effs.append(e)
        tgs = tg if tgs is None else np.maximum(tgs, tg)
        n_items += len(S)
    return dict(eff=np.stack(effs, -1), targets=tgs, status=status, n_items=n_items)


# ---------------------------------------------------------------------------------------
# gradient x input and integrated gradients, all features at once
# ---------------------------------------------------------------------------------------
def _loss_from_z(cache, Sb, Tb, Z):
    """Summed valid NLL for sequences Sb from latents Z [B,Tb,d_sae] (grad flows to Z)."""
    dic = cache.dic
    coef = cache.coef[Sb, :Tb] if dic.mode == "bypass" else None
    r = dic.splice_fp32(Z, coef)
    ids, am = cache.ids[Sb, :Tb], cache.am[Sb, :Tb]
    nll, val = modelio.suffix_nll(cache.suf, r.to(cache.dtype), ids, am, need_grad=True)
    return (nll * val.to(nll.device)).sum(), nll, val


def gradient_metrics(cache, steps_ladder=(16, 32, 64, 128), max_tokens=2048,
                     completeness_tol=0.01, top_overlap_tol=0.96, alive=None):
    """gxi (three estimands) and IG with step doubling. Accumulation is per position, so the
    result is invariant to batching (gate 8); N = all valid targets in the partition."""
    d_sae = cache.dic.d_sae
    dev = cache.dic.device
    part = cache.part
    n_docs = len(part.docs)
    N = cache.n_targets
    order = np.argsort(-part.lengths, kind="stable")
    batches = []
    i = 0
    while i < len(order):
        Tb = int(part.lengths[order[i]])
        nb = max(1, max_tokens // max(Tb, 1))
        batches.append((order[i:i + nb], Tb))
        i += nb

    def dense_z(Sb, Tb):
        idx = cache.Zidx[Sb, :Tb].long()
        Z = torch.zeros(len(Sb), Tb, d_sae, device=dev)
        Z.scatter_(2, idx.clamp_min(0), torch.where(idx >= 0, cache.Zval[Sb, :Tb], 0.0))
        return Z

    mag = torch.zeros(d_sae, n_docs, dtype=torch.float64, device=dev)
    sgn = torch.zeros(d_sae, n_docs, dtype=torch.float64, device=dev)
    for sel, Tb in batches:
        Sb = torch.as_tensor(sel, device=dev)
        Z = dense_z(Sb, Tb).requires_grad_(True)
        loss, _, _ = _loss_from_z(cache, Sb, Tb, Z)
        loss.backward()
        prod = (Z.detach() * Z.grad) * cache.am[Sb, :Tb, None].float()
        doc = torch.as_tensor(part.doc_index[sel], device=dev)
        mag.index_add_(1, doc, prod.abs().sum(1).double().T)
        sgn.index_add_(1, doc, prod.sum(1).double().T)
    out = dict(gxi_magnitude=(mag / N).cpu().numpy(), gxi_signed=(sgn / N).cpu().numpy())

    # integrated gradients with step doubling
    ig_hist = []
    for m in steps_ladder:
        acc = torch.zeros(d_sae, n_docs, dtype=torch.float64, device=dev)
        acc_abs = torch.zeros(d_sae, n_docs, dtype=torch.float64, device=dev)
        resid_num, resid_den = 0.0, 0.0
        for sel, Tb in batches:
            Sb = torch.as_tensor(sel, device=dev)
            Z0 = dense_z(Sb, Tb)
            with torch.no_grad():
                L1, _, _ = _loss_from_z(cache, Sb, Tb, Z0)
                L0, _, _ = _loss_from_z(cache, Sb, Tb, torch.zeros_like(Z0))
            g_acc = torch.zeros_like(Z0)
            for kstep in range(m):
                alpha = (kstep + 0.5) / m
                Zp = (alpha * Z0).detach().requires_grad_(True)       # the PATH latent
                loss, _, _ = _loss_from_z(cache, Sb, Tb, Zp)
                loss.backward()
                g_acc += Zp.grad
            ig = Z0 * g_acc / m * cache.am[Sb, :Tb, None].float()
            doc = torch.as_tensor(part.doc_index[sel], device=dev)
            acc.index_add_(1, doc, ig.sum(1).double().T)
            acc_abs.index_add_(1, doc, ig.abs().sum(1).double().T)
            resid_num += float(ig.sum()) - (float(L1) - float(L0))
            resid_den += abs(float(L1) - float(L0))
        ig_sig = (acc / N).cpu().numpy()
        rel = abs(resid_num) / max(resid_den, 1e-30)
        score = np.abs(ig_sig.sum(1))
        if alive is not None:
            score = np.where(alive, score, -np.inf)
        top = set(np.argsort(-score)[:50].tolist())
        ig_hist.append(dict(steps=m, completeness_rel_error=rel, top50=top, ig=ig_sig,
                            ig_abs=(acc_abs / N).cpu().numpy()))
        log("ig", "steps %d completeness rel err %.4f" % (m, rel))
        if len(ig_hist) >= 2:
            ov = len(ig_hist[-1]["top50"] & ig_hist[-2]["top50"]) / 50.0
            ig_hist[-1]["top50_overlap_prev"] = ov
            if rel < completeness_tol and ov >= top_overlap_tol:
                break
    final = ig_hist[-1]
    out["ig_signed"] = final["ig"]
    out["ig_magnitude"] = final["ig_abs"]
    out["ig_steps"] = final["steps"]
    out["ig_convergence"] = [dict(steps=h["steps"], completeness_rel_error=h["completeness_rel_error"],
                                  top50_overlap_prev=h.get("top50_overlap_prev"))
                             for h in ig_hist]
    out["ig_converged"] = bool(final["completeness_rel_error"] < completeness_tol and
                               final.get("top50_overlap_prev", 0) >= top_overlap_tol)
    return out


def score_from_eff(eff, targets_total, status):
    """Feature score = summed effect / total valid targets; NaN where NOT_EVALUATED."""
    e = eff.sum(axis=1) if eff.ndim == 2 else eff.sum(axis=1).mean(-1)
    s = e / max(targets_total, 1)
    s = s.astype(np.float64)
    s[status == STATUS_NOT_EVALUATED] = np.nan
    return s
