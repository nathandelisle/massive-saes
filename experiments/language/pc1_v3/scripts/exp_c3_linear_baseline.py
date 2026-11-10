#!/usr/bin/env python
"""Compare PCA and SAE reconstruction, and measure the change in model loss."""

import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from v3.common import registry, io_utils as io                      # noqa: E402
from v3.common.data import open_activations                          # noqa: E402
from v3.common.pca import estimate_pcs                               # noqa: E402

EXP = "c3_linear_baseline"
K_REPORT = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


# --------------------------------------------------------------------------
# linear algebra
# --------------------------------------------------------------------------

def _eigh_desc(A, device="cuda"):
    """Eigendecomposition, eigenvalues DESCENDING, float64.

    Tries the GPU first (cusolver syevd on a 4096^2 float64 matrix is seconds
    even at 1/64 fp64 throughput) and falls back to numpy, because a
    driver/cusolver mismatch must degrade to slow, not to wrong.
    """
    try:
        t = torch.tensor(A, dtype=torch.float64, device=device)
        w, V = torch.linalg.eigh(t)
        w = w.cpu().numpy()
        V = V.cpu().numpy()
        del t
        torch.cuda.empty_cache()
    except Exception as e:                                  # pragma: no cover
        print(f"    [eigh] GPU path failed ({str(e)[:70]}); using numpy",
              flush=True)
        w, V = np.linalg.eigh(A)
    order = np.argsort(w)[::-1]
    return np.ascontiguousarray(w[order]), np.ascontiguousarray(V[:, order])


def orthonormal_complement(v1):
    """Exact orthonormal basis Q (d x d-1) of v1's orthogonal complement.

    Built by Householder reflection rather than by eigendecomposing P C P and
    discarding the near-zero eigenpair.  P C P has one EXACT null direction
    (v1), but if the smallest genuine content eigenvalue is also tiny, eigh is
    free to return an arbitrary rotation inside that near-degenerate pair and
    the "content" basis quietly acquires a component along PC1 -- which is the
    one contamination this whole experiment exists to exclude.  Householder
    makes Q perp v1 true to machine precision by construction.
    """
    a = np.asarray(v1, dtype=np.float64).copy()
    a /= np.linalg.norm(a)
    d = a.shape[0]
    sigma = -1.0 if a[0] > 0 else 1.0        # avoid cancellation in a - sigma*e1
    u = a.copy()
    u[0] -= sigma
    nu = np.linalg.norm(u)
    if nu < 1e-12:                            # a is already +-e1
        return np.ascontiguousarray(np.eye(d)[:, 1:])
    w = u / nu
    H = np.eye(d) - 2.0 * np.outer(w, w)      # H e1 = sigma * a
    return np.ascontiguousarray(H[:, 1:])


@torch.no_grad()
def fit_covariance(adir, n_fit, slice_, device, batch, seed=0, dtype="float64"):
    """Compute centered covariance, using float64 by default.

    With --cov-dtype float32, chunk products use float32 and are accumulated
    in float64. Evaluation uses float64.
    """
    acts, mean = open_activations(adir)
    total, d = acts.shape
    n_fit = int(min(n_fit, total))
    if slice_ == "uniform":
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total, size=n_fit, replace=False))
        chunks = [idx[i:i + batch] for i in range(0, n_fit, batch)]
    elif slice_ == "head":
        chunks = [np.arange(i, min(i + batch, n_fit))
                  for i in range(0, n_fit, batch)]
    else:
        raise ValueError(slice_)
    dt = torch.float64 if dtype == "float64" else torch.float32
    mu = torch.tensor(np.asarray(mean), dtype=dt, device=device)
    C = torch.zeros(d, d, dtype=torch.float64, device=device)
    seen = 0
    for ii in chunks:
        x = torch.tensor(np.asarray(acts[ii], dtype=np.float32),
                         device=device).to(dt) - mu
        C += (x.T @ x).double()
        seen += x.shape[0]
        del x
    C /= max(seen - 1, 1)
    out = C.cpu().numpy()
    del C
    torch.cuda.empty_cache()
    return out, seen


def content_basis(C, v1, device):
    """Principal components OF THE CONTENT SUBSPACE, descending.

    Returns U (d x d-1, orthonormal, exactly perp to v1) and its eigenvalues.
    Equivalent to eigendecomposing P C P and dropping the null pair, but done in
    the complement's own coordinates so orthogonality to v1 is structural.
    """
    Q = orthonormal_complement(v1)                       # d x (d-1)
    Cs = Q.T @ (C @ Q)                                   # (d-1) x (d-1)
    Cs = 0.5 * (Cs + Cs.T)                               # kill asymmetry drift
    lam, W = _eigh_desc(Cs, device=device)
    return np.ascontiguousarray(Q @ W), lam


def random_rotations(d1, n_draws, device, seed=0):
    """n_draws Haar-random orthogonal (d1 x d1) matrices, on device, float64.

    Used as U @ O, so every random basis lies exactly inside the content
    subspace; a random basis of R^d would leak a component along v1 and inflate
    the zero-learning floor.
    """
    out = []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(n_draws):
        A = torch.randn(d1, d1, generator=g, dtype=torch.float32)
        Qr, R = torch.linalg.qr(A.to(device))
        Qr = Qr * torch.sign(torch.diagonal(R)).unsqueeze(0)   # unique QR
        out.append(Qr.double())
        del A, R
    return out


# --------------------------------------------------------------------------
# the single evaluation pass
# --------------------------------------------------------------------------

@torch.no_grad()
def eval_pass(adir, v1, U, V, rand_Os, saes, device, n_eval, batch,
              slice_="tail"):
    """One pass over the eval slice; returns the FULL k-curve for every method.

    The trick that makes the whole k-sweep free: for an ORTHONORMAL basis the
    captured energy at budget k is a prefix sum of per-direction energies, so
    accumulating
        s[j] = sum_t <x_perp(t), u_j>^2                       (fixed support)
        a[j] = sum_t (j-th LARGEST squared coefficient of token t)  (adaptive)
    gives captured(k) = cumsum(.)[k] / total for every k at once, in one pass.

    Projections are float64.  Not caution for its own sake: the closure check
    below (an orthonormal basis must account for ALL the energy) is the only
    thing standing between a subtly wrong basis and a plausible wrong number,
    and in float32 its own rounding is ~4e-4 relative -- the same size as the
    effects being measured, so the check would stop being a check.
    """
    acts, mean = open_activations(adir)
    total, d = acts.shape
    n = int(min(n_eval, total))
    start = total - n if slice_ == "tail" else 0
    mu32 = torch.tensor(np.asarray(mean), dtype=torch.float32, device=device)
    v1_32 = torch.tensor(np.asarray(v1), dtype=torch.float32, device=device)
    v1_64 = v1_32.double()
    Ut = torch.tensor(U, dtype=torch.float64, device=device)          # d x d-1
    Vt = torch.tensor(V, dtype=torch.float64, device=device)          # d x d
    d1 = Ut.shape[1]

    s_content = torch.zeros(d1, dtype=torch.float64, device=device)
    a_content = torch.zeros(d1, dtype=torch.float64, device=device)
    s_full = torch.zeros(d, dtype=torch.float64, device=device)
    s_rand = [torch.zeros(d1, dtype=torch.float64, device=device)
              for _ in rand_Os]
    tot_perp = tot_full = 0.0
    sae_acc = {k: dict(err_c=0.0, err_t=0.0, err_pc1=0.0, l0=0.0, nz=0)
               for k in saes}
    seen = 0

    for i in range(start, start + n, batch):
        j = min(i + batch, start + n)
        x32 = torch.tensor(np.asarray(acts[i:j], dtype=np.float32),
                           device=device) - mu32
        c1_32 = x32 @ v1_32
        xp32 = x32 - c1_32[:, None] * v1_32[None, :]

        x = x32.double()
        xp = xp32.double()
        tot_full += x.pow(2).sum().item()
        tot_perp += xp.pow(2).sum().item()

        co = xp @ Ut                                   # [B, d-1] float64
        co2 = co.pow(2)
        s_content += co2.sum(0)
        a_content += torch.sort(co2, dim=1, descending=True).values.sum(0)
        for r, O in enumerate(rand_Os):
            s_rand[r] += (co @ O).pow(2).sum(0)
        del co, co2

        s_full += (x @ Vt).pow(2).sum(0)

        for name, (sae, cond) in saes.items():
            recon, z = sae(xp32 if cond == "bypass" else x32)
            if cond == "bypass":
                recon = recon + c1_32[:, None] * v1_32[None, :]
            e = (recon - x32).double()
            e1sq = (e @ v1_64).pow(2).sum().item()
            etot = e.pow(2).sum().item()
            sae_acc[name]["err_t"] += etot
            sae_acc[name]["err_pc1"] += e1sq
            sae_acc[name]["err_c"] += etot - e1sq
            sae_acc[name]["l0"] += (z != 0).double().sum().item()
            sae_acc[name]["nz"] += z.shape[0]
            del recon, z, e
        seen += x32.shape[0]
        del x32, xp32, c1_32, x, xp

    # --- closure: an orthonormal basis must account for ALL of the energy
    err_c = abs(s_content.sum().item() - tot_perp) / max(tot_perp, 1e-30)
    err_f = abs(s_full.sum().item() - tot_full) / max(tot_full, 1e-30)
    assert err_c < 1e-8, f"content basis leaks energy: rel err {err_c:.2e}"
    assert err_f < 1e-8, f"full basis leaks energy: rel err {err_f:.2e}"

    def curve(acc, denom):
        c = torch.cumsum(acc, 0).cpu().numpy() / max(denom, 1e-30)
        return np.clip(c, 0.0, 1.0)                        # captured fraction

    out = dict(
        n_eval_tokens=seen, eval_slice=slice_, d_model=int(d),
        closure_rel_err=dict(content=err_c, full=err_f),
        tot_perp=tot_perp, tot_full=tot_full,
        pc1_var_weight_eval=(tot_full - tot_perp) / max(tot_full, 1e-30),
        fve_pca_fixed=curve(s_content, tot_perp),
        fve_pca_adaptive=curve(a_content, tot_perp),
        fve_pca_full=curve(s_full, tot_full),
        fve_rand=[curve(sr, tot_perp) for sr in s_rand],
        sae={})
    for name, acc in sae_acc.items():
        out["sae"][name] = dict(
            fvu_content=acc["err_c"] / max(tot_perp, 1e-30),
            fvu_total=acc["err_t"] / max(tot_full, 1e-30),
            fvu_pc1=acc["err_pc1"] / max(tot_full - tot_perp, 1e-30),
            l0=acc["l0"] / max(acc["nz"], 1))
    # adaptive can never be worse than fixed at any budget (same basis, better
    # support); if it is, the sort or an accumulator is wrong
    bad = np.flatnonzero(out["fve_pca_adaptive"] < out["fve_pca_fixed"] - 1e-9)
    assert bad.size == 0, f"adaptive < fixed at k={list(bad[:5] + 1)}"
    return out


# --------------------------------------------------------------------------
# downstream: splice the LINEAR reconstructions into the residual stream
# --------------------------------------------------------------------------

@contextlib.contextmanager
def linear_splice(model, spec, mu, v1, mode, U=None, V=None, k=64, layer=None):
    """Residual-stream splice for the non-SAE conditions.

    Deliberately NOT added to v3/common/splice.py: that module is on the
    critical path of a1/a5/r4/r5 and of every published number in the paper, and
    a new mode string there is a new way for those to break.  The hook body
    honours the same contract (tuple-aware, dtype-preserving, mean re-added).

    modes:
      mean_only     h -> mu                      (destroy everything: CE ceiling)
      pc1_only      h -> mu + <x,v1> v1          (rank-1, no dictionary)
      pca_fixed     h -> mu + <x,v1> v1 + U_k U_k^T x_perp
      pca_adaptive  h -> mu + <x,v1> v1 + top-k-coefficient reconstruction in U
      pca_full      h -> mu + V_k V_k^T x        (plain rank-k linear AE)
    """
    from v3.common.data import get_blocks
    from v3.common.registry import block_index
    from v3.common.splice import hook_device
    blocks = get_blocks(model, spec.block_path)
    li = block_index(spec, layer)
    # NOT next(model.parameters()).device -- under device_map='auto' that is
    # the embedding shard, not the block we hook.  See splice.hook_device.
    dev = hook_device(model, spec, layer)
    mu_t = torch.as_tensor(np.asarray(mu), dtype=torch.float32, device=dev)
    v1_t = torch.as_tensor(np.asarray(v1), dtype=torch.float32, device=dev)
    Uk = Ufull = Vk = None
    if mode == "pca_fixed":
        Uk = torch.as_tensor(U[:, :k], dtype=torch.float32, device=dev)
    elif mode == "pca_adaptive":
        Ufull = torch.as_tensor(U, dtype=torch.float32, device=dev)
    elif mode == "pca_full":
        Vk = torch.as_tensor(V[:, :k], dtype=torch.float32, device=dev)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        od = h.dtype
        x = h.float() - mu_t
        if mode == "mean_only":
            new = torch.zeros_like(x) + mu_t
        elif mode == "pc1_only":
            c1 = x @ v1_t
            new = c1[..., None] * v1_t + mu_t
        elif mode == "pca_full":
            new = (x @ Vk) @ Vk.T + mu_t
        else:
            c1 = x @ v1_t
            xp = x - c1[..., None] * v1_t
            if mode == "pca_fixed":
                r = (xp @ Uk) @ Uk.T
            else:
                co = xp @ Ufull
                idx = co.abs().topk(k, dim=-1).indices
                m = torch.zeros_like(co).scatter_(-1, idx, co.gather(-1, idx))
                r = m @ Ufull.T
            new = r + c1[..., None] * v1_t + mu_t
        new = new.to(od)
        return (new,) + output[1:] if is_tuple else new

    handle = blocks[li].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def downstream_ce(spec, mean, v1, U, V, saes, k, device, n_seqs, chunk,
                  device_map=None):
    """CE for clean / SAE / linear-baseline splices on the same eval sequences."""
    from v3.common.data import load_model_and_tokenizer, eval_sequences
    from v3.common.splice import ce_loss, spliced_ce
    model, tok = load_model_and_tokenizer(spec, device=device,
                                          device_map=device_map)
    seqs, am = eval_sequences(tok, n_seqs=n_seqs)
    out = {}
    try:
        out["clean"] = ce_loss(model, seqs, chunk=chunk, am=am)
        for name, (sae, cond) in saes.items():
            out[f"sae_{name}"] = spliced_ce(
                model, spec, seqs, sae=sae, mu=mean, v1=v1,
                mode=("bypass" if cond == "bypass" else "standard"),
                chunk=chunk, am=am)
        for mode in ("mean_only", "pc1_only", "pca_fixed", "pca_adaptive",
                     "pca_full"):
            with linear_splice(model, spec, mean, v1, mode, U=U, V=V, k=k):
                out[mode] = ce_loss(model, seqs, chunk=chunk, am=am)
    finally:
        del model
        torch.cuda.empty_cache()
    base = out["clean"]
    return dict(ce=out, splice_cost={kk: v - base for kk, v in out.items()
                                     if kk != "clean"},
                n_seqs=int(seqs.shape[0]), n_tokens=int(am.sum()), k=int(k))


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------

def at(curve, k):
    """Captured-variance fraction at budget k (1-indexed), clamped to range."""
    if k is None or k < 1:
        return 0.0
    return float(curve[min(int(k), len(curve)) - 1])


def match_k(curve, target):
    """Smallest k whose captured fraction reaches `target`; None if never."""
    idx = np.flatnonzero(np.asarray(curve) >= target)
    return int(idx[0] + 1) if idx.size else None


def bits_per_token(k, n_atoms, float_bits=32):
    """Description length of a k-sparse code: coefficients plus index choice."""
    k, n_atoms = int(k), int(n_atoms)
    idx_bits = math.log2(math.comb(n_atoms, k)) if 0 < k <= n_atoms else 0.0
    return dict(coef_bits=float(k * float_bits), index_bits=float(idx_bits),
                total_bits=float(k * float_bits + idx_bits))


def _subsample_curves(ev, d1, d):
    """Log-spaced subsample of the k-curves so the JSON stays a few hundred KB
    instead of tens of MB, keeping every reported k exactly."""
    keep = set(int(round(v)) for v in np.unique(np.logspace(0, math.log10(d1), 200)))
    keep |= {k for k in K_REPORT if k <= d1} | {registry.TOPK_K, d1}
    sub = sorted(k for k in keep if 1 <= k <= d1)
    ev["curve_k"] = sub
    for key in ("fve_pca_fixed", "fve_pca_adaptive"):
        ev[key] = [at(ev[key], k) for k in sub]
    ev["fve_rand"] = [[at(c, k) for k in sub] for c in ev["fve_rand"]]
    subf = sorted(set(sub) | {d})
    ev["curve_k_full"] = subf
    ev["fve_pca_full"] = [at(ev["fve_pca_full"], k) for k in subf]


def run_model(spec, args):
    dev = args.device
    t0 = time.time()
    # batch scales down with d_model: the eval pass holds two [B, d-1] float64
    # coefficient matrices plus a sort workspace, which at d=4096 is already
    # 400 MB per 4096 rows before the SAE latents are allocated
    batch = int(max(512, min(8192, 8192 * 1024 // max(spec.d_model, 1))))

    dmap = "auto" if spec.min_gpu_gb > 100 else None
    adir = io.ensure_acts(spec, device=dev, device_map=dmap)
    evals, evecs, mean = estimate_pcs(adir)
    v1 = evecs[:, 0]
    R = float(evals[0] / max(evals.sum() - evals[0], 1e-30))
    print(f"  [{spec.key}] d={spec.d_model} batch={batch} R={R:.1f} "
          f"acts={adir}", flush=True)

    # ---- SAEs (never trained implicitly: a missing checkpoint is a missing
    # checkpoint, not a silent 15-GPU-hour detour inside an analysis script)
    saes, missing = {}, []
    for cond in ("standard", "bypass"):
        s = io.load_cached_sae(spec, args.seed, device=dev, arch="topk",
                               condition=cond, k=registry.TOPK_K)
        if s is None and args.train_missing:
            s, _, _, _, _ = io.get_sae(spec, args.seed, arch="topk",
                                       condition=cond, device=dev,
                                       k=registry.TOPK_K)
        if s is None:
            missing.append(cond)
        else:
            s.eval()
            saes[cond] = (s, cond)
    if missing:
        print(f"  [{spec.key}] MISSING SAE checkpoints: {missing} "
              f"(baselines still computed)", flush=True)

    row = dict(model=spec.key, d_model=spec.d_model, d_sae=spec.d_sae,
               target_layer=spec.target_layer, seed=args.seed,
               topk_k=registry.TOPK_K, R=R,
               pc1_var_explained=float(evals[0] / evals.sum()),
               missing_saes=missing, fits={},
               fit_tokens=args.fit_tokens, eval_tokens=args.eval_tokens,
               eval_slice=args.eval_slice)

    fit_slices = ["uniform"] + ([] if args.no_gen_check else ["head"])
    primary = None
    kk = registry.TOPK_K
    for fs in fit_slices:
        ta = time.time()
        C, n_fit = fit_covariance(adir, args.fit_tokens, fs, dev, batch,
                                  seed=args.seed, dtype=args.cov_dtype)
        U, lam_c = content_basis(C, v1, dev)
        lam_f, V = _eigh_desc(0.5 * (C + C.T), device=dev)
        del C
        # the content basis must be exactly orthogonal to PC1
        leak = float(np.abs(U.T @ v1).max())
        assert leak < 1e-8, f"content basis leaks onto PC1: max|<u,v1>|={leak:.2e}"
        print(f"    fit[{fs}] n={n_fit:,} cov+eigh {time.time() - ta:.0f}s "
              f"leak={leak:.1e}", flush=True)

        d1 = U.shape[1]
        rand_Os = random_rotations(d1, args.rand_draws, dev, seed=args.seed)
        ev = eval_pass(adir, v1, U, V, rand_Os, saes, dev,
                       args.eval_tokens, batch, slice_=args.eval_slice)
        del rand_Os
        torch.cuda.empty_cache()

        ev["fit_slice"] = fs
        ev["n_fit_tokens"] = n_fit
        ev["content_eigen_top20"] = [float(x) for x in lam_c[:20]]
        ev["content_participation_ratio"] = float(
            lam_c.sum() ** 2 / max((lam_c ** 2).sum(), 1e-30))
        ev["k_for_content_var"] = {
            f"{q:g}": match_k(ev["fve_pca_fixed"], q)
            for q in (0.5, 0.8, 0.9, 0.95, 0.99)}
        ks = sorted(set([k for k in K_REPORT if k <= d1] + [kk]))
        ev["table"] = {
            str(k): dict(
                pca_fixed=at(ev["fve_pca_fixed"], k),
                pca_adaptive=at(ev["fve_pca_adaptive"], k),
                pca_full_total=at(ev["fve_pca_full"], k),
                rand_mean=float(np.mean([at(c, k) for c in ev["fve_rand"]])),
                rand_analytic=float(min(k, d1) / d1))
            for k in ks}
        # headline comparison at the SAE's own budget
        cmp_ = {}
        f_fix, f_ada = at(ev["fve_pca_fixed"], kk), at(ev["fve_pca_adaptive"], kk)
        f_rnd = float(np.mean([at(c, kk) for c in ev["fve_rand"]]))
        for name, sv in ev["sae"].items():
            fve_sae = 1.0 - sv["fvu_content"]
            cmp_[name] = dict(
                l0=sv["l0"], fve_content_sae=fve_sae,
                fve_content_pca_fixed=f_fix, fve_content_pca_adaptive=f_ada,
                fve_content_rand=f_rnd,
                gain_over_fixed_pp=100 * (fve_sae - f_fix),
                gain_over_adaptive_pp=100 * (fve_sae - f_ada),
                # error-reduction ratio: what fraction of the linear baseline's
                # residual survives under the SAE.  >1 means the SAE is WORSE.
                fvu_ratio_vs_fixed=sv["fvu_content"] / max(1 - f_fix, 1e-12),
                fvu_ratio_vs_adaptive=sv["fvu_content"] / max(1 - f_ada, 1e-12),
                k_pca_fixed_to_match=match_k(ev["fve_pca_fixed"], fve_sae),
                k_pca_adaptive_to_match=match_k(ev["fve_pca_adaptive"], fve_sae),
                bits=dict(sae=bits_per_token(round(sv["l0"]) or kk, spec.d_sae),
                          pca_adaptive=bits_per_token(kk, d1),
                          pca_fixed=bits_per_token(kk, kk)))
        ev["matched_budget"] = cmp_
        # the metric-is-saturable line: rank-1 PCA scored on TOTAL variance
        ev["pca_full_rank1_total_fve"] = at(ev["fve_pca_full"], 1)
        ev["pca_full_k_total_fve"] = at(ev["fve_pca_full"], kk)
        if args.drop_curves:
            for key in ("fve_pca_fixed", "fve_pca_adaptive", "fve_pca_full",
                        "fve_rand"):
                ev.pop(key, None)
        else:
            _subsample_curves(ev, d1, ev["d_model"])
        row["fits"][fs] = ev
        if fs == "uniform":
            primary = (U, V)
        if fs != fit_slices[-1]:
            del U, V

    # ---- downstream CE, using the primary (exposure-matched) bases
    if not args.no_ce and primary is not None:
        try:
            row["downstream"] = downstream_ce(
                spec, mean, v1, primary[0], primary[1], saes, kk, dev,
                args.eval_seqs, args.ce_chunk, device_map=dmap)
            print("    splice cost: " + "  ".join(
                f"{k}={v:+.4f}"
                for k, v in sorted(row["downstream"]["splice_cost"].items())),
                flush=True)
        except Exception as e:
            row["downstream_error"] = f"{type(e).__name__}: {e}"
            print(f"    [ce] FAILED: {row['downstream_error']}", flush=True)

    # ---- consistency with the number this experiment is a baseline FOR.
    # c3 recomputes the SAE's content FVU from scratch on the same 100k tail
    # slice c2 used, so the two must agree.  They are produced by different code
    # paths (c2 accumulates against v1 directly, c3 against a Householder
    # complement), which is exactly why agreeing is worth checking: a silent
    # disagreement means one of the two content subspaces is not the paper's.
    c2 = io.read_json("c2_fvu_decomposition", "fve_table")
    if c2 and spec.key in c2 and "uniform" in row["fits"]:
        chk = {}
        for cond, sv in row["fits"]["uniform"]["sae"].items():
            try:
                ref = c2[spec.key]["conditions"][cond]["mean"]["fvu_content"]
            except (KeyError, TypeError):
                continue
            chk[cond] = dict(c3=sv["fvu_content"], c2=ref,
                             abs_diff=abs(sv["fvu_content"] - ref))
        row["c2_crosscheck"] = chk
        for cond, v in chk.items():
            flag = "" if v["abs_diff"] < 5e-3 else "   <-- MISMATCH"
            print(f"    c2 crosscheck {cond}: fvu_content c3={v['c3']:.4f} "
                  f"c2={v['c2']:.4f} d={v['abs_diff']:.4f}{flag}", flush=True)

    row["wall_seconds"] = time.time() - t0
    return row


def summarize(row):
    p = row["fits"].get("uniform") or next(iter(row["fits"].values()))
    kk, t = row["topk_k"], None
    t = p["table"][str(kk)]
    parts = [f"SAE-{n} {100 * c['fve_content_sae']:.1f}%"
             for n, c in p["matched_budget"].items()]
    parts += [f"PCA-fixed {100 * t['pca_fixed']:.1f}%",
              f"PCA-adapt {100 * t['pca_adaptive']:.1f}%",
              f"rand {100 * t['rand_mean']:.1f}%"]
    return (f"[{row['model']}] R={row['R']:.1f} "
            f"w1={p['pc1_var_weight_eval']:.4f} | content-FVE @k={kk}: "
            + "  ".join(parts)
            + f" | rank-1 PCA total-FVE {100 * p['pca_full_rank1_total_fve']:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=registry.REFERENCE_SEED)
    ap.add_argument("--fit-tokens", type=int, default=1_000_000)
    ap.add_argument("--eval-tokens", type=int, default=100_000,
                    help="100k tail matches exp_c2 exactly, so the SAE columns "
                         "here must reproduce c2's fve_content")
    ap.add_argument("--eval-slice", default="tail", choices=["tail", "head"])
    ap.add_argument("--rand-draws", type=int, default=2)
    ap.add_argument("--cov-dtype", default="float64",
                    choices=["float64", "float32"])
    ap.add_argument("--eval-seqs", type=int, default=256)
    ap.add_argument("--ce-chunk", type=int, default=8)
    ap.add_argument("--no-ce", action="store_true")
    ap.add_argument("--no-gen-check", action="store_true",
                    help="skip the head-fit (strictly-disjoint) refit")
    ap.add_argument("--drop-curves", action="store_true")
    ap.add_argument("--train-missing", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    for spec in registry.iter_models(args.models, include_70b=False):
        try:
            row = run_model(spec, args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            io.write_json(EXP, spec.key, dict(model=spec.key,
                                              error=f"{type(e).__name__}: {e}"))
            continue
        io.write_json(EXP, spec.key, row)
        print(summarize(row), flush=True)

    # merge, never overwrite: one model per box is the normal case and the
    # combined table must accumulate across boxes and restarts
    d = os.path.join(io.ROOT, "results", EXP)
    table = {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".json") and f != "table.json":
                try:
                    r = json.load(open(os.path.join(d, f)))
                except Exception:
                    continue
                if isinstance(r, dict) and "model" in r:
                    table[r["model"]] = r
    io.write_json(EXP, "table", table)
    print(f"  merged table: {len(table)} models", flush=True)


if __name__ == "__main__":
    main()
