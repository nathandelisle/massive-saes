#!/usr/bin/env python
"""EXP-B1 — Architecture law: JumpReLU and Gated.  [NEW — registered predictions]

Thesis:   The inversion is a property of the OBJECTIVE, not of SAEs per se.
          Support-only sparsity (TopK, BatchTopK, JumpReLU's L0) leaves the
          PC1 tiling continuum loss-free; magnitude penalties (ReLU+L1, Gated's
          L1 on the gate path) price off-axis cancellation and break it.

REGISTERED PREDICTIONS (write into the paper before results exist):
          JumpReLU: top-50 sharing LOW under standard training (inversion),
                    repaired by bypass. Highest stakes: JumpReLU is what
                    Gemma Scope ships.
          Gated:    top-50 sharing HIGH already under standard training
                    (no inversion), like ReLU+L1.
          If either prediction fails, the objective-symmetry account is wrong
          and Section 5 must be rewritten — this is the falsifiable core.

Protocol: identical to v2 App G (same data, steps, batch, lr, dict size).
          Sparsity coefficient calibrated to hit L0 ~= 64 (matching TopK k=64),
          then full 20k-step runs, 5 seeds x {standard, bypass}. Reference rows
          for topk / batchtopk / relu_l1 are pulled from the shared cache
          (retrained if absent) so the final table is one uniform
          5-architecture x 2-condition grid.

SPARSITY-MATCHING GUARD (added after review — the law is only testable if L0 is
matched; an "inversion" difference at unmatched L0 is just an L0 difference):
  * calibration runs PER (arch, condition). The bypass condition removes PC1,
    which carries 90-99.9% of the variance, so the activation scale collapses
    and the coefficient that lands L0=64 on standard activations is wrong by
    orders of magnitude on bypass activations. TopK/BatchTopK are scale-free
    (k is exact); only the magnitude/threshold architectures need this — i.e.
    exactly the ones carrying the paper's contrast.
  * the calibrator BRACKETS the target (widening the grid geometrically) and
    then refines by secant in log-coefficient space. If it cannot bracket, the
    architecture is REFUSED (--allow-uncalibrated to override) and the failure
    is recorded in the results JSON.
  * L0 is MEASURED from the alive-pass firing rates (rates.sum() is the exact
    expected number of active features per token) on 100k tokens in the correct
    condition -- NOT read from the training log, which is stale by
    log_every-1 steps (for a 2000-step probe, log[-1] is step 1000).
  * every cell's measured L0 must land in [target*(1-tol), target*(1+tol)];
    otherwise `l0_matched` is False for that cell and `sparsity_matched` is
    False for the table. Do not put an unmatched table in the paper.

ATTRIBUTION GUARD (added after review): a rise in top-50 sharing has two
non-equivalent causes -- (a) the tiling symmetry broke (the law), or (b) the
importance RANKING moved off the dense cluster onto content features that were
always reproducible. We therefore also record, per cell, the dense fraction of
the top-50, sharing split by dense vs content, FVU, and the top-50 dCE scale
(a collapsed dCE scale means the ranking is noise and "sharing" is vacuous).

Cost:     see reviews/b1_b2.md. pythia-1.4b, 5 archs, fresh cache, chunk=64:
          ~10-20 L40S-h, dominated by the per-feature ablation loop
          (|cand| ~ 6.5k features x one full model forward over the eval set
          per feature x 10 arch-condition cells).

Run:      python scripts/exp_b1_architecture_law.py --models pythia-1.4b --chunk 64
"""

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from v3.common import registry, io_utils as io                      # noqa: E402
from v3.common.data import (load_model_and_tokenizer, eval_sequences,  # noqa: E402
                            activation_batches)
from v3.common.fastabl import (ablation_over, screen_candidates,      # noqa: E402
                               to_array, tuned_chunk, z_stats, dense_share)
from v3.common.matching import pairwise_over_seeds, max_cos           # noqa: E402
from v3.common.pca import estimate_pcs, dense_cluster                 # noqa: E402
from v3.common.saes import decoder_np, train_sae                      # noqa: E402

EXP = "b1_architecture_law"

# Starting brackets for the L0 calibration. These are only a STARTING bracket:
# the calibrator widens geometrically until the target L0 is bracketed, because
# the stationary coefficient scales with the activation scale (lambda* ~
# theta* * 2|<residual, d_j>| for the L0 penalty, ~ 2|<residual, d_j>| for L1,
# and the trainer's MSE is SUMMED over d_model, not averaged). Do not assume
# these ranges bracket anything on a model you have not calibrated before.
CAL_GRID = {
    "jumprelu": [3e-4, 3e-3, 3e-2, 3e-1, 3.0],
    "gated":    [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "relu_l1":  [1e-3, 4e-3, 1.6e-2, 6.4e-2, 2.56e-1],
}
COEF_ARG = {"jumprelu": "l0_coef", "gated": "l1_coef", "relu_l1": "l1_coef"}

# WHICH KNOB ACTUALLY SETS SPARSITY.
# For the magnitude architectures it is the penalty coefficient. For JumpReLU
# it is NOT: JumpReLUSAE.recalibrate re-seats theta on a schedule to hit
# `target_l0`, and the measured consequence (documented in that docstring, and
# reproduced in every overnight calibration trace) is that L0 barely responds
# to l0_coef at all --
#     gpt2/standard      L0 46.8 -> 44.9  across EIGHT decades of l0_coef
#     pythia-1.4b/std    L0 12.9 -> 12.1  across EIGHT decades
# -- so searching l0_coef for a target of 64 is searching a knob that is not
# connected to anything, and 5 of 6 jumprelu cells were dropped from the table
# as "calibration failed" when the real failure was aiming at the wrong screw.
# The threshold schedule IS the sparsity control, so calibrate `target_l0`.
SPARSITY_KNOB = {"jumprelu": "target_l0", "gated": "l1_coef",
                 "relu_l1": "l1_coef"}
# l0_coef is then held fixed for jumprelu; it is inert by construction here and
# the paper reports the row as "JumpReLU (scheduled threshold)".
JUMPRELU_L0_COEF = 3e-4
SCALE_FREE = ("topk", "batchtopk")      # k is exact; no calibration needed

PREDICTIONS = {
    "topk":      "inversion (support-only) — v2 baseline",
    "batchtopk": "inversion (support-only) — v2 App N",
    "jumprelu":  "PREDICT inversion (L0 is support-only)",
    "relu_l1":   "no inversion (magnitude penalty)",
    "gated":     "PREDICT no inversion (L1 on gate path)",
}


# -----------------------------------------------------------------------------
# L0 calibration
# -----------------------------------------------------------------------------

def _probe_batches(adir, steps, seed, condition, v1, device, batch_size=None):
    b = activation_batches(adir, batch_size or registry.BATCH_SIZE, steps,
                          seed, device=device)
    if condition == "bypass":
        b = io._BypassBatches(b, v1, device)
    return b


def probe_l0(spec, arch, adir, coef, condition, v1, device, steps):
    """Train a short probe and MEASURE its L0 at the final parameters.

    Deliberately does not use train_sae's log: with log_every=1000 the last
    logged step of a `steps`-step run is 1000*floor((steps-1)/1000), i.e. the
    log is stale by up to 999 steps (verified).
    """
    batches = _probe_batches(adir, steps, 123, condition, v1, device)
    try:
        sae, _ = train_sae(batches, spec.d_model, spec.d_sae, arch=arch,
                           seed=123, device=device, verbose=False,
                           tag=f"cal-{arch}-{condition}",
                           **{COEF_ARG[arch]: coef})
    except RuntimeError as e:
        # train_sae raises on a non-finite loss. A probe diverging must fail
        # THAT PROBE, not the whole B1 run -- otherwise one bad coefficient in
        # the grid kills an overnight queue. Divergence under a sparsity
        # penalty means the coefficient was too large (the code collapsed), so
        # report L0=0: that lands the point on the "too sparse" side and the
        # search correctly moves the coefficient DOWN.
        print(f"  [cal {arch}/{condition}] coef {coef:.3e} -> DIVERGED "
              f"({e}); recording L0=0", flush=True)
        torch.cuda.empty_cache()
        return 0.0
    l0s = []
    with torch.no_grad():
        for x in _probe_batches(adir, 4, 999, condition, v1, device):
            l0s.append(float((sae.encode(x) != 0).float().sum(-1).mean()))
    del sae
    torch.cuda.empty_cache()
    return float(np.mean(l0s))


def calibrate(spec, arch, adir, target, condition, v1, device,
              steps=4000, tol=0.12, max_widen=4, max_refine=4):
    """Bracket-then-secant search for the coefficient landing L0 ~= target.

    Returns (coef, rows, ok). ok=False means the target was never bracketed or
    never reached within tol -- the caller must NOT report this architecture as
    sparsity-matched.
    """
    rows = []

    def ev(c):
        l0 = probe_l0(spec, arch, adir, c, condition, v1, device, steps)
        rows.append((float(c), float(l0)))
        print(f"  [cal {arch}/{condition}] coef {c:.3e} -> L0 {l0:.1f} "
              f"(target {target})", flush=True)
        return l0

    pts = [(c, ev(c)) for c in CAL_GRID[arch]]
    if any(abs(l - target) <= tol * target for _, l in pts):
        c, l = min(pts, key=lambda t: abs(t[1] - target))
        return c, rows, True

    # sparsity is monotone in the coefficient: widen upward while still too
    # dense, downward while already too sparse.
    lo = [(c, l) for c, l in pts if l > target]      # too dense
    hi = [(c, l) for c, l in pts if l < target]      # too sparse
    for _ in range(max_widen):
        if lo and hi:
            break
        if not hi:                                   # every probe too dense
            c = max(c for c, _ in pts) * 10.0
        else:                                        # every probe too sparse
            c = min(c for c, _ in pts) / 10.0
        l = ev(c)
        pts.append((c, l))
        (hi if l < target else lo).append((c, l))
    if not (lo and hi):
        print(f"  !! CALIBRATION FAILED for {arch}/{condition}: target L0="
              f"{target} not bracketed by coef in "
              f"[{min(c for c,_ in pts):.1e}, {max(c for c,_ in pts):.1e}]; "
              f"L0 range observed [{min(l for _,l in pts):.1f}, "
              f"{max(l for _,l in pts):.1f}]", flush=True)
        return min(pts, key=lambda t: abs(t[1] - target))[0], rows, False

    a = max(lo, key=lambda t: t[0])                  # densest side, largest coef
    b = min(hi, key=lambda t: t[0])                  # sparsest side, smallest coef
    for _ in range(max_refine):
        # secant in (log coef, log L0)
        la, lb = np.log(a[0]), np.log(b[0])
        ya, yb = np.log(max(a[1], 1e-6)), np.log(max(b[1], 1e-6))
        t = np.log(target)
        # DEGENERATE-SECANT GUARD. When L0 barely responds to the coefficient
        # (measured for jumprelu at realistic activation scale: L0 moves 2.4x
        # across FIVE decades of l0_coef, because theta is frozen and the
        # sparsity is coming from the encoder), yb - ya ~ 0 and the secant step
        # is inf/nan. np.clip(nan) is nan, and a nan coefficient trains a nan
        # SAE whose measured "L0" is d_sae -- a silent, expensive garbage probe.
        if not np.isfinite(yb - ya) or abs(yb - ya) < 1e-6:
            print(f"  !! CALIBRATION DEGENERATE for {arch}/{condition}: L0 is "
                  f"insensitive to the coefficient over [{min(a[0],b[0]):.2e}, "
                  f"{max(a[0],b[0]):.2e}] (L0 {a[1]:.1f} -> {b[1]:.1f}). The "
                  f"coefficient is not the knob controlling sparsity here; for "
                  f"jumprelu this is the frozen-theta failure. Refusing to "
                  f"secant.", flush=True)
            break
        c = float(np.exp(la + (t - ya) * (lb - la) / (yb - ya)))
        if not np.isfinite(c):
            break
        c = float(np.clip(c, min(a[0], b[0]) * 1.05, max(a[0], b[0]) * 0.95))
        l = ev(c)
        if abs(l - target) <= tol * target:
            return c, rows, True
        if l > target:
            a = (c, l)
        else:
            b = (c, l)
    best = min(rows, key=lambda t: abs(t[1] - target))
    ok = abs(best[1] - target) <= tol * target
    if not ok:
        print(f"  !! CALIBRATION did not converge for {arch}/{condition}: "
              f"best coef {best[0]:.3e} -> L0 {best[1]:.1f} vs target {target}",
              flush=True)
    return best[0], rows, ok


def _elasticity(pts, near, target):
    """d log L0 / d log knob, estimated from points at ONE training length.

    Mixing lengths is what broke the old corrective step. The short probe runs
    4000 steps and the table runs 20000, and L0 keeps falling after step 4000
    under every magnitude penalty, so the two curves are different functions.
    The old code took its second secant point from whichever grid entry had the
    L0 closest to the full-length measurement -- i.e. it drew a line between a
    4k point and a 20k point and called the result a slope. Consequences, both
    from last night's traces:

      pythia-1.4b/gated/bypass: the nearest point WAS the same coefficient
        (short probe 4.8188 -> 70.1, full length 4.8188 -> 37.6). log-coef
        difference 0, so the step was exactly zero: the corrective pass
        retrained the identical (cached) SAE, measured the identical L0, and
        reported failure.
      gpt2/gated/bypass: the mixed-length slope came out with the WRONG SIGN
        and the step moved 9.16 -> 10.13, i.e. further from target (L0 27.3
        when 64 was wanted -- it needed LESS penalty, and got more).

    So: take both points from `pts` at the same length, require distinct knob
    values, and require a response worth extrapolating.
    """
    cand = [(c, l) for c, l in pts if l > 0 and np.isfinite(l)]
    if len(cand) < 2:
        return None
    cand.sort(key=lambda t: abs(np.log(max(t[0], 1e-30)) -
                                np.log(max(near, 1e-30))))
    a = cand[0]
    b = next((q for q in cand[1:]
              if abs(np.log(max(q[0], 1e-30)) -
                     np.log(max(a[0], 1e-30))) > 1e-3
              and abs(np.log(q[1]) - np.log(a[1])) > 1e-3), None)
    if b is None:
        return None
    sl = ((np.log(b[1]) - np.log(a[1])) /
          (np.log(b[0]) - np.log(a[0])))
    return float(sl) if np.isfinite(sl) and abs(sl) > 0.05 else None


def calibrate_full_length(spec, arch, adir, coef, target, condition, v1, args,
                          rows):
    """Correct the short-probe knob value at the REAL training length.

    The short probe is trained for --cal-steps (default 4000) but the table is
    trained for registry.STEPS (20000). L0 keeps falling after step 4000 under
    every magnitude/threshold penalty, so a probe-calibrated coefficient
    systematically overshoots -- and the naive guard only notices AFTER all
    five seeds of the cell have been trained and ablated. That is the expensive
    order of operations.

    So: train the REFERENCE SEED ONLY at full length, measure its L0 the same
    way run_cell does (io.alive rates, exact), and if it is outside tolerance,
    step the knob and retrain. The step uses a same-length slope:

      * two or more FULL-LENGTH points at distinct knob values -> secant on
        those (exactly the right slope, no extrapolation);
      * otherwise -> the local elasticity of the SHORT-PROBE curve, applied
        multiplicatively to the full-length L0. The two curves are offset but
        their log-log slopes agree closely, which is what makes this legitimate
        where the old mixed-length secant was not.

    For jumprelu the knob is `target_l0` and the map is near-proportional
    (achieved/requested was 0.20 on pythia-1.4b, 0.73 on gpt2, 1.11 on
    gemma-2-2b), so the fallback elasticity of 1.0 converges in one step.
    """
    short_pts = list(rows)
    full_pts = []
    knob = coef
    lo_k, hi_k = args.knob_min, args.knob_max
    for it in range(args.recal_passes + 1):
        knob = float(f"{knob:.6g}")        # cache-key stability (see report)
        akw = _akw(arch, knob)
        try:
            sae, v1_, mean, _adir, _e = io.get_sae(
                spec, registry.REFERENCE_SEED, arch=arch, condition=condition,
                device=args.device, **akw)
        except RuntimeError as e:
            # jumprelu refuses a target_l0 it cannot seat a threshold for; that
            # is a bound on the knob, not a reason to abandon the cell.
            print(f"  [full-length cal {arch}/{condition}] knob {knob:.4g} "
                  f"REFUSED ({e}); backing off", flush=True)
            torch.cuda.empty_cache()
            knob = knob / 2.0 if arch == "jumprelu" else knob * 2.0
            continue
        rates, _al = io.alive(sae, adir, condition=condition, v1=v1,
                              device=args.device)
        l0 = float(rates.sum())
        full_pts.append((knob, l0))
        del sae
        torch.cuda.empty_cache()
        print(f"  [full-length cal {arch}/{condition}] {SPARSITY_KNOB[arch]} "
              f"{knob:.4g} -> L0 {l0:.1f} at {registry.STEPS} steps "
              f"(target {target})", flush=True)
        if abs(l0 - target) <= args.l0_tol * target:
            return knob, short_pts + full_pts, True
        if it == args.recal_passes:
            break

        # FLATNESS GATE, before any fallback. Two full-length points at
        # distinct knob values with the same L0 mean this knob does not control
        # sparsity for this architecture, and no extrapolation will find a
        # value that works. Detect it on the SECOND training rather than
        # burning the whole recal budget: each of these is a full 20k-step run.
        if len(full_pts) >= 2:
            ks = [k for k, _ in full_pts]
            ls = [l for _, l in full_pts]
            if (max(ks) / max(min(ks), 1e-30) > 1.05
                    and max(ls) / max(min(ls), 1e-9) < 1.05):
                print(f"  !! [{arch}/{condition}] L0 is INERT to "
                      f"{SPARSITY_KNOB[arch]}: {min(ks):.4g}->{max(ks):.4g} "
                      f"moved L0 only {min(ls):.1f}->{max(ls):.1f}. This knob "
                      f"does not control sparsity here; target {target} is "
                      f"unreachable. Not spending more full-length runs.",
                      flush=True)
                break

        sl = _elasticity(full_pts, knob, target)          # same length: best
        src = "full-length secant"
        if sl is None:
            sl = _elasticity(short_pts, knob, target)     # same length: probe
            src = "short-probe elasticity"
        if sl is None and arch == "jumprelu":
            sl, src = 1.0, "jumprelu proportional fallback"
        if sl is None:
            print(f"  !! [{arch}/{condition}] no usable same-length slope; "
                  f"L0 does not respond to {SPARSITY_KNOB[arch]} over the "
                  f"probed range. Stopping rather than guessing.", flush=True)
            break
        step = (np.log(target) - np.log(max(l0, 1e-6))) / sl
        step = float(np.clip(step, -np.log(30.0), np.log(30.0)))
        nxt = float(np.exp(np.log(knob) + step))
        if not np.isfinite(nxt) or nxt <= 0:
            break
        nxt = float(np.clip(nxt, lo_k, hi_k))
        if abs(np.log(max(nxt, 1e-30)) - np.log(max(knob, 1e-30))) < 1e-4:
            print(f"  !! [{arch}/{condition}] corrective step is a no-op "
                  f"({knob:.4g} -> {nxt:.4g}); knob is at a bound or the slope "
                  f"is degenerate. Stopping.", flush=True)
            break
        print(f"     step via {src} (slope {sl:.3f}): "
              f"{knob:.4g} -> {nxt:.4g}", flush=True)
        knob = nxt
    best = min(full_pts, key=lambda t: abs(t[1] - target)) if full_pts else (knob, float("nan"))
    ok = bool(full_pts) and abs(best[1] - target) <= args.l0_tol * target
    if not ok:
        print(f"  !! FULL-LENGTH CALIBRATION FAILED for {arch}/{condition}: "
              f"best {SPARSITY_KNOB[arch]} {best[0]:.4g} lands L0 {best[1]:.1f}"
              f", target {target} +/-{100*args.l0_tol:.0f}%.", flush=True)
    return best[0], short_pts + full_pts, ok


def _akw(arch, knob):
    """Training kwargs for one value of the architecture's sparsity knob."""
    if arch == "jumprelu":
        # target_l0 is the knob (see SPARSITY_KNOB); l0_coef is held fixed.
        # bandwidth is training-relevant and MUST be in the SAE cache key, or
        # changing it silently returns the SAE trained with the old one.
        return {"l0_coef": JUMPRELU_L0_COEF, "bandwidth": 1e-3,
                "target_l0": float(knob)}
    return {COEF_ARG[arch]: knob}


# -----------------------------------------------------------------------------
# is the jumprelu row actually testing JumpReLU?
# -----------------------------------------------------------------------------

@torch.no_grad()
def _jumprelu_testability(sae, adir, condition, v1, device, n=32768):
    """Decide whether this SAE's sparsity comes from the LEARNED THRESHOLD or
    from the encoder having collapsed (in which case it is a ReLU SAE wearing a
    JumpReLU costume and the registered prediction is untestable).

    Reference protocol (Rajamanoharan et al. 2024 App. I; Gemma Scope App. A):
    theta is initialised to 1e-3 on activations RENORMALISED to E||x||^2 = 1.
    This suite trains on raw centered activations, so theta_init = e^-4 = 0.018
    is a fixed constant against a pre-activation scale it never sees. With
    theta log-parameterised and Adam, |d log theta| <= ~lr per step, so theta
    can move at most exp(lr * steps) = exp(3e-4 * 20000) = 403x over a full run
    and 3.3x over a 4000-step calibration probe. If theta* is further away than
    that, no coefficient reaches L0=64 by thresholding -- the L0 you measure is
    the encoder dying instead.

    Reports:
      gate_frac  fraction of POSITIVE pre-activations that theta kills.
                 ~0 => theta is inert, the row is a ReLU SAE.
      theta_over_pre_sd   theta relative to the pre-activation scale.
      theta_moved_x       theta / theta_init.
    """
    import numpy as np
    from v3.common.data import open_activations
    if not hasattr(sae, "log_theta"):
        return None
    acts, mu = open_activations(adir)
    n = min(n, acts.shape[0])
    x = torch.tensor(np.asarray(acts[:n], dtype=np.float32), device=device)
    x = x - torch.tensor(mu, dtype=torch.float32, device=device)
    if condition == "bypass":
        v1t = torch.tensor(v1, dtype=torch.float32, device=device)
        x = x - (x @ v1t)[:, None] * v1t[None, :]
    pre = torch.relu(sae.pre(x))
    theta = sae.log_theta.exp()
    pos = pre > 0
    n_pos = float(pos.sum())
    killed = float((pos & (pre <= theta)).sum())
    pre_sd = float(pre[pos].std()) if n_pos else float("nan")
    out = dict(
        gate_frac=(killed / n_pos) if n_pos else float("nan"),
        theta_median=float(theta.median()), theta_max=float(theta.max()),
        theta_moved_x=float(theta.median()) / float(np.exp(-4.0)),
        pre_positive_sd=pre_sd,
        theta_over_pre_sd=float(theta.median()) / max(pre_sd, 1e-12),
        l0_from_theta=float((pre > theta).float().sum(-1).mean()),
        l0_from_relu_only=float(pos.float().sum(-1).mean()),
    )
    out["testable"] = bool(out["gate_frac"] > 0.2)
    if not out["testable"]:
        print(f"  !! JUMPRELU ROW IS NOT TESTABLE: theta kills only "
              f"{100*out['gate_frac']:.2f}% of positive pre-activations "
              f"(theta={out['theta_median']:.4g} vs pre-activation sd "
              f"{pre_sd:.4g}; moved {out['theta_moved_x']:.2f}x from init). "
              f"L0 is set by the ReLU, not the threshold, so this row does NOT "
              f"test 'JumpReLU inherits the inversion'. Apply the theta-init "
              f"patch in reviews/b1_b2.md before running the headline table.",
              flush=True)
    del x, pre
    torch.cuda.empty_cache()
    return out


# -----------------------------------------------------------------------------
# one arch x condition cell
# -----------------------------------------------------------------------------

def run_cell(model, spec, seqs, adir, evecs, arch, akw, condition, args,
             am=None):
    decs, alives, l0_meas, l0_log = [], [], [], []
    imp_ref, fvu_ref = None, None
    theta_diag, n_screened = None, 0
    v1 = mean = None
    for s in _seeds(args):
        sae, v1, mean, adir, extra = io.get_sae(
            spec, s, arch=arch, condition=condition, device=args.device, **akw)
        rates, al = io.alive(sae, adir, condition=condition, v1=v1,
                             device=args.device)
        # rates.sum() is the exact expected number of active features per token
        # over the 100k alive sample in the correct condition. This is the L0
        # the sparsity-matching guard is allowed to trust.
        l0_meas.append(float(rates.sum()))
        decs.append(decoder_np(sae)); alives.append(al)
        if extra.get("log"):
            l0_log.append(extra["log"][-1].get("l0"))
            if s == registry.REFERENCE_SEED:
                fvu_ref = extra["log"][-1].get("fvu")
        if s == registry.REFERENCE_SEED:
            st = z_stats(model, spec, seqs, sae, mean, v1=v1, mode=condition,
                         chunk=args.chunk, am=am)
            er = st["rate"]             # padding excluded from the firing rate
            fires = (er > 0) & al
            if args.screen:
                # Variance screen (fastabl): rank by E[z^2], ablate only the
                # top-M. Licensed by gate0 phase V, which measures the screen's
                # recall of the true top-50 against an EXHAUSTIVE ablation on
                # this model. Do not raise --screen above what gate0 validated.
                cand = screen_candidates(st["ez2"], fires, args.screen)
            else:
                cand = np.nonzero(fires)[0]
            print(f"  [{arch}/{condition}] ablating {len(cand)} candidate "
                  f"features of {int(fires.sum())} firing+alive "
                  f"({'screened' if args.screen else 'exhaustive'}); "
                  f"one full model forward each", flush=True)
            def _prog(n, tot, _out, _base):
                print(f"    [zero] {n}/{tot}", flush=True)

            deltas, base = ablation_over(
                model, spec, seqs, sae, mean, cand.tolist(), v1=v1,
                mode=condition, metric="zero", chunk=args.chunk, stats=st,
                am=am, on_partial=(_prog if args.progress else None),
                partial_every=max(int(args.progress or 1), 1))
            # UNEVALUATED FEATURES MUST BE -inf, NOT 0. With a screen active,
            # zero-filling silently makes every unmeasured feature outrank every
            # feature with a negative dCE, so the "top-50" could contain
            # features we never ablated.
            imp_ref = to_array(sae.d_sae, deltas)
            n_screened = len(cand)
            if arch == "jumprelu":
                theta_diag = _jumprelu_testability(sae, adir, condition, v1,
                                                   args.device)

    dmask, cos2, _ = dense_cluster(decs[0], evecs[:, 0], alives[0])
    stats = pairwise_over_seeds(decs, alives, [imp_ref] * len(decs),
                                ref_seed_pos=0, k=args.topk)
    # imp_ref carries -inf for every feature the variance screen never
    # evaluated, so the old signed ratio summed -inf into BOTH ends and every
    # cell in this table published dense_importance_share = -Infinity.
    ds = dense_share(imp_ref, dmask, alives[0])
    dshare = ds["positive"]
    if ds.get("signed_suppressed"):
        print(f"  !! [{arch}/{condition}] {ds['signed_suppressed']}", flush=True)

    # PER-FEATURE GEOMETRY. The mechanism claim is that a magnitude objective
    # represents PC1 with ONE canonical atom (cos^2 -> 1, a unique solution, so
    # every seed finds it) while support-only sparsity tiles PC1 with several
    # atoms at cos^2 ~ 0.03-0.1 (a continuum, so no seed agrees). exp_e1 shows
    # cross-seed sharing collapses to ~0.01 in that tiling band and recovers to
    # 0.75-1.00 for canonical atoms -- but only for the TopK cells, because only
    # exp_a5 saved per-atom arrays. Saving them here extends that test to the
    # magnitude architectures, which is where the prediction actually bites.
    _mc = np.stack([max_cos(decs[0], decs[j]) for j in range(1, len(decs))]) \
        if len(decs) > 1 else np.zeros((1, len(decs[0])), dtype=np.float32)
    np.savez(io.path("results", EXP,
                     f"feat_{spec.key}_{arch}_{condition}.npz"),
             pc1_cos2=cos2, max_cos=_mc.astype(np.float32),
             alive=alives[0], dense=dmask, imp_zero=imp_ref)

    # ---- attribution diagnostics: is a HIGH top-50 sharing the law, or just a
    # ---- reshuffled / collapsed importance ranking?
    top = np.asarray(stats["pairs"][0]["top_idx"])
    shared_by_pair = np.asarray([p["top_shared"] for p in stats["pairs"]],
                                dtype=float)          # [n_pairs, k]
    top_is_dense = dmask[top]
    def _m(sel):
        # None, not nan: a cell whose top-50 is 100% dense has NO content atoms
        # to average, and json.dump writes bare NaN, which is not valid JSON.
        return (float(shared_by_pair[:, sel].mean()) if sel.any() else None)
    imp_alive = imp_ref[alives[0] & np.isfinite(imp_ref)]
    diag = dict(
        top50_dense_frac=float(top_is_dense.mean()),
        top50_shared_dense=_m(top_is_dense),
        top50_shared_content=_m(~top_is_dense),
        top50_dce_max=float(imp_ref[top].max()),
        top50_dce_min=float(imp_ref[top].min()),
        dce_p99_over_median=float(
            np.percentile(imp_alive, 99) /
            max(abs(np.median(imp_alive)), 1e-12)) if imp_alive.size else None,
        n_cand=int(np.isfinite(imp_ref).sum()), n_screened=int(n_screened),
        screen=int(args.screen), n_alive=int(alives[0].sum()),
        jumprelu_testability=theta_diag,
    )

    lo, hi = args.target_l0 * (1 - args.l0_tol), args.target_l0 * (1 + args.l0_tol)
    matched = bool(all(lo <= v <= hi for v in l0_meas))
    out = dict(
        top50_mean=stats["top_k_mean"], top50_std=stats["top_k_std"],
        dictwide_mean=stats["dict_wide_mean"],
        dictwide_std=stats["dict_wide_std"],
        dense_importance_share=ds["positive"],
        dense_importance_share_signed=ds["signed"],
        dense_importance_denominator=ds["denominator_signed"],
        dense_importance_n_evaluated=ds["n_evaluated"],
        l0_measured=l0_meas, l0_measured_mean=float(np.mean(l0_meas)),
        l0_train_log_stale=l0_log, fvu=fvu_ref,
        l0_matched=matched, **diag)
    flag = "" if matched else "   <-- L0 NOT MATCHED, cell is not comparable"
    print(f"  [{arch}/{condition}] top-50 "
          f"{100*stats['top_k_mean']:.1f} +/- {100*stats['top_k_std']:.1f} | "
          f"dense share {dshare:.2f} | L0 {np.mean(l0_meas):.1f} | "
          f"top50 dense frac {diag['top50_dense_frac']:.2f} | "
          f"fvu {fvu_ref}{flag}", flush=True)
    return out


def _seeds(args):
    """Seed set for this run, reference seed first.

    pairwise_over_seeds(ref_seed_pos=0) reads position 0 as the reference, so
    the ordering is load-bearing, not cosmetic.
    """
    ss = list(getattr(args, "seeds", None) or registry.SEEDS_FULL)
    if registry.REFERENCE_SEED not in ss:
        raise SystemExit(
            f"--seeds must include the reference seed "
            f"{registry.REFERENCE_SEED}; got {ss}")
    return [registry.REFERENCE_SEED] + [x for x in ss
                                        if x != registry.REFERENCE_SEED]


def run_model(spec, args):
    assert _seeds(args)[0] == registry.REFERENCE_SEED, (
        "pairwise_over_seeds(ref_seed_pos=0) assumes position 0 is the "
        "reference seed; _seeds() must guarantee that")
    model, tok = load_model_and_tokenizer(spec, device=args.device)
    seqs, am = eval_sequences(tok, n_seqs=args.eval_seqs)
    if not args.chunk:
        # a hardcoded 64 is ~8 GB of fp32 logits on pythia (vocab 50k) but
        # ~25 GB on gemma-2-9b (vocab 256k), on top of 18.5 GB of weights.
        args.chunk = tuned_chunk(spec, vocab_size=int(model.config.vocab_size))
    adir = io.ensure_acts(spec, device=args.device)
    evals, evecs, _ = estimate_pcs(adir)
    v1 = evecs[:, 0]
    pad = 1.0 - float(am.float().mean())
    print(f"== {spec.key}: PC1 var explained {evals[0]/evals.sum():.4f} | "
          f"chunk={args.chunk} | eval pad_frac={pad:.3f} ==", flush=True)

    # MERGE, DO NOT OVERWRITE.
    # Two things made a plain overwrite destructive. (1) A relaunched box
    # replays from cache and rewrites this file, so a box holding 2 of 10 cells
    # clobbered a finished 10-cell table -- that is how GPT-2 and Gemma-2-2B
    # went from complete to 8/10 and 4/10 with no error anywhere. (2) A targeted
    # re-run of one architecture (`--archs gated`) would otherwise publish a
    # two-cell table over an eight-cell one. Seeding from what is already on
    # disk, and skipping cells that are already filled unless --force, makes
    # this script resumable and makes single-architecture top-ups safe.
    prev = io.read_json(EXP, spec.key) or {}
    prev_meta = prev.get("_meta", {})
    if prev and prev_meta.get("target_l0") not in (None, args.target_l0):
        print(f"!! existing {spec.key}.json was built at target_l0="
              f"{prev_meta.get('target_l0')} != {args.target_l0}; NOT merging, "
              f"starting fresh", flush=True)
        prev = {}
    table = dict(_meta=dict(target_l0=args.target_l0, l0_tol=args.l0_tol,
                            eval_seqs=args.eval_seqs, chunk=args.chunk,
                            pc1_var_explained=float(evals[0] / evals.sum())))
    for a, v in prev.items():
        if a != "_meta" and isinstance(v, dict):
            table[a] = v
    kept = [(a, c) for a in table if a != "_meta"
            for c in ("standard", "bypass")
            if isinstance(table[a].get(c), dict)
            and "top50_mean" in table[a][c]]
    if kept:
        print(f"   merging with {len(kept)} cell(s) already on disk: "
              f"{', '.join(f'{a}/{c}' for a, c in kept)}", flush=True)

    for arch in args.archs:
        table.setdefault(arch, {})["prediction"] = PREDICTIONS.get(arch, "")
        for condition in ["standard", "bypass"]:
            done_cell = table[arch].get(condition)
            if (not args.force and isinstance(done_cell, dict)
                    and "top50_mean" in done_cell):
                print(f"== SKIP {spec.key}/{arch}/{condition}: already on disk "
                      f"(top-50 {100*done_cell['top50_mean']:.1f}). "
                      f"--force to recompute. ==", flush=True)
                continue
            if arch in SCALE_FREE:
                akw, cal_ok = dict(k=registry.TOPK_K), True
            else:
                if SPARSITY_KNOB[arch] == "target_l0":
                    # The short-probe grid sweeps l0_coef, which is inert for
                    # this architecture (see SPARSITY_KNOB). Running it costs
                    # five 4000-step trainings and returns a flat line, so skip
                    # straight to full-length calibration of the real knob,
                    # seeded at the target itself (the map is ~proportional).
                    coef, cal_rows = float(args.target_l0), []
                    cal_ok = False
                else:
                    coef, cal_rows, cal_ok = calibrate(
                        spec, arch, adir, args.target_l0, condition, v1,
                        args.device, steps=args.cal_steps, tol=args.l0_tol)
                if args.recal_passes:
                    # ALWAYS run the full-length pass, not only when the short
                    # probe converged. A short probe that missed still carries
                    # a usable elasticity, and the cells that failed last night
                    # failed at full length regardless of the probe verdict --
                    # gating on cal_ok just threw away the recovery attempt.
                    coef, cal_rows, cal_ok = calibrate_full_length(
                        spec, arch, adir, coef, args.target_l0, condition, v1,
                        args, cal_rows)
                coef = float(f"{coef:.6g}")
                akw = _akw(arch, coef)
                io.write_json(EXP, f"cal_{spec.key}_{arch}_{condition}",
                              dict(grid=cal_rows, chosen=coef, ok=cal_ok,
                                   knob=SPARSITY_KNOB[arch],
                                   target_l0=args.target_l0,
                                   cal_steps=args.cal_steps,
                                   full_steps=registry.STEPS,
                                   recal_passes=args.recal_passes))
            table[arch][f"{condition}_kwargs"] = akw
            table[arch][f"{condition}_calibrated"] = cal_ok
            if not cal_ok and not args.allow_uncalibrated:
                print(f"== SKIP {spec.key}/{arch}/{condition}: calibration "
                      f"failed; the L0-matched comparison this row exists to "
                      f"support is not available. Re-run with a wider "
                      f"CAL_GRID, more --cal-steps, or "
                      f"--allow-uncalibrated to record it anyway. ==",
                      flush=True)
                table[arch][condition] = dict(skipped="calibration_failed")
                continue
            print(f"== {spec.key} / {arch} / {condition} "
                  f"({PREDICTIONS.get(arch,'')}) kwargs={akw} ==", flush=True)
            table[arch][condition] = run_cell(model, spec, seqs, adir, evecs,
                                              arch, akw, condition, args,
                                              am=am)
            # CHECKPOINT AFTER EVERY CELL. The table used to be written once,
            # after all five architectures finished, so a per-model timeout
            # threw away every completed cell -- and on a 7B model a single
            # architecture is hours. `_meta.complete` marks whether the
            # table-level sparsity guard has run yet, so a partial file can
            # never be mistaken for a finished grid.
            table["_meta"]["complete"] = False
            io.write_json(EXP, spec.key, table)
    # table-level guard. Runs over EVERY architecture in the merged table, not
    # just the ones this invocation touched -- otherwise a `--archs gated`
    # top-up would stamp `publishable` after inspecting one row and ignore the
    # eight cells it merged in.
    graded = [a for a in table if a != "_meta" and isinstance(table[a], dict)]
    cells = [(a, c, table[a][c]) for a in graded for c in
             ("standard", "bypass")
             if isinstance(table[a].get(c), dict) and "l0_matched" in table[a][c]]
    bad = [(a, c, round(v["l0_measured_mean"], 1))
           for a, c, v in cells if not v["l0_matched"]]
    untestable = [(a, c) for a, c, v in cells
                  if isinstance(v.get("jumprelu_testability"), dict)
                  and not v["jumprelu_testability"]["testable"]]
    table["_meta"]["sparsity_matched"] = not bad
    table["_meta"]["unmatched_cells"] = bad
    table["_meta"]["untestable_cells"] = untestable
    table["_meta"]["publishable"] = not bad and not untestable
    if bad:
        print("\n!! SPARSITY NOT MATCHED across the table. Unmatched cells "
              f"(arch, condition, measured L0) = {bad}\n"
              "!! An architecture difference in top-50 sharing at unmatched L0 "
              "is confounded with an L0 difference. DO NOT publish this grid "
              "as the architecture law.\n", flush=True)
    else:
        print(f"\n[ok] all {len(cells)} cells within "
              f"{100*args.l0_tol:.0f}% of L0={args.target_l0}\n", flush=True)
    if untestable:
        print(f"!! ARCHITECTURE NOT EXERCISED in cells {untestable}: the "
              f"JumpReLU threshold is inert, so that row tests a ReLU SAE and "
              f"cannot confirm or refute the registered prediction.\n",
              flush=True)
    table["_meta"]["complete"] = True
    io.write_json(EXP, spec.key, table)
    del model
    torch.cuda.empty_cache()
    return table["_meta"]["publishable"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=["pythia-1.4b"])
    p.add_argument("--archs", nargs="*",
                   default=["topk", "batchtopk", "relu_l1", "jumprelu", "gated"])
    p.add_argument("--eval-seqs", type=int, default=registry.EVAL_SEQS_IMPORTANCE)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--chunk", type=int, default=0,
                   help="eval sequences per forward in the latent/ablation "
                        "passes. 0 (default) = fastabl.tuned_chunk, which "
                        "sizes the chunk to the model's vocab so a 256k-vocab "
                        "model does not OOM on the fp32 logits copy.")
    p.add_argument("--screen", type=int, default=1500,
                   help="ablate only the top-N alive features by E[z^2] "
                        "(fastabl variance screen). 0 = exhaustive. Must not "
                        "exceed the M that gate0 phase V validated at recall "
                        "1.0 for the true top-50 on this model.")
    p.add_argument("--knob-min", type=float, default=1e-6,
                   help="lower bound on the sparsity knob during full-length "
                        "calibration (l1/l0 coefficient, or target_l0)")
    p.add_argument("--knob-max", type=float, default=1e5,
                   help="upper bound on the sparsity knob")
    p.add_argument("--recal-passes", type=int, default=3,
                   help="corrective calibration passes at the FULL training "
                        "length (reference seed only) after the short probe. "
                        "0 disables; the short probe alone systematically "
                        "overshoots L0 because sparsity keeps rising after "
                        "step --cal-steps.")
    p.add_argument("--target-l0", type=float, default=float(registry.TOPK_K))
    p.add_argument("--l0-tol", type=float, default=0.25,
                   help="fractional tolerance for the sparsity-matching guard")
    p.add_argument("--cal-steps", type=int, default=4000,
                   help="steps per calibration probe. NOTE for jumprelu: theta "
                        "is log-parameterised and Adam moves log_theta by at "
                        "most ~lr per step, so a probe of N steps can only "
                        "move theta by exp(lr*N) (2000 steps -> 1.7x, 4000 -> "
                        "2.9x). If theta_init is orders of magnitude off the "
                        "pre-activation scale, NO probe length fixes it -- see "
                        "the saes.py patch for a data-dependent theta init.")
    p.add_argument("--force", action="store_true",
                   help="recompute cells that are already present in the "
                        "results JSON. Without it, a re-run MERGES with what is "
                        "on disk and skips filled cells, which is what makes a "
                        "targeted `--archs gated` top-up safe.")
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="override the seed set (default registry.SEEDS_FULL, "
                        "5 seeds). B1 trains archs x conditions x seeds SAEs, "
                        "so 5 archs x 2 x 5 = 50 trainings -- at ~32 min each "
                        "on a 7B model that is 27 h, past any sane watchdog. "
                        "Must include the reference seed 42, which "
                        "pairwise_over_seeds takes as position 0.")
    p.add_argument("--allow-uncalibrated", action="store_true",
                   help="record cells whose L0 calibration failed instead of "
                        "skipping them (they are still flagged in the JSON)")
    p.add_argument("--progress", type=int, default=500,
                   help="print ablation progress every N features")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    ok = True
    for mkey in args.models:
        ok = bool(run_model(registry.MODELS[mkey], args)) and ok
    # Non-zero exit so a driver script cannot consume an unmatched/untestable
    # architecture table as if it were the law.
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
