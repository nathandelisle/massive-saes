#!/usr/bin/env python
"""Sweep the magnitude penalty for Pythia-1.4B SAEs."""

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from v3.common import registry, io_utils as io                      # noqa: E402
from v3.common.data import load_model_and_tokenizer, eval_sequences  # noqa: E402
from v3.common.data import activation_batches                        # noqa: E402
from v3.common.fastabl import (ablation_over, screen_candidates,      # noqa: E402
                               to_array, tuned_chunk, z_stats)
from v3.common.matching import sharing_stats                          # noqa: E402
from v3.common.pca import estimate_pcs, dense_cluster                 # noqa: E402
from v3.common.fastabl import dense_share as dense_share_stat        # noqa: E402
from v3.common.matching import max_cos                               # noqa: E402
from v3.common.saes import decoder_np                                 # noqa: E402

EXP = "b2_l1_phase"

# Extended two decades below the original floor of 1e-4: an exactly-flat
# degeneracy is broken by an infinitesimal penalty, so beta* may be tiny and the
# transition must be bracketed from BELOW or the "phase diagram" is one step.
#
# ...and extended THREE decades ABOVE the old ceiling of 3e-2, which is the
# more likely failure. The old grid was anchored to relu_l1's calibrated lambda
# (~1e-3-1e-2), but that is the WRONG reference: lambda only has to exceed the
# marginal benefit of the ~8128th-best feature to kill it, whereas under a hard
# TopK mask the only features beta ever acts on are the surviving top-64, whose
# marginal benefit 2|d_i . residual| is orders of magnitude larger. A beta that
# sparsifies ReLU+L1 can be a no-op under TopK. `l1_over_mse` (below) is the
# dimensionless knob position and is what the phase diagram should be read
# against; if it never reaches ~1e-2 the knob was never turned and a flat
# diagram falsifies nothing.
BETAS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0,
         10.0]
# LAMBDAS: measured, not assumed. The comment above says the old grid was
# "anchored to relu_l1's calibrated lambda (~1e-3-1e-2)" -- but that anchor was
# itself wrong. B1's own calibration on this model landed relu_l1 at
# lambda* = 3.19 for L0 = 56, and its probe grid records lambda = 1.6e-2 giving
# L0 = 3680. So the entire old sweep sat in the no-sparsity-at-all regime:
# every lambda row came back with L0 ~ 3900-3980 of d_sae 8192 and cross-seed
# sharing of exactly 0.000, i.e. nothing to compare. The grid now brackets the
# calibrated value across the transition (L0 ~ 2700 down to ~0.5).
LAMBDAS = [0.1, 0.32, 1.0, 3.2, 10.0]
PAIR = [42, 137]


@torch.no_grad()
def l1_pressure(sae, adir, beta, device, n_batches=4, batch_size=4096):
    """Dimensionless position of the B2 knob.

    The trainer's MSE is SUMMED over d_model, so it is O(1e3-1e6) on raw
    centered residual activations, while the penalty is beta * ||z||_1 over the
    k=64 survivors. `l1_over_mse` is the ratio of the two loss TERMS at the
    trained optimum -- the only scale-free way to say whether beta did anything.
    Without it, a flat phase diagram is uninterpretable: it cannot distinguish
    "the magnitude penalty does not break the tiling symmetry" (which would
    falsify the law) from "beta was numerically negligible" (which says
    nothing). Also returns E[||z||_1] and the mean surviving magnitude so the
    right beta range can be read straight off a beta=0 run.
    """
    import torch.nn.functional as F
    mses, l1s, l0s = [], [], []
    for x in activation_batches(adir, batch_size, n_batches, 777, device=device):
        recon, z = sae(x)
        mses.append(float(F.mse_loss(recon, x, reduction="none").sum(-1).mean()))
        l1s.append(float(z.abs().sum(-1).mean()))
        l0s.append(float((z != 0).float().sum(-1).mean()))
    mse, l1, l0 = float(np.mean(mses)), float(np.mean(l1s)), float(np.mean(l0s))
    return dict(mse_sum=mse, mean_l1=l1, l0_batch=l0,
                mean_abs_z=l1 / max(l0, 1e-9),
                l1_term=beta * l1,
                l1_over_mse=(beta * l1 / mse) if mse > 0 else float("nan"))


def _point_tag(arch, akw):
    """Filesystem-safe identifier for one sweep point (architecture + knob)."""
    knob = akw.get("beta", akw.get("l1_coef", akw.get("l0_coef", 0.0)))
    return f"{arch}_{float(knob):.6g}".replace(".", "p").replace("-", "m")


def eval_point(model, spec, seqs, adir, evecs, arch, akw, args, am=None):
    decs, alives, l0_meas = [], [], []
    imp = None
    v1 = mean = None
    dense_share = dense_share_signed = fvu = l0_log = None
    dmask = None
    press = dict(mse_sum=float("nan"), mean_l1=float("nan"),
                 l0_batch=float("nan"), mean_abs_z=float("nan"),
                 l1_term=float("nan"), l1_over_mse=float("nan"))
    for s in PAIR:
        sae, v1, mean, adir, extra = io.get_sae(spec, s, arch=arch,
                                                condition="standard",
                                                device=args.device, **akw)
        rates, al = io.alive(sae, adir, device=args.device)
        # exact expected active features per token over the 100k alive sample;
        # the training log is stale by up to log_every-1 steps (for a 20k-step
        # run, log[-1] is step 19000), so it is recorded but not trusted.
        l0_meas.append(float(rates.sum()))
        decs.append(decoder_np(sae)); alives.append(al)
        if s == PAIR[0]:
            if extra.get("log"):
                fvu = extra["log"][-1].get("fvu")
                l0_log = extra["log"][-1].get("l0")
            stz = z_stats(model, spec, seqs, sae, mean, v1=v1,
                          mode="standard", chunk=args.chunk, am=am)
            er = stz["rate"]            # padding excluded from the firing rate
            fires = (er > 0) & al
            cand = (screen_candidates(stz["ez2"], fires, args.screen)
                    if args.screen else np.nonzero(fires)[0])
            print(f"    ablating {len(cand)} of {int(fires.sum())} firing+alive "
                  f"features ({'screened' if args.screen else 'exhaustive'})",
                  flush=True)

            def _prog(n, tot, _out, _base):
                print(f"    [zero] {n}/{tot}", flush=True)

            deltas, base = ablation_over(model, spec, seqs, sae, mean,
                                         cand.tolist(), v1=v1, mode="standard",
                                         metric="zero", chunk=args.chunk,
                                         stats=stz, am=am,
                                         on_partial=(_prog if args.progress
                                                     else None),
                                         partial_every=max(int(args.progress
                                                               or 1), 1))
            # -inf, not 0, for unevaluated features: with the screen on,
            # zero-filling lets a feature we never ablated outrank a measured
            # one whose dCE is negative.
            imp = to_array(sae.d_sae, deltas)
            dmask, _, _ = dense_cluster(decs[0], evecs[:, 0], al)
            # fastabl.dense_share, not a bare ratio: `max(denom, 1e-12)` does
            # not guard a NEGATIVE denominator, and signed dCE sums cancel
            # through zero (this exact expression published -6.2e10 in a5 and
            # -5.3e11 in b3). `positive` is the estimator to quote.
            _ds = dense_share_stat(imp, dmask, al)
            dense_share = _ds["positive"]
            dense_share_signed = _ds["signed"]
            press = l1_pressure(sae, adir, akw.get("beta", akw.get("l1_coef", 0.0)),
                                args.device)
    st = sharing_stats(decs[0], decs[1], alives[0], alives[1], imp, k=args.topk)

    # ---- PER-FEATURE GEOMETRY, saved so the phase sweep can be joined to the
    # ---- E1 two-regime picture. The whole mechanism claim is that beta works
    # ---- BY moving PC1 out of the tiling band (many atoms at cos^2 ~ 0.03-0.1,
    # ---- cross-seed sharing ~0) and into the canonical regime (one atom at
    # ---- cos^2 -> 1, perfectly shared). Testing that needs per-atom cos^2 and
    # ---- cross-seed max-cosine at every beta, not just the summary row --
    # ---- without this the sweep can show sharing recovering but cannot say
    # ---- through what geometry.
    cos_pc1 = decs[0] @ evecs[:, 0]
    mc = max_cos(decs[0], decs[1])
    tag = _point_tag(arch, akw)
    np.savez(io.path("results", EXP, f"feat_{spec.key}_{tag}.npz"),
             pc1_cos2=cos_pc1 ** 2, max_cos=mc.astype(np.float32),
             alive=alives[0], dense=dmask, imp_zero=imp)

    # ---- attribution diagnostics (see module docstring, threats (1) and (2))
    top = np.asarray(st["top_idx"])
    shared = np.asarray(st["top_shared"], dtype=float)
    top_is_dense = dmask[top]
    imp_alive = imp[alives[0] & np.isfinite(imp)]
    # E1 band occupancy at THIS beta: the mechanism variable
    _al = alives[0]
    _c2 = cos_pc1 ** 2
    _sh = (mc >= registry.SHARED_COS_THRESHOLD).astype(float)
    _band = _al & (_c2 >= 0.01) & (_c2 < 0.3)
    _canon = _al & (_c2 >= 0.3)
    return dict(
        top50=st["top_k"], dictwide=st["dict_wide"],
        dense_importance_share=dense_share,
        dense_importance_share_signed=dense_share_signed,
        tiling_band_n=int(_band.sum()),
        tiling_band_sharing=(float(_sh[_band].mean()) if _band.any() else None),
        canonical_n=int(_canon.sum()),
        canonical_sharing=(float(_sh[_canon].mean()) if _canon.any() else None),
        max_cos2_pc1=float(_c2[_al].max()) if _al.any() else None,
        top10_pc1_mass=float(np.sort(_c2[_al])[::-1][:10].sum()) if _al.any() else None,
        fvu=fvu, **press,
        l0_measured=l0_meas, l0_measured_mean=float(np.mean(l0_meas)),
        l0_train_log_stale=l0_log,
        top50_dense_frac=float(top_is_dense.mean()),
        top50_shared_dense=(float(shared[top_is_dense].mean())
                            if top_is_dense.any() else float("nan")),
        top50_shared_content=(float(shared[~top_is_dense].mean())
                              if (~top_is_dense).any() else float("nan")),
        top50_dce_max=float(imp[top].max()), top50_dce_min=float(imp[top].min()),
        dce_p99=float(np.percentile(imp_alive, 99)) if imp_alive.size else None,
        base_ce=float(base), n_cand=int(len(cand)), screen=int(args.screen),
        n_dense=int(dmask.sum()), n_alive=int(alives[0].sum()),
    )


def main():
    p = argparse.ArgumentParser()
    # `--models` is the harness contract: scripts/run_models.sh invokes every
    # experiment as `--models <m>`. This script only took `--model`, so the
    # launcher's call died with argparse exit code 2 -- and run_models.sh moved
    # on printing "ALL MODELS DONE", while the box synced back the results it
    # had RESTORED from another prefix. The run looked complete and was a no-op.
    # Accept both spellings so the harness can never miss this script again.
    p.add_argument("--model", default="pythia-1.4b")
    p.add_argument("--models", nargs="*", default=None,
                   help="alias for --model (harness compatibility); only the "
                        "first entry is used, this experiment is single-model")
    p.add_argument("--betas", nargs="*", type=float, default=BETAS)
    p.add_argument("--lambdas", nargs="*", type=float, default=LAMBDAS)
    p.add_argument("--eval-seqs", type=int, default=registry.EVAL_SEQS_IMPORTANCE)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--chunk", type=int, default=0,
                   help="eval sequences per forward in the ablation loop; "
                        "0 = fastabl.tuned_chunk (vocab-aware, avoids OOM on "
                        "big-vocab models)")
    p.add_argument("--screen", type=int, default=1500,
                   help="ablate only the top-N alive features by E[z^2] "
                        "(fastabl variance screen); 0 = exhaustive. Must not "
                        "exceed the M gate0 phase V validated at recall 1.0.")
    p.add_argument("--pressure-guard", type=float, default=1e-2,
                   help="the largest beta must reach this L1-term/MSE-term "
                        "ratio, or the knob was never turned and a flat phase "
                        "diagram falsifies nothing")
    p.add_argument("--fvu-guard", type=float, default=1.5,
                   help="flag any beta whose FVU exceeds this multiple of the "
                        "beta=0 FVU: the code has been destroyed, not re-tiled")
    p.add_argument("--l0-guard", type=float, default=0.12,
                   help="flag any point whose measured L0 drifts more than "
                        "this fraction from k (TopK yields L0<k when fewer "
                        "than k pre-activations are positive, which happens "
                        "once beta is large enough to collapse the encoder)")
    p.add_argument("--assert-topk-baseline", action="store_true",
                   help="require the beta=0 point to reproduce exp_a5's topk "
                        "top-50 sharing (they are bit-identical SAEs)")
    p.add_argument("--progress", type=int, default=500)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.models:
        if len(args.models) > 1:
            print(f"!! B2 is single-model; --models got {args.models}, using "
                  f"{args.models[0]} and IGNORING the rest", flush=True)
        args.model = args.models[0]
    spec = registry.MODELS[args.model]
    model, tok = load_model_and_tokenizer(spec, device=args.device)
    seqs, am = eval_sequences(tok, n_seqs=args.eval_seqs)
    if not args.chunk:
        args.chunk = tuned_chunk(spec, vocab_size=int(model.config.vocab_size))
    adir = io.ensure_acts(spec, device=args.device)
    evals, evecs, _ = estimate_pcs(adir)
    print(f"[b2] chunk={args.chunk} screen={args.screen} "
          f"eval pad_frac={1.0 - float(am.float().mean()):.3f}", flush=True)

    rows = []
    for beta in args.betas:
        akw = dict(k=registry.TOPK_K, beta=beta)
        print(f"[topk+L1 beta={beta:.0e}]", flush=True)
        r = eval_point(model, spec, seqs, adir, evecs, "topk_l1", akw, args,
                       am=am)
        r.update(sweep="topk_l1", coef=beta)
        rows.append(r)
        print(f"  -> top50 {100*r['top50']:.0f}% dense_share "
              f"{r['dense_importance_share']:.2f} top50_dense_frac "
              f"{r['top50_dense_frac']:.2f} L0 {r['l0_measured_mean']:.1f} "
              f"fvu {r['fvu']} max_dCE {r['top50_dce_max']:.4f} "
              f"L1/MSE {r['l1_over_mse']:.2e} |z| {r['mean_abs_z']:.3g}",
              flush=True)
    for lam in args.lambdas:
        print(f"[relu+L1 lam={lam:.0e}]", flush=True)
        r = eval_point(model, spec, seqs, adir, evecs, "relu_l1",
                       dict(l1_coef=lam), args, am=am)
        r.update(sweep="relu_l1", coef=lam)
        rows.append(r)
        print(f"  -> top50 {100*r['top50']:.0f}% L0 {r['l0_measured_mean']:.1f} "
              f"fvu {r['fvu']}", flush=True)

    # ---------------- guards: is the phase transition interpretable? ----------
    tl1 = [r for r in rows if r["sweep"] == "topk_l1"]
    base = next((r for r in tl1 if r["coef"] == 0.0), None)
    warn = []
    if base is not None and base["fvu"]:
        for r in tl1:
            if r["fvu"] and r["fvu"] > args.fvu_guard * base["fvu"]:
                r["degenerate_fvu"] = True
                warn.append(f"beta={r['coef']:.0e}: FVU {r['fvu']:.3f} = "
                            f"{r['fvu']/base['fvu']:.2f}x the beta=0 FVU")
    for r in rows:
        k = registry.TOPK_K if r["sweep"] == "topk_l1" else None
        if k and abs(r["l0_measured_mean"] - k) > args.l0_guard * k:
            r["l0_drifted"] = True
            warn.append(f"beta={r['coef']:.0e}: measured L0 "
                        f"{r['l0_measured_mean']:.1f} != k={k} (the support is "
                        f"no longer held fixed, so this is not a pure "
                        f"magnitude knob)")
    if base is not None:
        for r in tl1:
            if (r["coef"] > 0 and base["top50_dce_max"] > 0 and
                    r["top50_dce_max"] < 0.1 * base["top50_dce_max"]):
                r["degenerate_dce"] = True
                warn.append(f"beta={r['coef']:.0e}: max top-50 dCE "
                            f"{r['top50_dce_max']:.4g} is <10% of the beta=0 "
                            f"value -- the importance ranking is noise and "
                            f"'sharing' is vacuous")
            if (r["coef"] > 0 and base["top50_dense_frac"] > 0.2 and
                    r["top50_dense_frac"] < 0.2 * base["top50_dense_frac"]):
                r["ranking_moved_off_dense"] = True
                warn.append(f"beta={r['coef']:.0e}: top-50 dense fraction fell "
                            f"{base['top50_dense_frac']:.2f} -> "
                            f"{r['top50_dense_frac']:.2f} -- higher sharing may "
                            f"be a RANKING change (top-50 became content "
                            f"features), not a broken tiling symmetry. Compare "
                            f"top50_shared_dense across beta before claiming "
                            f"the law.")
    # ---- KNOB-ENGAGEMENT GUARD. This is the one that decides whether a NULL
    # ---- result is evidence. Every other guard here explains away a RISE in
    # ---- sharing; this one explains away the ABSENCE of one.
    press = [r["l1_over_mse"] for r in tl1
             if r["coef"] > 0 and np.isfinite(r.get("l1_over_mse", np.nan))]
    max_press = max(press) if press else 0.0
    engaged = max_press >= args.pressure_guard
    if not engaged:
        moved = base is not None and any(
            r["coef"] > 0 and abs(r["top50"] - base["top50"]) > 0.05
            for r in tl1)
        warn.append(
            f"KNOB NEVER ENGAGED: the largest beta reaches an L1/MSE loss-term "
            f"ratio of only {max_press:.2e} (< --pressure-guard "
            f"{args.pressure_guard:.0e}). The magnitude penalty is numerically "
            f"negligible against a d_model-SUMMED MSE, so a flat phase diagram "
            f"here is NOT evidence against the law -- it is evidence the "
            f"experiment did not run. Extend --betas upward by "
            f"{max(1, int(np.ceil(np.log10(args.pressure_guard / max(max_press, 1e-30)))))} "
            f"decades and re-run before reporting anything." +
            ("" if not moved else " (Sharing did move, so the low-pressure "
             "regime may still be informative -- report the ratio, not beta.)"))
    # DEGENERACY GUARD for the relu_l1 arm. This arm exists to be compared
    # against the TopK+beta arm at matched sparsity; if its lambda grid never
    # brings L0 anywhere near the TopK target, every row is a dictionary with
    # half its atoms active and cross-seed sharing collapses to 0.000 -- which
    # is not a data point, it is an unrun experiment. The first run had exactly
    # this shape (L0 3827-3981 against a target of 64) and it read as a result.
    l1rows = [r for r in rows if r.get("sweep") == "relu_l1"]
    if l1rows:
        l0s = [r["l0_measured_mean"] for r in l1rows]
        tgt = float(registry.TOPK_K)
        if min(l0s) > 2 * tgt or max(l0s) < 0.5 * tgt:
            warn.append(
                f"RELU_L1 ARM DEGENERATE: lambda grid "
                f"[{min(args.lambdas):.3g}, {max(args.lambdas):.3g}] produced "
                f"L0 in [{min(l0s):.0f}, {max(l0s):.0f}], which never brackets "
                f"the TopK target L0={tgt:.0f}. These rows are NOT sparsity-"
                f"matched to the beta arm and must not be compared against it. "
                f"Re-run with --lambdas bracketing the calibrated value from "
                f"B1's cal_*_relu_l1_*.json for this model.")
    if warn:
        print("\n!! B2 INTERPRETATION WARNINGS (each competes with the law as "
              "an explanation of a sharing rise):")
        for w in warn:
            print("   - " + w)
        print()
    if args.assert_topk_baseline and base is not None:
        ref = io.read_json("a5_metric_battery", spec.key)
        print(f"[baseline check] beta=0 top50 = {100*base['top50']:.1f}%; "
              f"compare exp_a5 topk/standard top-50 for {spec.key} "
              f"({'a5 results present' if ref else 'a5 results MISSING'}). "
              f"These SAEs are bit-identical; a mismatch means a protocol "
              f"drift between b2 and a5, not a finding.", flush=True)
    io.write_json(EXP, f"{spec.key}_phase",
                  dict(rows=rows, warnings=warn,
                       meta=dict(eval_seqs=args.eval_seqs, chunk=args.chunk,
                                 betas=args.betas, lambdas=args.lambdas,
                                 topk=args.topk, screen=args.screen,
                                 knob_engaged=bool(engaged),
                                 max_l1_over_mse=float(max_press))))
    return 0 if engaged else 3


if __name__ == "__main__":
    sys.exit(main())
