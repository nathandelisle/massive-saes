"""Train learned-threshold JumpReLU SAEs."""
import argparse
import csv
import math
import pathlib
import time

import numpy as np

from rcommon import add_v3, log, write_json
from xforms import BUILD


def build_faithful(d_model, d_sae, seed, l0_coef, theta_init=1e-3, bandwidth=1e-3):
    """A learned-threshold JumpReLU that reuses the archived App. J autograd kernels.

    The kernels in `v3/common/saes.py` (`_JumpReLUSTE`, `_StepSTE`) are correct and are
    imported, not reimplemented: dy/dx = H(x - theta) exactly, the L0 term's x-gradient is
    identically zero so the penalty moves only theta, and dy/dtheta uses the rectangle
    kernel at the stated bandwidth. What changes here is everything around them.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from v3.common.saes import BaseSAE, _JumpReLUSTE, _StepSTE

    class FaithfulJumpReLU(BaseSAE):
        arch = "jumprelu_learned"

        def __init__(self):
            super().__init__(d_model, d_sae, seed=seed)
            self.log_theta = nn.Parameter(torch.full((d_sae,), math.log(theta_init)))
            self.l0_coef = l0_coef
            self.register_buffer("bandwidth_t", torch.tensor(float(bandwidth)))
            self._pre = None

        @property
        def bandwidth(self):
            return float(self.bandwidth_t)

        def encode(self, x):
            pre = F.relu(self.pre(x))
            self._pre = pre
            return _JumpReLUSTE.apply(pre, self.log_theta.exp(), self.bandwidth)

        def sparsity_loss(self, z, pre):
            return self.l0_coef * _StepSTE.apply(
                self._pre, self.log_theta.exp(), self.bandwidth).sum(-1).mean()

        @torch.no_grad()
        def init_theta_from_batch(self, x, target_l0):
            """ONE-SHOT data-dependent initialization of theta, at step 0 only.

            WHY THIS IS NEEDED, and why it is not the scheduled variant. Adam bounds
            |d log theta| by about `lr` per step, so over a 20,000-step run theta can move
            at most exp(3e-4 * 20000) = 403x, and over a 3,000-step calibration probe only
            3.3x. Measured here on GPT-2: at l0_coef = 100, theta moved 1.11x in 1,000 steps
            and L0 sat at 899 against a target of 64. The L0 penalty is NOT inert -- theta is
            rate-limited, and Adam normalizes gradient magnitude, so raising the coefficient
            further buys nothing.

            Rajamanoharan et al. (2407.14435 App. I) avoid this by renormalizing activations
            so E||x||^2 = 1 and then using theta_init = 1e-3, which lands theta near the
            operating point for THEIR encoder initialization. Our encoder init is the
            archived tied init, so the same fixed constant lands in a different place. Setting
            theta from the pre-activation quantile at the target L0 reproduces the intent of
            their initialization rather than its literal constant.

            This fires ONCE, before the first optimizer step, and theta is a free learned
            parameter for every step thereafter. That is the whole difference from the
            archived scheduled variant, which re-seats theta from this same quantile every
            250 steps throughout training and thereby sets sparsity by schedule.
            """
            pre = F.relu(self.pre(x)).flatten().float()
            if pre.numel() > 1_000_000:
                pre = pre[torch.randint(0, pre.numel(), (1_000_000,), device=pre.device)]
            pos_frac = float((pre > 0).float().mean())
            want = float(target_l0) / self.d_sae
            if want >= pos_frac:
                raise RuntimeError(
                    "jumprelu theta init impossible: target L0 %g of d_sae %d needs %.4f of "
                    "units active but only %.4f of pre-activations are positive"
                    % (target_l0, self.d_sae, want, pos_frac))
            th = float(torch.quantile(pre, min(max(1.0 - want, 0.0), 1.0 - 1e-6)))
            if not (th > 0) or not math.isfinite(th):
                raise RuntimeError("jumprelu theta init produced theta=%r" % th)
            self.log_theta.data.fill_(math.log(th))
            self.bandwidth_t.fill_(th)      # the reference ties the bandwidth to the scale
            return th

    return FaithfulJumpReLU()


def train_faithful(batches, d_model, d_sae, seed, l0_coef, steps, device, tag,
                   lr=3e-4, log_every=1000, target_l0=64.0):
    """The archived trainer's loop, minus the recalibration branch that defines the
    scheduled variant. Decoder rows are renormalized every step, as in the protocol.

    theta is initialized once from the first batch (see `init_theta_from_batch`) and is a
    free learned parameter for every step after that. Nothing re-seats it during training.
    """
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    sae = build_faithful(d_model, d_sae, seed, l0_coef).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    hist, t0 = [], time.time()
    sae.train()
    th0 = None
    for step, x in enumerate(batches):
        if step == 0:
            th0 = sae.init_theta_from_batch(x, target_l0)
        recon, z = sae(x)
        mse = F.mse_loss(recon, x, reduction="none").sum(-1).mean()
        loss = mse + sae.sparsity_loss(z, None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.renorm_decoder()
        if step % log_every == 0 or step < 5:
            with torch.no_grad():
                l0v = (z != 0).float().sum(-1)
                rec = dict(step=step, mse=float(mse),
                           fvu=float(mse / x.pow(2).sum(-1).mean()),
                           l0=float(l0v.mean()), l0_p10=float(torch.quantile(l0v, .10)),
                           l0_p90=float(torch.quantile(l0v, .90)),
                           theta_mean=float(sae.log_theta.exp().mean()))
            hist.append(rec)
            if step % log_every == 0:
                log("trkC", "    [%s] step %6d fvu=%.4f L0=%.1f theta=%.3g %.1fm"
                    % (tag, step, rec["fvu"], rec["l0"], rec["theta_mean"],
                       (time.time() - t0) / 60))
            if not math.isfinite(rec["mse"]):
                raise RuntimeError("faithful jumprelu diverged at step %d (%s)" % (step, tag))
    sae.eval()
    thr = float(sae.log_theta.exp().mean())
    return sae, hist, dict(theta_init_mean=th0, theta_final_mean=thr,
                           theta_moved_x=thr / max(th0, 1e-30))


def measure_l0(sae, batches_fn, device, n=8):
    import torch
    vals = []
    with torch.no_grad():
        for i, x in enumerate(batches_fn()):
            z = sae.encode(x)
            vals.append((z != 0).float().sum(-1).cpu().numpy())
            if i + 1 >= n:
                break
    v = np.concatenate(vals) if vals else np.zeros(1)
    return dict(mean=float(v.mean()), p10=float(np.quantile(v, .10)),
                p50=float(np.quantile(v, .50)), p90=float(np.quantile(v, .90)),
                frac_zero=float((v == 0).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 0])
    p.add_argument("--arms", nargs="+",
                   default=["jumprelu_learned", "jumprelu_learned_bypass",
                            "jumprelu_scheduled"])
    p.add_argument("--target-l0", type=float, default=64.0)
    p.add_argument("--cal-seed", type=int, default=1009)
    p.add_argument("--cal-steps", type=int, default=3000)
    p.add_argument("--cal-full", action="store_true",
                   help="calibrate at the FULL training length. A 3,000-step probe "
                        "does NOT predict the 20,000-step endpoint: measured on GPT-2, "
                        "a setting that probed to L0 69 landed at L0 1.6 at full "
                        "length, because theta keeps climbing under the penalty for "
                        "6.7x longer. Costs one full training per probe.")
    p.add_argument("--cal-max-rounds", type=int, default=7)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--eval-seqs", type=int, default=64)
    p.add_argument("--out", default="results/track_c")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hours", type=float, default=10.0)
    a = p.parse_args()
    if a.cal_full:
        a.cal_steps = a.steps
        a.cal_max_rounds = min(a.cal_max_rounds, 5)

    import torch
    add_v3()
    from v3.common import registry
    from v3.common.data import load_model_and_tokenizer, eval_sequences, open_activations
    from v3.common.io_utils import ensure_acts
    from v3.common.pca import estimate_pcs
    from v3.common.saes import train_sae, save_sae, load_sae
    from v3.common.fastabl import tuned_chunk
    from v3.common.matching import max_cos
    from track_b import eval_arm, make_batches

    spec = registry.MODELS[a.model]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False
    t_start = time.time()

    adir = ensure_acts(spec, device=dev)
    _e, evecs, mean = estimate_pcs(adir)
    v1_np = evecs[:, 0]
    acts, _ = open_activations(adir)
    fit_rows = np.asarray(acts[:200_000], dtype=np.float32)
    torch.cuda.empty_cache()
    model, tok = load_model_and_tokenizer(spec, device=dev)
    model.eval()
    seqs, am = eval_sequences(tok, n_seqs=a.eval_seqs)
    chunk = tuned_chunk(spec)
    mu = torch.tensor(mean, dtype=torch.float32, device=dev)
    v1 = torch.tensor(v1_np, dtype=torch.float32, device=dev)

    sn = BUILD["scalarnorm"]().fit(fit_rows, mean)
    raw = BUILD["raw"]()
    log("trkC", "%s: normalization scalar s=%.4f (E||x/s||^2 = 1 on training rows)"
        % (a.model, sn.s))

    def batches(seed, bypass, steps, xf):
        return make_batches(adir, mean, xf, a.batch, steps, seed, dev,
                            v1_norm=None) if not bypass else _bypass(
            make_batches(adir, mean, xf, a.batch, steps, seed, dev), v1, sn.s if xf is sn else 1.0)

    res = dict(model=a.model, hf_id=spec.hf_id, layer=spec.target_layer,
               target_l0=a.target_l0, tolerance="+/-10%", cal_seed=a.cal_seed,
               steps=a.steps, batch=a.batch, normalization=sn.meta(),
               scheduled_variant_note=(
                   "the archived JumpReLU re-seats theta from a pre-activation quantile "
                   "every 250 steps; its sparsity is set by that schedule, not by the L0 "
                   "penalty. It is reproduced here as a comparator and labelled "
                   "'scheduled threshold' everywhere."),
               theta_init_note=(
                   "theta is set ONCE at step 0 from the pre-activation quantile at the "
                   "target L0 and is learned thereafter. The literal 1e-3 of the reference "
                   "is calibrated to its own encoder init; measured here it left GPT-2 at "
                   "L0 = 899 with theta rate-limited to 1.11x in 1,000 steps."),
               arms={}, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    rows = []

    # --- calibration: geometric bracket on a separate seed --------------------------
    cal = {}
    for arm in a.arms:
        if arm == "jumprelu_scheduled":
            continue
        byp = arm.endswith("bypass")
        target = a.target_l0 - (1.0 if byp else 0.0)      # 63 expected sparse + 1 carried
        trace = []

        def probe(init_l0, coef, tag):
            """One short run; returns its measured L0 or None if it diverged."""
            try:
                sae, _h, tm = train_faithful(
                    batches(a.cal_seed, byp, a.cal_steps, sn), spec.d_model, spec.d_sae,
                    a.cal_seed, coef, a.cal_steps, dev, tag, log_every=1000,
                    target_l0=init_l0)
            except RuntimeError as e:
                trace.append(dict(stage=tag, init_l0=init_l0, l0_coef=coef,
                                  status="diverged", err=str(e)))
                return None
            st = measure_l0(sae, lambda: batches(a.cal_seed + 1, byp, 8, sn), dev)
            trace.append(dict(stage=tag, init_l0=init_l0, l0_coef=coef, l0=st["mean"],
                              theta_moved_x=tm["theta_moved_x"], status="ok"))
            log("trkC", "  cal %s [%s] init_L0=%.0f l0_coef=%.4g -> L0 %.1f "
                        "(target %.1f, theta moved %.2fx)"
                % (arm, tag, init_l0, coef, st["mean"], target, tm["theta_moved_x"]))
            del sae
            torch.cuda.empty_cache()
            return st["mean"]

        # STAGE 1 -- find a theta initialization whose UNPENALIZED run lands ABOVE target.
        #
        # Measured on the first pass: initializing theta at the target quantile put L0 at
        # exactly 64 at step 0 and training then drove it DOWN to 40 (GPT-2), 14.5
        # (Pythia-1.4B) and 16.3 (Qwen), where it sat regardless of l0_coef across six
        # decades. The L0 penalty can only ever REMOVE activations, so initializing at the
        # target leaves it nothing to do and the bracket can never close from below. This
        # is the same shape as the archived scheduled variant's failure, arrived at from the
        # opposite direction, and it is a property of the search, not of the architecture.
        #
        # The reference recipe starts theta SMALL (L0 high) and lets the penalty sparsify
        # down to the operating point. Stage 1 reproduces that by raising the initial L0
        # until the unpenalized endpoint clears the target; stage 2 then brackets the
        # penalty, so sparsity is genuinely set by the L0 term and not by the initialization.
        init_l0, headroom = float(target), None
        for _ in range(9):
            if time.time() - t_start > a.hours * 3600 * 0.30:
                break
            v = probe(init_l0, 0.0, "stage1_init")
            if v is not None and v > target * 1.15:
                headroom = (init_l0, v)
                break
            # x2, not x4. Measured on Pythia-1.4B: init 256 gave an unpenalized endpoint
            # of 55.5 (just under the target) and init 1024 gave 220.9, so the x4 ladder
            # skipped straight past the usable range and left the penalty a 3.5x reduction
            # it could not make. GPT-2's x4 ladder happened to land on 256 -> 58 and worked.
            init_l0 = min(init_l0 * 2.0, 0.45 * spec.d_sae)
            if init_l0 >= 0.45 * spec.d_sae:
                v = probe(init_l0, 0.0, "stage1_init")
                if v is not None and v > target * 1.15:
                    headroom = (init_l0, v)
                break

        # STAGE 2 -- bracket the L0 penalty downward from that initialization.
        lo, hi, coef = None, None, 1e-6
        if headroom is not None:
            init_l0 = headroom[0]
            for rnd in range(a.cal_max_rounds):
                if time.time() - t_start > a.hours * 3600 * 0.38:
                    break
                v = probe(init_l0, coef, "stage2_penalty")
                if v is None:
                    coef /= 10.0
                    continue
                if v > target:
                    lo = coef                 # too dense -> need a bigger penalty
                    coef = coef * 10 if hi is None else math.sqrt(coef * hi)
                else:
                    hi = coef
                    coef = coef / 10 if lo is None else math.sqrt(coef * lo)
                if lo is not None and hi is not None and hi / lo < 1.6:
                    break

        ok = [t for t in trace if t.get("status") == "ok" and t.get("stage") == "stage2_penalty"]
        if not ok:
            ok = [t for t in trace if t.get("status") == "ok"]
        best = min(ok, key=lambda t: abs(t["l0"] - target)) if ok else None
        cal[arm] = dict(
            target_l0=target, trace=trace, chosen=best,
            stage1_headroom=headroom, bracketed=(lo is not None and hi is not None),
            status=("calibrated" if best and abs(best["l0"] - target) <= 0.1 * target
                    else "INCOMPLETE_nearest_operating_point"),
            note=("two-stage: the theta initialization is chosen on the calibration seed so "
                  "the UNPENALIZED endpoint clears the target, then the L0 penalty is "
                  "bracketed down to it, so the penalty sets sparsity. Both are frozen "
                  "across the measurement seeds. The short-probe L0 estimates the "
                  "full-length value; the full-length L0 is reported per seed and is the "
                  "number that must be compared."))
        log("trkC", "  cal %s -> %s (%s)" % (arm, best, cal[arm]["status"]))
    res["calibration"] = cal

    # --- measurement ---------------------------------------------------------------
    ranks = {}
    for arm in a.arms:
        byp = arm.endswith("bypass")
        res["arms"][arm] = dict(cells={}, bypass=byp,
                                l0_coef=(cal.get(arm, {}).get("chosen") or {}).get("l0_coef"),
                                theta_init_l0=(cal.get(arm, {}).get("chosen") or {})
                                .get("init_l0"),
                                calibration_status=cal.get(arm, {}).get("status"))
        for seed in a.seeds:
            if time.time() - t_start > a.hours * 3600:
                res["arms"][arm]["cells"][str(seed)] = dict(status="unmeasured_budget")
                continue
            t0 = time.time()
            path = pathlib.Path("saes_c/%s_%s_s%d.pt" % (a.model, arm, seed))
            xf = raw if arm == "jumprelu_scheduled" else sn
            try:
                if path.exists():
                    sae, _x = load_sae(str(path), device=dev)
                    src, tmeta = "cached", {}
                elif arm == "jumprelu_scheduled":
                    sae, tlog = train_sae(batches(seed, byp, a.steps, raw), spec.d_model,
                                          spec.d_sae, arch="jumprelu", l0_coef=1e-3,
                                          target_l0=a.target_l0, seed=seed, device=dev,
                                          total_steps=a.steps,
                                          tag="%s/%s/s%d" % (a.model, arm, seed))
                    src, tmeta = "trained", dict(variant="scheduled threshold")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_sae(sae, str(path), extra=dict(arm=arm, log=tlog[-3:]))
                else:
                    ch = cal[arm].get("chosen") or {}
                    coef = ch.get("l0_coef")
                    init_l0 = ch.get("init_l0", a.target_l0 - (1.0 if byp else 0.0))
                    if coef is None:
                        res["arms"][arm]["cells"][str(seed)] = dict(
                            status="not_run_calibration_failed")
                        continue
                    sae, tlog, tmeta = train_faithful(
                        batches(seed, byp, a.steps, sn), spec.d_model, spec.d_sae, seed,
                        coef, a.steps, dev, "%s/%s/s%d" % (a.model, arm, seed),
                        target_l0=init_l0)   # the CALIBRATED init, frozen across seeds
                    src = "trained"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_sae(sae, str(path), extra=dict(arm=arm, l0_coef=coef,
                                                        theta=tmeta, log=tlog[-3:]))
            except RuntimeError as e:
                res["arms"][arm]["cells"][str(seed)] = dict(status="FAILED", error=str(e))
                log("trkC", "  %s/s%d FAILED: %s" % (arm, seed, e))
                continue
            sae.eval()
            l0d = measure_l0(sae, lambda: batches(seed + 500, byp, 8, xf), dev)
            ev = eval_arm(model, spec, seqs, am, sae, mu, xf,
                          (v1 if byp else None), ("bypass" if byp else "standard"),
                          None, chunk, v1_stats=v1)
            D = sae.W_dec.detach().float()
            Dn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)
            c2 = (Dn @ v1).pow(2).cpu().numpy()
            ranks[(arm, seed)] = Dn.cpu().numpy()
            cell = dict(status="measured", source=src, seconds=time.time() - t0,
                        l0_distribution=l0d, theta=tmeta,
                        l0_within_tolerance=bool(
                            abs(l0d["mean"] - (a.target_l0 - (1 if byp else 0)))
                            <= 0.1 * a.target_l0),
                        frac_alive_above_0p01=float((c2 >= 0.01).mean()),
                        **{k: v for k, v in ev.items() if k != "sufficient_stats"})
            res["arms"][arm]["cells"][str(seed)] = cell
            rows.append(dict(model=a.model, arm=arm, seed=seed, source=src,
                             l0_mean=l0d["mean"], l0_p10=l0d["p10"], l0_p90=l0d["p90"],
                             l0_in_tolerance=cell["l0_within_tolerance"],
                             theta_moved_x=tmeta.get("theta_moved_x"),
                             **{k: v for k, v in ev.items() if k != "sufficient_stats"}))
            log("trkC", "  %s/s%d [%s] %.1f min | L0 %.1f [%.0f,%.0f] FVU %.4f content "
                        "%.4f CE %.4f alive %d"
                % (arm, seed, src, (time.time() - t0) / 60, l0d["mean"], l0d["p10"],
                   l0d["p90"], ev["fvu_total"], ev["fvu_content"] or float("nan"),
                   ev["splice_ce"], ev["n_alive"]))
            write_json(out / ("track_c_%s.json" % a.model), res)
            _wrows(out / ("track_c_rows_%s.csv" % a.model), rows)
            del sae
            torch.cuda.empty_cache()

    rec = []
    for arm in a.arms:
        ks = [(arm, s) for s in a.seeds if (arm, s) in ranks]
        for ka in ks:
            for kb in ks:
                if ka == kb:
                    continue
                dw = max_cos(ranks[ka], ranks[kb])
                rec.append(dict(model=a.model, arm=arm, source_seed=ka[1],
                                target_seed=kb[1],
                                dictionary_wide_shared_0p7=float((np.abs(dw) >= 0.7).mean()),
                                median_cos=float(np.median(np.abs(dw)))))
    _wrows(out / ("track_c_recurrence_%s.csv" % a.model), rec)
    res["wall_minutes"] = (time.time() - t_start) / 60
    write_json(out / ("track_c_%s.json" % a.model), res)
    _wrows(out / ("track_c_rows_%s.csv" % a.model), rows)
    log("trkC", "DONE %s in %.1f min" % (a.model, res["wall_minutes"]))


def _bypass(batches, v1_t, scale=1.0):
    """PC1 bypass inside whatever space the batches are in. The direction is the raw v1,
    which is correct for the fixed-scalar normalization because a scalar does not rotate."""
    for x in batches:
        yield x - (x @ v1_t)[:, None] * v1_t[None, :]


def _wrows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
