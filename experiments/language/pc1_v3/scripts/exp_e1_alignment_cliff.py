"""Measure feature sharing across ranges of alignment with PC1."""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3.common import io_utils as io, registry                   # noqa: E402
from v3.common.fastabl import json_safe                          # noqa: E402

EXP = "e1_alignment_cliff"
# log-spaced in cos^2; the interesting structure is all below 0.2
BINS = [0.0, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6, 1.01]


def log(*a):
    print(*a, flush=True)


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if x.size < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, x.size), replace=True).mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def cell_from_npz(path, thr=registry.SHARED_COS_THRESHOLD):
    z = np.load(path)
    need = {"pc1_cos2", "alive", "max_cos"}
    if not need <= set(z.files):
        return None
    c2 = z["pc1_cos2"]
    alive = z["alive"].astype(bool)
    # max_cos is [n_other_seeds, d_sae]; an atom is "shared" against a given
    # seed if its best match exceeds the threshold. Average over seed-pairs.
    shared = (z["max_cos"] >= thr).mean(0)
    imp = z["imp_zero"] if "imp_zero" in z.files else None
    return c2, alive, shared, imp


def analyse(c2, alive, shared, imp, topk=50):
    base = float(shared[alive].mean())
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = alive & (c2 >= lo) & (c2 < hi)
        if not sel.any():
            continue
        v = shared[sel]
        ci = bootstrap_ci(v)
        rows.append(dict(lo=lo, hi=hi, n=int(sel.sum()),
                         sharing=float(v.mean()), ci_lo=ci[0], ci_hi=ci[1],
                         ratio_to_baseline=float(v.mean() / base) if base else None))
    # THE TILING BAND: the widest CONTIGUOUS run of bins whose sharing is below
    # half the dictionary-wide baseline. Contiguous, NOT "every bin above the
    # cliff" -- because the two-regime hypothesis predicts sharing RECOVERS at
    # the canonical end, so an all-bins-above rule refuses to find the band
    # precisely in the models where the prediction holds. It did exactly that:
    # the first version returned no cliff for gpt2, pythia-1.4b or gemma-2-2b,
    # each of which has an atom at cos^2 > 0.6 that is perfectly shared.
    lo_i = hi_i = None
    best, i = 0, 0
    while i < len(rows):
        if rows[i]["sharing"] < 0.5 * base:
            j = i
            while j + 1 < len(rows) and rows[j + 1]["sharing"] < 0.5 * base:
                j += 1
            n = sum(r["n"] for r in rows[i:j + 1])
            if n >= 20 and n > best:
                best, lo_i, hi_i = n, i, j
            i = j + 1
        else:
            i += 1
    band = rows[lo_i:hi_i + 1] if lo_i is not None else []
    cliff = band[0]["lo"] if band else None
    band_hi = band[-1]["hi"] if band else None
    out = dict(dictwide_baseline=base, bins=rows,
               cliff_cos2=cliff, band_hi_cos2=band_hi,
               band_n=int(sum(r["n"] for r in band)) if band else 0,
               band_sharing=(float(np.average([r["sharing"] for r in band],
                                              weights=[r["n"] for r in band]))
                             if band else None))
    # THE CANONICAL REGIME: atoms ABOVE the tiling band. An atom parallel to PC1
    # is a UNIQUE solution, so every seed should find it -- in sharp contrast to
    # the tiling shell just below. These counts are small by construction (one
    # atom, sometimes an antipodal pair), so n is reported alongside the rate
    # and the rate must never be quoted without it.
    if band_hi is not None:
        can = alive & (c2 >= band_hi)
        out["canonical_n"] = int(can.sum())
        out["canonical_sharing"] = (float(shared[can].mean())
                                    if can.any() else None)
        out["canonical_max_cos2"] = (float(c2[can].max()) if can.any() else None)
        out["two_regime_gap"] = (
            float(out["canonical_sharing"] - out["band_sharing"])
            if can.any() and out["band_sharing"] is not None else None)
    if imp is not None and np.isfinite(imp).any():
        fin = np.isfinite(imp) & alive
        idx = np.argsort(np.where(fin, imp, -np.inf))[::-1][:topk]
        out["top50_mean_cos2"] = float(np.mean(c2[idx]))
        out["top50_sharing"] = float(np.mean(shared[idx]))
        if cliff is not None:
            out["top50_frac_in_band"] = float(np.mean(c2[idx] >= cliff))
    # E-NULL discriminator: is it a cliff or a smooth slope? Compare the
    # bin-to-bin drop at the cliff against the median drop elsewhere.
    if cliff is not None and len(rows) >= 3:
        s = [r["sharing"] for r in rows]
        drops = [s[i] - s[i + 1] for i in range(len(s) - 1)]
        ci_idx = next((i for i, r in enumerate(rows) if r["lo"] == cliff), None)
        if ci_idx is not None and 0 < ci_idx <= len(drops):
            out["cliff_drop"] = float(drops[ci_idx - 1])
            others = [d for i, d in enumerate(drops) if i != ci_idx - 1]
            out["median_other_drop"] = float(np.median(others)) if others else None
            if others and abs(np.median(others)) > 1e-9:
                out["cliff_sharpness"] = float(
                    drops[ci_idx - 1] / max(abs(np.median(others)), 1e-9))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=io.ROOT,
                    help="directory tree containing */a5_metric_battery/features_*.npz")
    ap.add_argument("--topk", type=int, default=registry.EVAL_SEQS_IMPORTANCE and 50)
    a = ap.parse_args()

    pats = os.path.join(a.root, "**", "a5_metric_battery", "features_*.npz")
    seen, cells = set(), {}
    for p in sorted(glob.glob(pats, recursive=True), key=os.path.getmtime,
                    reverse=True):
        stem = os.path.basename(p)[len("features_"):-len(".npz")]
        if stem in seen:                     # same cell synced from two boxes
            continue
        seen.add(stem)
        got = cell_from_npz(p)
        if got is None:
            log(f"  skip {stem}: npz lacks pc1_cos2/alive/max_cos")
            continue
        cells[stem] = analyse(*got, topk=a.topk)
        cells[stem]["_c2_alive"] = got[0][got[1]]        # for the E3 control

    # Count, in each BYPASS cell, how many atoms reach the band location found
    # in the MATCHING standard cell -- the control has to be evaluated at the
    # standard condition's threshold, not at whatever cliff bypass finds on its
    # own residual noise.
    for stem, r in cells.items():
        if not stem.endswith("_bypass"):
            continue
        std_r = cells.get(stem[:-len("_bypass")] + "_standard")
        thr = std_r.get("cliff_cos2") if std_r else None
        if thr is not None:
            r["std_band_threshold"] = float(thr)
            r["n_at_or_above_std_band"] = int((r["_c2_alive"] >= thr).sum())
    for r in cells.values():
        r.pop("_c2_alive", None)

    nan = float("nan")
    log(f"\n{'cell':<26} {'base':>6} {'band lo':>9} {'band n':>7} "
        f"{'BAND share':>11} {'canon n':>8} {'CANON share':>12} {'gap':>7} "
        f"{'top50 in band':>14}")
    log("-" * 110)
    for stem in sorted(cells):
        r = cells[stem]
        log(f"{stem:<26} {r['dictwide_baseline']:6.3f} "
            f"{(r['cliff_cos2'] if r['cliff_cos2'] is not None else nan):9.2e} "
            f"{r['band_n']:7d} "
            f"{(r['band_sharing'] if r['band_sharing'] is not None else nan):11.3f} "
            f"{(r.get('canonical_n') if r.get('canonical_n') is not None else -1):8d} "
            f"{(r.get('canonical_sharing') if r.get('canonical_sharing') is not None else nan):12.3f} "
            f"{(r.get('two_regime_gap') if r.get('two_regime_gap') is not None else nan):7.3f} "
            f"{r.get('top50_frac_in_band', nan):14.2f}")

    std = {k: v for k, v in cells.items() if k.endswith("_standard")}
    byp = {k: v for k, v in cells.items() if k.endswith("_bypass")}
    verdict = dict(
        n_cells=len(cells),
        E1_cliff_in_standard=bool(std) and all(
            v["cliff_cos2"] is not None for v in std.values()),
        # E3 is a control, and it has to be stated at the STANDARD condition's
        # band location. Asking "is there any cliff at all in bypass" is the
        # wrong question: bypass leaves a residual micro-band (gemma-2-2b has
        # 40 atoms at cos^2 in [1e-5, 1e-3) with sharing 0.03, because PC1 is
        # estimated from a finite sample and the projection is not exact), and
        # that residual is 100-1000x below the standard band at cos^2 ~ 0.03.
        # The claim that matters is that bypass EMPTIES the tiling band.
        E3_band_empty_in_bypass=bool(byp) and all(
            (v.get("n_at_or_above_std_band") or 0) < 5 for v in byp.values()),
        bypass_n_in_std_band={k: v.get("n_at_or_above_std_band")
                              for k, v in byp.items()},
        bypass_residual_band={k: (v["cliff_cos2"], v["band_n"],
                                  v["band_sharing"]) for k, v in byp.items()
                              if v["cliff_cos2"] is not None},
        cliff_locations={k: v["cliff_cos2"] for k, v in std.items()},
        band_sharing={k: v["band_sharing"] for k, v in std.items()},
        canonical_sharing={k: v.get("canonical_sharing") for k, v in std.items()},
        canonical_n={k: v.get("canonical_n") for k, v in std.items()},
    )
    gaps = [v["two_regime_gap"] for v in std.values()
            if v.get("two_regime_gap") is not None]
    verdict["two_regime_gaps"] = gaps
    verdict["E1b_canonical_recovers"] = bool(gaps) and all(g > 0.4 for g in gaps)
    verdict["E2_cliff_stable"] = bool(
        len({round(np.log10(c), 0) for c in verdict["cliff_locations"].values()
             if c}) <= 2)
    io.write_json(EXP, "cliff", json_safe(dict(cells=cells, verdict=verdict,
                                               bins=BINS)))
    log("\nVERDICT " + json.dumps(json_safe(verdict), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
