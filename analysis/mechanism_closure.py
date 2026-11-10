"""Summarize individual and joint component ablations."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

RAW = Path(sys.argv[1]) / "raw"
OUT = Path(sys.argv[2])
rows, flags = [], []
for f in sorted(RAW.glob("rx-wsx-*/wsx/*/*/mechanism.json")):
    model, cs = f.parts[-3], f.parts[-2]
    cond, seed = cs.rsplit("_s", 1)
    d = json.loads(f.read_text())
    z = np.load(f.with_name("mechanism.npz"))
    feats = z["features"]
    pos = {int(j): i for i, j in enumerate(feats)}
    g = d.get("groups") or {}
    r = dict(model=model, condition=cond, seed=int(seed), n_dense=d.get("n_dense"))
    for gname, cap in (("top50", None), ("dense", 50)):
        ids = [int(x) for x in (g.get(gname) or [])][:cap]
        ix = [pos[j] for j in ids if j in pos]
        if not ix:
            continue
        full, par, perp = (z["dce_%s" % k][ix] for k in ("full", "par", "perp"))
        ok = np.isfinite(full) & np.isfinite(par) & np.isfinite(perp)
        tag = "%s%s" % (gname, "" if cap is None else "_first%d" % cap)
        r["%s_n_features" % tag] = int(ok.sum())
        r["%s_sum_full" % tag] = float(full[ok].sum())
        r["%s_sum_par" % tag] = float(par[ok].sum())
        r["%s_sum_perp" % tag] = float(perp[ok].sum())
        r["%s_perp_over_full_sums" % tag] = float(perp[ok].sum() / full[ok].sum()) if full[ok].sum() != 0 else None
        r["%s_par_over_full_sums" % tag] = float(par[ok].sum() / full[ok].sum()) if full[ok].sum() != 0 else None
        fr = perp[ok] / np.where(full[ok] == 0, np.nan, full[ok])
        r["%s_median_feature_perp_over_full" % tag] = float(np.nanmedian(fr))
        r["%s_frac_features_perp_gt_par" % tag] = float((perp[ok] > par[ok]).mean())
        if (~ok).any():
            flags.append(dict(model=model, condition=cond, seed=int(seed), what="per-feature %s" % tag,
                              n_nonfinite=int((~ok).sum())))
    for k, v in (d.get("group_dce") or {}).items():
        valid = v is not None and np.isfinite(v)
        r["JOINT_%s" % k] = float(v) if valid else "INVALID"
        if not valid:
            flags.append(dict(model=model, condition=cond, seed=int(seed), what="group_dce.%s" % k, value=str(v)))
    canc = d.get("cancellation") or {}
    for gname in ("top50", "dense", "ctl_count"):
        r["cancel_%s_ratio_energy" % gname] = (canc.get(gname) or {}).get("ratio_energy")
    rows.append(r)


def write(name, rs):
    keys = list(dict.fromkeys(k for x in rs for k in x))
    tmp = OUT / (name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rs)
    tmp.replace(OUT / name)


write("mechanism_individual_vs_joint.csv", rows)
write("mechanism_invalid_flags.csv", flags or [dict(model="", what="none")])
print(len(rows), "dictionaries;", len(flags), "flags")
