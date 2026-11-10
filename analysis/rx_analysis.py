"""Summarize language-model measurements from the saved results and per-feature arrays."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

import os
FEAT = Path(os.environ.get("RX_FEAT", str(Path(__file__).resolve().parent / "features")))
SEEDS = [42, 137, 0, 1, 2]
MODELS = ["gpt2", "pythia-70m-deduped", "pythia-160m", "pythia-1.4b", "pythia-6.9b", "gemma-2-2b", "gemma-2-9b",
          "llama-3.1-8b", "qwen-2.5-7b", "llama-3.1-70b"]


def unit(d):
    return d / np.linalg.norm(d, axis=1, keepdims=True).clip(1e-12)


def maxcos(a, b):
    out = np.empty(len(a))
    for i in range(0, len(a), 1024):
        out[i:i + 1024] = (a[i:i + 1024] @ b.T).max(1)
    return out


def metric_scores(raw, model, cond):
    """Per-feature score vectors for seed 42 from the packed imp outputs."""
    d = raw / ("rx-imp-%s-%s-s42" % (model, cond)) / "imp" / model / ("%s_s42" % cond)
    out = {}
    for name in ("zero", "mean", "sp", "mg", "st"):
        f = d / ("rank_%s.npz" % name)
        if f.exists():
            z = np.load(f)
            s = np.asarray(z["score"], np.float64)
            out[name] = (s, np.isfinite(s), int(z["draws"]) if "draws" in z.files else 1)
    f = d / "rank_grad.npz"
    if f.exists():
        z = np.load(f)
        for k, nm in (("gxi_magnitude", "gxi"), ("ig_magnitude", "IG")):
            kk = k if k in z.files else (k + "_feature_total" if k + "_feature_total" in z.files else None)
            if kk:
                v = np.asarray(z[kk], np.float64)
                v = v.reshape(v.shape[0], -1).sum(1) if v.ndim > 1 else v
                out[nm] = (v, np.isfinite(v), 1)
    plan = d / "plan.json"
    return out, (json.loads(plan.read_text()) if plan.exists() else {})


def importance_table(raw, ks=(10, 50)):
    rows = []
    for m in MODELS:
        for c in ("standard", "bypass"):
            src = FEAT / m / ("%s_s42.npz" % c)
            pre = FEAT / m / ("shared_%s_s42.npz" % c)     # precompute_shared.py: same definition, no decoders
            if not src.exists() and not pre.exists():
                continue
            scores, plan = metric_scores(raw, m, c)
            if not scores:
                continue
            if pre.exists():
                P = np.load(pre)
                shared, alive, tgts = P["shared"], P["alive"].astype(bool), [None] * int(P["n_targets"])
            else:
                A = np.load(src)
                da, alive = unit(A["decoder"].astype(np.float64)), A["alive"].astype(bool)
                tgts = [np.load(FEAT / m / ("%s_s%d.npz" % (c, s))) for s in SEEDS[1:] if (FEAT / m / ("%s_s%d.npz" % (c, s))).exists()]
                mc = [maxcos(da, unit(t["decoder"].astype(np.float64))) >= 0.7 for t in tgts]
                shared = np.mean(mc, axis=0)                 # fraction of the 4 targets matching each atom
            base = float(shared[alive].mean())
            for name, (s, meas, draws) in scores.items():
                ok = alive & meas
                n_ok = int(ok.sum())
                row = dict(model=m, condition=c, metric=name, draws=draws, n_alive=int(alive.sum()), n_ranked=n_ok,
                           exhaustive=bool(n_ok == alive.sum()), universe=plan.get("resample_universe") if name in ("sp", "mg", "st") else "",
                           dict_baseline=base, n_targets=len(tgts))
                order = np.argsort(-np.where(ok, s, -np.inf))
                for k in ks:
                    top = order[:min(k, n_ok)]
                    row["top%d" % k] = float(shared[top].mean()) if len(top) else None
                    row["top%d_minus_base" % k] = (row["top%d" % k] - base) if len(top) else None
                rows.append(row)
    return rows


def write(rows, path):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(pack_root):
    raw = Path(pack_root) / "raw"
    outd = Path(pack_root) / "analysis"
    outd.mkdir(exist_ok=True)
    rows = importance_table(raw)
    write(rows, outd / "importance_metric_sharing.csv")
    lines = ["model | cond | metric | ranked/alive | universe | dict base | top10 | top50 | top50-base"]
    for r in rows:
        f = lambda x: "-" if x is None else ("%.3f" % x if isinstance(x, float) else str(x))
        lines.append(" | ".join([r["model"], r["condition"], r["metric"], "%d/%d" % (r["n_ranked"], r["n_alive"]),
                                 str(r["universe"]), f(r["dict_baseline"]), f(r.get("top10")), f(r.get("top50")),
                                 f(r.get("top50_minus_base"))]))
    (outd / "importance_metric_sharing.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
