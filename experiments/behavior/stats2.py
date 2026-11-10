"""Compute agreement-edit summaries and bootstrap intervals."""
import argparse
import csv
import glob
import json
import pathlib

import numpy as np


def seed_vertex_bootstrap(per_seed, draws=4000, seed=20260921):
    """per_seed: {seed: value}. Resamples seeds with replacement."""
    ks = sorted(per_seed)
    if len(ks) < 2:
        return None
    rng = np.random.default_rng(seed)
    v = np.array([per_seed[k] for k in ks], float)
    d = v[rng.integers(0, len(v), size=(draws, len(v)))].mean(1)
    return dict(mean=float(v.mean()), lo95=float(np.quantile(d, .025)),
                hi95=float(np.quantile(d, .975)), n_seeds=len(ks),
                per_seed={str(k): float(per_seed[k]) for k in ks},
                note="seed-vertex bootstrap; %d seeds is few, treat the interval as descriptive" % len(ks))


def paired_contrast(per_seed_a, per_seed_b, draws=4000, seed=20260921):
    """Paired difference a-b, resampling seed vertices jointly so the pairing is preserved."""
    ks = sorted(set(per_seed_a) & set(per_seed_b))
    if len(ks) < 2:
        return None
    rng = np.random.default_rng(seed)
    d0 = np.array([per_seed_a[k] - per_seed_b[k] for k in ks], float)
    d = d0[rng.integers(0, len(d0), size=(draws, len(d0)))].mean(1)
    return dict(mean_diff=float(d0.mean()), lo95=float(np.quantile(d, .025)),
                hi95=float(np.quantile(d, .975)), n_seeds=len(ks),
                per_seed_diff={str(k): float(per_seed_a[k] - per_seed_b[k]) for k in ks},
                sign_consistent=bool(np.all(d0 > 0) or np.all(d0 < 0)))


def cluster_bootstrap_from_items(per_item, clusters, draws=2000, seed=20260921):
    """Resample example CLUSTERS (template x lemma), identically across methods.

    per_item: {method: {item_id: bool}}; clusters: {item_id: cluster_key}.
    """
    methods = sorted(per_item)
    if not methods:
        return None
    ids = sorted(clusters)
    by_cluster = {}
    for i in ids:
        by_cluster.setdefault(clusters[i], []).append(i)
    keys = sorted(by_cluster)
    if len(keys) < 3:
        return dict(status="too_few_clusters", n_clusters=len(keys))
    rng = np.random.default_rng(seed)
    out = {m: [] for m in methods}
    for _ in range(draws):
        pick = [keys[j] for j in rng.integers(0, len(keys), len(keys))]
        sel = [i for k in pick for i in by_cluster[k]]
        for m in methods:
            vals = [per_item[m].get(i) for i in sel]
            vals = [v for v in vals if v is not None]
            out[m].append(float(np.mean(vals)) if vals else np.nan)
    return dict(status="ok", n_clusters=len(keys),
                per_method={m: dict(mean=float(np.nanmean(out[m])),
                                    lo95=float(np.nanquantile(out[m], .025)),
                                    hi95=float(np.nanquantile(out[m], .975))) for m in methods})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/exp_a2")
    p.add_argument("--out", default="tables")
    p.add_argument("--b", type=int, default=5)
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, contrasts = [], []
    for f in sorted(glob.glob(str(pathlib.Path(a.results) / "expA2_*_sva.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        model = d["model"]
        methods = {}
        EPS = ("target_success", "target_success_eligible", "control_preserved", "joint_success")
        for key, arm in d["arms"].items():
            seed, meth = key.split("|")
            base = "random_pool" if meth.startswith("random_pool") else meth
            r = arm["results"].get(str(a.b))
            if r:
                for ep in EPS:
                    methods.setdefault((base, ep), {}).setdefault(int(seed), []).append(r.get(ep))
            # Dose is a predeclared sensitivity, carried with an explicit suffix so it can never be
            # confused with the dose-1 primary. Dose 2 is an EXTRAPOLATION beyond the natural donor
            # coefficient and is labelled as such wherever it appears.
            for dz, dr in (arm.get("dose_curve") or {}).items():
                for ep in EPS:
                    methods.setdefault((base, "%s@dose%s%s" % (ep, dz,
                                        "_EXTRAPOLATION" if float(dz) > 1 else "")),
                                       {}).setdefault(int(seed), []).append(dr.get(ep))
        agg = {}
        for (meth, ep), per_seed in methods.items():
            ps = {s: float(np.mean([x for x in v if x is not None]))
                  for s, v in per_seed.items() if any(x is not None for x in v)}
            if not ps:
                continue
            agg[(meth, ep)] = ps
            bs = seed_vertex_bootstrap(ps)
            rows.append(dict(model=model, method=meth, endpoint=ep, b=a.b,
                             **({k: v for k, v in bs.items() if k != "per_seed"} if bs else {}),
                             per_seed=json.dumps(bs["per_seed"]) if bs else None))
        eps_all = sorted({ep for _, ep in agg})
        for ep in [e for e in eps_all if e.startswith(("target_success_eligible", "joint_success",
                                                       "control_preserved"))]:
            for meth in ("posthoc", "bypass"):
                A, B = agg.get((meth, ep)), agg.get(("standard", ep))
                if not A or not B:
                    continue
                c = paired_contrast(A, B)
                if c:
                    contrasts.append(dict(model=model, contrast="%s_minus_standard" % meth,
                                          endpoint=ep, b=a.b,
                                          **{k: v for k, v in c.items() if k != "per_seed_diff"},
                                          per_seed_diff=json.dumps(c["per_seed_diff"])))
    for name, data in (("v2_endpoints.csv", rows), ("v2_contrasts.csv", contrasts)):
        if not data:
            print("skip (no rows):", name)
            continue
        fields = list(dict.fromkeys(k for r in data for k in r))
        with (out / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
        print("wrote %s (%d rows)" % (name, len(data)))
    for c in contrasts:
        if c["endpoint"] in ("joint_success", "joint_success@dose2_EXTRAPOLATION"):
            print("%-13s %-22s joint %+.3f [%+.3f,%+.3f] sign-consistent=%s"
                  % (c["model"], c["contrast"], c["mean_diff"], c["lo95"], c["hi95"],
                     c["sign_consistent"]))


if __name__ == "__main__":
    main()
