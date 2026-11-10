"""Summarize sparse-probe accuracy and paired confidence intervals."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

PACK, LABELS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
RAW = PACK / "raw"
rng = np.random.default_rng(20261011)
NB = 2000
rows, model_avg = [], {}
for jd in sorted(RAW.glob("rx-probefit-*/probe/*")):
    model = jd.name
    for task in ("sst2", "ag_news", "amazon", "emotion"):
        fs, fb = jd / "standard" / ("%s.json" % task), jd / "bypass" / ("%s.json" % task)
        if not (fs.exists() and fb.exists()):
            continue
        S, Bp = json.loads(fs.read_text())["tables"], json.loads(fb.read_text())["tables"]
        assert S["test_ids"] == Bp["test_ids"], (model, task, "test ids differ")
        y = np.load(LABELS / ("%s_test_labels.npy" % task))[np.asarray(S["test_ids"])]
        for variant in ("mean_64", "last_128"):
            for k in ("1", "20", "100"):
                corr = {}
                for cname, C in (("standard", S), ("bypass", Bp)):
                    mats = []
                    for dname, e in C["variants"][variant]["per_dict"].items():
                        r = e["mi"][k]
                        c = np.asarray(r["pred"]) == y
                        assert abs(c.mean() - r["acc"]) < 1e-9, (model, task, variant, k, dname)
                        mats.append(c)
                    corr[cname] = np.vstack(mats)           # (5 dicts, n_test)
                n = y.shape[0]
                d_point = corr["bypass"].mean() - corr["standard"].mean()
                idx = rng.integers(0, n, (NB, n))
                bs = corr["bypass"][:, idx].mean((0, 2)) - corr["standard"][:, idx].mean((0, 2))
                ps, pb = corr["standard"].mean(1), corr["bypass"].mean(1)
                bl = S["variants"][variant]["baselines"]
                rows.append(dict(model=model, task=task, variant=variant, k=int(k), n_test=n,
                                 standard_acc=float(ps.mean()), bypass_acc=float(pb.mean()),
                                 bypass_minus_standard=float(d_point),
                                 ci95=[round(float(v), 4) for v in np.quantile(bs, [0.025, 0.975])],
                                 standard_seed_accs=[round(float(v), 4) for v in ps],
                                 bypass_seed_accs=[round(float(v), 4) for v in pb],
                                 dense_raw=bl["dense_raw"]["acc"], dense_pc1_removed=bl["dense_pc1_removed"]["acc"],
                                 scalar_pc1=bl["scalar_pc1"]["acc"]))
                if variant == "mean_64" and k == "20":
                    model_avg.setdefault(model, []).append((d_point, bs))
avg = []
for m, v in model_avg.items():
    pts = np.mean([x[0] for x in v])
    bs = np.mean([x[1] for x in v], axis=0)
    avg.append(dict(model=m, n_tasks=len(v), variant="mean_64", k=20, mean_task_diff_bypass_minus_standard=float(pts),
                    ci95=[round(float(x), 4) for x in np.quantile(bs, [0.025, 0.975])]))


def write(name, rs):
    keys = list(dict.fromkeys(k for x in rs for k in x))
    tmp = OUT / (name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rs)
    tmp.replace(OUT / name)


write("probe_paired.csv", rows)
write("probe_k20_task_average.csv", avg)
print(len(rows), "rows;", len(avg), "models")
