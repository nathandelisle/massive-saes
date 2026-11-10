"""Summarize feature sharing under zero, mean, gradient, and resampling scores."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

FULL, FEAT, SUMMARY, OUT = map(Path, sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["gpt2", "pythia-70m-deduped", "pythia-160m", "pythia-1.4b", "pythia-6.9b", "gemma-2-2b", "gemma-2-9b",
          "llama-3.1-8b", "qwen-2.5-7b"]
SEEDS = [42, 137, 0, 1, 2]
NB = 1000


def write(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    tmp = OUT / (name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(OUT / name)


def top50(score, ok, k=50):
    """Same rule as the pair exporter: descending score, ties by feature index, finite + alive only."""
    ix = np.flatnonzero(ok & np.isfinite(score))
    return ix[np.lexsort((ix, -score[ix]))][:k]


def shared_of(model, cond):
    P = np.load(FEAT / model / ("shared_%s_s42.npz" % cond))
    return P["shared"], P["alive"].astype(bool)


summ = {(r["model"], r["condition"]): r for r in csv.DictReader(open(SUMMARY))}

# ---------------- headline: zero ablation, seed-42 reference, article bootstrap ----------------
rng = np.random.default_rng(20261008)
head = []
for m in MODELS:
    per = {}
    for c in ("standard", "bypass"):
        z = np.load(FULL / ("%s_%s" % (m, c)) / "rank_zero.npz")
        sh, alive = shared_of(m, c)
        per[c] = dict(eff=z["eff"], score=z["score"], sh=sh, alive=alive, docs=z["docs"])
    assert (per["standard"]["docs"] == per["bypass"]["docs"]).all(), m          # same 75 articles
    nd = per["standard"]["eff"].shape[1]
    pt = {c: float(per[c]["sh"][top50(per[c]["score"], per[c]["alive"])].mean()) for c in per}
    for c in per:   # the bootstrap code path must reproduce the point estimate and the exporter's table
        s_full = per[c]["eff"].sum(1)
        assert np.array_equal(top50(s_full, per[c]["alive"] & np.isfinite(per[c]["score"])),
                              top50(per[c]["score"], per[c]["alive"])), (m, c, "rank rule")
        ref = float(summ[(m, c)]["top50"])
        assert abs(pt[c] - ref) < 1e-9, (m, c, pt[c], ref)
    bs = {"standard": [], "bypass": [], "diff": []}
    for _ in range(NB):
        idx = rng.integers(0, nd, nd)
        v = {}
        for c in per:
            s = per[c]["eff"][:, idx].sum(1)
            v[c] = float(per[c]["sh"][top50(s, per[c]["alive"] & np.isfinite(per[c]["score"]))].mean())
            bs[c].append(v[c])
        bs["diff"].append(v["bypass"] - v["standard"])
    q = {k: np.quantile(v, [0.025, 0.975]) for k, v in bs.items()}
    head.append(dict(model=m, standard_top50=pt["standard"], standard_ci95=[round(float(x), 4) for x in q["standard"]],
                     bypass_top50=pt["bypass"], bypass_ci95=[round(float(x), 4) for x in q["bypass"]],
                     diff=pt["bypass"] - pt["standard"], diff_ci95=[round(float(x), 4) for x in q["diff"]],
                     standard_dict_base=float(summ[(m, "standard")]["dict_sharing"]),
                     bypass_dict_base=float(summ[(m, "bypass")]["dict_sharing"]),
                     n_articles=nd, n_boot=NB,
                     scope="seed-42 source vs seeds 137/0/1/2; signed cos>=0.7; exhaustive zero ablation"))
write("headline_ref42_ci.csv", head)
print("headline done", flush=True)

# ---------------- 1D: reference seed vs all ordered pairs, with per-source spread ----------------
rows = []
for m in MODELS:
    for c in ("standard", "bypass"):
        s = summ[(m, c)]
        per_src = []
        for a in SEEDS:
            v = []
            for b in SEEDS:
                if a == b:
                    continue
                d = json.loads((FEAT / m / "pairs" / ("%s_s%d_vs_s%d.json" % (c, a, b))).read_text())
                v.append(d["top"]["50"]["sharing"])
            per_src.append(float(np.mean(v)))
        rows.append(dict(model=m, condition=c, n_ref_pairs=int(s["n_ref_pairs"]), n_all_pairs=int(s["n_all_pairs"]),
                         top50_ref42=float(s["top50"]), top50_ref42_min=float(s["top50_min"]),
                         top50_ref42_max=float(s["top50_max"]), top50_allpairs=float(s["top50_allpairs"]),
                         dict_ref42=float(s["dict_sharing"]), dict_allpairs=float(s["dict_sharing_allpairs"]),
                         top50_by_source_seed=json.dumps(dict(zip(SEEDS, np.round(per_src, 4).tolist()))),
                         top50_source_min=min(per_src), top50_source_max=max(per_src),
                         top50_source_sd=float(np.std(per_src, ddof=1)),
                         in_nine_model_set=True, note="nine-model set includes Pythia-70M, excludes Llama-70B"))
write("ref_vs_allpairs.csv", rows)

# ---------------- 1E: resampling operators ----------------
rng = np.random.default_rng(20261009)
res = []
for m in MODELS:
    for c in ("standard", "bypass"):
        d = FULL / ("%s_%s" % (m, c))
        sh, alive = shared_of(m, c)
        plan = json.loads((d / "plan.json").read_text())
        Ureg = np.load(d / "resample_universe.npy") if (d / "resample_universe.npy").exists() else None
        print("resampling", m, c, flush=True)
        for op in ("mean", "sp", "mg", "st"):
            f = d / ("rank_%s.npz" % op)
            if not f.exists():
                continue
            z = np.load(f)
            score = z["score"]
            meas = alive & np.isfinite(score)
            t = top50(score, alive)
            row = dict(model=m, condition=c, operator=op, plan_universe=plan.get("resample_universe"),
                       n_ranked=int(meas.sum()), n_alive=int(alive.sum()),
                       top50=float(sh[t].mean()), ranked_set_baseline=float(sh[meas].mean()),
                       dict_baseline=float(sh[alive].mean()))
            if op != "mean" and z["eff"].ndim == 3:
                rows_m = np.flatnonzero(meas)
                E = z["eff"][rows_m].sum(1)       # (measured, draws): article-summed effect per draw
                nr = E.shape[1]
                row["draws"] = nr
                bs = []
                s = np.full(len(score), np.nan)
                for _ in range(NB):
                    r_ = rng.integers(0, nr, nr)
                    s[rows_m] = E[:, r_].mean(1)
                    bs.append(float(sh[top50(s, alive)].mean()))
                row["top50_draw_ci95"] = [round(float(x), 4) for x in np.quantile(bs, [0.025, 0.975])]
            # U recall on exhaustive rankings: how much of the full top-50 the registered U contains,
            # and what top-50-within-U would have reported
            if plan.get("resample_universe") == "all_alive" and Ureg is not None:
                inU = np.zeros_like(alive)
                inU[Ureg] = True
                row["U_size"] = int((inU & alive).sum())
                row["U_recall_of_exhaustive_top50"] = float(inU[t].mean())
                tu = top50(np.where(inU, score, np.nan), alive)
                row["top50_within_U"] = float(sh[tu].mean())
                row["U_baseline"] = float(sh[inU & alive].mean())
            res.append(row)
write("resampling_closure.csv", res)
print("headline", len(head), "ref_vs_all", len(rows), "resampling", len(res))
