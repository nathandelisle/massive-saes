"""Assemble the original agreement-edit results into tables."""
import csv
import glob
import json
import pathlib
import statistics as st

HERE = pathlib.Path(__file__).resolve().parent
T = HERE / "tables"
MODELS = ["gpt2", "pythia-1.4b", "gemma-2-2b", "qwen-2.5-7b"]
METHODS = ["standard", "posthoc", "bypass", "random_pool"]


def w(name, rows):
    if not rows:
        print("skip (no rows):", name)
        return
    T.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (T / name).open("w", newline="", encoding="utf-8") as fh:
        c = csv.DictWriter(fh, fieldnames=fields)
        c.writeheader()
        for r in rows:
            c.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    print("wrote %s (%d rows)" % (name, len(rows)))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def main():
    main_rows, tr_rows, pool_rows, cov = [], [], [], []
    seen = set()
    for f in sorted(glob.glob(str(HERE / "results/exp_a2/expA2_*_sva.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        model = d["model"]
        seen.add(model)
        seeds = d["seeds"]
        arms = d["arms"]

        ac = d.get("archive_check") or {}
        for cond in ("standard", "bypass"):
            c = ac.get(cond) or {}
            pool_rows.append(dict(model=model, condition=cond,
                                  archive_id_overlap=c.get("top50_id_overlap"),
                                  archive_of=c.get("top50_of"),
                                  spearman_over_common=c.get("spearman_over_common"),
                                  status=c.get("status"),
                                  note="seed 42 only; membership defines the pool, ordering does not"))

        def cells(meth, key, dose=None):
            outp = {}
            for s in seeds:
                names = ([("random_pool%d" % i) for i in range(3)] if meth == "random_pool" else [meth])
                vals = []
                for n in names:
                    arm = arms.get("%d|%s" % (s, n))
                    if not arm:
                        continue
                    src = (arm["dose_curve"].get(dose) if dose else arm["results"].get("5"))
                    if src:
                        vals.append(src.get(key))
                v = mean(vals)
                if v is not None:
                    outp[s] = v
            return outp

        for meth in METHODS:
            row = dict(model=model, method=meth,
                       position_set=d["site_feasibility"]["position_set_used"],
                       n_test=d["arms"][list(arms)[0]]["results"]["5"]["n_items"],
                       unique_test_recipients=d["splits_meta"]["per_split"]["test"]["unique_recipients"],
                       all_splits_disjoint=d["disjointness"]["all_disjoint"])
            for key, lbl in (("target_success", "target"), ("control_preserved", "control"),
                             ("joint_success", "joint")):
                p1 = cells(meth, key)
                row["%s_dose1" % lbl] = mean(list(p1.values()))
                row["%s_dose1_per_seed" % lbl] = json.dumps({str(k): round(v, 4)
                                                             for k, v in sorted(p1.items())})
                p2 = cells(meth, key, dose="2")
                row["%s_dose2_EXTRAPOLATION" % lbl] = mean(list(p2.values()))
            for b in ("1", "3", "5"):
                v = []
                for s in seeds:
                    names = ([("random_pool%d" % i) for i in range(3)] if meth == "random_pool" else [meth])
                    for n in names:
                        arm = arms.get("%d|%s" % (s, n))
                        if arm and arm["results"].get(b):
                            v.append(arm["results"][b]["joint_success"])
                row["joint_b%s" % b] = mean(v)
            main_rows.append(row)

        byv = {}
        for t in d.get("transfer", []):
            base = "random_pool" if t["method"].startswith("random_pool") else t["method"]
            byv.setdefault((base, t["variant"]), []).append(t)
        for (meth, var), ts in sorted(byv.items()):
            tv = [x.get("target_success") for x in ts if x.get("target_success") is not None]
            jv = [x.get("joint_success") for x in ts if x.get("joint_success") is not None]
            tr_rows.append(dict(model=model, method=meth, variant=var, n_ordered_pairs=len(ts),
                                source_target=mean([x["source_target_success"] for x in ts]),
                                target_target=mean(tv), target_joint=mean(jv),
                                mean_collisions=mean([x["collisions"] for x in ts]),
                                mean_distinct_edited=mean([x["n_distinct_edited"] for x in ts]),
                                median_cosine=mean([st.median(x["cosines"]) for x in ts if x.get("cosines")]),
                                note=("six ordered pairs over three seeds share vertices; not six "
                                      "independent replicates")))
    for m in MODELS:
        cov.append(dict(model=m, task="sva",
                        status=("measured" if m in seen else "not_run"),
                        reason="" if m in seen else "cell not reached in the measured window"))
        cov.append(dict(model=m, task="ioi", status="not_run",
                        reason="v2 confirmation is SVA-only; v1 IOI retained as exploratory"))
    w("v2_main.csv", main_rows)
    w("v2_transfer.csv", tr_rows)
    w("v2_pools.csv", pool_rows)
    w("v2_coverage.csv", cov)

    if main_rows:
        print("\n%-13s %-12s %7s %7s %7s | %7s" % ("model", "method", "target", "control", "joint", "joint@2"))
        for r in main_rows:
            f = lambda x: ("%7.3f" % x) if x is not None else "     na"
            print("%-13s %-12s %s %s %s | %s" % (r["model"], r["method"], f(r["target_dose1"]),
                                                 f(r["control_dose1"]), f(r["joint_dose1"]),
                                                 f(r["joint_dose2_EXTRAPOLATION"])))


if __name__ == "__main__":
    main()
