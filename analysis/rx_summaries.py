"""Assemble language-model result tables. Some inputs must be downloaded separately."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

B = "ssup-research-079415246777"


def rd(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def write(rows, path):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def md_table(rows, cols, fmt=lambda x: x):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def f3(x):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return "%.3f" % x
    return str(x)


def jobs(raw, prefix):
    return sorted(p for p in raw.iterdir() if p.name.startswith(prefix))


def mechanism(raw):
    rows = []
    for j in jobs(raw, "rx-wsx-"):
        for f in j.rglob("mechanism.json"):
            m = rd(f)
            if not m:
                continue
            model, cs = f.parent.parent.name, f.parent.name
            r = dict(model=model, dict=cs, n_dense=m.get("n_dense"))
            for g in ("dense", "top50", "ctl_count", "ctl_rate", "ctl_energy"):
                c = m["cancellation"].get(g, {})
                r[g + "_N2_over_S2"] = c.get("ratio_energy")
                r[g + "_PR"] = c.get("aggregate_participation_ratio")
                r[g + "_cofire"] = c.get("mean_pairwise_cofiring_jaccard")
                for comp in ("full", "par", "perp"):
                    r["%s_dCE_%s" % (g, comp)] = m["group_dce"].get("%s_%s" % (g, comp))
            rows.append(r)
    return rows


def patching(raw):
    rows = []
    for j in jobs(raw, "rx-wsx-"):
        for f in j.rglob("patching/*.json"):
            if f.name == "PROVENANCE.json":
                continue
            p = rd(f)
            if not p or "task" not in p:
                continue
            r = dict(model=f.parent.parent.parent.name, dict=f.parent.parent.name, task=p["task"], status=p.get("status"),
                     test_aligned=(p.get("aligned_counts") or {}).get("test"),
                     native_test_eligible=((p.get("native") or {}).get("test") or {}).get("eligible"),
                     spliced_acc=p.get("spliced_recipient_acc_test"))
            c = p.get("controls") or {}
            for k in ("none", "full_residual", "pc1_only", "perp_only"):
                r["ctl_%s_switch" % k] = (c.get(k) or {}).get("target_switch")
            pi = p.get("patch_importance") or {}
            v = pi.get("top50_validation_effect")
            r["top50_valid_effect_mean"] = float(np.mean(v)) if v else None
            # PRIMARY endpoint (rx/patching.py): held-out test pairs, native-eligible cohort, budget 5,
            # dose 1 -> results['b5_d1_eligible']. The greedy trace is DISCOVERY data (budget 10) and is
            # exported only under a discovery_* name (audit 2026-10-08: the old *_joint_final column read it).
            for pool, u in (p.get("usability") or {}).items():
                res = (u or {}).get("results") or {}
                for cohort in ("eligible", "unrestricted"):
                    q = res.get("b5_d1_%s" % cohort) or {}
                    for k in ("n", "target_switch", "foil_retention", "joint_success"):
                        r["%s_b5d1_%s_%s" % (pool, cohort, k)] = q.get(k)
                tr = (u or {}).get("greedy_trace") or []
                if tr:
                    r["%s_discovery_b%d_joint" % (pool, tr[-1].get("step"))] = tr[-1].get("joint")
            rows.append(r)
    return rows


def train(raw):
    rows = []
    for j in jobs(raw, "rx-train-"):
        for f in j.rglob("summary.json"):
            s = rd(f)
            if not s:
                continue
            for cond, c in (s.get("cells") or {}).items():
                l0 = c.get("l0_measured") or []
                rows.append(dict(model=s.get("model"), kind=s.get("kind"), layer=s.get("layer"), arch=c.get("arch"),
                                 condition=cond, pc1_var=s.get("pc1_var_explained"), kwargs=json.dumps(c.get("kwargs")),
                                 l0_mean=float(np.mean(l0)) if l0 else None, l0_min=min(l0) if l0 else None,
                                 l0_max=max(l0) if l0 else None, l0_matched_pm5=c.get("l0_matched_pm5"),
                                 calibration_ok=c.get("calibration_ok"),
                                 top50_shared=c.get("top50_shared_mean_over_sources"),
                                 top50_shared_sd_over_sources=c.get("top50_shared_sd_over_sources"),
                                 dictwide=float(np.mean([b["dictwide_mean"] for b in c.get("by_source", [])])) if c.get("by_source") else None,
                                 top50_dense_frac=float(np.mean([b["top50_dense_frac"] for b in c.get("by_source", [])])) if c.get("by_source") else None,
                                 reused_archived=len(c.get("reused_archived") or [])))
            rf = s.get("refit")
            if rf:
                rr = rf.get("refits") or []
                rows.append(dict(model=s.get("model"), kind="refit", arch="topk", condition="standard",
                                 refit_n_cluster=rf.get("n_cluster"), refit_cont_mse=(rf.get("continuation") or {}).get("mse"),
                                 refit_rel_loss_diff=json.dumps([round(x["rel_loss_diff"], 4) for x in rr]),
                                 refit_within_tol=sum(bool(x["within_tolerance"]) for x in rr),
                                 refit_hungarian_cos=json.dumps([round(x["hungarian_cos_mean"], 3) for x in rr]),
                                 refit_span_overlap=json.dumps([round(x["span_overlap_mean_cos2"], 3) for x in rr]),
                                 refit_null=(rf.get("random_null_mean_cos2") or {}).get("expected")))
    return rows


def aux(raw):
    a1rows, a3rows = [], []
    for j in jobs(raw, "rx-aux-"):
        for f in j.rglob("a1.json"):
            a = rd(f)
            if not a:
                continue
            model = f.parent.name
            for t, tr in a["targets"].items():
                r = dict(model=model, target=t, n_test=a.get("n_test"),
                         nonfinite_test=((a.get("nonfinite_targets") or {}).get(t) or {}).get("test"))
                for fs in ("pc1", "pc2", "norm", "position", "pc1_removed", "raw"):
                    r["r2_" + fs] = (tr.get(fs) or {}).get("r2_mean")
                r["r2_random_raw_q95"] = (tr.get("random_raw_distribution") or {}).get("q95")
                r["r2_random_orth_q95"] = (tr.get("random_orth_distribution") or {}).get("q95")
                r["pc1_spline_r2"] = (tr.get("pc1") or {}).get("spline_r2")
                a1rows.append(r)
        for f in j.rglob("a3.json"):
            a = rd(f)
            if not a:
                continue
            r = dict(model=f.parent.name, **{k: v for k, v in a["rows"].items()})
            w = (a.get("writers") or {}).get("conditions") or {}
            r["writer_candidates"] = json.dumps((a.get("writers") or {}).get("candidates"))
            for k, v in w.items():
                r["W_%s_ce" % k] = v.get("ce")
                r["W_%s_pc1coef" % k] = v.get("site_pc1_abs_coef")
                r["W_%s_sink" % k] = v.get("sink_mass_block_b1")
            a3rows.append(r)
    return a1rows, a3rows


def probe(raw):
    rows = []
    for j in jobs(raw, "rx-probefit-"):
        for f in j.rglob("*.json"):
            p = rd(f)
            if not p or "tables" not in p:
                continue
            model, cond = f.parent.parent.name, f.parent.name
            for vname, v in p["tables"]["variants"].items():
                r = dict(model=model, condition=cond, task=p["task"], variant=vname,
                         dense_raw=v["baselines"]["dense_raw"]["acc"],
                         dense_pc1_removed=v["baselines"]["dense_pc1_removed"]["acc"],
                         scalar_pc1=v["baselines"]["scalar_pc1"]["acc"])
                for k in ("1", "20", "100"):
                    accs = [d["mi"][k]["acc"] for d in v["per_dict"].values() if k in d["mi"] and "acc" in d["mi"][k]]
                    rnd = [d["random_alive"][k]["mean"] for d in v["per_dict"].values() if k in d.get("random_alive", {})]
                    r["mi_k%s_mean" % k] = float(np.mean(accs)) if accs else None
                    r["mi_k%s_sd" % k] = float(np.std(accs)) if accs else None
                    r["random_k%s_mean" % k] = float(np.mean(rnd)) if rnd else None
                    pc = [d["mi_plus_pc1_scalar"][k]["acc"] for d in v["per_dict"].values() if "mi_plus_pc1_scalar" in d and k in d["mi_plus_pc1_scalar"]]
                    r["mi_plus_pc1_k%s_mean" % k] = float(np.mean(pc)) if pc else None
                r["n_dicts"] = len(v["per_dict"])
                rows.append(r)
            h = p.get("historical") or {}
            for dname, d in (h.get("per_dict") or {}).items():
                pass
    return rows


def auroc(pos, neg):
    if not pos or not neg:
        return None
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return float(gt)


def judge(raw, cache):
    rows = []
    cache.mkdir(parents=True, exist_ok=True)
    for j in jobs(raw, "rx-judge-"):
        jd = j / "judge"
        meta = rd(jd / "judge_meta.json") or {}
        for pair, explainer in (("mistral_judge_qwen.json", "qwen"), ("qwen_judge_mistral.json", "mistral")):
            sc = rd(jd / pair)
            ex = rd(jd / ("%s_explain.json" % explainer))
            if not sc:
                continue
            ctxs = {}
            for spec in meta.get("dicts", []):
                m, c, s = spec.split(":")
                cf = cache / ("%s_%s_s%s.json" % (m, c, s))
                if not cf.exists():
                    subprocess.run(["aws", "s3", "cp", "s3://%s/rx-shared/%s/contexts/%s_s%s.json" % (B, m, c, s), str(cf), "--quiet"])
                d = rd(cf)
                if d:
                    for r in d["features"]:
                        ctxs["%s|%d" % (spec, r["feature"])] = r
            for key, scores in sc.items():
                r = ctxs.get(key)
                if r is None:
                    continue
                kinds = [e.get("kind") for e in r["eval"]]
                if len(kinds) != len(scores):
                    continue
                pos = [s for s, k in zip(scores, kinds) if k == "positive"]
                rneg = [s for s, k in zip(scores, kinds) if k == "random_negative"]
                hneg = [s for s, k in zip(scores, kinds) if k == "hard_negative"]
                spec, feat = key.rsplit("|", 1)
                e = (ex or {}).get(key, {})
                rows.append(dict(dict=spec, feature=int(feat), explainer=explainer, judge=("mistral" if explainer == "qwen" else "qwen"),
                                 membership=",".join(r.get("membership", [])), dense_flag=r.get("dense_flag"), status=r.get("status"),
                                 n_pos=len(pos), auroc_random_neg=auroc(pos, rneg), auroc_hard_neg=auroc(pos, hneg),
                                 no_pattern=e.get("no_pattern"), parse_ok=e.get("parse_ok"),
                                 explanation=(e.get("explanation") or "")[:200]))
    return rows


def summarize_judge(rows):
    out = []
    groups = {}
    for r in rows:
        m, c, s = r["dict"].split(":")
        for memb in r["membership"].split(","):
            groups.setdefault((m, c, r["explainer"], memb), []).append(r)
    for (m, c, ex, memb), rs in sorted(groups.items()):
        a = [x["auroc_random_neg"] for x in rs if x["auroc_random_neg"] is not None]
        h = [x["auroc_hard_neg"] for x in rs if x["auroc_hard_neg"] is not None]
        out.append(dict(model=m, condition=c, explainer=ex, membership=memb, n=len(rs),
                        auroc_random_mean=float(np.mean(a)) if a else None, auroc_hard_mean=float(np.mean(h)) if h else None,
                        no_pattern_frac=float(np.mean([bool(x["no_pattern"]) for x in rs])),
                        dense_frac=float(np.mean([bool(x["dense_flag"]) for x in rs]))))
    return out


def main():
    root = Path(sys.argv[1])
    cache = Path(sys.argv[sys.argv.index("--contexts-cache") + 1]) if "--contexts-cache" in sys.argv else root.parent / "_contexts_cache"
    raw = root / "raw"
    outd = root / "analysis"
    outd.mkdir(exist_ok=True)
    mech, pat, tr = mechanism(raw), patching(raw), train(raw)
    a1, a3 = aux(raw)
    pr = probe(raw)
    jr = judge(raw, cache)
    js = summarize_judge(jr)
    for name, rows in (("mechanism", mech), ("patching", pat), ("train", tr), ("aux_a1", a1), ("aux_a3", a3), ("probe", pr),
                       ("judge_per_feature", jr), ("judge_summary", js)):
        write(rows, outd / ("%s.csv" % name))
    md = ["# RX reviewer-experiment summary tables (auto-generated)", ""]
    md += ["## Mechanism: cancellation (N2/S2) and group ablation dCE by component", "",
           md_table(mech, ["model", "dict", "n_dense", "dense_N2_over_S2", "ctl_count_N2_over_S2", "ctl_energy_N2_over_S2",
                           "dense_dCE_full", "dense_dCE_par", "dense_dCE_perp", "top50_dCE_full", "top50_dCE_par", "top50_dCE_perp",
                           "ctl_count_dCE_full"], f3), ""]
    md += ["## Paired patching", "", md_table(pat, ["model", "dict", "task", "status", "test_aligned", "native_test_eligible",
                                                   "spliced_acc", "ctl_full_residual_switch", "ctl_pc1_only_switch",
                                                   "ctl_none_switch", "ctl_perp_only_switch", "top50_valid_effect_mean",
                                                   "standard_b5d1_eligible_joint_success", "posthoc_b5d1_eligible_joint_success",
                                                   "bypass_b5d1_eligible_joint_success", "random_0_b5d1_eligible_joint_success"], f3),
           "Joint success = PRIMARY held-out test endpoint (eligible cohort, budget 5, dose 1). Discovery-set values are "
           "in patching.csv under *_discovery_* columns only.", ""]
    md += ["## Architecture / layer training (A4 / A2) and dense refit", "",
           md_table([r for r in tr if r.get("kind") != "refit"], ["model", "kind", "layer", "arch", "condition", "l0_mean", "l0_min",
                                                                   "l0_max", "l0_matched_pm5", "top50_shared", "dictwide", "top50_dense_frac"], f3), "",
           md_table([r for r in tr if r.get("kind") == "refit"], ["model", "refit_n_cluster", "refit_rel_loss_diff", "refit_within_tol",
                                                                  "refit_hungarian_cos", "refit_span_overlap", "refit_null"], f3), ""]
    md += ["## A1: what PC1 predicts (out-of-sample R2; random_q95 = 95th pct of 32 random 1-D projections)", "",
           md_table(a1, ["model", "target", "nonfinite_test", "r2_pc1", "pc1_spline_r2", "r2_pc2", "r2_norm", "r2_position",
                         "r2_random_raw_q95", "r2_random_orth_q95", "r2_pc1_removed", "r2_raw"], f3), ""]
    md += ["## A3: CE under deletions / replacements at the SAE site", "",
           md_table(a3, ["model", "native", "pc1_delete_centred", "pc1_coef_to_position_mean", "pc2_delete_centred",
                         "random_dir_delete_0", "splice_standard_s42", "splice_bypass_s42"], f3), ""]
    md += ["## Sparse probing (corrected protocol; mean over 5 seeds)", "",
           md_table(pr, ["model", "condition", "task", "variant", "dense_raw", "scalar_pc1", "mi_k1_mean", "mi_k20_mean", "mi_k100_mean",
                         "random_k20_mean", "mi_plus_pc1_k20_mean"], f3), ""]
    md += ["## Auto-interpretation (seed 42; AUROC of judge scores, positives vs negatives)", "",
           md_table(js, ["model", "condition", "explainer", "membership", "n", "auroc_random_mean", "auroc_hard_mean",
                         "no_pattern_frac", "dense_frac"], f3), ""]
    (outd / "RX_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", outd)


if __name__ == "__main__":
    main()
