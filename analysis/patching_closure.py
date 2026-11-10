"""Compare behavioral edits on the examples shared by both conditions."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

PACK = Path(sys.argv[1])
ITEMS = Path(sys.argv[2])
FEAT = Path(sys.argv[3])
OUT = Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
RAW = PACK / "raw"
B = 2000
rng = np.random.default_rng(20261008)


def write(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    tmp = OUT / (name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(OUT / name)


def dicts():
    """-> list of (model, cond, seed, task, json_path, job)"""
    out = []
    for f in sorted(RAW.glob("rx-wsx-*/wsx/*/*/patching/*.json")):
        if f.name == "PROVENANCE.json":
            continue
        model, cs = f.parts[-4], f.parts[-3]
        cond, seed = cs.rsplit("_s", 1)
        out.append((model, cond, int(seed), f.stem, f, f.parts[-6]))
    return out


def items_path(job, model, cond, seed, task):
    return ITEMS / job / "wsx" / model / ("%s_s%d" % (cond, seed)) / "patching" / ("%s_items.npz" % task)


primary, curves, ctrls, natives = [], [], [], []
recs = {}          # (model, task, cond, seed) -> dict(items, template, pools{name: joint indicator array})
for model, cond, seed, task, f, job in dicts():
    p = json.loads(f.read_text())
    if p.get("status") != "ok":
        primary.append(dict(model=model, task=task, condition=cond, seed=seed, status=p.get("status"),
                            why=json.dumps(p.get("why"))))
        continue
    nt = (p.get("native") or {}).get("test") or {}
    natives.append(dict(model=model, task=task, condition=cond, seed=seed, test_n=nt.get("n"),
                        test_eligible=nt.get("eligible"), donor_acc=nt.get("donor_acc"),
                        recipient_acc=nt.get("recipient_acc"), foil_acc=nt.get("foil_acc"),
                        spliced_recipient_acc_test=p.get("spliced_recipient_acc_test")))
    for pool, u in (p.get("usability") or {}).items():
        res = u.get("results") or {}
        tr = u.get("greedy_trace") or []
        for cohort in ("eligible", "unrestricted"):
            r = res.get("b5_d1_%s" % cohort) or {}
            primary.append(dict(model=model, task=task, condition=cond, seed=seed, pool=pool,
                                endpoint="PRIMARY test b5 d1", cohort=cohort, n=r.get("n"),
                                target_switch=r.get("target_switch"), foil_retention=r.get("foil_retention"),
                                joint_success=r.get("joint_success"),
                                mean_target_margin=r.get("mean_target_margin"),
                                mean_foil_margin=r.get("mean_foil_margin"),
                                mask_b5=json.dumps(u.get("mask_b5")),
                                discovery_b10_joint_LABELLED_DISCOVERY=tr[-1].get("joint") if tr else None,
                                discovery_b10_steps=tr[-1].get("step") if tr else None))
        for k, r in res.items():
            b, d, cohort = k.split("_")
            curves.append(dict(model=model, task=task, condition=cond, seed=seed, pool=pool,
                               budget=int(b[1:]), dose=float(d[1:]), cohort=cohort,
                               secondary=not (k == "b5_d1_eligible"), **{x: r.get(x) for x in (
                                   "n", "target_switch", "foil_retention", "joint_success",
                                   "mean_target_margin", "mean_foil_margin")}))
    c = p.get("controls") or {}
    row = dict(model=model, task=task, condition=cond, seed=seed)
    for k in ("none", "self", "full_residual", "pc1_only", "perp_only"):
        row["%s_switch" % k] = (c.get(k) or {}).get("target_switch")
        row["%s_mean_margin" % k] = (c.get(k) or {}).get("mean_margin")
        row["%s_n" % k] = (c.get(k) or {}).get("n")
    pc = c.get("pc1_coef_abs_diff_clean_vs_corrupt") or {}
    for k in ("mean", "median", "p95", "frac_zero"):
        row["pc1_coef_absdiff_%s" % k] = pc.get(k)
    ctrls.append(row)
    ip = items_path(job, model, cond, seed, task)
    if ip.exists():
        z = np.load(ip)
        pools = sorted({k[:-7] for k in z.files if k.endswith("_target")})
        recs[(model, task, cond, seed)] = dict(
            items=z["item"].astype(int), template=z["template"],
            joint={q: (z["%s_target" % q] > 0) & (z["%s_foil" % q] > 0) for q in pools},
            target={q: z["%s_target" % q] > 0 for q in pools})

write("patching_primary.csv", primary)
write("patching_secondary_curves.csv", curves)
write("patching_controls.csv", ctrls)
write("patching_native.csv", natives)

# ---- 1B: eligibility reconciliation across every dictionary of a (model, task) ----
elig_rows, paired = [], []
for (model, task) in sorted({(k[0], k[1]) for k in recs}):
    keys = sorted(k for k in recs if k[:2] == (model, task))
    sets = {k: set(recs[k]["items"].tolist()) for k in keys}
    inter = set.intersection(*sets.values())
    union = set.union(*sets.values())
    for k in keys:
        # cross-check: per-item record count equals the json's reported primary n
        prim = [r for r in primary if (r.get("model"), r.get("task"), r.get("condition"), r.get("seed"))
                == (model, task, k[2], k[3]) and r.get("cohort") == "eligible"]
        elig_rows.append(dict(model=model, task=task, condition=k[2], seed=k[3], n_eligible=len(sets[k]),
                              json_primary_n=prim[0]["n"] if prim else None,
                              n_common_all_dicts=len(inter), n_union=len(union),
                              not_in_common=len(sets[k] - inter),
                              missing_vs_union=json.dumps(sorted(union - sets[k])[:20])))
    # paired comparison on the common cohort: seed-matched standard vs bypass, each pool vs its
    # condition's primary pool; item-level bootstrap stratified by template family
    common_ix = np.array(sorted(inter))
    for seed in sorted({k[3] for k in keys}):
        S, Bp = recs.get((model, task, "standard", seed)), recs.get((model, task, "bypass", seed))
        if S is None or Bp is None:
            continue

        def on_common(R, pool, what="joint"):
            pos = {it: i for i, it in enumerate(R["items"])}
            return np.array([R[what][pool][pos[i]] for i in common_ix])
        tmpl = np.array([S["template"][list(S["items"]).index(i)] for i in common_ix])
        comps = [("bypass", "bypass", "standard", "standard")]
        if "posthoc" in S["joint"]:
            comps += [("bypass", "bypass", "standard", "posthoc"), ("standard", "posthoc", "standard", "standard")]
        for c1, p1, c2, p2 in comps:
            a = on_common(Bp if c1 == "bypass" else S, p1)
            b = on_common(Bp if c2 == "bypass" else S, p2)
            d = a.astype(float) - b.astype(float)
            boots = []
            groups = [np.nonzero(tmpl == t)[0] for t in np.unique(tmpl)]
            for _ in range(B):
                ix = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
                boots.append(d[ix].mean())
            lo, hi = np.quantile(boots, [0.025, 0.975])
            paired.append(dict(model=model, task=task, seed=seed, n_common=len(common_ix),
                               n_templates=len(groups), contrast="%s:%s - %s:%s" % (c1, p1, c2, p2),
                               joint_a=float(a.mean()), joint_b=float(b.mean()), diff=float(d.mean()),
                               ci95_lo=float(lo), ci95_hi=float(hi),
                               a_only=int((a & ~b).sum()), b_only=int((b & ~a).sum()),
                               note="item bootstrap stratified by template family (few templates: not a "
                                    "template-cluster bootstrap); one SAE seed per row"))
write("patching_eligibility.csv", elig_rows)
write("patching_paired_common_cohort.csv", paired)

# ---- patching-ranked top-50 sharing (seed-42 source dictionaries, primary protocol) ----
share_rows = []
for f in sorted(RAW.glob("rx-wsx-*/wsx/*/*_s42/patching/*_patch_scores.npz")):
    model, cs, task = f.parts[-4], f.parts[-3], f.name.split("_patch_scores")[0]
    cond = cs.rsplit("_s", 1)[0]
    pre = FEAT / model / ("shared_%s_s42.npz" % cond)
    if not pre.exists():
        continue
    P = np.load(pre)
    shared, alive_f = P["shared"], P["alive"].astype(bool)
    z = np.load(f)
    signed = np.asarray(z["signed"], float)
    alive = np.asarray(z["alive"], bool)
    assert alive.shape == alive_f.shape and bool((alive == alive_f).all()), (model, cond, "alive mismatch")
    for kind, sc in (("signed", signed), ("absolute", np.asarray(z["absolute"], float))):
        ok = alive & np.isfinite(sc)
        top = np.argsort(-np.where(ok, sc, -np.inf))[:50]
        share_rows.append(dict(model=model, task=task, condition=cond, ranking="patch_%s" % kind,
                               n_ranked=int(ok.sum()), n_alive=int(alive.sum()),
                               top50_shared=float(shared[top].mean()), dict_baseline=float(shared[alive].mean()),
                               top10_shared=float(shared[top[:10]].mean())))
write("patching_ranked_sharing.csv", share_rows)
print("primary rows", len(primary), "dicts with items", len(recs), "paired", len(paired), "sharing", len(share_rows))
