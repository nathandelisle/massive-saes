"""Build the Gated comparison tables from saved result JSONs."""
import csv
import glob
import json
import os
import subprocess

B = "s3://" + os.environ.get("S3_BUCKET", "")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
RAW = os.environ.get("GATED_RAW", os.path.join(ROOT, "results/additional/objectives/gated/raw"))
T = os.environ.get("GATED_OUT", os.path.join(ROOT, "generated/gated"))
os.makedirs(RAW, exist_ok=True)
os.makedirs(T, exist_ok=True)

PREFIXES = ["qwdiag", "qwrep-s42", "qwrep-s137", "qwrep-s0", "qwdiv-s0", "qwfix-s42", "qwfix-s137", "qwfix-s0",
            "g2meas", "g2alloc", "g2beta-0.03", "g2beta-0.1", "g2beta-0.3", "g2beta-1", "g2route-rb", "g2route-w",
            "p69g-c5-s42", "p69g-c6-s42", "p69g-c7-s42", "p69g-c7-s137", "p69g-c8-s42", "p69g-c8-s137", "p69g-c8-s0",
            "p69g-c8f-s42", "qwfix2-s42", "qwfix2-s137", "qwfix2-s0", "qwfix2lo-s0",
            "g2swap", "p14swap", "qwswap-i0d42", "qwswap-i42d0", "p69alloc", "p69b-s42", "p69b-s137", "p69b-s0",
            "g2lr", "p14lr"]
for p in (PREFIXES if os.environ.get("GATED_FETCH") == "1" and os.environ.get("S3_BUCKET") else []):
    subprocess.run(["aws", "s3", "cp", "--recursive", "--only-show-errors", f"{B}/{p}/results/gated_investigation/", os.path.join(RAW, p) + "/",
                    "--exclude", "*", "--include", "*.json"], check=False)
# Only result JSONs are downloaded; historical job logs are not needed.


if not glob.glob(os.path.join(RAW, "**", "*.json"), recursive=True):
    raise FileNotFoundError("No Gated raw results. Set GATED_RAW; saved tables are in results/additional/objectives/gated/.")

def J(p, name):
    f = os.path.join(RAW, p, name)
    return json.load(open(f)) if os.path.exists(f) else None


def write(name, rows):
    if not rows:
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(os.path.join(T, name), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


# ---- Pythia-6.9B calibration grid and cells
cal = []
for f in sorted(glob.glob(os.path.join(RAW, "p69g-*", "p69_c*_s*.json"))) + sorted(glob.glob(os.path.join(RAW, "p69_c*_s*.json"))):
    d = json.load(open(f))
    cal.append(dict(l1_coef=d["l1_coef"], seed=d["seed"], l0_exact=d["l0_exact"], l0_archived_counter=d["l0_archived_counter"],
                    n_alive=d["n_alive"], fvu_total=d["fvu_total"], fvu_content=d["fvu_content"], source=os.path.relpath(f, RAW)))
seen = set(); cal = [c for c in cal if not ((c["l1_coef"], c["seed"]) in seen or seen.add((c["l1_coef"], c["seed"])))]
write("pythia69_gated_calibration.csv", sorted(cal, key=lambda c: (c["l1_coef"], c["seed"])))
cells = []
for f in sorted(glob.glob(os.path.join(RAW, "p69g-*", "p69_c*_cell.json"))):
    d = json.load(open(f)); c = d["cell"]
    cells.append(dict(l1_coef=d["l1_coef"], seeds=d["seeds"], top50=c["top50_mean"], top50_sd_over_pairs=c["top50_std"], dictwide=c["dictwide_mean"],
                      top50_minus_dictwide=c["top50_mean"] - c["dictwide_mean"], l0_archived_counter_per_seed=c["l0_measured"],
                      l0_exact_per_seed=[p["l0_exact"] for p in d["per_seed_exact"]], n_alive_per_seed=[p["n_alive"] for p in d["per_seed_exact"]],
                      fvu_content_per_seed=[p["fvu_content"] for p in d["per_seed_exact"]], top50_dense_frac=c["top50_dense_frac"],
                      siblings_trained_on_cell_box=d.get("siblings_trained_here")))
write("pythia69_gated_cells.csv", cells)

# ---- Qwen
q = []
dg = J("qwdiag", "qwen_diag_archived.json")
if dg:
    for c in dg["checkpoints"]:
        q.append(dict(run="archived", seed=c["seed"], checkpoint=c["checkpoint"], sha256=c["checkpoint_sha256"], mapping_ok=c["mapping_ok"],
                      l0_exact=c["l0_exact"], l0_archived_counter=c["l0_archived_counter"], n_alive=c["n_alive"], n_never_fired=c["n_never_fired"],
                      fvu_total=c["fvu_total"], fvu_content=c["fvu_content"], heldout_fvu_content=c["heldout_fvu_content"],
                      decoder_rows_near_init=c["init_similarity"]["decoder"]["frac_rows_cos_gt_0p9"], gates_ever_open=c["frac_features_gate_ever_open"],
                      aux_mse_per_token=c["gate_totals"]["aux_mse_per_token"], mse_per_token=c["gate_totals"]["mse_per_token"],
                      train_log_tail=c["train_log_tail"]))
    write("qwen_pairwise_archived.csv", dg["pairwise"])
for s in (42, 137, 0):
    r = J(f"qwrep-s{s}", f"qwen_rep_s{s}.json")
    if r:
        q.append(dict(run="reproduced_original_config", seed=s, l0_exact=r["l0_exact"], n_alive=r["n_alive"], n_never_fired=r["n_never_fired"],
                      fvu_total=r["fvu_total"], fvu_content=r["fvu_content"], heldout_fvu_content=r["heldout_fvu_content"],
                      decoder_rows_near_init=r["init_similarity"]["decoder"]["frac_rows_cos_gt_0p9"], gates_ever_open=r["frac_features_gate_ever_open"],
                      aux_mse_per_token=r["gate_totals"]["aux_mse_per_token"], mse_per_token=r["gate_totals"]["mse_per_token"],
                      bitwise_equal_to_archived=r["vs_archived_same_seed"]["bitwise_equal"], vs_archived_ref42=r.get("archived_ref42_vs_this")))
for pre, role in (("qwfix", "fix"), ("qwfix2", "fix2")):
    for s in (42, 137, 0):
        r = J(f"{pre}-s{s}", f"qwen_{role}_s{s}.json")
        if r:
            q.append(dict(run=f"corrected_{role}", seed=s, protocol=r.get("protocol"), clip_events=r.get("clip_events"), l0_exact=r["l0_exact"],
                          n_alive=r["n_alive"], n_never_fired=r["n_never_fired"], fvu_total=r["fvu_total"], fvu_content=r["fvu_content"],
                          heldout_fvu_content=r["heldout_fvu_content"], decoder_rows_near_init=r["init_similarity"]["decoder"]["frac_rows_cos_gt_0p9"],
                          gates_ever_open=r["frac_features_gate_ever_open"], aux_mse_per_token=r["gate_totals"]["aux_mse_per_token"],
                          mse_per_token=r["gate_totals"]["mse_per_token"]))
r = J("qwfix2lo-s0", "qwen_fix2_lr5e-05_s0.json")
if r:
    q.append(dict(run="corrected_fix2_lr5e-05_seed0_only", seed=0, protocol=r.get("protocol"), clip_events=r.get("clip_events"), l0_exact=r["l0_exact"],
                  n_alive=r["n_alive"], n_never_fired=r["n_never_fired"], fvu_total=r["fvu_total"], fvu_content=r["fvu_content"],
                  heldout_fvu_content=r["heldout_fvu_content"], decoder_rows_near_init=r["init_similarity"]["decoder"]["frac_rows_cos_gt_0p9"],
                  gates_ever_open=r["frac_features_gate_ever_open"], aux_mse_per_token=r["gate_totals"]["aux_mse_per_token"],
                  mse_per_token=r["gate_totals"]["mse_per_token"]))
write("qwen_gated_runs.csv", q)
dv = J("qwdiv-s0", "qwen_diag0_s0.json")
if dv:
    write("qwen_seed0_divergence_every100.csv", [{k: v for k, v in r.items() if k != "grad_by_param"} | {f"grad_{k}": v for k, v in r["grad_by_param"].items()}
                                                 for r in dv["diag_every_100_steps"]])
qc = []
for p, name, label in (("qwrep-s42", "qwen_rep_cell.json", "reproduced_original"), ("qwfix-s42", "qwen_fix_cell.json", "corrected_fix"),
                       ("qwfix2-s42", "qwen_fix2_cell.json", "corrected_fix2")):
    d = J(p, name)
    if d:
        c = d["cell"]
        qc.append(dict(row=label, top50=c["top50_mean"], top50_sd_over_pairs=c["top50_std"], dictwide=c["dictwide_mean"], l0_archived_counter=c["l0_measured"],
                       top50_dense_frac=c["top50_dense_frac"]))
write("qwen_gated_cells.csv", qc)

# ---- GPT-2 / Pythia-1.4B Gated mechanism
m = J("g2meas", "gpt2_gated_meas.json")
rows, cls_rows = [], []
if m:
    for model, rec in m["models"].items():
        for arch in ("gated", "relu_l1", "topk"):
            for r in rec.get(arch, []):
                row = dict(model=model, arch=arch, seed=r["seed"], l0_exact=r["l0_exact"], l0_archived_counter=r["l0_archived_counter"], n_alive=r["n_alive"],
                           fvu_total=r["fvu_total"], fvu_content=r["fvu_content"], heldout_fvu_content=r["heldout_fvu_content"],
                           decoder_norm_max_dev=r["decoder_norms"]["max_abs_dev_from_1"], seed_in_checkpoint=r["seed_in_checkpoint"])
                if arch == "gated":
                    row.update({f"gate_{k}": v for k, v in r["gate_totals"].items()})
                    for cname, c in r["by_class"].items():
                        cls_rows.append(dict(model=model, seed=r["seed"], group=cname, **c))
                if arch == "relu_l1":
                    row.update({f"relu_{k}": v for k, v in r["loss_split"].items()})
                rows.append(row)
        rows.append(dict(model=model, arch="activation_scale", **rec["scale"]))
    write("gated_checkpoint_measurements.csv", rows)
    write("gated_gate_vs_coefficient_by_class.csv", cls_rows)
    gf = []
    for model, rec in m["models"].items():
        for t, g in rec["gated"][0].get("grad_flow", {}).items():
            gf.append(dict(model=model, loss_term=t, **g))
    write("gated_gradient_flow.csv", gf)
beta = []
for p, b in (("g2beta-0.03", "0.03"), ("g2beta-0.1", "0.1"), ("g2beta-0.3", "0.3"), ("g2beta-1", "1")):
    d = J(p, f"gpt2_gated_magl1_beta{b}.json")
    if d:
        c = d["cell"]; ps = d["per_seed"]; comp = d["composition"]
        beta.append(dict(beta=float(b), top50=c["top50_mean"], top50_sd_over_pairs=c["top50_std"], dictwide=c["dictwide_mean"],
                         l0_exact_mean=sum(x["l0_exact"] for x in ps) / len(ps), n_alive_mean=sum(x["n_alive"] for x in ps) / len(ps),
                         heldout_fvu_content_mean=sum(x["heldout_fvu_content"] for x in ps) / len(ps),
                         **{f"comp_{k}": comp[k] for k in comp}, top50_vs_archived_gated_s42=ps[0].get("top50_vs_archived_gated_s42"),
                         per_seed_l0=[x["l0_exact"] for x in ps], per_seed_alive=[x["n_alive"] for x in ps],
                         per_seed_heldout_content=[x["heldout_fvu_content"] for x in ps]))
write("gpt2_gated_magl1_sweep.csv", beta)
route = []
for p, ar in (("g2route-rb", "gated_mag_rb"), ("g2route-w", "gated_mag_w")):
    d = J(p, f"gpt2_{ar}_beta0.1.json")
    if d:
        c = d["cell"]; ps = d["per_seed"]
        route.append(dict(variant=ar, beta=d["beta"], top50=c["top50_mean"], top50_sd_over_pairs=c["top50_std"], dictwide=c["dictwide_mean"],
                          l0_exact_mean=sum(x["l0_exact"] for x in ps) / len(ps), n_alive_mean=sum(x["n_alive"] for x in ps) / len(ps),
                          heldout_fvu_content_mean=sum(x["heldout_fvu_content"] for x in ps) / len(ps),
                          massive_atoms_for_90pct_pc1_s42=ps[0]["massive_rows"]["massive_atoms_for_90pct_pc1_median"],
                          massive_total_l1_s42=ps[0]["massive_rows"]["massive_total_l1_mean"], top50_frac_massive_specific=ps[0].get("top50_frac_massive_specific"),
                          **{f"comp_{k}": v for k, v in d["composition"].items()}))
write("gpt2_route_test.csv", route)
al = J("g2alloc", "alloc.json")
if al:
    ar = []
    for model, res in al.items():
        for lab, e in res["models"].items():
            for r in e["per_seed"]:
                ms = [x["massive_specific_sharing"] for x in e["massive_specific_pairwise"] if x["source"] == r["seed"]]
                ar.append(dict(model=model, objective=lab, seed=r["seed"], massive_row_fraction=res["massive_row_fraction"],
                               massive_share_of_pc1_energy=res["massive_share_of_pc1_energy"],
                               **{k: v for k, v in r.items() if k != "seed"}, massive_specific_sharing_vs_others=ms,
                               top50_frac_massive_specific_s42=e.get("top50_frac_massive_specific_s42") if r["seed"] == 42 else None))
    write("massive_row_allocation.csv", ar)

# ---- seed swap (init seed x data-order seed) and massive-row geometry
sw, geo = [], []
for p in ("g2swap", "p14swap", "qwswap-i0d42", "qwswap-i42d0"):
    for f in sorted(glob.glob(os.path.join(RAW, p, "seedswap_*.json"))):
        d = json.load(open(f))
        geo.append(dict(model=d["model"], source=p, **d["massive_geometry"]))
        for r in d["runs"]:
            mr = r["massive_rows"]
            sw.append(dict(model=d["model"], init_seed=r["init_seed"], data_seed=r["data_seed"], source=r["source"], l0_exact=r["l0_exact"],
                           n_alive=r["n_alive"], n_never_fired=r.get("n_never_fired"), fvu_total=r["fvu_total"], fvu_content=r["fvu_content"],
                           heldout_fvu_content=r["heldout_fvu_content"], massive_atoms_for_90pct_pc1=mr["massive_atoms_for_90pct_pc1_median"],
                           massive_total_l1=mr["massive_total_l1_mean"], massive_band_active=mr["massive_band_active_atoms_mean"],
                           massive_content_active=mr["massive_content_active_atoms_mean"], n_massive_specific=mr["n_massive_specific_atoms"]))
write("seed_swap.csv", sw)
pa = J("p69alloc", "p69_alloc.json")
if pa:
    geo.append(dict(model="pythia-6.9b", source="p69alloc", **pa["massive_geometry"]))
write("massive_row_geometry.csv", geo)
p69 = []
if pa:
    for lab, e in pa["objectives"].items():
        for r in e["per_seed"]:
            ms = [x["massive_specific_sharing"] for x in e["massive_specific_pairwise"] if x["source"] == r["seed"]]
            p69.append(dict(objective=lab, seed=r["seed"], l0_exact=r["l0_exact"], n_alive=r["n_alive"], fvu_content=r["fvu_content"],
                            **{k: v for k, v in r.items() if k.startswith("massive_") or k.startswith("n_massive")},
                            massive_specific_sharing_vs_others=ms, top50_frac_massive_specific_s42=e.get("top50_frac_massive_specific_s42") if r["seed"] == 42 else None,
                            **({f"comp_{k}": v for k, v in e.get("composition", {}).items()} if r["seed"] == 42 else {})))
pb = J("p69b-s42", "p69_magl1_beta0.427.json")
if pb:
    for r in pb["per_seed"]:
        ms = [x["massive_specific_sharing"] for x in pb["massive_specific_pairwise"] if x["source"] == r["seed"]]
        p69.append(dict(objective="gated_magl1_beta0.427", seed=r["seed"], l0_exact=r["l0_exact"], n_alive=r["n_alive"], fvu_content=r["fvu_content"],
                        heldout_fvu_content=r["heldout_fvu_content"], **{k: v for k, v in r["massive_rows"].items()},
                        massive_specific_sharing_vs_others=ms, top50_frac_massive_specific_s42=r.get("top50_frac_massive_specific"),
                        top50_vs_plain_gated_c8_s42=r.get("top50_vs_plain_gated_c8_s42"),
                        **({f"comp_{k}": v for k, v in pb["composition"].items()} if r["seed"] == 42 else {}),
                        **({"cell_top50": pb["cell"]["top50_mean"], "cell_top50_sd_over_pairs": pb["cell"]["top50_std"], "cell_dictwide": pb["cell"]["dictwide_mean"]}
                           if r["seed"] == 42 else {})))
write("pythia69_mechanism.csv", p69)

# ---- lower-lr Gated cells on GPT-2 / Pythia-1.4B, and the seed-0 traces under the original trainer
lr_rows, lr_trace = [], []
for p in ("g2lr", "p14lr"):
    for f in sorted(glob.glob(os.path.join(RAW, p, "lrcell_*.json"))):
        d = json.load(open(f))
        c = d.get("cell") or {}
        for r in d["runs"]:
            mr = r["massive_rows"]
            lr_rows.append(dict(model=d["model"], arch=r["arch"], seed=r["seed"], lr=r["lr"], l0_exact=r["l0_exact"], n_alive=r["n_alive"],
                                fvu_content=r["fvu_content"], heldout_fvu_content=r["heldout_fvu_content"],
                                aux_mse_per_token=r["gate_totals"]["aux_mse_per_token"], mse_per_token=r["gate_totals"]["mse_per_token"],
                                massive_atoms_for_90pct_pc1=mr["massive_atoms_for_90pct_pc1_median"], massive_total_l1=mr["massive_total_l1_mean"],
                                max_grad_norm=max((x["grad_norm"] for x in r.get("diag_every_100_steps") or []), default=None),
                                cell_top50=c.get("top50_mean"), cell_top50_sd=c.get("top50_std"), cell_dictwide=c.get("dictwide_mean"),
                                **({f"comp_{k}": v for k, v in (d.get("composition") or {}).items()} if r["seed"] == 42 and r["arch"] == "gated_lr" else {})))
            for x in r.get("diag_every_100_steps") or []:
                lr_trace.append(dict(model=d["model"], arch=r["arch"], seed=r["seed"], lr=r["lr"], **{k: v for k, v in x.items() if k != "grad_by_param"},
                                     **{f"grad_{k}": v for k, v in x["grad_by_param"].items()}))
write("lr_cells_gpt2_pythia14.csv", lr_rows)
write("lr_cells_traces_every100.csv", lr_trace)
print("tables:", sorted(os.listdir(T)))
