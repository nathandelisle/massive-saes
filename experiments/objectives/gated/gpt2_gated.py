"""Why does Gated stay confounded on GPT-2 (top-50 sharing 0.11) when it is reproducible on Pythia-1.4B (0.86)?

--role meas : existing checkpoints only (plus one control retrain).
    GPT-2 Gated (l1 5.62397, seeds 42/137/0), ReLU+l1 (5.06247), TopK (k=64); Pythia-1.4B Gated (3.32599, 5 seeds),
    ReLU+l1 (3.18648). For each: checkpoint/config consistency, training-log tail, exact L0, alive, store and
    held-out FVU. For Gated additionally: gate (penalized) vs reconstruction coefficient per feature, exp(r_mag),
    b_enc, b_gate, the implied L1 price per unit of reconstruction coefficient, loss split, by PC1-alignment class
    and for the top 50; gradient flow of each loss term. Activation scale and effective penalty strength per model.
    Control: retrain GPT-2 Gated seed 42 on this box (beta = 0) and compare with the archived checkpoint.
--role beta --beta B : the controlled test. Gated + B * sum|z| on the actual reconstruction coefficients, gate L1
    (5.62397) and auxiliary loss retained, same seeds / init / data order / schedule. Runs the ORIGINAL run_cell
    (reference ablation, top-50, matching), then exact L0, alive, held-out content FVU, gate statistics, top-50
    composition by PC1 band, and whether the top-50 features are the same objects as in the archived Gated run.
Pre-registered sweep: B in {0.03, 0.1, 0.3, 1.0} (the Section 5.1 grid plus 1.0).
"""
import argparse
import os
import shutil
import time

import numpy as np
import torch

import gi_common as G
from gi_common import io, registry

import sys
sys.path.insert(0, G.HERE)
import exp_b1_architecture_law as B1                                     # noqa: E402
from v3.common.data import load_model_and_tokenizer, eval_sequences     # noqa: E402
from v3.common.fastabl import tuned_chunk                               # noqa: E402

GPT2, P14 = registry.MODELS["gpt2"], registry.MODELS["pythia-1.4b"]
GATED = {"gpt2": 5.62397, "pythia-1.4b": 3.32599}
RELU = {"gpt2": 5.06247, "pythia-1.4b": 3.18648}
PREFIX = {"gpt2": "b1v2-gpt2", "pythia-1.4b": "b1v2-p14b"}
SEEDS = {"gpt2": [42, 137, 0], "pythia-1.4b": [42, 137, 0, 1, 2]}


def band_comp(npz_path):
    z = np.load(npz_path)
    cos2, alive, imp, mc = z["pc1_cos2"], z["alive"], z["imp_zero"].astype(float), z["max_cos"]
    imp[~alive] = -np.inf
    top = np.argsort(imp)[::-1][:50]
    cl = G.classes(cos2)
    shared = mc >= G.THR
    out = dict(top50=float(shared[:, top].mean()), dictwide=float(shared[:, alive].mean()))
    for c in ("canonical", "band", "content"):
        m = cl[top] == c
        out[f"top50_frac_{c}"] = float(m.mean())
        out[f"top50_sharing_{c}"] = float(shared[:, top][:, m].mean()) if m.any() else None
        am = alive & (cl == c)
        out[f"n_{c}"] = int(am.sum())
        out[f"sharing_{c}"] = float(shared[:, am].mean()) if am.any() else None
    return out, top


def scale_stats(st):
    x = np.asarray(st["acts"][:G.N_EXACT], dtype=np.float32) - st["mean"].astype(np.float32)
    n2 = (x.astype(np.float64) ** 2).sum(1)
    a = x @ st["v1"].astype(np.float32)
    return dict(rms_norm=float(np.sqrt(n2.mean())), median_norm=float(np.median(np.sqrt(n2))),
                pc1_coef_rms=float(np.sqrt((a.astype(np.float64) ** 2).mean())), pc1_coef_abs_p99=float(np.quantile(np.abs(a), 0.99)),
                content_rms=float(np.sqrt((n2 - a.astype(np.float64) ** 2).mean())), w1=st["w1"])


def meas():
    out = dict(role="meas", models={})
    for spec in (GPT2, P14):
        m = spec.key
        t0 = time.time()
        st = G.store(spec)
        X = G.harvest_split(spec, "test", 200_000)
        print(f"[{G.now()}] {m} store + held-out ready ({(time.time()-t0)/60:.1f}m)", flush=True)
        rec = dict(scale=scale_stats(st), gated=[], relu_l1=[], topk=[])
        comp, top_g = band_comp(os.path.join(G.ASSETS, f"feat_{m}_gated_standard.npz"))
        rec["gated_archived_cell_composition"] = comp
        rec["relu_l1_archived_cell_composition"] = band_comp(os.path.join(G.ASSETS, f"feat_{m}_relu_l1_standard.npz"))[0]
        arch_dir = os.path.join(io.ROOT, "archived_saes")
        for arch, coef in (("gated", GATED[m]), ("relu_l1", RELU[m]), ("topk", None)):
            if arch == "topk" and m != "gpt2":
                continue
            for s in SEEDS[m]:
                kw = dict(k=64) if arch == "topk" else dict(l1_coef=coef)
                h = G.cache_name(spec, s, arch, **kw)
                p = G.fetch(f"{PREFIX[m]}/saes/{h}", os.path.join(arch_dir, h))
                sae, extra, cfg = G.load_ckpt(p)
                fid, rate = G.store_fidelity(sae, st)
                r = dict(seed=s, checkpoint=h, sha256=G.sha256(p), cfg_in_checkpoint=cfg,
                         seed_in_checkpoint=(extra.get("cfg") or {}).get("seed"), train_log_tail=G.log_tail(extra),
                         decoder_norms=G.decoder_norms(sae), l0_archived_counter=G.archived_l0(sae, st)[0], **fid,
                         **G.heldout_fidelity(sae, X, st["mean"], st["v1"]))
                cos2 = (G.dec_np(sae) @ st["v1"]) ** 2
                alive = rate > registry.ALIVE_FIRING_RATE
                if arch == "gated":
                    per, tot = G.gate_stats(sae, st)
                    r["gate_totals"] = tot
                    r["by_class"] = G.summarize_per_feature(per, cos2, alive, top_g if s == 42 else None)
                    if s == 42:
                        r["grad_flow"] = G.grad_flow(sae, st)
                        r["top50_features"] = [dict(id=int(f), cos2=float(cos2[f]), **{k: (float(v[f]) if np.isfinite(v[f]) else None)
                                                                                     for k, v in per.items()}) for f in top_g]
                elif arch == "relu_l1":
                    r["loss_split"] = G.l1_price_relu(sae, st)
                rec[arch].append(r)
                print(f"[{G.now()}] {m} {arch} s{s}: L0 {fid['l0_exact']:.2f} alive {fid['n_alive']} fvu {fid['fvu_total']:.5f} "
                      f"content {fid['fvu_content']:.4f} held-out content {r['heldout_fvu_content']:.4f}"
                      + (f" | price/unit z {r['gate_totals']['aggregate_price_per_unit_z']:.3f} pen/mse {r['gate_totals']['penalty_over_mse']:.3f}"
                         if arch == "gated" else ""), flush=True)
                del sae
                torch.cuda.empty_cache()
        if m == "gpt2":
            # control: retrain Gated seed 42 here (beta 0) and compare with the archived checkpoint
            sae, _v, _mm, _a, extra = io.get_sae(spec, 42, arch="gated", condition="standard", device="cuda", l1_coef=GATED[m])
            new = G.dec_np(sae)
            fid, _ = G.store_fidelity(sae, st)
            del sae
            h = G.cache_name(spec, 42, "gated", l1_coef=GATED[m])
            old_sae, _e, _c = G.load_ckpt(os.path.join(arch_dir, h))
            old = G.dec_np(old_sae)
            del old_sae
            rc = (new * old).sum(1)
            rec["control_retrain_s42"] = dict(**fid, median_rowwise_cos_vs_archived=float(np.median(rc)),
                                              frac_rows_cos_gt_0p99=float((rc > 0.99).mean()),
                                              maxcos_match_frac=float((G.max_cos(new, old) >= G.THR).mean()))
        out["models"][m] = rec
        G.dump(out, "gpt2_gated_meas.json")
        del st
    return out


def beta_run(beta):
    spec = GPT2
    t0 = time.time()
    st = G.store(spec)
    X = G.harvest_split(spec, "test", 200_000)
    args = __import__("argparse").Namespace(
        models=["gpt2"], archs=["gated_magl1"], eval_seqs=registry.EVAL_SEQS_IMPORTANCE, topk=50, chunk=0, screen=1500,
        knob_min=1e-6, knob_max=1e5, recal_passes=0, target_l0=64.0, l0_tol=0.25, cal_steps=4000, force=True,
        seeds=[42, 137, 0], allow_uncalibrated=False, device="cuda", progress=500)
    model, tok = load_model_and_tokenizer(spec, device="cuda")
    seqs, am = eval_sequences(tok, n_seqs=args.eval_seqs)
    args.chunk = tuned_chunk(spec, vocab_size=int(model.config.vocab_size))
    akw = dict(l1_coef=GATED["gpt2"], beta=float(beta))
    cell = B1.run_cell(model, spec, seqs, st["adir"], st["evecs"], "gated_magl1", akw, "standard", args, am=am)
    npz = io.path("results", "gated_investigation", f"feat_gpt2_gated_magl1_beta{beta:g}.npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    shutil.copy(io.path("results", B1.EXP, "feat_gpt2_gated_magl1_standard.npz"), npz)
    comp, top = band_comp(npz)
    # are the top-50 features the same objects as the archived Gated run's top 50 (same seed 42)?
    h = G.cache_name(spec, 42, "gated", l1_coef=GATED["gpt2"])
    old_sae, _e, _c = G.load_ckpt(G.fetch(f"b1v2-gpt2/saes/{h}", os.path.join(io.ROOT, "archived_saes", h)))
    old = G.dec_np(old_sae)
    del old_sae
    _comp_old, top_old = band_comp(os.path.join(G.ASSETS, "feat_gpt2_gated_standard.npz"))
    per_seed = []
    for s in args.seeds:
        sae, _v, _m, _a, extra = io.get_sae(spec, s, arch="gated_magl1", condition="standard", device="cuda", **akw)
        fid, rate = G.store_fidelity(sae, st)
        r = dict(seed=s, **fid, **G.heldout_fidelity(sae, X, st["mean"], st["v1"]), train_log_tail=G.log_tail(extra))
        per, tot = G.gate_stats(sae, st)
        r["gate_totals"] = tot
        cos2 = (G.dec_np(sae) @ st["v1"]) ** 2
        r["by_class"] = G.summarize_per_feature(per, cos2, rate > registry.ALIVE_FIRING_RATE, top if s == 42 else None)
        if s == 42:
            new = G.dec_np(sae)
            mc_old = G.max_cos(new[top], old)                      # best match of each new top-50 atom in the old dictionary
            arg_old = np.argmax((new[top] / np.linalg.norm(new[top], axis=1, keepdims=True)) @
                                (old / np.linalg.norm(old, axis=1, keepdims=True)).T, axis=1)
            r["top50_vs_archived_gated_s42"] = dict(
                frac_matched_in_old_dictionary=float((mc_old >= G.THR).mean()),
                frac_matched_to_old_top50=float(np.mean([(m >= G.THR) and (a in set(top_old.tolist())) for m, a in zip(mc_old, arg_old)])))
        per_seed.append(r)
        del sae
        torch.cuda.empty_cache()
    G.dump(dict(role="beta", beta=beta, l1_coef=GATED["gpt2"], seeds=args.seeds, cell=cell, composition=comp,
                per_seed=per_seed, minutes=(time.time() - t0) / 60), f"gpt2_gated_magl1_beta{beta:g}.json")
    print(f"[{G.now()}] beta {beta:g}: top-50 {cell['top50_mean']:.3f} dict-wide {cell['dictwide_mean']:.3f} "
          f"L0 exact {np.mean([p['l0_exact'] for p in per_seed]):.2f} alive {np.mean([p['n_alive'] for p in per_seed]):.0f} "
          f"held-out content {np.mean([p['heldout_fvu_content'] for p in per_seed]):.4f} | {comp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["meas", "beta"], required=True)
    ap.add_argument("--beta", type=float, default=0.0)
    a = ap.parse_args()
    if a.role == "meas":
        meas()
    else:
        beta_run(a.beta)


if __name__ == "__main__":
    main()
