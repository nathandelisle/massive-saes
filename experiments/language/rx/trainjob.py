"""SAE-training workstreams, on the v3 protocol (same store, trainer, eval sequences and
zero-ablation code that produced the archived grid), so new cells are comparable to old ones.

  --kind a4   --arch {topk,batchtopk,relu_l1}   canonical middle site, 5 seeds x 2 conditions.
              relu_l1: l1 calibrated on training-store data only (short probes, then full-length
              passes on the reference seed) to measured L0 = 64 +/- 5%; every calibration point is
              published. A cell outside tolerance is recorded as UNMATCHED, never as matched.
              topk also runs the dense-cluster refit (below).
  --kind a2   --site {pre,post}                 the pre-rise / post-fall block chosen in prep from
              the CALIBRATION geometry profile (before any SAE sharing was looked at); TopK
              standard + bypass x 5 seeds; zero-ablation inversion and sharing.

Every seed is ranked by its OWN zero-ablation (v3 eval sequences). Sharing is reported for every
source seed against the other four, and the mean over sources. Each seed's ranking and decoder
statistics are checkpointed, so a relaunched box resumes.

Dense-cluster refit (a4/topk, standard s42): the cluster = alive atoms with cos^2(d_j, v1) > 5x the
alive median. Everything outside the cluster is frozen (all other encoder rows, encoder biases,
decoder rows, and b_dec). Free: the cluster's encoder rows, encoder biases and decoder rows.
Five fresh initialisations (the trainer's own init distribution, seeds 1000..1004) plus one
continuation from the trained values, all for REFIT_STEPS on the same batches. Predeclared
practical tolerance: final held-out MSE within 1% of the continuation run. Reported: final
loss, decoder agreement (max-cos and Hungarian), aggregate cluster output agreement, effective
ranks, principal-angle overlap of the spans, and a matched-dimension random-subspace null
(dimension-specific: E[mean cos^2] = k/d, plus 50 empirical draws).
"""
import argparse
import json
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from . import common, modelio as M
from .common import log

REFIT_STEPS = 5000
REFIT_TOL = 0.01
SEEDS = [42, 137, 0, 1, 2]


def _v3():
    sys.path.insert(0, str(common.PKG / "pc1_v3" / "scripts"))
    from v3.common import registry, io_utils as io
    from v3.common.data import load_model_and_tokenizer, eval_sequences
    from v3.common.fastabl import (ablation_over, screen_candidates, z_stats, to_array,
                                   dense_share, tuned_chunk)
    from v3.common.matching import pairwise_over_seeds, max_cos
    from v3.common.pca import estimate_pcs, dense_cluster
    from v3.common.saes import decoder_np
    import exp_b1_architecture_law as b1
    return types.SimpleNamespace(**locals())


def fetch_archived_saes(v, spec, arch, layer, akw):
    """Pre-seed the v3 SAE cache with archived checkpoints whose config hash matches what
    get_sae would compute, so archived dictionaries are reused byte-for-byte, not retrained."""
    inv = json.loads((common.PKG / "inventory.json").read_text())
    by_hash = {r["config_hash"]: r for r in inv["rows"] if r["model"] == spec.key}
    got = []
    for cond in ("standard", "bypass"):
        for s in SEEDS:
            p = v.io.sae_cache_path(spec, s, arch=arch, condition=cond, layer=layer, **akw)
            h = p.replace("\\", "/").rsplit("/", 1)[1][:-3]
            r = by_hash.get(h)
            if r is None or common.pathlib.Path(p).exists():
                continue
            src = r.get("rx_prefix") or r["prefixes"][0]
            common.s3_get("%s/saes/%s.pt" % (src, h), common.pathlib.Path(p), required=False)
            if common.pathlib.Path(p).exists():
                got.append((cond, s, h))
    return got


def cell(v, model, spec, seqs, am, arch, cond, akw, layer, screen, chunk, out, device):
    """One (arch, condition) cell: 5 seeds trained-or-reused, each ranked by its own zero
    ablation; sharing for every source seed."""
    decs, alives, imps, l0s, fvus, per = [], [], [], [], [], []
    v1_ref = None
    for s in SEEDS:
        f = out / ("%s_%s_s%d.npz" % (arch, cond, s))
        sae, v1, mean, adir, extra = v.io.get_sae(spec, s, arch=arch, condition=cond, layer=layer,
                                                  device=device, **akw)
        v1_ref = v1 if v1_ref is None else v1_ref
        if common.fresh(f.parent, f.name, "train"):
            z = np.load(f)
            dec, al, imp, l0 = z["dec"], z["alive"], z["imp"], float(z["l0"])
        else:
            rates, al = v.io.alive(sae, adir, condition=cond, v1=v1, device=device)
            l0 = float(rates.sum())
            dec = v.decoder_np(sae)
            st = v.z_stats(model, spec, seqs, sae, mean, v1=v1, mode=cond, layer=layer,
                           chunk=chunk, am=am)
            fires = (st["rate"] > 0) & al
            cand = v.screen_candidates(st["ez2"], fires, screen) if screen else np.nonzero(fires)[0]
            deltas, base = v.ablation_over(model, spec, seqs, sae, mean, cand.tolist(), v1=v1,
                                           mode=cond, metric="zero", layer=layer, chunk=chunk,
                                           stats=st, am=am)
            imp = v.to_array(sae.d_sae, deltas)
            common.save_npz(f, dec=dec.astype(np.float32), alive=al, imp=imp, l0=l0,
                            rate=rates, n_cand=len(cand), base_ce=float(base) if np.isscalar(base) else 0.0)
            common.pin(f.parent, f.name, "train")
        fvu = (extra.get("log") or [{}])[-1].get("fvu") if isinstance(extra, dict) else None
        decs.append(dec); alives.append(al); imps.append(imp); l0s.append(l0); fvus.append(fvu)
        del sae
        torch.cuda.empty_cache()
    by_source = []
    for i in range(len(SEEDS)):
        st = v.pairwise_over_seeds(decs, alives, imps, ref_seed_pos=i, k=50)
        dmask, cos2, _ = v.dense_cluster(decs[i], v1_ref, alives[i])
        top = np.argsort(-np.where(alives[i] & np.isfinite(imps[i]), imps[i], -np.inf))[:50]
        ds = v.dense_share(imps[i], dmask, alives[i])
        by_source.append(dict(seed=SEEDS[i], top50_shared_mean=st["top_k_mean"],
                              top50_shared_std=st["top_k_std"], dictwide_mean=st["dict_wide_mean"],
                              top50_dense_frac=float(dmask[top].mean()),
                              dense_importance_share=ds["positive"], n_dense=int(dmask.sum()),
                              n_alive=int(alives[i].sum())))
    lo, hi = 64 * 0.95, 64 * 1.05
    return dict(arch=arch, condition=cond, layer=layer, kwargs=akw, screen=int(screen),
                l0_measured=l0s, l0_matched_pm5=bool(all(lo <= x <= hi for x in l0s)),
                fvu_final_log=fvus, by_source=by_source,
                top50_shared_mean_over_sources=float(np.mean([b["top50_shared_mean"] for b in by_source])),
                top50_shared_sd_over_sources=float(np.std([b["top50_shared_mean"] for b in by_source])))


def calibrate_relu(v, spec, adir, cond, v1, device, out):
    f = out / ("cal_relu_l1_%s.json" % cond)
    if common.fresh(f.parent, f.name, "train"):
        c = json.loads(f.read_text())
        return c["chosen"], c["ok"]
    args = types.SimpleNamespace(l0_tol=0.05, knob_min=1e-6, knob_max=1e5, recal_passes=3,
                                 cal_steps=4000, device=device, target_l0=64.0)
    coef, rows, ok = v.b1.calibrate(spec, "relu_l1", adir, 64.0, cond, v1, device,
                                    steps=args.cal_steps, tol=args.l0_tol)
    coef, rows, ok = v.b1.calibrate_full_length(spec, "relu_l1", adir, coef, 64.0, cond, v1,
                                                args, rows)
    coef = float("%.6g" % coef)
    common.write_json(f, dict(chosen=coef, ok=bool(ok), points=rows, tol=0.05,
                              note="all calibration points, (coef, measured L0); training-store "
                                   "data only; no recovery/test data used"))
    common.pin(f.parent, f.name, "train")
    return coef, ok


# ---------------------------------------------------------------------------------------
# dense-cluster refit
# ---------------------------------------------------------------------------------------
def refit(v, spec, device, out):
    f = out / "refit.json"
    if common.fresh(f.parent, f.name, "train"):
        return json.loads(f.read_text())
    t0 = time.time()
    sae0, v1, mean, adir, _ = v.io.get_sae(spec, 42, arch="topk", condition="standard",
                                           device=device, k=64)
    rates, al = v.io.alive(sae0, adir, condition="standard", v1=v1, device=device)
    dmask, cos2, _ = v.dense_cluster(v.decoder_np(sae0), v1, al)
    C = np.nonzero(dmask)[0]
    k, d = len(C), spec.d_model
    from v3.common.data import open_activations
    acts, _ = open_activations(adir)
    n = acts.shape[0]
    ev_idx = np.arange(n - 50_000, n)                       # held-out rows: never sampled below
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    Xev = torch.as_tensor(np.asarray(acts[ev_idx], np.float32), device=device) - mean_t
    Ct = torch.as_tensor(C, device=device)

    def batches(seed):
        rng = np.random.default_rng(seed)
        for _ in range(REFIT_STEPS):
            idx = np.sort(rng.integers(0, n - 50_000, size=4096))
            yield torch.as_tensor(np.asarray(acts[idx], np.float32), device=device) - mean_t

    def run(init_seed):
        from v3.common.saes import ARCHS
        sae = ARCHS["topk"](spec.d_model, spec.d_sae, k=64, seed=0).to(device)
        sae.load_state_dict(sae0.state_dict())
        if init_seed is not None:
            fresh = ARCHS["topk"](spec.d_model, spec.d_sae, k=64, seed=init_seed).to(device)
            with torch.no_grad():
                sae.W_dec[Ct] = fresh.W_dec[Ct]; sae.W_enc[Ct] = fresh.W_enc[Ct]; sae.b_enc[Ct] = 0
        mask = torch.zeros(spec.d_sae, 1, device=device); mask[Ct] = 1
        sae.b_dec.requires_grad_(False)
        opt = torch.optim.Adam([sae.W_dec, sae.W_enc, sae.b_enc], lr=3e-4)
        for x in batches(77):                                # identical batches for every run
            recon, z = sae(x)
            loss = F.mse_loss(recon, x, reduction="none").sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            sae.W_dec.grad *= mask; sae.W_enc.grad *= mask; sae.b_enc.grad *= mask[:, 0]
            opt.step()
            sae.renorm_decoder()
        with torch.no_grad():
            mse, agg = 0.0, []
            for i in range(0, len(Xev), 8192):
                x = Xev[i:i + 8192]
                r, z = sae(x)
                mse += float(F.mse_loss(r, x, reduction="none").sum(-1).sum())
                agg.append((z[:, Ct] @ sae.W_dec[Ct]).float())
        return dict(mse=mse / len(Xev), dec=sae.W_dec[Ct].detach().float().cpu().numpy(),
                    agg=torch.cat(agg).cpu().numpy())

    def span_overlap(A, B):
        qa, _ = np.linalg.qr(A.T); qb, _ = np.linalg.qr(B.T)
        s = np.linalg.svd(qa.T @ qb, compute_uv=False)
        return float((s ** 2).mean())

    def eff_rank(A):
        s = np.linalg.svd(A, compute_uv=False) ** 2
        return float(s.sum() ** 2 / (s ** 2).sum())
    cont = run(None)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(50):
        R1 = rng.standard_normal((k, d)); R2 = rng.standard_normal((k, d))
        null.append(span_overlap(R1, R2))
    res = dict(n_cluster=int(k), d=int(d), refit_steps=REFIT_STEPS, tolerance_rel=REFIT_TOL,
               free_params="cluster rows of W_enc, b_enc, W_dec; everything else frozen incl. b_dec",
               continuation=dict(mse=cont["mse"], eff_rank_dec=eff_rank(cont["dec"])),
               random_null_mean_cos2=dict(expected=k / d, empirical_mean=float(np.mean(null)),
                                          empirical_q95=float(np.quantile(null, .95))),
               refits=[])
    from scipy.optimize import linear_sum_assignment
    for s in range(1000, 1005):
        r = run(s)
        Cm = np.abs(r["dec"] @ cont["dec"].T)
        ri, ci = linear_sum_assignment(-Cm)
        a, b = r["agg"], cont["agg"]
        res["refits"].append(dict(
            init_seed=s, mse=r["mse"], rel_loss_diff=(r["mse"] - cont["mse"]) / cont["mse"],
            within_tolerance=abs(r["mse"] - cont["mse"]) / cont["mse"] <= REFIT_TOL,
            maxcos_mean=float(Cm.max(1).mean()), hungarian_cos_mean=float(Cm[ri, ci].mean()),
            frac_matched_07=float((Cm.max(1) > 0.7).mean()),
            agg_output_rel_mse=float(((a - b) ** 2).sum() / max((b ** 2).sum(), 1e-12)),
            agg_output_cos=float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
            eff_rank_dec=eff_rank(r["dec"]), eff_rank_agg=eff_rank(a[:20000]),
            span_overlap_mean_cos2=span_overlap(r["dec"], cont["dec"])))
    res["seconds"] = time.time() - t0
    common.write_json(f, res)
    common.pin(f.parent, f.name, "train")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["a4", "a2"], required=True)
    ap.add_argument("--arch", default="topk")
    ap.add_argument("--site", choices=["pre", "post"], default=None)
    ap.add_argument("--screen", type=int, default=0, help="0 = exhaustive zero ablation")
    a = ap.parse_args(argv)
    v = _v3()
    spec = v.registry.MODELS[a.model]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.kind == "a2":
        pf = common.WORK / "prep" / a.model / "prep.json"
        common.s3_get("%s/%s/prep/prep.json" % (common.SHARED, a.model), pf)
        prep = json.loads(pf.read_text())
        sites = prep["sites"]
        layer = int(sites["pre_rise_block"] if a.site == "pre" else sites["post_fall_block"])
        tag = "a2_%s_L%d" % (a.site, layer)
    else:
        layer, tag = None, "a4_%s" % a.arch
    out = common.RESULTS / "train" / a.model / tag
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / "meta.json", dict(model=a.model, kind=a.kind, arch=a.arch, site=a.site,
                                              layer=layer if layer is not None else spec.target_layer,
                                              screen=a.screen, env=common.env_info()))
    big = spec.min_gpu_gb > 100
    # the harvest loads (and frees) its own copy of the model, so it must run BEFORE ours is
    # resident; ours is loaded with explicit per-GPU headroom so the SAE trainer fits on cuda:0
    adir = v.io.ensure_acts(spec, layer=layer, device=dev, device_map="auto" if big else None)
    torch.cuda.empty_cache()
    model, tok = M.load(spec, device=dev, device_map="auto" if big else None,
                        headroom=0.75 if big else 0.80)
    seqs, am = v.eval_sequences(tok, n_seqs=v.registry.EVAL_SEQS_IMPORTANCE)
    chunk = v.tuned_chunk(spec, vocab_size=int(model.config.vocab_size))
    evals, evecs, _ = v.estimate_pcs(adir)
    v1 = evecs[:, 0]
    arch = "topk" if a.kind == "a2" else a.arch
    res = dict(model=a.model, kind=a.kind, layer=layer if layer is not None else spec.target_layer,
               pc1_var_explained=float(evals[0] / evals.sum()), cells={})
    for cond in ("standard", "bypass"):
        cf = out / ("cell_%s.json" % cond)
        if common.fresh(cf.parent, cf.name, "train"):
            res["cells"][cond] = json.loads(cf.read_text())
            continue
        if arch == "relu_l1":
            coef, ok = calibrate_relu(v, spec, adir, cond, v1, dev, out)
            akw = v.b1._akw("relu_l1", coef)
        else:
            akw, ok = dict(k=v.registry.TOPK_K), True
        got = fetch_archived_saes(v, spec, arch, layer, akw)
        log("train", "%s/%s/%s: reused %d archived checkpoints" % (a.model, tag, cond, len(got)))
        c = cell(v, model, spec, seqs, am, arch, cond, akw, layer, a.screen, chunk, out, dev)
        c["calibration_ok"] = bool(ok)
        c["reused_archived"] = [list(x) for x in got]
        common.write_json(cf, c)
        common.pin(cf.parent, cf.name, "train")
        res["cells"][cond] = c
    if a.kind == "a4" and arch == "topk":
        res["refit"] = refit(v, spec, dev, out)
    common.write_json(out / "summary.json", res)
    return dict(tag=tag, top50={c: res["cells"][c]["top50_shared_mean_over_sources"] for c in res["cells"]})
