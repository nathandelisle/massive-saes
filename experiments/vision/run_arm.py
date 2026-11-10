"""One arm of the corrected vision comparison. Conditions (all at the same site, same images, same SAE):

  A standard      dinov2-base, SAE on patch tokens
  B random        dinov2-base, bypass a random unit direction (control for "removing any direction helps")
  C pc1           dinov2-base, bypass PC1 (estimated on TRAIN images only)
  D ttr           dinov2-base + Jiang test-time registers, standard SAE on patch tokens
  E registers     dinov2-with-registers-base, standard SAE on patch tokens ONLY (learned registers excluded)

A-C share one activation pass and run in one box. D and E each need their own model pass.

usage: run_arm.py --arm abc|ttr|registers [--steps 10000] [--seeds 0 1 2]
"""
import argparse
import json
import os
import pathlib
import subprocess
import time

import numpy as np
import torch

import vitlib as V
from vitlib import DEV

BUCKET = os.environ.get("S3_BUCKET", "")
CAMPAIGN = os.environ.get("VIT_CAMPAIGN", "vit2")
ROOT = pathlib.Path(os.environ.get("PC1_V3_ROOT", "work"))
BASE = "facebook/dinov2-base"
REG = "facebook/dinov2-with-registers-base"
CFG = dict(k=32, d_sae_mult=8, lr=3e-4, sae_bs=4096, steps=10000)
SITE = int(os.environ.get("VIT_SITE", "9"))          # hidden_states index; see the note in the report
N_IMAGES = int(os.environ.get("VIT_IMAGES", "640"))
N_TRAIN_IMAGES = int(os.environ.get("VIT_TRAIN_IMAGES", "512"))
DATASET = (os.environ.get("VIT_DATASET", "frgfm/imagenette"), os.environ.get("VIT_CONFIG", "320px"),
           os.environ.get("VIT_SPLIT", "train"))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def s3(*args):
    if not BUCKET:
        return
    subprocess.run(["aws", "s3", *args, "--only-show-errors"], check=False)


def manifest_stage(build):
    key = f"s3://{BUCKET}/{CAMPAIGN}/images.json"
    local = ROOT / "images.json"
    if build:
        man = V.build_image_manifest(*DATASET, N_IMAGES,
                                     fallbacks=[("frgfm/imagenette", "160px", "train"),
                                                ("uoft-cs/cifar100", None, "train"),
                                                ("ethz/food101", None, "validation")])
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(man))
        s3("cp", str(local), key)
        log(f"saved image manifest: {man['dataset']} {man['split']} n={man['n_images']}")
    else:
        bundled = pathlib.Path(__file__).resolve().parent / "images.json"
        if not local.exists() and bundled.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(bundled.read_bytes())
        if not local.exists() and not BUCKET:
            raise FileNotFoundError(
                f"Missing {local}. Supply the original image manifest, or use "
                "--build-manifest to select a new sample. This will not reproduce "
                "the paper\'s original sample."
            )
        deadline = time.time() + 45 * 60
        while not local.exists():
            s3("cp", key, str(local))
            if local.exists():
                break
            if time.time() > deadline:
                raise SystemExit("image manifest never appeared; the builder arm failed")
            log("waiting for the shared image manifest ...")
            time.sleep(60)
        man = json.loads(local.read_text())
    return man


def load_model(name):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float32).to(DEV).eval()
    return model, proc


def fit_arm(name, A_tr, A_te, mu, v, v1_eval, seeds, out, steps):
    """Train `seeds` SAEs on TRAIN activations; evaluate on HELD-OUT; match decoders both ways."""
    saes, snaps, hists = [], {}, {}
    for s in seeds:
        sae, hist, snap = V.train_sae(A_tr, mu, v, s, dict(CFG, steps=steps), name, checkpoints=(3000, steps), log=log)
        saes.append(sae); hists[s] = hist; snaps[s] = snap
    res = dict(arm=name, seeds=list(seeds), steps=steps, curves=hists, per_seed={}, checkpoints={})
    alive = []
    for s, sae in zip(seeds, saes):
        ev = V.eval_sae(sae, A_te, CFG, v1=v1_eval)
        res["per_seed"][str(s)] = ev
        with torch.no_grad():
            f = torch.zeros(sae.W_d.shape[0], device=DEV)
            for i in range(0, len(A_te), CFG["sae_bs"]):
                _, z, _ = sae(A_te[i:i + CFG["sae_bs"]].to(DEV), CFG["k"])
                f += (z > 0).float().sum(0)
        alive.append((f > 0).cpu())
    res["matching"] = V.matching(saes, alive_masks=alive)
    # the 3,000-step checkpoint, for continuity with the archived pilot
    for step in sorted({3000, steps}):
        mids = []
        for s in seeds:
            if step not in snaps[s]:
                continue
            m = V.SAE(A_tr.shape[1], A_tr.shape[1] * CFG["d_sae_mult"], mu.to(DEV),
                      None if v is None else v.to(DEV)).to(DEV)
            m.load_state_dict(snaps[s][step])
            mids.append(m)
        if len(mids) == len(seeds):
            res["checkpoints"][str(step)] = dict(
                matching=V.matching(mids), per_seed={str(s): V.eval_sae(m, A_te, CFG, v1=v1_eval)
                                                     for s, m in zip(seeds, mids)})
    V.dump(res, str(out))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["abc", "ttr", "registers"])
    p.add_argument("--steps", type=int, default=CFG["steps"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--build-manifest", action="store_true")
    p.add_argument("--n-neurons-grid", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--ttr-only-n", type=int, default=0,
                   help="run ONLY this neuron count, as arm D_ttr_n<N> (does not overwrite D_ttr)")
    p.add_argument("--suppression-target", type=float, default=0.10,
                   help="smallest neuron count whose outlier fraction falls below this fraction of baseline")
    a = p.parse_args()
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    man = manifest_stage(a.build_manifest)
    tr_man, te_man = V.split_images(man, N_TRAIN_IMAGES)
    log(f"images: {len(tr_man['images'])} train / {len(te_man['images'])} held-out from "
        f"{man['dataset']}:{man['split']}")
    imgs_tr, imgs_te = V.load_images(tr_man), V.load_images(te_man)

    if a.arm == "registers":
        model, proc = load_model(REG)
        A_tr, pre_tr, _, grid = V.collect_patch_activations(model, proc, imgs_tr, [SITE], log=log)
        A_te, _, _, _ = V.collect_patch_activations(model, proc, imgs_te, [SITE], log=log)
        A_tr, A_te = A_tr[SITE], A_te[SITE]
        geo, v1, mu = V.geometry(A_tr)
        reg_geo, _, _ = V.geometry(pre_tr[SITE][1:].reshape(-1, A_tr.shape[1])) if pre_tr[SITE].shape[0] else (None, None, None)
        V.dump(dict(arm="registers", model=REG, site=SITE, grid=grid, patch_geometry=geo,
                    prefix_token_geometry_separate=reg_geo,
                    note="patch tokens only; CLS and the 4 learned registers excluded (archive kept the registers)"),
               str(results / "geometry_registers.json"))
        fit_arm("E_registers", A_tr, A_te, mu, None, v1, a.seeds, results / "arm_E_registers.json", a.steps)
        return

    if a.arm == "ttr":
        import jiang
        model, proc = load_model(BASE)
        # Their released DINOv2 config (configs/dinov2_large.yaml): detect_outliers_layer -2, top_k 50,
        # num_registers 1, highest_layer 17 of 24 (0.71 of depth) -> 8 of 12 for Base. The absolute norm
        # threshold (150) is ViT-L specific, so Base uses the relative 10x-median rule and records it.
        highest = int(round(0.71 * len(jiang.layers_of(model))))
        ttr = jiang.TTR(model, n_registers=1, variant="per_neuron")
        ttr.discover(proc, imgs_tr[:256], detect_layer=-2, norm_mult=10.0, highest_layer=highest, log=log)

        # (i) Compare the LITERAL released implementation against the per-neuron variant on the SAME
        # calibration batch, at their own top_k, before anything touches an SAE.
        probe_imgs = imgs_tr[:32]
        variant_cmp = {}
        for variant in ("literal", "per_neuron"):
            ttr.variant, ttr.per_neuron_max = variant, (variant == "per_neuron")
            ttr.select_top(50)
            variant_cmp[variant] = jiang.validate(ttr, proc, probe_imgs, detect_layer=-2, log=log)
        d_out = abs((variant_cmp["literal"]["outlier_frac_ratio"] or 0)
                    - (variant_cmp["per_neuron"]["outlier_frac_ratio"] or 0))
        d_reg = abs(variant_cmp["literal"]["added_register_mean_norm"]
                    - variant_cmp["per_neuron"]["added_register_mean_norm"]) / max(
                        variant_cmp["per_neuron"]["added_register_mean_norm"], 1e-9)
        d_cls = abs(variant_cmp["literal"]["cls_cosine_baseline_vs_intervened"]
                    - variant_cmp["per_neuron"]["cls_cosine_baseline_vs_intervened"])
        materially_different = bool(d_out > 0.05 or d_reg > 0.10 or d_cls > 0.01)
        log(f"[variant] literal vs per-neuron: d_outlier_ratio={d_out:.4f} d_register_norm={d_reg:.4f} "
            f"d_cls_cos={d_cls:.4f} materially_different={materially_different}")

        # (ii) Freeze the arm's settings now, on outlier suppression over TRAIN images only.
        # PRIMARY = the LITERAL released implementation. On this calibration batch it passes every gate,
        # while the per-neuron reading of the paper does not (its single register absorbs less: mean norm
        # 37.1 vs a 47.5 max patch norm, so the register no longer dominates). Since the released code is
        # what Jiang et al. actually ship, the arm runs it as-is and keeps the per-neuron numbers as a
        # recorded sensitivity rather than making the primary arm an adaptation. Frozen before any SAE.
        ttr.variant, ttr.per_neuron_max = "literal", False
        grid_rep, chosen = [], None
        for n in a.n_neurons_grid:
            sel = ttr.select_top(n)
            rep = jiang.validate(ttr, proc, probe_imgs, detect_layer=-2, log=lambda *x: None)
            grid_rep.append(dict(n_neurons=n, outlier_frac_ratio=rep["outlier_frac_ratio"],
                                 cls_cos=rep["cls_cosine_baseline_vs_intervened"], passes=rep["passes"],
                                 selected=sel["layers"]))
            log(f"[calibrate] n={n} outlier {rep['outlier_frac_baseline']:.4f} -> "
                f"{rep['outlier_frac_intervened']:.4f} (ratio {rep['outlier_frac_ratio']}) "
                f"cls_cos {rep['cls_cosine_baseline_vs_intervened']:.4f} passes={rep['passes']}")
            # NOTE (bug fixed after the first D run): `(x or 1)` makes a PERFECT suppression (0.0)
            # falsy, so this rule never fired and the arm fell back to their released top_k=50.
            # The n=50 result already collected is therefore Jiang et al.'s own released setting;
            # --ttr-only-n reruns the minimal setting separately as a robustness arm.
            ratio = rep["outlier_frac_ratio"]
            if chosen is None and rep["passes"] and (1.0 if ratio is None else ratio) <= a.suppression_target:
                chosen = n
        if chosen is None:
            chosen = 50
            log(f"[calibrate] suppression target not reached; falling back to their released top_k={chosen}")
        sel = ttr.select_top(chosen)
        final = jiang.validate(ttr, proc, probe_imgs, detect_layer=-2, log=log)
        V.dump(dict(arm="ttr", model=BASE, site=SITE, detect_outliers_layer=-2, highest_layer=highest,
                    released_config_reference="configs/dinov2_large.yaml (detect -2, top_k 50, 1 register, highest 17/24)",
                    chosen_n_neurons=chosen, selection=sel, calibration_grid=grid_rep,
                    variant_comparison=variant_cmp, variant_deltas=dict(outlier_ratio=d_out, register_norm=d_reg,
                                                                        cls_cosine=d_cls),
                    materially_different=materially_different,
                    primary_variant="literal",
                    arm_label=("Jiang et al. as released (literal sign-max reduced over positions and over that "
                               "layer's selected neurons, one scalar written into the register)"),
                    per_neuron_variant_status=("recorded sensitivity, NOT used for the SAE arm; it is our reading "
                                               "of the paper's per-neuron wording and fails the register-dominance "
                                               "gate on this batch" if materially_different else
                                               "recorded sensitivity; agrees with the released code on this batch"),
                    validation=final, suppression_target=a.suppression_target),
               str(results / "ttr_calibration.json"))
        if not final["passes"]:
            V.dump(dict(blocked=True, why="intervention failed its validation gate", validation=final),
                   str(results / "arm_D_ttr_BLOCKED.json"))
            raise SystemExit("TTR validation gate failed; not training SAEs on it")
        # Run BOTH the frozen minimal setting and their released top_k, since both fully suppress outliers;
        # the conclusion should not depend on which of the two is chosen.
        plan = ([("D_ttr_n%d" % a.ttr_only_n, a.ttr_only_n)] if a.ttr_only_n
                else [("D_ttr", chosen), ("D_ttr_topk50", 50)])
        for label, n_sel in plan:
            if label == "D_ttr_topk50" and n_sel == chosen:
                continue
            sel_n = ttr.select_top(n_sel)
            rep = jiang.validate(ttr, proc, probe_imgs, detect_layer=-2, log=log)
            if not rep["passes"]:
                V.dump(dict(blocked=True, n_neurons=n_sel, validation=rep), str(results / f"arm_{label}_BLOCKED.json"))
                log(f"[{label}] gate failed at n={n_sel}; skipping")
                continue
            fwd = lambda px: ttr.intervened_hidden(px)
            A_tr, _, trail_tr, grid = V.collect_patch_activations(model, proc, imgs_tr, [SITE], forward_fn=fwd,
                                                                 extra_trailing=ttr.R, log=log)
            A_te, _, _, _ = V.collect_patch_activations(model, proc, imgs_te, [SITE], forward_fn=fwd,
                                                        extra_trailing=ttr.R, log=log)
            A_tr2, A_te2 = A_tr[SITE], A_te[SITE]
            geo, v1, mu = V.geometry(A_tr2)
            tgeo, _, _ = V.geometry(trail_tr[SITE])
            V.dump(dict(arm=label, n_neurons=n_sel, selection=sel_n, validation=rep, patch_geometry=geo,
                        added_register_geometry_separate=tgeo, site=SITE, grid=grid),
                   str(results / f"geometry_{label}.json"))
            fit_arm(label, A_tr2, A_te2, mu, None, v1, a.seeds, results / f"arm_{label}.json", a.steps)
        return

    # ---- arm abc: one activation pass on the base checkpoint, three SAE conditions
    model, proc = load_model(BASE)
    A_tr, _, _, grid = V.collect_patch_activations(model, proc, imgs_tr, [SITE], log=log)
    A_te, _, _, _ = V.collect_patch_activations(model, proc, imgs_te, [SITE], log=log)
    A_tr, A_te = A_tr[SITE], A_te[SITE]
    geo, v1, mu = V.geometry(A_tr)                        # PC1 from TRAIN images only
    g = torch.Generator().manual_seed(4242)
    vr = torch.randn(A_tr.shape[1], generator=g)
    vr = vr / vr.norm()
    V.dump(dict(arm="abc", model=BASE, site=SITE, grid=grid, patch_geometry=geo,
                n_train_rows=int(len(A_tr)), n_heldout_rows=int(len(A_te))),
           str(results / "geometry_abc.json"))
    for nm, v in (("A_standard", None), ("B_random_bypass", vr), ("C_pc1_bypass", v1)):
        fit_arm(nm, A_tr, A_te, mu, v, v1, a.seeds, results / f"arm_{nm}.json", a.steps)


if __name__ == "__main__":
    main()
