"""Qwen-2.5-7B Gated / standard (l1_coef 5.64369, seeds 42/137/0): is the low-sharing seed a failed run?

--role diag : the three ARCHIVED checkpoints (s3://.../b1v2-qwen/saes). Seed/checkpoint mapping (config hash and
              the cfg stored inside each file), sha256, training-log tail, decoder normalization, distance from
              initialization, exact L0 / alive / never-fired, store and held-out FVU, gate-vs-magnitude statistics,
              and sharing for EVERY ordered seed pair (top-50 uses the archived reference importance of seed 42).
--role rep  : retrain one seed under the original configuration on this box (same package, trainer, data order,
              init seed), run the same diagnostics, and compare it with the archived checkpoint of that seed.
              With --cell (seed 42 only) it waits for the reproduced seeds 137 and 0 from sibling prefixes and runs
              the ORIGINAL exp_b1 run_cell on the three reproduced seeds, giving a complete reproduced table row.
"""
import argparse
import itertools
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

COEF = 5.64369
SEEDS = [42, 137, 0]
SPEC = registry.MODELS["qwen-2.5-7b"]


def diagnose(sae, extra, cfg, seed, st, X_test, label):
    fid, rate = G.store_fidelity(sae, st)
    ho = G.heldout_fidelity(sae, X_test, st["mean"], st["v1"])
    arch_l0, arch_alive = G.archived_l0(sae, st)
    per, tot = G.gate_stats(sae, st)
    cos2 = (G.dec_np(sae) @ st["v1"]) ** 2
    alive = rate > registry.ALIVE_FIRING_RATE
    init, _ = G.init_similarity(sae, seed)
    return dict(label=label, seed=seed, cfg_in_checkpoint=cfg, train_log_tail=G.log_tail(extra),
                decoder_norms=G.decoder_norms(sae), init_similarity=init,
                nonfinite_params=int(sum((~torch.isfinite(p)).sum().item() for p in sae.parameters())),
                l0_archived_counter=arch_l0, n_alive_archived_counter=arch_alive, **fid, **ho,
                gate_totals=tot, gate_by_class=G.summarize_per_feature(per, cos2, alive),
                frac_features_gate_ever_open=float((per["fire_rate"] > 0).mean()),
                r_mag_exp_quantiles=np.quantile(per["exp_r_mag"], [0.05, 0.5, 0.95]).tolist(),
                b_gate_quantiles=np.quantile(per["b_gate"], [0.05, 0.5, 0.95]).tolist())


def pairwise(decs, alives, imp_ref, seeds):
    out = []
    for i, j in itertools.permutations(range(len(seeds)), 2):
        mc = G.max_cos(decs[i], decs[j])
        dw = float((mc >= G.THR)[alives[i]].mean())
        rec = dict(source=seeds[i], target=seeds[j], dictwide=dw)
        if seeds[i] == 42:
            imp = imp_ref.copy(); imp[~alives[i]] = -np.inf
            top = np.argsort(imp)[::-1][:50]
            rec["top50"] = float((mc >= G.THR)[top].mean())
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["diag", "rep"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell", action="store_true")
    ap.add_argument("--sibling-prefix", default="")
    ap.add_argument("--wait-min", type=float, default=120)
    a = ap.parse_args()
    t0 = time.time()
    names = {s: G.cache_name(SPEC, s, "gated", l1_coef=COEF) for s in SEEDS}
    st = G.store(SPEC)
    X_test = G.harvest_split(SPEC, "test", 200_000)
    print(f"[{G.now()}] store + held-out ready ({(time.time()-t0)/60:.1f}m) w1 {st['w1']:.4f}", flush=True)
    npz = np.load(os.path.join(G.ASSETS, "feat_qwen-2.5-7b_gated_standard.npz"))
    imp_ref = npz["imp_zero"].astype(float)
    arch_dir = os.path.join(io.ROOT, "archived_saes")

    if a.role == "diag":
        recs, decs, alives = [], [], []
        for s in SEEDS:
            p = G.fetch(f"b1v2-qwen/saes/{names[s]}", os.path.join(arch_dir, names[s]))
            sae, extra, cfg = G.load_ckpt(p)
            r = diagnose(sae, extra, cfg, s, st, X_test, "archived")
            r.update(checkpoint=names[s], checkpoint_sha256=G.sha256(p),
                     mapping_ok=(cfg.get("l1_coef") == COEF and (extra.get("cfg") or {}).get("seed") == s))
            recs.append(r)
            decs.append(G.dec_np(sae))
            _f, rate = G.store_fidelity(sae, st)
            alives.append(rate > registry.ALIVE_FIRING_RATE)
            print(f"[{G.now()}] archived s{s}: L0 exact {r['l0_exact']:.2f} alive {r['n_alive']} never {r['n_never_fired']} "
                  f"fvu {r['fvu_total']:.5f} content {r['fvu_content']:.4f} held-out content {r['heldout_fvu_content']:.4f} "
                  f"init-cos>0.9 {r['init_similarity']['decoder']['frac_rows_cos_gt_0p9']:.3f}", flush=True)
            del sae
            torch.cuda.empty_cache()
        pw = pairwise(decs, alives, imp_ref, SEEDS)
        # does the archived per-atom file agree with these decoders? (npz max_cos rows are 42 vs 137, 42 vs 0)
        npz_check = [float(np.abs(G.max_cos(decs[0], decs[j + 1]) - npz["max_cos"][j]).max()) for j in range(2)]
        G.dump(dict(role="diag", coef=COEF, seeds=SEEDS, checkpoints=recs, pairwise=pw,
                    npz_max_cos_max_abs_diff=npz_check, alive_npz_matches_ref=bool(np.array_equal(npz["alive"], alives[0]) )),
               "qwen_diag_archived.json")
        return

    # ---- rep: retrain one seed under the original configuration
    s = a.seed
    sae, _v1, _m, _a, extra = io.get_sae(SPEC, s, arch="gated", condition="standard", device="cuda", l1_coef=COEF)
    cfg = torch.load(io.sae_cache_path(SPEC, s, arch="gated", condition="standard", l1_coef=COEF),
                     map_location="cpu", weights_only=False)["cfg"]
    r = diagnose(sae, extra, cfg, s, st, X_test, "reproduced")
    dec_new = G.dec_np(sae)
    del sae
    torch.cuda.empty_cache()
    p = G.fetch(f"b1v2-qwen/saes/{names[s]}", os.path.join(arch_dir, names[s]))
    old, _e, _c = G.load_ckpt(p)
    dec_old = G.dec_np(old)
    del old
    row_cos = (dec_new * dec_old).sum(1)
    r["vs_archived_same_seed"] = dict(median_rowwise_cos=float(np.median(row_cos)), frac_rows_cos_gt_0p99=float((row_cos > 0.99).mean()),
                                      maxcos_match_frac=float((G.max_cos(dec_new, dec_old) >= G.THR).mean()),
                                      bitwise_equal=bool(np.array_equal(dec_new, dec_old)))
    # sharing of this reproduced seed against the ARCHIVED reference seed 42, with the archived reference ranking
    if s != 42:
        p42 = G.fetch(f"b1v2-qwen/saes/{names[42]}", os.path.join(arch_dir, names[42]))
        ref, _e, _c = G.load_ckpt(p42)
        d42 = G.dec_np(ref); del ref
        shr, _mc, _top = G.sharing_pair(d42, npz["alive"], imp_ref, dec_new)
        r["archived_ref42_vs_this"] = shr
    G.dump(r, f"qwen_rep_s{s}.json")
    print(f"[{G.now()}] reproduced s{s}: L0 exact {r['l0_exact']:.2f} alive {r['n_alive']} fvu {r['fvu_total']:.5f} "
          f"content {r['fvu_content']:.4f} | vs archived: {r['vs_archived_same_seed']} | {r.get('archived_ref42_vs_this')}", flush=True)
    if not (a.cell and s == 42):
        return

    need = {t: names[t] for t in (137, 0)}
    deadline = time.time() + 60 * a.wait_min
    while True:
        missing = [t for t, h in need.items() if not os.path.exists(os.path.join(io.ROOT, "saes", h))]
        for t in list(missing):
            key = f"{a.sibling_prefix}{t}/saes/{need[t]}"
            if G.s3_exists(key):
                G.fetch(key, os.path.join(io.ROOT, "saes", need[t]))
                print(f"[{G.now()}] fetched reproduced sibling seed {t}", flush=True)
        missing = [t for t, h in need.items() if not os.path.exists(os.path.join(io.ROOT, "saes", h))]
        if not missing or time.time() > deadline:
            break
        time.sleep(60)
    args = __import__("argparse").Namespace(
        models=["qwen-2.5-7b"], archs=["gated"], eval_seqs=registry.EVAL_SEQS_IMPORTANCE, topk=50, chunk=0, screen=1500,
        knob_min=1e-6, knob_max=1e5, recal_passes=0, target_l0=64.0, l0_tol=0.25, cal_steps=4000, force=True,
        seeds=SEEDS, allow_uncalibrated=False, device="cuda", progress=500)
    model, tok = load_model_and_tokenizer(SPEC, device="cuda")
    seqs, am = eval_sequences(tok, n_seqs=args.eval_seqs)
    args.chunk = tuned_chunk(SPEC, vocab_size=int(model.config.vocab_size))
    cell = B1.run_cell(model, SPEC, seqs, st["adir"], st["evecs"], "gated", dict(l1_coef=COEF), "standard", args, am=am)
    shutil.copy(io.path("results", B1.EXP, f"feat_{SPEC.key}_gated_standard.npz"),
                io.path("results", "gated_investigation", "feat_qwen-2.5-7b_gated_standard_reproduced.npz"))
    G.dump(dict(role="rep_cell", coef=COEF, seeds=SEEDS, cell=cell, siblings_trained_here=missing), "qwen_rep_cell.json")
    print(f"[{G.now()}] reproduced cell: top-50 {cell['top50_mean']:.3f} +/- {cell['top50_std']:.3f} dict-wide {cell['dictwide_mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
