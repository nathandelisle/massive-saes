"""Replication of the GPT-2 Gated mechanism on Pythia-6.9B, whose recalibrated Gated cell also inverts.

--role alloc : massive-row allocation (alloc.py) for plain Gated l1 8 (the recalibrated cell, 3 seeds, polled from S3),
               and the archived ReLU+l1 and TopK standard checkpoints. Also the massive-row geometry.
--role beta --seed S [--cell] : Gated + beta * sum|z| (gated_magl1), gate l1 8, beta 0.427 = the GPT-2 beta 0.3 point in
               units of the gate coefficient (beta / l1 = 0.0533), fixed before any Pythia-6.9B result. Same seeds, init,
               data order, schedule, auxiliary loss. The seed-42 box waits for siblings, runs the ORIGINAL run_cell, and
               reports exact L0, alive, held-out content FVU, band composition, the massive-row allocation and whether the
               top-50 occupants are the plain-Gated top-50 objects.
"""
import argparse
import json
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
import alloc as A                                                       # noqa: E402
import gpt2_gated as GG                                                 # noqa: E402
import seed_swap as SW                                                  # noqa: E402
from v3.common.data import load_model_and_tokenizer, eval_sequences     # noqa: E402
from v3.common.fastabl import tuned_chunk                               # noqa: E402

SPEC = registry.MODELS["pythia-6.9b"]
COEF = 8.0
BETA = 0.427
PLAIN = {42: "p69g-c8f-s42", 137: "p69g-c8-s137", 0: "p69g-c8-s0"}
PLAIN_NPZ = "p69g-c8f-s42/results/gated_investigation/feat_pythia-6.9b_gated_standard_c8.npz"
# the l1 7 cell (the one that inverts): all three checkpoints were written by the c7 cell box
PLAIN7 = {42: "p69g-c7-s42", 137: "p69g-c7-s42", 0: "p69g-c7-s42"}
PLAIN7_NPZ = "p69g-c7-s42/results/gated_investigation/feat_pythia-6.9b_gated_standard_c7.npz"


def wait_for(key, minutes):
    deadline = time.time() + 60 * minutes
    while not G.s3_exists(key):
        if time.time() > deadline:
            return False
        time.sleep(60)
    return True


def alloc_ckpt(key, st, massive, label):
    p = G.fetch(key, os.path.join(io.ROOT, "alloc_saes", os.path.basename(key)))
    sae, extra, cfg = G.load_ckpt(p)
    r, spec_mask, _rm, _ro = A.alloc(sae, st, massive, st["v1"])
    fid, _ = G.store_fidelity(sae, st)
    dec = G.dec_np(sae)
    del sae
    torch.cuda.empty_cache()
    print(f"[{G.now()}] {label}: massive atoms for 90% PC1 {r['massive_atoms_for_90pct_pc1_median']:.0f} massive L1 "
          f"{r['massive_total_l1_mean']:.0f} band/canon active {r['massive_band_active_atoms_mean']:.1f}/{r['massive_canonical_active_atoms_mean']:.1f} "
          f"massive-specific {r['n_massive_specific_atoms']} | L0 {fid['l0_exact']:.2f} alive {fid['n_alive']}", flush=True)
    return dict(label=label, key=key, **r, **fid), spec_mask, dec


def pairwise(masks, decs, seeds):
    out = []
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            if i != j:
                mc = G.max_cos(decs[i], decs[j])
                out.append(dict(source=seeds[i], target=seeds[j],
                                massive_specific_sharing=float((mc[masks[i]] >= G.THR).mean()) if masks[i].any() else None))
    return out


def top50(npz_path):
    z = np.load(npz_path)
    imp = z["imp_zero"].astype(float)
    imp[~z["alive"]] = -np.inf
    return np.argsort(imp)[::-1][:50]


def role_alloc7():
    """Same allocation for the l1 7 Gated cell, which inverts, against the l1 8 cell, which does not."""
    st = G.store(SPEC)
    geo, massive = SW.massive_geometry(st)
    out = dict(model="pythia-6.9b", massive_geometry=geo, objectives={})
    for label, coef, pre, npz_key in (("gated_c7", 7.0, PLAIN7, PLAIN7_NPZ), ("gated_c8", COEF, PLAIN, PLAIN_NPZ)):
        recs, masks, decs, seeds = [], [], [], []
        for s in (42, 137, 0):
            key = f"{pre[s]}/saes/{G.cache_name(SPEC, s, 'gated', l1_coef=coef)}"
            if not G.s3_exists(key):
                print(f"[{G.now()}] {label} s{s}: {key} missing", flush=True)
                continue
            r, m, d = alloc_ckpt(key, st, massive, f"{label} s{s}")
            r["seed"] = s
            recs.append(r); masks.append(m); decs.append(d); seeds.append(s)
        e = dict(l1_coef=coef, per_seed=recs, massive_specific_pairwise=pairwise(masks, decs, seeds))
        if seeds and seeds[0] == 42 and G.s3_exists(npz_key):
            p = G.fetch(npz_key, os.path.join(io.ROOT, "alloc_npz", label + ".npz"))
            e["top50_frac_massive_specific_s42"] = float(masks[0][top50(p)].mean())
            e["composition"] = GG.band_comp(p)[0]
        out["objectives"][label] = e
        G.dump(out, "p69_alloc_c7_vs_c8.json")


def role_alloc():
    st = G.store(SPEC)
    geo, massive = SW.massive_geometry(st)
    print(f"[{G.now()}] massive geometry: {geo}", flush=True)
    out = dict(model="pythia-6.9b", massive_geometry=geo, objectives={})
    arch_tab = json.load(open(os.path.join(G.ASSETS, "archived_pythia-6.9b.json")))
    items = []
    for a in ("relu_l1", "topk"):
        kw = arch_tab[a]["standard_kwargs"]
        items.append((a, [(s, f"b1v2-p69/saes/{G.cache_name(SPEC, s, a, **kw)}") for s in (42, 137, 0)], f"b1v2-p69/results/b1_architecture_law/feat_pythia-6.9b_{a}_standard.npz"))
    items.append(("gated_c8", [(s, f"{PLAIN[s]}/saes/{G.cache_name(SPEC, s, 'gated', l1_coef=COEF)}") for s in (42, 137, 0)], PLAIN_NPZ))
    for label, keys, npz_key in items:
        recs, masks, decs, seeds = [], [], [], []
        for s, key in keys:
            if not wait_for(key, 150 if label == "gated_c8" else 0):
                print(f"[{G.now()}] {label} s{s}: {key} not found; skipped", flush=True)
                continue
            r, m, d = alloc_ckpt(key, st, massive, f"{label} s{s}")
            r["seed"] = s
            recs.append(r); masks.append(m); decs.append(d); seeds.append(s)
        e = dict(per_seed=recs, massive_specific_pairwise=pairwise(masks, decs, seeds))
        if seeds and seeds[0] == 42 and wait_for(npz_key, 180 if label == "gated_c8" else 0):
            p = G.fetch(npz_key, os.path.join(io.ROOT, "alloc_npz", label + ".npz"))
            e["top50_frac_massive_specific_s42"] = float(masks[0][top50(p)].mean())
            e["composition"] = GG.band_comp(p)[0]
        out["objectives"][label] = e
        G.dump(out, "p69_alloc.json")


def role_beta(a):
    t0 = time.time()
    st = G.store(SPEC)
    akw = dict(l1_coef=COEF, beta=BETA)
    sae, _v, _m, _a, extra = io.get_sae(SPEC, a.seed, arch="gated_magl1", condition="standard", device="cuda", **akw)
    fid, _ = G.store_fidelity(sae, st)
    arch_l0, _al = G.archived_l0(sae, st)
    G.dump(dict(model="pythia-6.9b", arch="gated_magl1", l1_coef=COEF, beta=BETA, seed=a.seed, l0_archived_counter=arch_l0, **fid,
                train_log_tail=G.log_tail(extra), minutes=(time.time() - t0) / 60), f"p69_magl1_s{a.seed}.json")
    print(f"[{G.now()}] magl1 s{a.seed}: L0 exact {fid['l0_exact']:.2f} alive {fid['n_alive']} content {fid['fvu_content']:.4f}", flush=True)
    del sae
    torch.cuda.empty_cache()
    if not a.cell:
        return
    need = {s: G.cache_name(SPEC, s, "gated_magl1", **akw) for s in (137, 0)}
    deadline = time.time() + 60 * a.wait_min
    while True:
        missing = [s for s, h in need.items() if not os.path.exists(os.path.join(io.ROOT, "saes", h))]
        for s in list(missing):
            key = f"{a.sibling_prefix}{s}/saes/{need[s]}"
            if G.s3_exists(key):
                G.fetch(key, os.path.join(io.ROOT, "saes", need[s]))
                print(f"[{G.now()}] fetched sibling seed {s}", flush=True)
        missing = [s for s, h in need.items() if not os.path.exists(os.path.join(io.ROOT, "saes", h))]
        if not missing or time.time() > deadline:
            break
        time.sleep(60)
    X = G.harvest_split(SPEC, "test", 200_000)
    args = argparse.Namespace(
        models=["pythia-6.9b"], archs=["gated_magl1"], eval_seqs=registry.EVAL_SEQS_IMPORTANCE, topk=50, chunk=0, screen=1500,
        knob_min=1e-6, knob_max=1e5, recal_passes=0, target_l0=64.0, l0_tol=0.25, cal_steps=4000, force=True,
        seeds=[42, 137, 0], allow_uncalibrated=False, device="cuda", progress=500)
    model, tok = load_model_and_tokenizer(SPEC, device="cuda")
    seqs, am = eval_sequences(tok, n_seqs=args.eval_seqs)
    args.chunk = tuned_chunk(SPEC, vocab_size=int(model.config.vocab_size))
    cell = B1.run_cell(model, SPEC, seqs, st["adir"], st["evecs"], "gated_magl1", akw, "standard", args, am=am)
    del model
    torch.cuda.empty_cache()
    npz = io.path("results", "gated_investigation", f"feat_pythia-6.9b_gated_magl1_beta{BETA:g}.npz")
    shutil.copy(io.path("results", B1.EXP, "feat_pythia-6.9b_gated_magl1_standard.npz"), npz)
    comp, top = GG.band_comp(npz)
    _geo, massive = SW.massive_geometry(st)
    old_top, old = None, None
    if wait_for(PLAIN_NPZ, 60):
        old_top = top50(G.fetch(PLAIN_NPZ, os.path.join(io.ROOT, "plain_npz", "plain_c8.npz")))
        k = f"{PLAIN[42]}/saes/{G.cache_name(SPEC, 42, 'gated', l1_coef=COEF)}"
        o, _e, _c = G.load_ckpt(G.fetch(k, os.path.join(io.ROOT, "plain_saes", os.path.basename(k))))
        old = G.dec_np(o)
        del o
    per_seed, masks, decs = [], [], []
    for s in args.seeds:
        sae, _v, _m, _a, ex = io.get_sae(SPEC, s, arch="gated_magl1", condition="standard", device="cuda", **akw)
        f, _ = G.store_fidelity(sae, st)
        r = dict(seed=s, **f, **G.heldout_fidelity(sae, X, st["mean"], st["v1"]), train_log_tail=G.log_tail(ex))
        al, m, _rm, _ro = A.alloc(sae, st, massive, st["v1"])
        r["massive_rows"] = al
        new = G.dec_np(sae)
        masks.append(m); decs.append(new)
        if s == 42:
            r["top50_frac_massive_specific"] = float(m[top].mean())
            if old is not None:
                mc_old = G.max_cos(new[top], old)
                arg_old = np.argmax((new[top] / np.linalg.norm(new[top], axis=1, keepdims=True)) @
                                    (old / np.linalg.norm(old, axis=1, keepdims=True)).T, axis=1)
                r["top50_vs_plain_gated_c8_s42"] = dict(
                    frac_matched_in_plain_dictionary=float((mc_old >= G.THR).mean()),
                    frac_matched_to_plain_top50=float(np.mean([(mm >= G.THR) and (aa in set(old_top.tolist())) for mm, aa in zip(mc_old, arg_old)])))
        per_seed.append(r)
        del sae
        torch.cuda.empty_cache()
    G.dump(dict(role="p69_beta", l1_coef=COEF, beta=BETA, beta_over_l1=BETA / COEF, seeds=args.seeds, cell=cell, composition=comp,
                per_seed=per_seed, massive_specific_pairwise=pairwise(masks, decs, args.seeds), siblings_trained_here=missing,
                minutes=(time.time() - t0) / 60), f"p69_magl1_beta{BETA:g}.json")
    print(f"[{G.now()}] p69 magl1 beta {BETA:g}: top-50 {cell['top50_mean']:.3f} dict-wide {cell['dictwide_mean']:.3f} "
          f"L0 exact {np.mean([p['l0_exact'] for p in per_seed]):.2f} alive {np.mean([p['n_alive'] for p in per_seed]):.0f} "
          f"held-out content {np.mean([p['heldout_fvu_content'] for p in per_seed]):.4f} | {comp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["alloc", "alloc7", "beta"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cell", action="store_true")
    ap.add_argument("--sibling-prefix", default="p69b-s")
    ap.add_argument("--wait-min", type=float, default=180)
    a = ap.parse_args()
    {"alloc": role_alloc, "alloc7": role_alloc7}.get(a.role, lambda: role_beta(a))()


if __name__ == "__main__":
    main()
