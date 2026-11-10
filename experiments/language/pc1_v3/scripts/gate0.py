"""Collect training activations, train SAEs, and measure reconstruction and feature sharing."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3.common import fastabl, io_utils, matching, registry
from v3.common.data import (eval_sequences, get_blocks, load_model_and_tokenizer,
                            tokenized_batches)
from v3.common.pca import dense_cluster, estimate_pcs, pc1_stats
from v3.common.saes import decoder_np

EXP = "gate0"
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:7.1f}m]", *a, flush=True)


def save(name, obj):
    io_utils.write_json(EXP, name, obj)


# --------------------------------------------------------------------------- L
@torch.no_grad()
def layer_profile(spec, n_tokens=40_000, device="cuda"):
    """PC1 variance share at EVERY hidden-state index, via streaming float64
    covariance (25 x d x d in float64 is ~0.8 GB at d=2048; materialising the
    activations instead would be 10 GB)."""
    model, tok = load_model_and_tokenizer(spec, device=device)
    d = spec.d_model
    n_hs = spec.n_layers + 1
    S1 = np.zeros((n_hs, d), dtype=np.float64)
    S2 = np.zeros((n_hs, d, d), dtype=np.float64)
    n = 0
    for ids, am in tokenized_batches(tok, corpus="wikitext", batch_seqs=8,
                                     max_tokens=n_tokens + registry.SEQ_LEN):
        ids, am = ids.to(device), am.to(device)
        out = model(ids, attention_mask=am, output_hidden_states=True)
        keep = am.bool()
        for li, hs in enumerate(out.hidden_states):
            X = hs[keep].float().double().cpu().numpy()
            S1[li] += X.sum(0)
            S2[li] += X.T @ X
        n += int(am.sum())
        if n >= n_tokens:
            break
    del model
    torch.cuda.empty_cache()

    rows = []
    for li in range(n_hs):
        mu = S1[li] / n
        C = (S2[li] - n * np.outer(mu, mu)) / (n - 1)
        ev = np.linalg.eigvalsh(C)[::-1]
        tot = ev.sum()
        rows.append(dict(hidden_state_index=li,
                         pc1_var_explained=float(ev[0] / tot),
                         pc2_var_explained=float(ev[1] / tot),
                         total_var=float(tot)))
    return rows, n


@torch.no_grad()
def verify_capture_parity(spec, device="cuda", n_seq=4):
    """EXACT check that this suite captures the tensor v2 captured.

    v2 (core/final.py) used BOTH of these, interchangeably:
        acts = outputs.hidden_states[target_layer + 1]
        get_layer_module(target_layer).register_forward_hook(...)
    i.e. the output of block index `target_layer`.

    This suite hooks `blocks[registry.block_index(spec)]`. So the test is a
    tensor identity, not a fingerprint match: run one forward, capture through
    the suite's own hook path, and compare against hidden_states[t+1] AND
    hidden_states[t]. Exactly one must match, and it must be the former.

    A fingerprint/profile comparison CANNOT catch this — the layer-profile
    tables in v2 were written with a different (0-indexed hidden_states)
    convention than the SAE pipeline, so scoring PC1 shares against them
    validates the wrong tensor and reports a false PASS.
    """
    model, tok = load_model_and_tokenizer(spec, device=device)
    blocks = get_blocks(model, spec.block_path)
    li = registry.block_index(spec)
    buf = {}

    def hook(module, inputs, output):
        buf["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = blocks[li].register_forward_hook(hook)
    ids, am = next(tokenized_batches(tok, corpus="wikitext", batch_seqs=n_seq))
    ids, am = ids.to(device), am.to(device)
    out = model(ids, attention_mask=am, output_hidden_states=True)
    handle.remove()
    hooked = buf["h"].float()
    hs = [h.float() for h in out.hidden_states]

    def close(a, b):
        return bool(torch.allclose(a, b, atol=1e-3, rtol=1e-3))

    matches = [i for i, h in enumerate(hs) if h.shape == hooked.shape
               and close(hooked, h)]
    v2_index = li + 1
    res = dict(
        block_hooked=li, target_layer=spec.target_layer,
        hidden_state_matches=matches, v2_hidden_state_index=v2_index,
        n_hidden_states=len(hs),
        ok=bool(matches == [v2_index] or (v2_index in matches)),
    )
    res["verdict"] = ("hook == hidden_states[%d] == v2's capture" % v2_index
                      if res["ok"] else
                      "MISMATCH: hook matched %s, v2 used %d" % (matches, v2_index))
    del model
    torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------------- R/V
def run_repro(spec, seeds, eval_seqs, screen_only, device="cuda", min_chars=1):
    """Train every seed, but ablate only the REFERENCE seed.

    The top-50 statistic ranks features of the reference SAE (A) and asks
    whether each is matched in the comparison SAE (B) — see
    `matching.sharing_stats`, which consumes `importance_a` only. So importance
    is needed for seed 42 and nowhere else, which halves the dominant cost.
    """
    model, tok = load_model_and_tokenizer(spec, device=device)
    # min_chars=1 == v2's `[t for t in val if t["text"].strip()][:64]`. v2's
    # eval set is mostly short heading lines and is therefore heavily padded;
    # that is fine (padding is masked) and it is what the published numbers
    # used, so the reproduction gate must not "improve" it.
    seqs, am = eval_sequences(tok, corpus="wikitext", n_seqs=eval_seqs,
                              min_chars=min_chars)
    chunk = fastabl.tuned_chunk(spec, vocab_size=int(model.config.vocab_size))
    pad_frac = 1.0 - float(am.float().mean())
    log(f"eval {tuple(seqs.shape)} real_tokens={int(am.sum())} "
        f"pad_frac={pad_frac:.3f} min_chars={min_chars} chunk={chunk} "
        f"vocab={model.config.vocab_size}")

    per = {}
    for condition in ("standard", "bypass"):
        for seed in seeds:
            key = f"{condition}_s{seed}"
            is_ref = (seed == seeds[0])
            log(f"--- {key}: train/load SAE (reference={is_ref})")
            sae, v1, mean, adir, extra = io_utils.get_sae(
                spec, seed, arch="topk", condition=condition, device=device,
                k=registry.TOPK_K)
            rates, alive_mask = io_utils.alive(sae, adir, condition=condition,
                                              v1=v1, device=device)
            dec = decoder_np(sae)
            dmask, cos2, thr = dense_cluster(dec, v1, alive_mask)
            log(f"    alive={int(alive_mask.sum())}/{sae.d_sae} "
                f"L0_train={extra.get('log', [{}])[-1].get('l0')} "
                f"dense_cluster={int(dmask.sum())}")

            cell = dict(model=spec.key, arch="topk", layer=spec.target_layer,
                        condition=condition, seed=seed, is_ref=is_ref,
                        d_sae=int(sae.d_sae), n_alive=int(alive_mask.sum()),
                        n_dense=int(dmask.sum()), dense_thr=thr,
                        cos2_pc1=cos2.tolist(), alive=alive_mask.tolist(),
                        dense=dmask.tolist(), rate_train=rates.tolist())
            evals, _evecs, _m = estimate_pcs(adir)
            cell["pc1_var_share"] = float(evals[0] / evals.sum())

            if is_ref:
                st = fastabl.z_stats(model, spec, seqs, sae, mean,
                                     v1=v1 if condition == "bypass" else None,
                                     mode=condition, chunk=chunk, am=am)
                # exhaustive over ALIVE features — a never-firing feature has
                # dCE == 0 exactly, so excluding it is not an approximation
                feats = (np.flatnonzero(alive_mask) if not screen_only else
                         fastabl.screen_candidates(st["ez2"], alive_mask,
                                                   screen_only))
                log(f"    ablating {len(feats)} features "
                    f"({'exhaustive' if not screen_only else 'screened'})")
                part_path = io_utils.path("results", EXP, f"partial_{key}.json")

                def on_partial(n, tot, out, base, _p=part_path, _k=key):
                    json.dump(dict(key=_k, done=n, total=tot, base_ce=base,
                                   dce=out), open(_p, "w"))
                    log(f"    ablation {n}/{tot}")

                imp, base = fastabl.ablation_over(
                    model, spec, seqs, sae, mean, feats,
                    v1=v1 if condition == "bypass" else None, mode=condition,
                    metric="zero", chunk=chunk, stats=st, on_partial=on_partial,
                    partial_every=500, am=am)
                cell.update(base_ce=base, n_evaluated=len(imp), dce=imp,
                            ez2=st["ez2"].tolist(), rate_eval=st["rate"].tolist())

            per[key] = cell
            save(f"cell_{key}", cell)
            del sae
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-1.4b")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 137])
    ap.add_argument("--eval-seqs", type=int, default=registry.EVAL_SEQS_IMPORTANCE)
    ap.add_argument("--eval-seqs-alt", type=int, default=32)
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument("--screen-only", type=int, default=0,
                    help="if >0, ablate only the top-N by E[z^2] (NOT the gate)")
    ap.add_argument("--min-chars", type=int, default=1,
                    help="1 == v2's eval-text filter; raise only for a "
                         "separately-reported robustness row")
    a = ap.parse_args()
    spec = registry.MODELS[a.model]
    log(f"gate0 model={spec.key} target_layer={spec.target_layer} "
        f"d_model={spec.d_model} d_sae={spec.d_sae} root={io_utils.ROOT}")

    # ---- L0: exact capture parity (cheap, definitive, blocking)
    log("phase L0: capture parity vs v2")
    par = verify_capture_parity(spec)
    save("capture_parity", par)
    log(f"  {par['verdict']}")
    if not par["ok"]:
        log("  !! STOPPING: this suite captures a different tensor than v2. "
            "Fix registry.BLOCK_OFFSET before spending anything.")
        return 2

    # ---- L: layer profile (a figure, and a sanity read on PC1 dominance)
    if not a.skip_profile:
        log("phase L: layer profile")
        rows, n = layer_profile(spec)
        save("layer_profile", dict(
            model=spec.key, n_tokens=n, rows=rows,
            note="hidden_state_index i == output of block i-1; the SAE target "
                 f"layer {spec.target_layer} is hidden_state_index "
                 f"{registry.hidden_state_index(spec)}"))
        tgt = registry.hidden_state_index(spec)
        pc1 = next((r["pc1_var_explained"] for r in rows
                    if r["hidden_state_index"] == tgt), float("nan"))
        log(f"  PC1 share at the SAE target (hs[{tgt}]) = {pc1:.4f}")

    # ---- R + V
    log("phase R/V: train, exhaustive ablation, sharing")
    per = run_repro(spec, a.seeds, a.eval_seqs, a.screen_only,
                    min_chars=a.min_chars)

    # decoders for matching (reload cheaply; SAEs are cached)
    decs, alives, imps, meta = {}, {}, {}, {}
    for condition in ("standard", "bypass"):
        for seed in a.seeds:
            key = f"{condition}_s{seed}"
            sae, v1, mean, adir, _e = io_utils.get_sae(
                spec, seed, arch="topk", condition=condition, k=registry.TOPK_K)
            decs[key] = decoder_np(sae)
            alives[key] = np.asarray(per[key]["alive"], dtype=bool)
            if "dce" in per[key]:
                imps[key] = fastabl.to_array(
                    per[key]["d_sae"],
                    {int(j): v for j, v in per[key]["dce"].items()})
            meta[key] = dict(pc1_var_share=per[key]["pc1_var_share"],
                             n_alive=per[key]["n_alive"],
                             n_dense=per[key]["n_dense"])
            del sae
            torch.cuda.empty_cache()

    res = {}
    for condition in ("standard", "bypass"):
        ka, kb = f"{condition}_s{a.seeds[0]}", f"{condition}_s{a.seeds[1]}"
        s = matching.sharing_stats(decs[ka], decs[kb], alives[ka], alives[kb],
                                   imps[ka], k=50)
        res[condition] = dict(top50=s["top_k"], dict_wide=s["dict_wide"],
                              top_idx=s["top_idx"], **meta[ka])
        log(f"  {condition:9s} top50={s['top_k']:.3f} "
            f"dict_wide={s['dict_wide']:.3f} (v2: std .05-.10 / bypass .74-.97,"
            f" dict-wide .44-.62)")

    gate = dict(
        top50_standard=res["standard"]["top50"],
        top50_bypass=res["bypass"]["top50"],
        dict_wide_standard=res["standard"]["dict_wide"],
        inversion_present=res["standard"]["top50"] < res["standard"]["dict_wide"],
        repair_present=res["bypass"]["top50"] > res["standard"]["top50"],
        v2_window_ok=(0.02 <= res["standard"]["top50"] <= 0.15
                      and 0.60 <= res["bypass"]["top50"] <= 1.0),
    )
    save("sharing", dict(gate=gate, conditions=res))
    log(f"  GATE: {gate}")

    # ---- V: screen validation (only meaningful if the ablation was exhaustive)
    if not a.screen_only:
        rep = {}
        for key, cell in per.items():
            if "dce" not in cell:
                continue
            rep[key] = fastabl.screen_report(
                {int(j): v for j, v in cell["dce"].items()},
                np.asarray(cell["ez2"]), np.asarray(cell["alive"], dtype=bool))
            log(f"  screen {key}: spearman(log E[z^2], log dCE)="
                f"{rep[key]['spearman_logez2_vs_dce']:.3f} "
                f"recall(m1000,top50)={rep[key]['recall']['m1000_top50']:.3f}")
        save("screen_validation", rep)

    # ---- E: eval-set sensitivity, on the screened candidates only (cheap)
    log("phase E: eval-seqs sensitivity")
    try:
        model, tok = load_model_and_tokenizer(spec)
        chunk = fastabl.tuned_chunk(spec, vocab_size=int(model.config.vocab_size))
        alt = {}
        for condition in ("standard", "bypass"):
            key = f"{condition}_s{a.seeds[0]}"
            sae, v1, mean, adir, _e = io_utils.get_sae(
                spec, a.seeds[0], arch="topk", condition=condition,
                k=registry.TOPK_K)
            seqs, am2 = eval_sequences(tok, corpus="wikitext",
                                       n_seqs=a.eval_seqs_alt)
            alive_mask = np.asarray(per[key]["alive"], dtype=bool)
            cand = fastabl.screen_candidates(np.asarray(per[key]["ez2"]),
                                             alive_mask, 800)
            st = fastabl.z_stats(model, spec, seqs, sae, mean,
                                 v1=v1 if condition == "bypass" else None,
                                 mode=condition, chunk=chunk, am=am2)
            imp, base = fastabl.ablation_over(
                model, spec, seqs, sae, mean, cand,
                v1=v1 if condition == "bypass" else None, mode=condition,
                metric="zero", chunk=chunk, stats=st, am=am2)
            arr = fastabl.to_array(per[key]["d_sae"], imp)
            arr[~alive_mask] = -np.inf
            top_alt = np.argsort(arr)[::-1][:50]
            top_ref = res[condition]["top_idx"]
            alt[condition] = dict(
                n_eval_seqs=a.eval_seqs_alt,
                overlap_with_64seq_top50=len(set(map(int, top_alt))
                                             & set(map(int, top_ref))) / 50.0,
                spearman_vs_64=fastabl.spearman(
                    arr[cand], imps[key][cand]))
            log(f"  {condition}: top50 overlap 32 vs 64 seqs = "
                f"{alt[condition]['overlap_with_64seq_top50']:.2f}")
            del sae
            torch.cuda.empty_cache()
        save("eval_seqs_sensitivity", alt)
    except Exception as e:                      # never let this sink the gate
        log(f"  phase E failed (non-fatal): {type(e).__name__}: {e}")

    save("STATUS", dict(status="complete", minutes=(time.time() - T0) / 60,
                        gate=gate))
    log("gate0 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
