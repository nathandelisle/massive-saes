#!/usr/bin/env python
"""Measure feature importance and cross-seed matching for the original model comparison."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from v3.common import registry, io_utils as io                     # noqa: E402
from v3.common.data import load_model_and_tokenizer, eval_sequences  # noqa: E402
from v3.common.fastabl import (ablation_over, z_stats,             # noqa: E402
                               gradient_importance, dense_share,
                               screen_candidates)
from v3.common.matching import sharing_stats, max_cos               # noqa: E402
from v3.common.pca import estimate_pcs, pc1_stats, dense_cluster    # noqa: E402
from v3.common.saes import decoder_np                              # noqa: E402
from v3.common.splice import spliced_ce                            # noqa: E402

EXP = "a5_metric_battery"
ABLATION_METRICS = ["zero", "mean", "resample"]
GRAD_METRICS = ["gradxinput", "intgrad"]
THR = registry.SHARED_COS_THRESHOLD


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _sae_key(spec, condition, seed, extra, args):
    """Exact identity of the SAE + eval protocol an ablation cache belongs to."""
    cfg = (extra or {}).get("cfg")
    h = io.config_hash(cfg) if cfg else f"{spec.key}-{condition}-{seed}-nocfg"
    return f"{spec.key}_{condition}_{h}_ev{args.eval_seqs}"


def _sanitize(arr, label):
    """Non-finite importance must never win an argsort: np.argsort places NaN
    LAST ascending, i.e. FIRST after the [::-1] used for top-k, so a single NaN
    (fp16 overflow in the spliced forward) silently hijacks the whole top-50."""
    arr = np.asarray(arr, dtype=np.float64)
    bad = ~np.isfinite(arr)
    n = int(bad.sum())
    if n:
        print(f"  !! {label}: {n} non-finite importance values -> -inf "
              f"(excluded from top-k). Check model dtype / spliced CE.",
              flush=True)
        arr = np.where(bad, -np.inf, arr)
    return arr, n


def _top_share(shared, imp, alive_ref, k):
    """Identical to matching.sharing_stats' top_k, but reusing a precomputed
    max-cos vector (sharing_stats recomputes the full [d_sae, d_sae] cosine
    matrix on every call -> 1 GB + ~15 s per metric per pair otherwise)."""
    masked = np.where(alive_ref, imp, -np.inf)
    top = np.argsort(masked)[::-1][:k]
    return float(shared[top].mean()), top


def _ablate(model, spec, seqs, sae, mu, v1, mode, metric, cand, stats,
            tag, args, am=None):
    """Sharded + checkpointed ablation sweep. Returns (delta array, base CE).

    Shards partition `cand` on a FIXED grid so a resumed run reproduces a fresh
    run bit-for-bit (the resample RNG is seeded per shard index). The spliced
    baseline is recomputed once per shard and asserted stable — that is the
    check that every delta in a cell is referenced to the same baseline.
    """
    cache = io.path("cache_ablate", f"{tag}_{metric}.npz")
    feats = np.asarray(cand, dtype=np.int64)
    keep = set(int(j) for j in feats)
    shards = [feats[i:i + args.ablate_shard]
              for i in range(0, len(feats), args.ablate_shard)]
    done, base = {}, None
    if os.path.exists(cache) and not args.force:
        z = np.load(cache)
        done = {int(a): float(b) for a, b in zip(z["feat"], z["delta"])
                if int(a) in keep}
        base = float(z["base"])
        print(f"  [{mode}] {metric}: resuming, {len(done)}/{len(feats)} cached",
              flush=True)
    for si, sh in enumerate(shards):
        if all(int(j) in done for j in sh):
            continue
        deltas, b = ablation_over(
            model, spec, seqs, sae, mu, [int(j) for j in sh], v1=v1, mode=mode,
            metric=metric, stats=stats, seed=args.ablate_seed + si, am=am,
            chunk=args.chunk)
        if base is None:
            base = b
        elif abs(b - base) > args.base_tol:
            raise RuntimeError(
                f"spliced baseline CE drifted across shards for {tag}/{metric}"
                f": {base:.8f} -> {b:.8f} (> {args.base_tol}). Deltas in this "
                f"cell are not referenced to a common baseline; stop and fix.")
        done.update({int(k): float(v) for k, v in deltas.items()})
        np.savez(cache, feat=np.array(sorted(done), dtype=np.int64),
                 delta=np.array([done[k] for k in sorted(done)]),
                 base=np.array(float(base)))
        print(f"    [{mode}/{metric}] shard {si + 1}/{len(shards)} "
              f"({len(done)}/{len(feats)} features)", flush=True)
    arr = np.zeros(sae.d_sae)
    for j, v in done.items():
        arr[j] = v
    return arr, float(base if base is not None else np.nan)


def gate0_parity(model, tok, spec, args):
    """README Gate 0 item 1 — capture-indexing parity artifact.

    Nothing else in the suite tests this, and the (42,137) reproduction gate
    CANNOT detect it (top-50/dict-wide ranges are far too wide to separate
    layer 12 from layer 13). This dumps the first sequences of the harvest, in
    harvest order, plus the tokens that produced them, and reports which HF
    `hidden_states` index the memmap actually equals — so the diff against the
    old repo's core/final.py is a one-liner offline.
    """
    import torch
    from v3.common.data import open_activations, tokenized_batches
    adir = io.ensure_acts(spec, device=args.device)
    acts, _ = open_activations(adir)
    # harvest batch 0. The store holds ONLY non-padding positions, so it is a
    # ragged concatenation of each sequence's real tokens, not [n_seq, T, d].
    ids, am = next(iter(tokenized_batches(tok, batch_seqs=32)))
    n_seq = 4
    lens = [int(x) for x in am[:n_seq].sum(1).tolist()]
    total = int(sum(lens))
    rows = np.asarray(acts[:total], dtype=np.float32)           # [total, d]
    dev = next(model.parameters()).device
    with torch.no_grad():
        hs = model(ids[:n_seq].to(dev), attention_mask=am[:n_seq].to(dev),
                   output_hidden_states=True).hidden_states
    L = spec.target_layer
    want = registry.hidden_state_index(spec)     # v2's index: block L output
    errs = {}
    for cand in (want - 1, want, want + 1):
        if 0 <= cand < len(hs):
            got = np.concatenate([hs[cand][i, :lens[i]].float().cpu().numpy()
                                  for i in range(n_seq)])
            errs[cand] = float(np.abs(got - rows).max() /
                               (np.abs(rows).max() + 1e-9))
    best = min(errs, key=errs.get)
    print(f"  [gate0] harvest memmap matches hidden_states[{best}] "
          f"(rel err {errs[best]:.2e}); target_layer={L}, expected "
          f"hidden_states[{want}]; all={errs}", flush=True)
    if best != want:
        print(f"  !! [gate0] CAPTURE-INDEX MISMATCH: registry says layer {L} "
              f"(hidden_states[{want}]) but the memmap equals "
              f"hidden_states[{best}].", flush=True)
    np.savez(io.path("results", EXP, f"gate0_{spec.key}.npz"),
             token_ids=ids[:n_seq].cpu().numpy(),
             attention_mask=am[:n_seq].cpu().numpy(),
             seq_lens=np.array(lens), acts=rows,
             target_layer=L, expected_hidden_state=want,
             matches_hidden_states=best,
             rel_err_by_index=np.array([[k, v] for k, v in sorted(errs.items())]))
    return dict(target_layer=L, expected_hidden_state=int(want),
                matches_hidden_states=int(best),
                rel_err={str(k): v for k, v in errs.items()})


# ----------------------------------------------------------------------------

def run_model(spec, args):
    print(f"== {spec.key} ==", flush=True)
    if args.seeds:
        seeds = args.seeds                      # e.g. Gate-0: --seeds 42 137
    else:
        seeds = registry.SEEDS_FULL if spec.key != "llama-3.1-70b" else registry.SEEDS_PAIR
    ref = registry.REFERENCE_SEED
    if ref not in seeds:
        raise SystemExit(f"--seeds must contain the reference seed {ref} "
                         f"(it is SAE 'A' in every cross-seed pair); got {seeds}")

    todo = []
    for condition in (args.conditions or ["standard", "bypass"]):
        prev = io.read_json(EXP, f"{spec.key}_{condition}")
        npz = os.path.join(io.ROOT, "results", EXP,
                           f"features_{spec.key}_{condition}.npz")
        have = set((prev or {}).get("metrics", {}))
        same_proto = bool(prev) and prev.get("eval_seqs") == args.eval_seqs
        if (same_proto and os.path.exists(npz) and not args.force
                and set(args.metrics) <= have):
            print(f"  [{condition}] complete for metrics {sorted(args.metrics)}"
                  f" — skipping (use --force to redo)", flush=True)
            continue
        todo.append(condition)
    if not todo and not args.gate0_dump:
        return

    device_map = "auto" if spec.min_gpu_gb > 100 else None
    model, tok = load_model_and_tokenizer(spec, device=args.device,
                                          device_map=device_map)
    seqs, am = eval_sequences(tok, n_seqs=args.eval_seqs)
    gate0 = gate0_parity(model, tok, spec, args) if args.gate0_dump else None
    ce_clean = spliced_ce(model, spec, seqs, mode="none", am=am,
                          chunk=args.chunk)
    print(f"  clean CE (no splice) {ce_clean:.4f}", flush=True)
    # A healthy LM on WikiText sits around 2.5-4.5 nats. Anything far above
    # that means the model itself is broken (dtype overflow, wrong attention
    # implementation), and every downstream number would be noise.
    MAX_CLEAN_CE = 8.0
    if not np.isfinite(ce_clean) or ce_clean > MAX_CLEAN_CE:
        raise SystemExit(
            f"clean CE {ce_clean} is not plausible for {spec.key}: the model is "
            f"broken before any SAE is involved (registry dtype is "
            f"{spec.dtype}; gemma-2 overflows in float16 and needs bfloat16 + "
            f"attn_implementation='eager' for logit softcapping). Fix before "
            f"spending GPU-h.")
    for condition in todo:
        decs, alives, rates_all = {}, {}, {}
        sae = v1 = mean = extra = adir = None
        for s in seeds:
            # only the reference SAE is kept resident; the others are only
            # needed as numpy decoders + alive masks (VRAM: 5x411 MB on 9B).
            sae_s, v1_s, mean_s, adir, extra_s = io.get_sae(
                spec, s, arch="topk", condition=condition,
                device=args.device, k=registry.TOPK_K,
                device_map=device_map)
            rates, al = io.alive(sae_s, adir, condition=condition, v1=v1_s,
                                 device=args.device)
            decs[s], alives[s], rates_all[s] = decoder_np(sae_s), al, rates
            if s == ref:                        # v1/mean must come from the
                sae, v1, mean, extra = sae_s, v1_s, mean_s, extra_s
            del sae_s

        evals, evecs, _ = estimate_pcs(adir)
        stats, _ = pc1_stats(evals, evecs, mean)
        mode = condition
        alive_ref = alives[ref]
        print(f"  [{condition}] alive per seed: "
              f"{ {s: int(alives[s].sum()) for s in seeds} }", flush=True)

        # one capture pass: eval-set firing + means + resample banks
        # (padding excluded from every per-token statistic)
        st = z_stats(model, spec, seqs, sae, mean, v1=v1, mode=mode, am=am,
                     chunk=args.chunk)
        eval_rates = st["rate"]
        firing_eval = eval_rates > 0            # dead-on-eval => ablation delta = 0 exactly
        cand = np.nonzero(firing_eval & alive_ref)[0]
        l0_eval = float(eval_rates.sum())
        n_cand_full = len(cand)
        screened_out = 0
        if args.screen and args.screen < n_cand_full:
            # Ablate only the top-m by E[z^2]. The cost of this experiment is
            # exactly (features x eval sequences) forward passes, so at
            # d_sae=32768 the unscreened battery is ~600-1250 GPU-h; this is the
            # lever exp_b1 and gate0 already pull. The screen is a MAGNITUDE
            # statistic, not an importance one, so it can in principle drop a
            # feature that fires rarely and still moves CE a lot -- state the
            # cap in the output rather than letting a truncated ranking read as
            # a complete one.
            keep = set(int(j) for j in screen_candidates(st["ez2"], alive_ref,
                                                         args.screen))
            cand = np.array([j for j in cand if int(j) in keep], dtype=np.int64)
            screened_out = n_cand_full - len(cand)
            print(f"  [{condition}] SCREENED: ablating top {len(cand)} of "
                  f"{n_cand_full} firing+alive by E[z^2]; {screened_out} "
                  f"features not ablated (importance recorded as 0)", flush=True)
        print(f"  [{condition}] {len(cand)} firing+alive features to ablate; "
              f"eval L0 {l0_eval:.1f}", flush=True)
        if len(cand) < args.topk:
            raise SystemExit(f"only {len(cand)} firing+alive features for "
                             f"{spec.key}/{condition}; top-{args.topk} would be "
                             f"padded with never-firing features. Aborting.")

        tag = _sae_key(spec, condition, ref, extra, args)
        imp, base_ce, nonfinite, zero_delta = {}, {}, {}, {}
        for metric in args.metrics:
            if metric in GRAD_METRICS:
                continue
            arr, base = _ablate(model, spec, seqs, sae, mean, v1, mode, metric,
                                cand, st, tag, args, am=am)
            zero_delta[metric] = int((arr[cand] == 0.0).sum())
            arr, nf = _sanitize(arr, f"{condition}/{metric}")
            imp[metric], base_ce[metric], nonfinite[metric] = arr, base, nf
            print(f"  [{condition}] {metric}: base spliced CE {base:.4f} "
                  f"(clean {ce_clean:.4f}); {zero_delta[metric]} of "
                  f"{len(cand)} firing features have EXACTLY zero delta",
                  flush=True)
        if any(m in GRAD_METRICS for m in args.metrics):
            # fastabl.gradient_importance is padding-masked AND fixes the
            # spurious factor of alpha in the IG path integral, so these two
            # columns are now on the same footing as the ablation metrics.
            gxi, ig = gradient_importance(model, spec, seqs, sae, mean,
                                          v1=v1, mode=mode, am=am)
            for name, a in (("gradxinput", gxi), ("intgrad", ig)):
                if name in args.metrics:
                    imp[name], nonfinite[name] = _sanitize(a, f"{condition}/{name}")

        # dense cluster + budget (extends App F, now both conditions)
        dmask, cos2, thr = dense_cluster(decs[ref], evecs[:, 0], alive_ref)
        budget, budget_pos, budget_den, budget_note = {}, {}, {}, {}
        for m, a in imp.items():
            # fastabl.dense_share: `positive` is the estimator to quote;
            # `signed` is suppressed to None when its denominator is a
            # near-zero or negative difference of opposite-sign dCE sums.
            ds = dense_share(a, dmask, alive_ref)
            budget[m] = ds["signed"]
            budget_pos[m] = ds["positive"]
            budget_den[m] = ds["denominator_signed"]
            if ds.get("signed_suppressed"):
                budget_note[m] = ds["signed_suppressed"]
                print(f"  !! [{condition}] {m}: {ds['signed_suppressed']}",
                      flush=True)

        # sharing stats per metric, mean+/-std over pairs involving reference.
        # max-cos is computed ONCE per pair and shared by every metric.
        others = [s for s in seeds if s != ref]
        # signed, against ALL of B's decoder directions — matching.max_cos's
        # default and therefore the sharing_stats cross-check below
        mc = np.stack([max_cos(decs[ref], decs[s]) for s in others])
        shared = mc >= THR
        table = {}
        for m, arr in imp.items():
            tops, dws = [], []
            for pi, s in enumerate(others):
                tk, top = _top_share(shared[pi], arr, alive_ref, args.topk)
                tops.append(tk)
                dws.append(float(shared[pi][alive_ref].mean())
                           if alive_ref.any() else 0.0)
            table[m] = dict(top50_mean=float(np.mean(tops)),
                            top50_std=float(np.std(tops)),
                            dictwide_mean=float(np.mean(dws)),
                            dictwide_std=float(np.std(dws)))
        # canonical cross-check: one call into matching.sharing_stats must
        # reproduce the inlined numbers exactly, or the fast path has drifted.
        m0 = sorted(imp)[0]
        chk = sharing_stats(decs[ref], decs[others[0]], alive_ref,
                            alives[others[0]], imp[m0], k=args.topk)
        tk0, _ = _top_share(shared[0], imp[m0], alive_ref, args.topk)
        if (abs(chk["top_k"] - tk0) > 1e-12 or
                abs(chk["dict_wide"] - float(shared[0][alive_ref].mean())) > 1e-12):
            raise RuntimeError("fast sharing path disagrees with "
                               "matching.sharing_stats — do not trust output")

        prev = io.read_json(EXP, f"{spec.key}_{condition}") or {}
        merge = prev if prev.get("eval_seqs") == args.eval_seqs else {}
        if prev and not merge:
            print(f"  !! [{condition}] existing json used eval_seqs="
                  f"{prev.get('eval_seqs')} != {args.eval_seqs}; NOT merging",
                  flush=True)
        out = dict(
            model=spec.key, condition=condition, pc1=stats, seeds=seeds,
            n_dense=int(dmask.sum()), dense_frac_dict=float(dmask.mean()),
            dense_frac_alive=float(dmask.sum() / max(alive_ref.sum(), 1)),
            dense_cos2_threshold=thr,
            dense_importance_share={**merge.get("dense_importance_share", {}),
                                    **budget},
            dense_importance_share_pos={
                **merge.get("dense_importance_share_pos", {}), **budget_pos},
            dense_importance_denominator={
                **merge.get("dense_importance_denominator", {}), **budget_den},
            dense_importance_share_suppressed={
                **merge.get("dense_importance_share_suppressed", {}),
                **budget_note},
            metrics={**merge.get("metrics", {}), **table},
            eval_seqs=args.eval_seqs, topk=args.topk,
            n_alive={str(s): int(alives[s].sum()) for s in seeds},
            n_firing_alive=int(len(cand)), l0_eval=l0_eval,
            ce_clean=float(ce_clean),
            ce_spliced_base={**merge.get("ce_spliced_base", {}), **base_ce},
            n_zero_delta={**merge.get("n_zero_delta", {}), **zero_delta},
            n_nonfinite_importance={**merge.get("n_nonfinite_importance", {}),
                                    **nonfinite})
        if gate0:
            out["gate0_capture_index"] = gate0
        io.write_json(EXP, f"{spec.key}_{condition}", out)

        # per-feature table for figures + downstream analysis (pc1_cos2 sweep)
        npz_path = io.path("results", EXP,
                           f"features_{spec.key}_{condition}.npz")
        carry = {}
        if os.path.exists(npz_path) and merge:
            with np.load(npz_path) as old:       # close before overwriting
                carry = {k: np.array(old[k]) for k in old.files
                         if k.startswith("imp_") and k[4:] not in imp}
        np.savez(npz_path,
                 pc1_cos2=cos2, alive=alive_ref, rate=rates_all[ref],
                 dense=dmask, eval_rate=eval_rates, max_cos=mc,
                 seeds_other=np.array(others), **carry,
                 **{f"imp_{m}": a for m, a in imp.items()})
    del model
    import torch; torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="override seed set (Gate-0 parity: --seeds 42 137); "
                        "must include the reference seed 42")
    p.add_argument("--include-70b", action="store_true")
    p.add_argument("--metrics", nargs="*",
                   default=ABLATION_METRICS + GRAD_METRICS)
    p.add_argument("--eval-seqs", type=int, default=registry.EVAL_SEQS_IMPORTANCE,
                   help="64 = v2 protocol; pass 256 to reproduce v1 numbers")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--ablate-shard", type=int, default=256,
                   help="features per checkpointed ablation shard (resume unit)")
    p.add_argument("--ablate-seed", type=int, default=0,
                   help="base RNG seed for resample ablation (per-shard offset)")
    p.add_argument("--conditions", nargs="*", default=None,
                   choices=["standard", "bypass"],
                   help="run only these conditions. Exists so the ablation can "
                        "be fanned out across containers: the shard cache is "
                        "keyed per (condition, metric), so one worker per cell "
                        "writes a disjoint cache file and a later assembly pass "
                        "sees every cell as a cache hit. Splitting the RESULTS "
                        "json across concurrent workers would race; splitting "
                        "the cache does not.")
    p.add_argument("--screen", type=int, default=0,
                   help="ablate only the top-m alive features by E[z^2] "
                        "(0 = no screen). The battery costs features x eval "
                        "sequences forward passes, so this is the only lever "
                        "that makes d_sae=32768 affordable; the cap is "
                        "recorded in the results json.")
    p.add_argument("--chunk", type=int, default=8,
                   help="eval sequences per forward pass. 8 is the legacy "
                        "default and is FLOP-identical to 32; raising it "
                        "trades VRAM for wall-clock (logits dominate: "
                        "~14*chunk*128*vocab bytes).")
    p.add_argument("--base-tol", type=float, default=1e-4,
                   help="max tolerated spliced-baseline CE drift across shards")
    p.add_argument("--gate0-dump", action="store_true",
                   help="write the capture-indexing parity artifact (Gate 0.1)")
    p.add_argument("--force", action="store_true",
                   help="recompute cached ablations and overwrite results")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    known = set(ABLATION_METRICS + GRAD_METRICS)
    bad = [m for m in args.metrics if m not in known]
    if bad:
        raise SystemExit(f"unknown --metrics {bad}; choose from {sorted(known)}")
    for spec in registry.iter_models(args.models, include_70b=args.include_70b):
        run_model(spec, args)


if __name__ == "__main__":
    main()
