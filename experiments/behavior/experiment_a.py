"""Primary experiment A: can a small set of features from a FROZEN top-50 pool make the intended edit,
and does that edit survive a change of SAE seed without refitting?

The analyst's toolbox is the frozen pool (standard / posthoc / bypass, from freeze_pools.py). Task-specific
selection is permitted INSIDE the pool and nowhere else: no feature outside it, no learned rotation, no
free residual vector. Every arm gets the same learner, the same budget, the same data and the same doses.

A2  learn a size-b binary mask inside the pool on the task-train split (greedy exact search, which is
    deterministic and has no optimizer hyperparameters to differ between arms)
A3  score it on the locked test split: correct-edit rate, control preservation, margins, dose curve,
    necessity (suppress on clean) and sufficiency (donor patch into corrupted)
A4  freeze the mask, match its features into another seed's dictionary by signed decoder cosine, and
    evaluate on the target seed WITHOUT refitting anything -- using the TARGET's own encoder coefficients

Endpoints, predeclared: correct-edit success with control preservation at b=5 (A3), and the same quantity
on the target seed (A4). Everything else is supporting.

usage: experiment_a.py --model gpt2 --task ioi --seeds 42 137 0 --out results/exp_a
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

V3 = os.environ.get("PC1_V3", str(pathlib.Path(__file__).resolve().parents[1] / "language" / "pc1_v3"))
HERE = pathlib.Path(__file__).resolve().parent
METHODS = ["standard", "posthoc", "bypass", "random_pool"]


def log(*a):
    print("[expA]", *a, flush=True)


def tokenize_fixed(tok, prompts, device):
    """One padded batch for the WHOLE prompt list, so activation tensors from different calls line up
    position-for-position. Right padding, so the final valid token index is attention_mask.sum(1)-1."""
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in enc.items()}


def batched_logprobs(model, tok, prompts, device, batch=32, capture=None, target=None, override=None,
                     override_mask=None, enc=None):
    """Final-position log-probs. `capture` collects the FULL hook-site activation [N, T, D]; `override`
    replaces it at the positions where `override_mask` is true (all valid positions if None).

    Both are indexed against a single fixed tokenization of the whole prompt list, so a captured tensor
    and an override tensor always refer to the same positions.
    """
    import torch
    import torch.nn.functional as F
    enc_all = enc if enc is not None else tokenize_fixed(tok, prompts, device)
    ids_all, am_all = enc_all["input_ids"], enc_all["attention_mask"]
    outs, caps = [], []
    for s in range(0, len(prompts), batch):
        sl = slice(s, min(s + batch, len(prompts)))
        ids, am = ids_all[sl], am_all[sl]
        kw = {k: v[sl] for k, v in enc_all.items()}
        last = am.sum(1) - 1
        handle = None
        if target is not None and (capture is not None or override is not None):
            def fn(module, inp, outp, _sl=sl, _am=am):
                h = outp[0] if isinstance(outp, tuple) else outp
                if capture is not None:
                    caps.append(h.detach().float().cpu())
                if override is not None:
                    new = h.clone()
                    rep = override[_sl].to(h.device, h.dtype)
                    m = _am.bool() if override_mask is None else override_mask[_sl].to(h.device)
                    m = m & _am.bool()
                    new[m] = rep[m]
                    return (new,) + outp[1:] if isinstance(outp, tuple) else new
                return outp
            handle = target.register_forward_hook(fn)
        try:
            with torch.no_grad():
                lg = model(**kw).logits.float()
        finally:
            if handle is not None:
                handle.remove()
        outs.append(F.log_softmax(lg[torch.arange(len(lg), device=lg.device), last], dim=-1).cpu())
    if capture is not None and caps:
        T_max = max(c.shape[1] for c in caps)
        caps = [torch.nn.functional.pad(c, (0, 0, 0, T_max - c.shape[1])) for c in caps]
        capture.append(torch.cat(caps))
    return torch.cat(outs)


def edit_success(lp, items):
    """Correct-edit rate under the frozen eligibility rule: only items whose UNEDITED answer was not
    already the target can earn a switch, and the unconditional rate is reported next to it."""
    import tasks
    return np.array([tasks.correct(lp[i], it) for i, it in enumerate(items)], dtype=bool)


def greedy_mask(score_fn, pool, b, log_every=10):
    """Deterministic greedy exact search inside the pool: add the single feature that most improves the
    training objective, b times. No optimizer, no learning rate, no initialization seed -- so no arm can
    win because it got a better hyperparameter draw."""
    chosen, remaining = [], list(pool)
    best_hist = []
    for step in range(b):
        scores = [(score_fn(chosen + [f]), f) for f in remaining]
        scores.sort(key=lambda t: -t[0])
        best, f = scores[0]
        chosen.append(f)
        remaining.remove(f)
        best_hist.append(dict(step=step + 1, feature=int(f), train_objective=float(best)))
    return chosen, best_hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=["ioi", "sva", "gt"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 0])
    p.add_argument("--pools", default="protocol/pools.json")
    p.add_argument("--pool-size", type=int, default=50)
    p.add_argument("--b", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--primary-b", type=int, default=5)
    p.add_argument("--doses", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-test", type=int, default=512)
    p.add_argument("--out", default="results/exp_a")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force-positions", choices=["last", "all"],
                   help="override the pre-registered position rule. SENSITIVITY ONLY: the result is "
                        "written with a _forced<set> suffix and is never a replacement for the locked "
                        "cell. Used to test whether a null cell is a ceiling artifact of the rule.")
    a = p.parse_args()

    import torch
    sys.path.insert(0, V3)
    sys.path.insert(0, str(HERE))
    from v3.common import registry
    from v3.common.data import load_model_and_tokenizer, get_blocks
    from v3.common.io_utils import ensure_acts
    from v3.common.pca import estimate_pcs
    from v3.common.saes import load_sae
    import tasks as T
    from adapters import SAEAdapter, run_gate
    from checkpoint_utils import config_hash, fetch_sae

    spec = registry.MODELS[a.model]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False

    adir = ensure_acts(spec, device=dev)
    _e, evecs, mu = estimate_pcs(adir)
    v1 = evecs[:, 0]
    torch.cuda.empty_cache()

    model, tok = load_model_and_tokenizer(spec, device=dev)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    target = get_blocks(model, spec.block_path)[spec.target_layer]

    data, meta = T.make_all(tok, n_per_task=6000, tasks=[a.task])
    splits = data.get(a.task)
    if splits is None:
        out.mkdir(parents=True, exist_ok=True)
        (out / ("UNSUPPORTED_%s_%s.json" % (a.model, a.task))).write_text(json.dumps(
            dict(model=a.model, task=a.task, status="unsupported_task", meta=meta.get(a.task)), indent=1))
        log("task %s is unsupported on %s (%s); recorded, not a failure"
            % (a.task, a.model, meta.get(a.task, {}).get("why", "")))
        return
    log("task %s splits: %s" % (a.task, {k: len(v) for k, v in splits.items()}))

    # --- competence gate on the unmodified model, before any SAE arm ------------------------------
    cal = splits["calibration"][:512]
    lp_cal = batched_logprobs(model, tok, [i["prompt"] for i in cal], dev)
    comp = T.competence(lp_cal, cal)
    log("competence: %s" % json.dumps(comp))

    pools_all = json.loads((HERE / a.pools).read_text())["pools"][a.model]
    rng = np.random.default_rng(20260921)

    report = dict(model=a.model, task=a.task, competence=comp, seeds=a.seeds,
                  pool_size=a.pool_size, primary_b=a.primary_b, arms={}, gates={},
                  task_meta=meta[a.task], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    adapters, pools = {}, {}
    for seed in a.seeds:
        for cond in ("standard", "bypass"):
            cfg = dict(model=spec.key, seed=seed, arch="topk", condition=cond, corpus="wikitext",
                       bos_free=False, layer=spec.target_layer, steps=registry.STEPS,
                       batch=registry.BATCH_SIZE, d_sae=spec.d_sae, k=64.0)
            h = config_hash(cfg)
            sae, _ = load_sae(str(fetch_sae(h, "saes/%s.pt" % h)), device=dev)
            adapters[(seed, cond)] = SAEAdapter(sae, mu, v1, cond, device=dev)

    # --- adapter gate: one per condition, on real task activations --------------------------------
    probe_prompts = [i["prompt"] for i in splits["calibration"][:16]]
    enc_probe = tokenize_fixed(tok, probe_prompts, dev)
    cap = []
    batched_logprobs(model, tok, probe_prompts, dev, capture=cap, target=target, enc=enc_probe)
    probe_valid = enc_probe["attention_mask"].bool()
    h_probe_full = cap[0].to(dev)
    h_probe = h_probe_full[probe_valid]
    for cond in ("standard", "bypass"):
        ad = adapters[(a.seeds[0], cond)]

        def fwd(h_override):
            if h_override is None:
                return batched_logprobs(model, tok, probe_prompts, dev, enc=enc_probe)
            full = h_probe_full.clone()
            full[probe_valid] = h_override
            return batched_logprobs(model, tok, probe_prompts, dev, target=target, override=full,
                                    override_mask=probe_valid, enc=enc_probe)
        rep = run_gate(ad, h_probe, fwd, log=log)
        report["gates"][cond] = rep
        if not rep["passed"]:
            (out / ("GATE_FAILED_%s_%s_%s.json" % (a.model, a.task, cond))).write_text(
                json.dumps(report, indent=1, default=str))
            raise SystemExit("adapter gate failed for %s/%s; refusing to run interventions" % (a.model, cond))

    # --- position selection, then site feasibility -------------------------------------------------
    # Donor patching requires clean and corrupt to align position-for-position, so items whose two
    # tokenizations differ in length are dropped here rather than silently mis-aligned.
    def aligned(items):
        keep = []
        for it in items:
            a_ = tok(it["prompt"])["input_ids"]
            b_ = tok(it["corrupt_prompt"])["input_ids"]
            if len(a_) == len(b_):
                keep.append(it)
        return keep

    for name in list(splits):
        n0 = len(splits[name])
        splits[name] = aligned(splits[name])
        if n0 != len(splits[name]):
            log("split %s: dropped %d/%d items whose clean/corrupt tokenizations differ in length"
                % (name, n0 - len(splits[name]), n0))

    def encode_pair(items):
        """One fixed tokenization per side, plus the hook-site activations for both sides."""
        pc = [i["corrupt_prompt"] for i in items]
        pk = [i["prompt"] for i in items]
        enc_c = tokenize_fixed(tok, pc, dev)
        enc_k = tokenize_fixed(tok, pk, dev)
        cap_c, cap_k = [], []
        batched_logprobs(model, tok, pc, dev, capture=cap_c, target=target, enc=enc_c)
        batched_logprobs(model, tok, pk, dev, capture=cap_k, target=target, enc=enc_k)
        return enc_c, enc_k, cap_c[0].to(dev), cap_k[0].to(dev)

    def pos_mask(enc, kind):
        am = enc["attention_mask"].bool()
        if kind == "all":
            return am
        m = torch.zeros_like(am)
        last = am.sum(1) - 1
        m[torch.arange(len(am), device=am.device), last] = True
        return m

    cal2 = splits["calibration"][:256]
    enc_c, enc_k, h_corr_cal, h_clean_cal = encode_pair(cal2)
    lp_clean_cal = batched_logprobs(model, tok, [i["prompt"] for i in cal2], dev, enc=enc_k)
    lp_corr = batched_logprobs(model, tok, [i["corrupt_prompt"] for i in cal2], dev, enc=enc_c)
    m_clean = np.array([T.margin(lp_clean_cal[i], it) for i, it in enumerate(cal2)])
    m_corr = np.array([T.margin(lp_corr[i], it) for i, it in enumerate(cal2)])
    gap = float(np.mean(m_clean) - np.mean(m_corr))

    feas = {}
    for kind in ("last", "all"):
        mk = pos_mask(enc_c, kind)
        lp_full = batched_logprobs(model, tok, [i["corrupt_prompt"] for i in cal2], dev,
                                   target=target, override=h_clean_cal, override_mask=mk, enc=enc_c)
        m_full = np.array([T.margin(lp_full[i], it) for i, it in enumerate(cal2)])
        feas[kind] = dict(full_patch_margin=float(np.mean(m_full)),
                          fraction_recovered=(float((np.mean(m_full) - np.mean(m_corr)) / gap)
                                              if abs(gap) > 1e-6 else None))
        log("site feasibility [%s]: recovers %.4f of the clean/corrupt gap"
            % (kind, feas[kind]["fraction_recovered"] or float("nan")))

    # Declared rule, fixed before any SAE arm: take the SMALLEST position set whose full-residual donor
    # patch recovers at least half the clean/corrupt margin gap; if neither does, keep "all" and mark the
    # cell as having no causal access at this site. The same set is then used by EVERY arm.
    POS_MIN_RECOVERY = 0.50
    chosen_pos = next((k for k in ("last", "all")
                       if (feas[k]["fraction_recovered"] or 0) >= POS_MIN_RECOVERY), None)
    no_access = chosen_pos is None
    if no_access:
        chosen_pos = "all"
    rule_pos = chosen_pos
    if a.force_positions:
        chosen_pos = a.force_positions
        log("POSITION SET FORCED to %s (rule would have chosen %s) -- sensitivity run, not a replacement"
            % (chosen_pos, rule_pos))
    report["site_feasibility"] = dict(
        clean_margin=float(np.mean(m_clean)), corrupt_margin=float(np.mean(m_corr)),
        clean_corrupt_gap=gap, by_position_set=feas, position_set_used=chosen_pos,
        position_set_by_rule=rule_pos, position_set_forced=bool(a.force_positions),
        min_recovery_rule=POS_MIN_RECOVERY, no_causal_access=no_access,
        note=("A full-residual donor patch is the ceiling for any feature intervention at this site. "
              "The position set is chosen by this rule on calibration data with the unmodified model, "
              "identically for all arms, and locked before any SAE outcome. A cell marked "
              "no_causal_access is reported as a limitation of the site, not of feature recovery; no "
              "new layer is selected on the basis of which SAE wins."))
    log("position set locked: %s (no_causal_access=%s)" % (chosen_pos, no_access))

    # --- per-seed importance and frozen pools -----------------------------------------------------
    # The archived per-feature arrays exist only for the reference seed, so the other seeds' pools are
    # rebuilt here with the SAME documented screen (top SCREEN candidates by activation variance, then
    # exact zero-ablation dCE) on the SELECTION split -- never on task test data.
    SCREEN = int(os.environ.get("PC1_SCREEN", "400"))
    sel_items = splits["train"][:128]
    sel_prompts = [i["prompt"] for i in sel_items]
    enc_sel = tokenize_fixed(tok, sel_prompts, dev)
    cap_sel = []
    batched_logprobs(model, tok, sel_prompts, dev, capture=cap_sel, target=target, enc=enc_sel)
    h_sel_full = cap_sel[0].to(dev)
    sel_valid = enc_sel["attention_mask"].bool()
    h_sel = h_sel_full[sel_valid]                       # [n_valid_tokens, d_model], flat
    lp_sel_native = batched_logprobs(model, tok, sel_prompts, dev, enc=enc_sel).to(dev)

    def ranking(seed, cond):
        """Screen by activation variance, then rank the survivors by exact zero-ablation dCE.

        Returns the full ranked candidate list plus each candidate's squared PC1 alignment, so the
        post-hoc pool can be taken as a FILTER of this same ranking rather than a second search. That
        guarantees post-hoc and standard share one dictionary and one ranking quantity, which is the
        property the protocol requires of them."""
        ad = adapters[(seed, cond)]
        z = ad.encode(h_sel)
        alive = (z > 0).any(dim=0)
        var = torch.where(alive, z.var(dim=0), torch.full_like(z[0], -1.0))
        cand = torch.topk(var, min(SCREEN, int(alive.sum()))).indices
        dce = torch.zeros(len(cand))
        for j in range(len(cand)):
            f = int(cand[j])
            z_new = z.clone()
            z_new[:, f] = 0.0
            h_flat = ad.edit(h_sel, z_new, z)
            h_e = h_sel_full.clone()
            h_e[sel_valid] = h_flat
            lp = batched_logprobs(model, tok, sel_prompts, dev, target=target, override=h_e,
                                  override_mask=sel_valid, enc=enc_sel).to(dev)
            dce[j] = float((lp_sel_native.exp() * (lp_sel_native - lp)).sum(-1).mean())
        order = torch.argsort(dce, descending=True)
        ranked = [int(cand[i]) for i in order]
        d = ad.sae.W_dec.float()
        c2 = ((d / d.norm(dim=1, keepdim=True)) @ ad.v1) ** 2
        return ranked, c2

    method_pools = {}
    for seed in a.seeds:
        ranked_std, c2_std = ranking(seed, "standard")
        ranked_byp, _ = ranking(seed, "bypass")
        method_pools[(seed, "standard")] = ranked_std[:a.pool_size]
        method_pools[(seed, "posthoc")] = [f for f in ranked_std
                                           if float(c2_std[f]) < 0.01][:a.pool_size]
        method_pools[(seed, "bypass")] = ranked_byp[:a.pool_size]
        zz = adapters[(seed, "standard")].encode(h_sel)
        alive_idx = torch.nonzero((zz > 0).any(dim=0)).flatten().cpu().numpy()
        method_pools[(seed, "random_pool")] = [
            int(x) for x in np.random.default_rng(1000 + seed).choice(
                alive_idx, size=min(a.pool_size, len(alive_idx)), replace=False)]
        log("seed %d pools frozen: %s"
            % (seed, {m: len(method_pools[(seed, m)]) for m in METHODS}))
    report["pools"] = {"%d|%s" % (s_, m): v for (s_, m), v in method_pools.items()}
    report["screen_size"] = SCREEN

    # --- A2/A3: learn a mask inside the frozen pool, evaluate on locked test ----------------------
    train_items = splits["train"][:a.n_train]
    test_items = splits["test"][:a.n_test]
    cache = {}

    def site_pair(items, key):
        if key not in cache:
            ec, ek, hr, hd = encode_pair(items)
            cache[key] = (ec, ek, hr, hd, pos_mask(ec, chosen_pos) & ec["attention_mask"].bool())
        return cache[key]

    def eval_patch(ad, items, key, features, dose=1.0):
        """Patch the CLEAN donor's coefficients into the CORRUPTED recipient on `features` only, at the
        locked position set. Positions outside that set keep the recipient's own activation."""
        ec, ek, h_r, h_d, mask = site_pair(items, key)
        prompts = [i["corrupt_prompt"] for i in items]
        if not features:
            return batched_logprobs(model, tok, prompts, dev, enc=ec), None
        hr_f, hd_f = h_r[mask], h_d[mask]
        h_flat, _, _ = ad.donor_patch(hr_f, hd_f, torch.tensor(list(features), device=dev), dose=dose)
        h_e = h_r.clone()
        h_e[mask] = h_flat
        lp = batched_logprobs(model, tok, prompts, dev, target=target, override=h_e,
                              override_mask=mask, enc=ec)
        return lp, ad.delta_norms(h_flat - hr_f)

    arms = {}
    for seed in a.seeds:
        for method in METHODS:
            cond = "bypass" if method == "bypass" else "standard"
            ad = adapters[(seed, cond)]
            pool = method_pools[(seed, method)]
            tr = train_items[:96]

            def objective(feats, _ad=ad, _tr=tr):
                lp, _ = eval_patch(_ad, _tr, ("train", id(_tr)), feats)
                return float(np.mean([T.margin(lp[i], it) for i, it in enumerate(_tr)]))

            masks, res = {}, {}
            bmax = max(a.b)
            chosen_all, hist_all = greedy_mask(objective, pool, bmax)
            for b in a.b:
                masks[str(b)] = dict(features=[int(f) for f in chosen_all[:b]],
                                     history=hist_all[:b])
            lp0, _ = eval_patch(ad, test_items, ("test", id(test_items)), [])
            ok0 = edit_success(lp0, test_items)
            m0 = float(np.mean([T.margin(lp0[i], it) for i, it in enumerate(test_items)]))
            for b, mk in masks.items():
                lp, dn = eval_patch(ad, test_items, ("test", id(test_items)), mk["features"])
                ok = edit_success(lp, test_items)
                eligible = ~ok0
                res[b] = dict(
                    features=mk["features"], edit_success=float(ok.mean()), n_test=len(test_items),
                    unedited_already_correct=float(ok0.mean()), eligible_n=int(eligible.sum()),
                    edit_success_on_eligible=(float(ok[eligible].mean()) if eligible.any() else None),
                    mean_margin=float(np.mean([T.margin(lp[i], it) for i, it in enumerate(test_items)])),
                    mean_margin_unedited=m0,
                    delta_norm_total=dn[0] if dn else None, delta_norm_pc1=dn[1] if dn else None,
                    delta_norm_offpc1=dn[2] if dn else None)
            arms["%d|%s" % (seed, method)] = dict(pool_size=len(pool), masks=masks, results=res)
            pb = res[str(a.primary_b)]
            log("seed %d %-12s b=%d edit_success=%.3f eligible=%.3f margin %.3f -> %.3f"
                % (seed, method, a.primary_b, pb["edit_success"],
                   pb["edit_success_on_eligible"] if pb["edit_success_on_eligible"] is not None else float("nan"),
                   pb["mean_margin_unedited"], pb["mean_margin"]))

    report["arms"] = arms

    # --- A4: transfer the frozen mask to another seed, no refitting -------------------------------
    transfer = []
    for method in METHODS:
        cond = "bypass" if method == "bypass" else "standard"
        for s_src in a.seeds:
            src_feats = arms["%d|%s" % (s_src, method)]["masks"][str(a.primary_b)]["features"]
            D_src = adapters[(s_src, cond)].sae.W_dec.float()
            Dn_src = D_src / D_src.norm(dim=1, keepdim=True)
            for s_tgt in a.seeds:
                if s_tgt == s_src:
                    continue
                ad_t = adapters[(s_tgt, cond)]
                D_t = ad_t.sae.W_dec.float()
                Dn_t = D_t / D_t.norm(dim=1, keepdim=True)
                best = (Dn_src[src_feats] @ Dn_t.T).max(dim=1)
                matched = [int(i) for i in best.indices]
                cos = [float(c) for c in best.values]
                for setting, feats in (("dictionary_retrieval", matched),
                                       ("closed_top50", [f for f in matched
                                                         if f in set(method_pools[(s_tgt, method)])])):
                    lp, dn = eval_patch(ad_t, test_items, ("test", id(test_items)), feats)
                    ok = edit_success(lp, test_items)
                    lp0t, _ = eval_patch(ad_t, test_items, ("test", id(test_items)), [])
                    ok0t = edit_success(lp0t, test_items)
                    eligible = ~ok0t
                    transfer.append(dict(
                        method=method, source_seed=s_src, target_seed=s_tgt, setting=setting,
                        source_features=src_feats, target_features=feats,
                        n_matched=len(feats), signed_cosines=cos,
                        min_cosine=min(cos), median_cosine=float(np.median(cos)),
                        target_collisions=len(matched) - len(set(matched)),
                        source_edit_success=arms["%d|%s" % (s_src, method)]["results"][str(a.primary_b)]["edit_success"],
                        target_edit_success=float(ok.mean()),
                        target_edit_success_on_eligible=(float(ok[eligible].mean()) if eligible.any() else None),
                        unedited_already_correct=float(ok0t.mean()),
                        delta_norm_total=dn[0] if dn else None))
                log("%-12s %d->%d retrieval %.3f / closed %.3f (source %.3f, min cos %.2f)"
                    % (method, s_src, s_tgt, transfer[-2]["target_edit_success"],
                       transfer[-1]["target_edit_success"], transfer[-1]["source_edit_success"],
                       min(cos)))
    report["transfer"] = transfer

    suffix = ("_forced%s" % a.force_positions) if a.force_positions else ""
    (out / ("expA_%s_%s%s.json" % (a.model, a.task, suffix))).write_text(
        json.dumps(report, indent=1, default=str))
    log("wrote expA_%s_%s%s.json" % (a.model, a.task, suffix))


if __name__ == "__main__":
    main()
