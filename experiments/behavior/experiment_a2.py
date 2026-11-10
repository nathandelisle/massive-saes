"""Run the original subject--verb agreement edits and transfer comparisons.
Uses the adapters and checkpoint helpers in this folder."""
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
N_RANDOM_DRAWS = 3          # several fixed random pools per seed, not one


def log(*a):
    print("[expA2]", *a, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 0])
    p.add_argument("--pools", required=True)
    p.add_argument("--pool-size", type=int, default=50)
    p.add_argument("--b", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--primary-b", type=int, default=5)
    p.add_argument("--doses", nargs="+", type=float, default=[0.25, 0.5, 1.0, 2.0])
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-test", type=int, default=512)
    p.add_argument("--out", default="results/exp_a2")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()

    import torch
    sys.path.insert(0, V3)
    sys.path.insert(0, str(HERE))
    from v3.common import registry
    from v3.common.data import load_model_and_tokenizer, get_blocks
    from v3.common.io_utils import ensure_acts
    from v3.common.pca import estimate_pcs
    from v3.common.saes import load_sae
    import tasks2 as T2
    from adapters import SAEAdapter, run_gate
    from experiment_a import batched_logprobs, tokenize_fixed, greedy_mask
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

    splits, meta = T2.build_sva(tok)
    if splits is None:
        (out / ("UNSUPPORTED_%s.json" % a.model)).write_text(json.dumps(meta, indent=1))
        raise SystemExit("sva unsupported on %s" % a.model)
    dj = T2.assert_disjoint(splits)
    if not dj["all_disjoint"]:
        raise SystemExit("splits are not disjoint: %s" % json.dumps(dj)[:400])
    log("splits %s ; all_disjoint=%s" % ({k: len(v) for k, v in splits.items()}, dj["all_disjoint"]))

    # token-length alignment for donor/recipient AND foil/recipient
    def aligned(rows):
        keep, drop = [], 0
        for it in rows:
            L = [len(tok(it[k])["input_ids"]) for k in ("prompt", "corrupt_prompt", "foil_prompt")]
            if len(set(L)) == 1:
                keep.append(it)
            else:
                drop += 1
        return keep, drop
    drops = {}
    for k in list(splits):
        splits[k], drops[k] = aligned(splits[k])
    log("tokenizer-alignment drops: %s" % drops)

    cal = T2.balanced_sample(splits["calibration"], 256)
    train = T2.balanced_sample(splits["train"], a.n_train)
    test = T2.balanced_sample(splits["test"], a.n_test)
    log("balanced samples: cal %d train %d test %d (unique recipients %d/%d/%d)"
        % (len(cal), len(train), len(test), len({i["corrupt_prompt"] for i in cal}),
           len({i["corrupt_prompt"] for i in train}), len({i["corrupt_prompt"] for i in test})))

    def acts(prompts, enc=None):
        e = enc or tokenize_fixed(tok, prompts, dev)
        cap = []
        batched_logprobs(model, tok, prompts, dev, capture=cap, target=target, enc=e)
        return e, cap[0].to(dev)

    def pos_mask(enc, kind):
        am = enc["attention_mask"].bool()
        if kind == "all":
            return am
        m = torch.zeros_like(am)
        m[torch.arange(len(am), device=am.device), am.sum(1) - 1] = True
        return m

    # ---- native competence + site feasibility, on the unmodified model only --------------------
    enc_r, h_r_cal = acts([i["corrupt_prompt"] for i in cal])
    enc_d, h_d_cal = acts([i["prompt"] for i in cal], )
    lp_r = batched_logprobs(model, tok, [i["corrupt_prompt"] for i in cal], dev, enc=enc_r)
    lp_d = batched_logprobs(model, tok, [i["prompt"] for i in cal], dev, enc=enc_d)
    # Which verb is correct depends on the SUBJECT NUMBER of the prompt being scored:
    #   donor prompt      -> answer_tok      (the donor's own subject number)  -> which="target"
    #   recipient prompt  -> foil_answer_tok (the recipient's subject number)  -> which="foil"
    # A first version scored the donor with foil_answer_tok and the recipient with answer_tok, which
    # reported donor_accuracy 0.0 and recipient_accuracy 0.0 -- both are the OPPOSITE quantity, not a
    # model failure. The same value is independently recorded downstream as
    # `control_correct_unedited`, so affected runs can be corrected without rerunning.
    comp = dict(
        native_recipient_accuracy=float(np.mean([T2.correct(lp_r[i], it, "foil")
                                                 for i, it in enumerate(cal)])),
        native_donor_accuracy=float(np.mean([T2.correct(lp_d[i], it, "target")
                                             for i, it in enumerate(cal)])),
        already_prefers_donor_answer=float(np.mean([T2.correct(lp_r[i], it)
                                                    for i, it in enumerate(cal)])),
        n=len(cal))
    comp["eligible"] = bool(min(comp["native_recipient_accuracy"],
                                comp["native_donor_accuracy"]) >= 0.70)
    log("competence: %s" % json.dumps(comp))

    feas = {}
    m_r = np.array([T2.margin(lp_r[i], it) for i, it in enumerate(cal)])
    m_d = np.array([T2.margin(lp_d[i], it) for i, it in enumerate(cal)])
    gap = float(np.mean(m_d) - np.mean(m_r))
    for kind in ("last", "all"):
        mk = pos_mask(enc_r, kind)
        lp = batched_logprobs(model, tok, [i["corrupt_prompt"] for i in cal], dev, target=target,
                              override=h_d_cal, override_mask=mk, enc=enc_r)
        mf = np.array([T2.margin(lp[i], it) for i, it in enumerate(cal)])
        feas[kind] = float((np.mean(mf) - np.mean(m_r)) / gap) if abs(gap) > 1e-6 else None
        log("site feasibility [%s]: %.4f" % (kind, feas[kind] or float("nan")))
    chosen_pos = next((k for k in ("last", "all") if (feas[k] or 0) >= 0.50), "all")
    log("position set locked: %s" % chosen_pos)

    # ---- frozen pools from the paper's ranking -------------------------------------------------
    P = json.loads(pathlib.Path(a.pools).read_text())
    adapters, pools = {}, {}
    for seed in a.seeds:
        for cond in ("standard", "bypass"):
            cfg = dict(model=spec.key, seed=seed, arch="topk", condition=cond, corpus="wikitext",
                       bos_free=False, layer=spec.target_layer, steps=registry.STEPS,
                       batch=registry.BATCH_SIZE, d_sae=spec.d_sae, k=64.0)
            h = config_hash(cfg)
            sae, _ = load_sae(str(fetch_sae(h, "saes/%s.pt" % h)), device=dev)
            adapters[(seed, cond)] = SAEAdapter(sae, mu, v1, cond, device=dev)
        s = P["seeds"][str(seed)]
        pools[(seed, "standard")] = s["standard"]["pools"][str(a.pool_size)]["standard"]
        pools[(seed, "posthoc")] = s["standard"]["pools"][str(a.pool_size)]["posthoc"]
        pools[(seed, "bypass")] = s["bypass"]["pools"][str(a.pool_size)]["bypass"]
        rng = np.random.default_rng(1000 + seed)
        alive = np.flatnonzero(np.load(pathlib.Path(a.pools).parent /
                                       ("rank_%s_standard_s%d.npz" % (a.model, seed)))["alive"])
        for d_ in range(N_RANDOM_DRAWS):
            pools[(seed, "random_pool%d" % d_)] = [int(x) for x in
                                                   rng.choice(alive, size=a.pool_size, replace=False)]
        log("seed %d pools: std %d posthoc %d bypass %d (+%d random draws)"
            % (seed, len(pools[(seed, "standard")]), len(pools[(seed, "posthoc")]),
               len(pools[(seed, "bypass")]), N_RANDOM_DRAWS))

    gate = {}
    for cond in ("standard", "bypass"):
        ad = adapters[(a.seeds[0], cond)]
        pv = enc_r["attention_mask"].bool()

        def fwd(h_override):
            if h_override is None:
                return batched_logprobs(model, tok, [i["corrupt_prompt"] for i in cal], dev, enc=enc_r)
            full = h_r_cal.clone()
            full[pv] = h_override
            return batched_logprobs(model, tok, [i["corrupt_prompt"] for i in cal], dev, target=target,
                                    override=full, override_mask=pv, enc=enc_r)
        gate[cond] = run_gate(ad, h_r_cal[pv], fwd, log=log)
        if not gate[cond]["passed"]:
            raise SystemExit("adapter gate failed for %s/%s" % (a.model, cond))

    # ---- evaluation machinery -------------------------------------------------------------------
    cache = {}

    def prep(items, tag):
        if tag not in cache:
            er, hr = acts([i["corrupt_prompt"] for i in items])
            ed, hd = acts([i["prompt"] for i in items])
            ef, hf = acts([i["foil_prompt"] for i in items])
            cache[tag] = (er, hr, hd, hf, pos_mask(er, chosen_pos) & er["attention_mask"].bool())
        return cache[tag]

    def run(ad, items, tag, feats, dose=1.0, which="target"):
        er, hr, hd, hf, mk = prep(items, tag)
        donor = hd if which == "target" else hf
        prompts = [i["corrupt_prompt"] for i in items]
        if not feats:
            return batched_logprobs(model, tok, prompts, dev, enc=er), None
        he, _, _ = ad.donor_patch(hr[mk], donor[mk], torch.tensor(list(feats), device=dev), dose=dose)
        full = hr.clone()
        full[mk] = he
        lp = batched_logprobs(model, tok, prompts, dev, target=target, override=full,
                              override_mask=mk, enc=er)
        return lp, ad.delta_norms(he - hr[mk])

    def endpoints(ad, items, tag, feats, dose=1.0):
        """Target switch, control preservation, and the predeclared joint endpoint."""
        lp_t, dn = run(ad, items, tag, feats, dose, "target")
        lp_c, _ = run(ad, items, tag, feats, dose, "foil")
        lp0, _ = run(ad, items, tag, [], 1.0)
        tgt = np.array([T2.correct(lp_t[i], it) for i, it in enumerate(items)])
        base = np.array([T2.correct(lp0[i], it) for i, it in enumerate(items)])
        ctl0 = np.array([T2.correct(lp0[i], it, "foil") for i, it in enumerate(items)])
        ctl = np.array([T2.correct(lp_c[i], it, "foil") for i, it in enumerate(items)])
        elig = ~base
        preserved = ctl == ctl0
        return dict(
            target_success=float(tgt.mean()),
            target_success_eligible=(float(tgt[elig].mean()) if elig.any() else None),
            n_eligible=int(elig.sum()), n_items=len(items),
            already_target=float(base.mean()),
            control_preserved=float(preserved.mean()),
            control_correct_unedited=float(ctl0.mean()),
            joint_success=(float((tgt & preserved)[elig].mean()) if elig.any() else None),
            mean_margin=float(np.mean([T2.margin(lp_t[i], it) for i, it in enumerate(items)])),
            mean_margin_unedited=float(np.mean([T2.margin(lp0[i], it) for i, it in enumerate(items)])),
            delta_norm=dn[0] if dn else None, delta_norm_pc1=dn[1] if dn else None,
            per_item=[dict(item_id=it["item_id"], target=bool(tgt[i]), eligible=bool(elig[i]),
                           control_preserved=bool(preserved[i])) for i, it in enumerate(items)])

    report = dict(model=a.model, task="sva", version="v2", seeds=a.seeds, competence=comp,
                  site_feasibility=dict(by_position_set=feas, position_set_used=chosen_pos),
                  splits_meta=meta, disjointness=dj, alignment_drops=drops, gates=gate,
                  pools={"%d|%s" % k: v for k, v in pools.items()},
                  pool_source=str(a.pools), archive_check=P.get("archive_check"),
                  arms={}, transfer=[], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    for seed in a.seeds:
        for method in METHODS:
            cond = "bypass" if method == "bypass" else "standard"
            ad = adapters[(seed, cond)]
            keys = ([("random_pool%d" % d_) for d_ in range(N_RANDOM_DRAWS)]
                    if method == "random_pool" else [method])
            for mk_name in keys:
                pool = pools[(seed, mk_name)]

                def obj(fs, _ad=ad):
                    lp, _ = run(_ad, train, "train", fs)
                    return float(np.mean([T2.margin(lp[i], it) for i, it in enumerate(train)]))

                chosen, hist = greedy_mask(obj, pool, max(a.b))
                res = {}
                for b in a.b:
                    res[str(b)] = endpoints(ad, test, "test", chosen[:b])
                    res[str(b)]["features"] = [int(f) for f in chosen[:b]]
                dose_curve = {("%g" % d): endpoints(ad, test, "test", chosen[:a.primary_b], dose=d)
                              for d in a.doses}
                report["arms"]["%d|%s" % (seed, mk_name)] = dict(
                    pool_size=len(pool), mask=[int(f) for f in chosen], history=hist,
                    results=res, dose_curve={k: {kk: vv for kk, vv in v.items() if kk != "per_item"}
                                             for k, v in dose_curve.items()})
                pb = res[str(a.primary_b)]
                # `x or nan` prints nan for a legitimate 0.0 -- the same falsy-zero trap that once made a
                # perfect suppression look like a missing value. Test for None explicitly.
                def _f(x):
                    return float("nan") if x is None else x
                log("seed %d %-14s b=%d target %.3f (elig %.3f, n=%d) control %.3f joint %.3f"
                    % (seed, mk_name, a.primary_b, pb["target_success"],
                       _f(pb["target_success_eligible"]), pb["n_eligible"],
                       pb["control_preserved"], _f(pb["joint_success"])))

    # ---- transfer: three named variants ---------------------------------------------------------
    for method in METHODS:
        cond = "bypass" if method == "bypass" else "standard"
        keys = ([("random_pool%d" % d_) for d_ in range(N_RANDOM_DRAWS)]
                if method == "random_pool" else [method])
        for mk_name in keys:
            for s_src in a.seeds:
                src = report["arms"]["%d|%s" % (s_src, mk_name)]["mask"][:a.primary_b]
                Ds = adapters[(s_src, cond)].sae.W_dec.float()
                Ds = Ds / Ds.norm(dim=1, keepdim=True)
                for s_tgt in a.seeds:
                    if s_tgt == s_src:
                        continue
                    ad_t = adapters[(s_tgt, cond)]
                    Dt = ad_t.sae.W_dec.float()
                    Dt = Dt / Dt.norm(dim=1, keepdim=True)
                    sims = Ds[src] @ Dt.T
                    best = sims.max(dim=1)
                    glob = [int(i) for i in best.indices]
                    tpool = pools[(s_tgt, mk_name)]
                    inpool = torch.tensor(tpool, device=sims.device)
                    bestin = sims[:, inpool].max(dim=1)
                    within = [int(inpool[i]) for i in bestin.indices]
                    variants = dict(
                        full_dictionary=glob,
                        global_then_filtered=[f for f in glob if f in set(tpool)],
                        within_target_pool=within)
                    for vname, feats in variants.items():
                        ded = sorted(set(feats))
                        r = endpoints(ad_t, test, "test", ded) if ded else None
                        report["transfer"].append(dict(
                            method=mk_name, source_seed=s_src, target_seed=s_tgt, variant=vname,
                            n_requested=len(feats), n_distinct_edited=len(ded),
                            collisions=len(feats) - len(ded),
                            cosines=[float(c) for c in (best.values if vname != "within_target_pool"
                                                        else bestin.values)],
                            source_target_success=report["arms"]["%d|%s" % (s_src, mk_name)]
                                                        ["results"][str(a.primary_b)]["target_success"],
                            **({k: v for k, v in r.items() if k != "per_item"} if r else
                               dict(target_success=None, note="no features after filtering"))))
            log("transfer done for %s" % mk_name)

    (out / ("expA2_%s_sva.json" % a.model)).write_text(json.dumps(report, indent=1, default=str))
    log("wrote expA2_%s_sva.json" % a.model)


if __name__ == "__main__":
    main()
