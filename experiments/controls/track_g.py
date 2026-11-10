"""Compare feature selection and resampling-ablation rankings."""
import argparse
import csv
import pathlib
import time

import numpy as np

from rcommon import (add_v3, config_hash, fetch_sae, load_pool_npz, log, paper_cfg,
                     write_json)

COS2_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]
PRIMARY_COS2 = 0.01


def _rank_from(imp, ids):
    order = sorted(range(len(ids)), key=lambda i: -imp[i])
    return np.array([ids[i] for i in order], dtype=int), np.array([imp[i] for i in order])


def spearman(a_rank, b_rank, common):
    if len(common) < 10:
        return None
    ar = {int(j): r for r, j in enumerate(a_rank)}
    br = {int(j): r for r, j in enumerate(b_rank)}
    x = np.array([ar[j] for j in common], float)
    y = np.array([br[j] for j in common], float)
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 0])
    p.add_argument("--conditions", nargs="+", default=["standard", "bypass"])
    p.add_argument("--out", default="results/track_g")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-seqs", type=int, default=64)
    p.add_argument("--confirm-seqs", type=int, default=256)
    p.add_argument("--full", action="store_true",
                   help="exhaustive: rank every alive feature (small models only)")
    p.add_argument("--expanded-extra", type=int, default=1500,
                   help="stratified random sample of screened-out features, large models")
    p.add_argument("--mean-ablation", action="store_true")
    p.add_argument("--hours", type=float, default=6.0)
    a = p.parse_args()

    import torch
    add_v3()
    from v3.common import registry
    from v3.common.data import load_model_and_tokenizer, eval_sequences
    from v3.common.io_utils import ensure_acts
    from v3.common.pca import estimate_pcs
    from v3.common.saes import load_sae
    from v3.common.fastabl import z_stats, screen_candidates, ablation_over, tuned_chunk
    from v3.common.matching import max_cos

    spec = registry.MODELS[a.model]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False
    t_start = time.time()

    adir = ensure_acts(spec, device=dev)
    _e, evecs, mu_np = estimate_pcs(adir)
    v1_np = evecs[:, 0]
    torch.cuda.empty_cache()
    model, tok = load_model_and_tokenizer(spec, device=dev)
    model.eval()
    seqs, am = eval_sequences(tok, n_seqs=a.eval_seqs)
    # An independent, larger confirmation corpus -- selections frozen on `seqs` are
    # re-evaluated here so the reported effect is not the one the ranking maximised.
    # `eval_sequences` IGNORES its `seed` argument and always returns the first n
    # documents, so asking for more and slicing is the only way to get a disjoint set:
    # passing seed=7 would have silently returned a superset of the ranking corpus.
    _all, _allam = eval_sequences(tok, n_seqs=a.eval_seqs + a.confirm_seqs)
    cseqs, cam = _all[a.eval_seqs:], _allam[a.eval_seqs:]
    assert cseqs.shape[0] > 0, "no documents left for the confirmation split"
    chunk = tuned_chunk(spec)
    mu = torch.tensor(mu_np, dtype=torch.float32, device=dev)
    v1 = torch.tensor(v1_np, dtype=torch.float32, device=dev)

    res = dict(model=a.model, hf_id=spec.hf_id, layer=spec.target_layer, d_sae=spec.d_sae,
               eval_seqs=a.eval_seqs, confirm_seqs=a.confirm_seqs, chunk=chunk,
               confirm_split="documents %d..%d of the wikitext validation order, disjoint from the ranking split" % (a.eval_seqs, a.eval_seqs + a.confirm_seqs),
               mode=("exhaustive_alive" if a.full else "expanded_screen_union"),
               coverage_claim=("every alive feature exact-ablated" if a.full else
                               "UNION of E[z^2] screen, reconstruction-energy screen and a "
                               "stratified random sample of excluded features -- an "
                               "expanded-screen audit, not full-dictionary coverage"),
               cos2_grid=COS2_GRID, primary_cos2=PRIMARY_COS2,
               cells={}, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    grows, trows = [], []
    decoders, alives, imps = {}, {}, {}

    for cond in a.conditions:
        for seed in a.seeds:
            key = "%s|%d" % (cond, seed)
            if time.time() - t_start > a.hours * 3600:
                res["cells"][key] = dict(status="unmeasured_budget")
                log("trkG", "BUDGET EXHAUSTED before %s" % key)
                continue
            t0 = time.time()
            cfg = paper_cfg(spec, seed, cond)
            h = config_hash(cfg)
            sae, _ = load_sae(str(fetch_sae(h, "saes/%s.pt" % h)), device=dev)
            sae.eval()
            vv = v1 if cond == "bypass" else None

            st = z_stats(model, spec, seqs, sae, mu, v1=vv, mode=cond, chunk=chunk, am=am)
            alive = st["rate"] > registry.ALIVE_FIRING_RATE
            n_alive = int(alive.sum())
            D = sae.W_dec.detach().float()
            Dn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)
            c2 = (Dn @ v1).pow(2).cpu().numpy()
            decoders[key] = Dn.cpu().numpy()
            alives[key] = alive

            screen_ids = screen_candidates(st["ez2"], alive, 1500)
            # reconstruction-energy screen: E[z^2] * ||d_j||^2 == E[||z_j d_j||^2]
            energy = st["ez2"] * (D.norm(dim=1) ** 2).cpu().numpy()
            energy_ids = screen_candidates(energy, alive, 1500)
            if a.full:
                cand = np.where(alive)[0]
                incl = {int(j): 1.0 for j in cand}
                sample_note = "exhaustive over alive"
            else:
                base = set(int(x) for x in screen_ids) | set(int(x) for x in energy_ids)
                rest = [int(j) for j in np.where(alive)[0] if int(j) not in base]
                rng = np.random.default_rng(1000 + seed)
                # stratified by E[z^2] decile so the sample is not all near-dead features
                take = []
                if rest:
                    q = np.quantile(st["ez2"][rest], np.linspace(0, 1, 11))
                    per = max(1, a.expanded_extra // 10)
                    for lo, hi in zip(q[:-1], q[1:]):
                        bucket = [j for j in rest if lo <= st["ez2"][j] <= hi]
                        if bucket:
                            take += [int(x) for x in rng.choice(
                                bucket, size=min(per, len(bucket)), replace=False)]
                cand = np.array(sorted(base | set(take)), dtype=int)
                incl = {int(j): (1.0 if int(j) in base else
                                 float(min(1.0, (a.expanded_extra / 10) /
                                           max(len(rest) / 10, 1)))) for j in cand}
                sample_note = ("screen union + stratified sample of %d of %d excluded"
                               % (len(take), len(rest)))
            log("trkG", "  %s: %d alive, %d candidates (%s)"
                % (key, n_alive, len(cand), sample_note))

            imp, base_ce = ablation_over(model, spec, seqs, sae, mu, cand, v1=vv,
                                         mode=cond, metric="zero", chunk=chunk, am=am,
                                         on_partial=lambda n, N, o, b:
                                         log("trkG", "    %d/%d" % (n, N)),
                                         partial_every=500)
            ids_full = np.array(sorted(imp, key=lambda j: -imp[j]), dtype=int)
            sc_full = np.array([imp[int(j)] for j in ids_full], float)
            imps[key] = imp

            # --- G1: what the paper's screen would have produced -------------------
            sset = set(int(x) for x in screen_ids)
            ids_screen = np.array([j for j in ids_full if int(j) in sset], dtype=int)
            true50 = set(int(x) for x in ids_full[:50])
            scr50 = set(int(x) for x in ids_screen[:50])
            missed = sorted(true50 - scr50)
            arch = load_pool_npz(a.model, cond, seed)
            arch_top = ([int(x) for x in arch["ranked_ids"][:50]] if arch is not None else None)

            cell = dict(sae_hash=h, n_alive=n_alive, n_candidates=int(len(cand)),
                        baseline_spliced_ce=base_ce, coverage=sample_note,
                        status="measured",
                        recall_at_50=len(true50 & scr50) / 50.0,
                        recall_at_10=len(set(int(x) for x in ids_full[:10]) &
                                         set(int(x) for x in ids_screen[:10])) / 10.0,
                        missed_by_screen=[dict(feature=int(j), rank=int(
                            np.where(ids_full == j)[0][0]), delta_ce=float(imp[int(j)]),
                            pc1_cos2=float(c2[j]), ez2_rank=int(
                                np.where(np.argsort(-st["ez2"]) == j)[0][0]))
                            for j in missed],
                        spearman_full_vs_screen=spearman(
                            ids_full, ids_screen, [int(x) for x in ids_screen]),
                        archive_top50_overlap=(len(true50 & set(arch_top))
                                               if arch_top else None),
                        seconds=time.time() - t0)

            # --- confirmation on an independent, larger corpus ---------------------
            conf = ablation_over(model, spec, cseqs, sae, mu, ids_full[:100], v1=vv,
                                 mode=cond, metric="zero", chunk=chunk, am=cam)[0]
            ids_c, _ = _rank_from([conf[int(j)] for j in ids_full[:100]],
                                  [int(j) for j in ids_full[:100]])
            cell["confirm_top50_overlap"] = len(
                set(int(x) for x in ids_c[:50]) & set(int(x) for x in ids_full[:50]))
            cell["confirm_spearman"] = spearman(ids_full[:100], ids_c,
                                                [int(j) for j in ids_full[:100]])

            # --- G3: cos^2 threshold sweep, no new primary chosen ------------------
            for thr in COS2_GRID:
                keep = [int(j) for j in ids_full if c2[int(j)] < thr]
                removed_alive = int(((c2 >= thr) & alive).sum())
                pool = keep[:50]
                trows.append(dict(
                    model=a.model, condition=cond, seed=seed, cos2_threshold=thr,
                    is_primary=(thr == PRIMARY_COS2),
                    frac_alive_removed=removed_alive / max(n_alive, 1),
                    n_alive_removed=removed_alive,
                    n_top50_removed=len([j for j in ids_full[:50] if c2[int(j)] >= thr]),
                    n_canonical_removed=int(((c2 >= 0.5) & alive).sum()),
                    pool_mean_delta_ce=float(np.mean([imp[j] for j in pool])) if pool else None,
                    pool_min_delta_ce=float(np.min([imp[j] for j in pool])) if pool else None,
                    pool_mean_pc1_cos2=float(np.mean([c2[j] for j in pool])) if pool else None))
            # random-direction alignment null at the same d and dictionary
            rng = np.random.default_rng(7)
            rv = rng.normal(size=(len(v1_np), 16))
            rv /= np.linalg.norm(rv, axis=0, keepdims=True)
            rc2 = (decoders[key] @ rv) ** 2
            cell["random_direction_cos2_null"] = dict(
                median=float(np.median(rc2[alive])), q99=float(np.quantile(rc2[alive], .99)),
                frac_above_primary=float((rc2[alive] >= PRIMARY_COS2).mean()),
                note="16 fixed random unit directions, same d and dictionary")

            # --- G2: mean ablation on the same frozen candidates -------------------
            if a.mean_ablation:
                mimp, mbase = ablation_over(model, spec, seqs, sae, mu, ids_full[:300],
                                            v1=vv, mode=cond, metric="mean", chunk=chunk,
                                            am=am, stats=st)
                ids_m, _ = _rank_from([mimp[int(j)] for j in ids_full[:300]],
                                      [int(j) for j in ids_full[:300]])
                cell["mean_ablation"] = dict(
                    n=300, baseline_ce=mbase,
                    top50_overlap_with_zero=len(set(int(x) for x in ids_m[:50]) &
                                                set(int(x) for x in ids_full[:50])),
                    spearman_vs_zero=spearman(ids_full[:300], ids_m,
                                              [int(j) for j in ids_full[:300]]),
                    mean_is_from="training rows (z_stats over the eval corpus is the "
                                 "ranking population; the mean here is E[z] including "
                                 "non-firing positions, which ADDS activation where the "
                                 "feature is silent)")
                for j in [int(x) for x in ids_full[:300]]:
                    grows.append(dict(model=a.model, condition=cond, seed=seed,
                                      feature=j, metric="mean", delta_ce=float(mimp[j]),
                                      pc1_cos2=float(c2[j])))

            for r, j in enumerate(ids_full):
                grows.append(dict(model=a.model, condition=cond, seed=seed, feature=int(j),
                                  metric="zero", rank=r, delta_ce=float(imp[int(j)]),
                                  pc1_cos2=float(c2[int(j)]), ez2=float(st["ez2"][int(j)]),
                                  firing_rate=float(st["rate"][int(j)]),
                                  in_paper_screen=bool(int(j) in sset),
                                  inclusion_prob=incl.get(int(j))))
            np.savez_compressed(out / ("trackg_%s_%s_s%d.npz" % (a.model, cond, seed)),
                                ranked_ids=ids_full, delta_ce=sc_full, ez2=st["ez2"],
                                rate=st["rate"], alive=alive, pc1_cos2=c2,
                                candidates=cand, paper_screen=screen_ids,
                                energy_screen=energy_ids)
            res["cells"][key] = cell
            log("trkG", "  %s done in %.1f min | recall@50 %.3f, %d missed by screen"
                % (key, (time.time() - t0) / 60, cell["recall_at_50"], len(missed)))
            write_json(out / ("track_g_%s.json" % a.model), res)
            _wrows(out / ("track_g_features_%s.csv" % a.model), grows)
            _wrows(out / ("track_g_thresholds_%s.csv" % a.model), trows)
            del sae
            torch.cuda.empty_cache()

    # --- recurrence under the full ranking vs the screened ranking -----------------
    rec = []
    for cond in a.conditions:
        keys = ["%s|%d" % (cond, s) for s in a.seeds if "%s|%d" % (cond, s) in imps]
        for i, ka in enumerate(keys):
            for kb in keys:
                if ka == kb:
                    continue
                impa = imps[ka]
                order = np.array(sorted(impa, key=lambda j: -impa[j]), dtype=int)
                Db = decoders[kb]
                for ranking, ids in (("full", order[:50]),):
                    mc = max_cos(decoders[ka][ids], Db, alive_b=alives[kb])
                    rec.append(dict(model=a.model, condition=cond, source=ka, target=kb,
                                    ranking=ranking, top_n=50,
                                    shared_at_0p7=float((np.abs(mc) >= 0.7).mean()),
                                    median_cos=float(np.median(np.abs(mc)))))
                dictwide = max_cos(decoders[ka][np.where(alives[ka])[0]], Db,
                                   alive_b=alives[kb])
                rec[-1]["dictionary_wide_shared_at_0p7"] = float((np.abs(dictwide) >= 0.7).mean())
                rec[-1]["top50_minus_dictionary_wide"] = (
                    rec[-1]["shared_at_0p7"] - rec[-1]["dictionary_wide_shared_at_0p7"])
    _wrows(out / ("track_g_recurrence_%s.csv" % a.model), rec)
    res["wall_minutes"] = (time.time() - t_start) / 60
    write_json(out / ("track_g_%s.json" % a.model), res)
    _wrows(out / ("track_g_features_%s.csv" % a.model), grows)
    _wrows(out / ("track_g_thresholds_%s.csv" % a.model), trows)
    log("trkG", "DONE %s in %.1f min" % (a.model, res["wall_minutes"]))


def _wrows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
