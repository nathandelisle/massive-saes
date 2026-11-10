"""Compare decoder matching before and after removing PC1."""
import argparse
import csv
import json
import pathlib
import subprocess
import time

import numpy as np

from rcommon import add_v3, config_hash, log, position_strata, write_json

BUCKET = "ssup-research-079415246777"
PREFIXES = ["b2gen", "b2v2", "pc1-b2"]
RETAIN = [0.01, 0.001, 0.05]
# the manuscript's GPT-2 sweep; the 13-beta Pythia-1.4B sweep is a DIFFERENT cohort
MANUSCRIPT_BETAS = [0.0, 0.03, 0.1, 0.3]


def beta_cfg(spec, seed, beta):
    from v3.common import registry
    return dict(model=spec.key, seed=seed, arch="topk_l1", condition="standard",
                corpus="wikitext", bos_free=False, layer=spec.target_layer,
                steps=registry.STEPS, batch=registry.BATCH_SIZE, d_sae=spec.d_sae,
                k=64.0, beta=float(beta))


def s3_get(key, dest):
    dest = pathlib.Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for pre in PREFIXES:
        r = subprocess.run(["aws", "s3", "cp", "s3://%s/%s/%s" % (BUCKET, pre, key),
                            str(dest), "--only-show-errors"], capture_output=True,
                           text=True, timeout=1800)
        if r.returncode == 0 and dest.exists():
            return dest
    return None


ARCHIVE_SEED = 42        # exp_b2_l1_phase.py: PAIR = [42, 137], and imp_zero / alive /
                         # pc1_cos2 are all computed from PAIR[0] alone.


def archived_ranking(model, beta):
    """Reuse the archived delta-CE ranking: `imp_zero`, `alive`, `pc1_cos2`. No reranking.

    The tag MUST match the producer's `_point_tag`:
        f"{arch}_{float(knob):.6g}".replace(".", "p").replace("-", "m")
    so beta=1e-06 is `topk_l1_1em06`, not `topk_l1_1e-06`. There is deliberately NO
    trailing-zero fallback: it would let a missing beta=10 archive silently resolve to
    beta=1's file, and a different penalty strength is not a substitute.

    These arrays belong to ARCHIVE_SEED and to no other seed. Feature index j has no shared
    identity across independently initialized dictionaries.
    """
    tag = ("topk_l1_" + format(float(beta), ".6g")).replace(".", "p").replace("-", "m")
    name = "feat_%s_%s.npz" % (model, tag)
    f = s3_get("results/b2_l1_phase/" + name, "archf/" + name)
    if f is None:
        return None, None
    return np.load(f, allow_pickle=True), name


def match(Da, Db, v1, retain):
    """Counterparts on RAW geometry; projected and PC1-only cosine for that fixed choice."""
    an = Da / np.linalg.norm(Da, axis=1, keepdims=True).clip(1e-12)
    bn = Db / np.linalg.norm(Db, axis=1, keepdims=True).clip(1e-12)
    C = an @ bn.T
    partner = C.argmax(1)
    raw = C[np.arange(len(an)), partner]
    ca, cb = an @ v1, bn @ v1
    pc1_only = ca * cb[partner]
    pa, pb = an - np.outer(ca, v1), bn - np.outer(cb, v1)
    na, nb = np.linalg.norm(pa, axis=1), np.linalg.norm(pb, axis=1)
    keep = (na >= retain) & (nb[partner] >= retain)
    proj = np.full(len(an), np.nan)
    i = np.where(keep)[0]
    if len(i):
        proj[i] = ((pa[i] / na[i, None]) * (pb[partner[i]] / nb[partner[i], None])).sum(1)
    return dict(partner=partner, raw=raw, proj=proj, pc1_only=pc1_only, keep=keep, na=na)


def pc1_preserving_null(D, v1, rng):
    """Keep each row's signed PC1 coefficient and orthogonal norm; randomize its direction.

    This is the null that matters: it has the same PC1 geometry as the real dictionary and
    no shared off-axis content, so any raw-cosine agreement it produces is agreement the
    shared direction alone can manufacture.
    """
    c = D @ v1
    P = D - np.outer(c, v1)
    n = np.linalg.norm(P, axis=1, keepdims=True)
    R = rng.normal(size=D.shape)
    R -= np.outer(R @ v1, v1)
    R *= n / np.linalg.norm(R, axis=1, keepdims=True).clip(1e-12)
    return np.outer(c, v1) + R


def collect_z(model, spec, seqs, am, sae, mu, feats, chunk):
    """Activation series for a fixed feature list on VALID (non-padded) tokens only."""
    import torch
    from v3.common.splice import splice
    dev = next(model.parameters()).device
    idx = torch.tensor(list(feats), device=dev)
    out = []
    with splice(model, spec, sae=sae, mu=mu, mode="standard", capture=True) as st:
        with torch.no_grad():
            for i in range(0, seqs.shape[0], chunk):
                mb = None if am is None else am[i:i + chunk].to(dev)
                model(seqs[i:i + chunk].to(dev), attention_mask=mb)
                z = st.z.detach().reshape(-1, sae.d_sae)[:, idx]
                if mb is not None:
                    z = z[mb.reshape(-1).bool()]
                out.append(z.float().cpu().numpy())
    return np.concatenate(out, 0)          # [n_valid_tokens, n_feats]


def act_agree(Za, Zb, ord_mask=None):
    """Pearson, support Jaccard, per matched column. Zero-variance columns are counted."""
    n = Za.shape[1]
    pe = np.full(n, np.nan)
    ja = np.full(n, np.nan)
    degen = 0
    for j in range(n):
        a, b = Za[:, j], Zb[:, j]
        if ord_mask is not None:
            a, b = a[ord_mask], b[ord_mask]
        sa, sb = a.std(), b.std()
        if sa < 1e-12 or sb < 1e-12:
            degen += 1
        else:
            pe[j] = float(np.corrcoef(a, b)[0, 1])
        fa, fb = a != 0, b != 0
        u = (fa | fb).sum()
        ja[j] = float((fa & fb).sum() / u) if u else np.nan
    return pe, ja, degen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 137])
    p.add_argument("--betas", nargs="+", type=float, default=MANUSCRIPT_BETAS)
    p.add_argument("--cohort", default="manuscript_gpt2_beta_sweep")
    p.add_argument("--out", default="results/track_f2")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-seqs", type=int, default=64)
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--hours", type=float, default=5.0)
    a = p.parse_args()

    import torch
    add_v3()
    from v3.common import registry
    from v3.common.data import load_model_and_tokenizer, eval_sequences, open_activations
    from v3.common.io_utils import ensure_acts
    from v3.common.pca import estimate_pcs
    from v3.common.saes import load_sae
    from v3.common.fastabl import tuned_chunk

    spec = registry.MODELS[a.model]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()

    adir = ensure_acts(spec, device=dev)
    _e, evecs, mu_np = estimate_pcs(adir)
    v1 = evecs[:, 0].astype(np.float64)
    acts, _ = open_activations(adir)
    tn = np.linalg.norm(np.asarray(acts[:200_000], np.float32) - mu_np[None, :], axis=1)
    massive_thresh = float(np.quantile(tn, 0.99))
    del tn
    torch.cuda.empty_cache()

    model, tok = load_model_and_tokenizer(spec, device=dev)
    model.eval()
    seqs, am = eval_sequences(tok, n_seqs=a.eval_seqs)
    chunk = tuned_chunk(spec)
    mu = torch.tensor(mu_np, dtype=torch.float32, device=dev)

    # ordinary-token mask over VALID tokens, in the same flattened order as collect_z
    rown = _row_norms(model, spec, seqs, am, mu, chunk)
    strata = position_strata(seqs.numpy(), None if am is None else am.numpy(), tok,
                             acts_norm=rown, massive_thresh=massive_thresh)
    valid = strata["all"].reshape(-1)
    ord_mask = strata["ordinary"].reshape(-1)[valid]

    res = dict(model=a.model, cohort=a.cohort, betas=a.betas, seeds=a.seeds,
               retain_thresholds=RETAIN, primary_retain=RETAIN[0], top=a.top,
               ranking_source="ARCHIVED imp_zero from b2_l1_phase; no re-ranking here",
               massive_threshold=massive_thresh,
               n_valid_tokens=int(valid.sum()), n_ordinary_tokens=int(ord_mask.sum()),
               nulls=["identity_permutation (must be 1.0)", "independent_random",
                      "pc1_preserving_random", "shuffled_activations"],
               cells={}, missing=[], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    rows = []

    for beta in a.betas:
        if time.time() - t0 > a.hours * 3600:
            res["cells"]["beta=%g" % beta] = dict(status="unmeasured_budget")
            continue
        arr, arrname = archived_ranking(a.model, beta)
        saes = {}
        for s in a.seeds:
            h = config_hash(beta_cfg(spec, s, beta))
            f = s3_get("saes/%s.pt" % h, "saes_f/%s.pt" % h)
            if f is None:
                res["missing"].append(dict(beta=beta, seed=s, config_hash=h))
                continue
            sae, _x = load_sae(str(f), device=dev)
            sae.eval()
            saes[s] = (sae, h)
        if len(saes) < 2:
            res["cells"]["beta=%g" % beta] = dict(status="checkpoints_missing",
                                                  found=sorted(saes))
            log("trkF2", "beta=%g: only %d checkpoints, skipping" % (beta, len(saes)))
            continue

        cell = dict(status="measured", archived_ranking=arrname,
                    sae_hashes={str(s): saes[s][1] for s in saes})
        rng = np.random.default_rng(int(beta * 1000) + 7)
        for sa in list(saes):
            for sb in list(saes):
                if sa == sb:
                    continue
                Da = saes[sa][0].W_dec.detach().float().cpu().numpy().astype(np.float64)
                Db = saes[sb][0].W_dec.detach().float().cpu().numpy().astype(np.float64)
                if arr is None:
                    cell.setdefault("skipped", []).append(
                        dict(source_seed=sa, target_seed=sb,
                             reason="missing_archive_for_this_beta"))
                    continue
                if sa != ARCHIVE_SEED:
                    # The archive carries seed 42's delta-CE ranking, alive mask and
                    # alignment. Applying its feature IDs to another source dictionary
                    # selects the wrong physical features, so this direction is skipped with
                    # a reason instead of producing a fabricated reverse top-50. A valid
                    # reverse direction needs that seed's OWN frozen ranking.
                    cell.setdefault("skipped", []).append(
                        dict(source_seed=sa, target_seed=sb,
                             reason="missing_source_ranking",
                             note=("archived imp_zero/alive/pc1_cos2 belong to seed %d; "
                                   "feature IDs do not correspond across independently "
                                   "trained dictionaries" % ARCHIVE_SEED)))
                    log("trkF2", "  beta=%g %d->%d SKIPPED: no ranking for source seed %d"
                        % (beta, sa, sb, sa))
                    continue
                sc = arr["imp_zero"].astype(np.float64).copy()
                sc[~arr["alive"]] = -np.inf
                ranked = np.argsort(sc)[::-1]
                c2 = arr["pc1_cos2"]
                alive = arr["alive"]
                rank_src = "archived_seed%d_delta_ce" % ARCHIVE_SEED
                sets = dict(top50=[int(x) for x in ranked[:a.top]],
                            tiling_band=[int(j) for j in np.where(
                                alive & (c2 >= 0.01) & (c2 < 0.5))[0]][:200],
                            canonical=[int(j) for j in np.where(alive & (c2 >= 0.5))[0]][:200],
                            low_align=[int(j) for j in np.where(alive & (c2 < 0.001))[0]][:200])

                nulls = dict(
                    identity_permutation=(Da, Da[rng.permutation(len(Da))]),
                    independent_random=(_unit(rng.normal(size=Da.shape)),
                                        _unit(rng.normal(size=Db.shape))),
                    pc1_preserving_random=(Da, pc1_preserving_null(Db, v1, rng)))

                for retain in RETAIN:
                    g = match(Da, Db, v1, retain)
                    gn = {k: match(x, y, v1, retain) for k, (x, y) in nulls.items()}
                    for name, idx in sets.items():
                        if not idx:
                            continue
                        i = np.array(idx)
                        pr = g["proj"][i]
                        fin = np.isfinite(pr)
                        row = dict(
                            model=a.model, cohort=a.cohort, beta=beta, source_seed=sa,
                            target_seed=sb, feature_set=name, ranking_source=rank_src,
                            retain_threshold=retain, is_primary=(retain == RETAIN[0]),
                            n=len(i), n_excluded_small_remainder=int((~g["keep"][i]).sum()),
                            raw_cos_mean=float(np.mean(g["raw"][i])),
                            raw_shared_0p7=float(np.mean(g["raw"][i] >= 0.7)),
                            proj_cos_mean=(float(np.mean(pr[fin])) if fin.any() else None),
                            proj_shared_0p7=(float(np.mean(pr[fin] >= 0.7))
                                             if fin.any() else None),
                            pc1_only_cos_mean=float(np.mean(g["pc1_only"][i])),
                            mean_retained_norm=float(np.mean(g["na"][i])))
                        for nm, gg in gn.items():
                            row["null_%s_raw" % nm] = float(np.mean(gg["raw"][i]))
                            q = gg["proj"][i]
                            row["null_%s_proj" % nm] = (float(np.nanmean(q))
                                                        if np.isfinite(q).any() else None)
                        rows.append(row)

                # --- activation agreement, on the FIXED matches -------------------------
                # `g` above is the LAST value of the retain loop (0.05). Recompute the
                # primary match explicitly: relying on loop order silently applied a 5%
                # exclusion where 1% was declared.
                g_primary = match(Da, Db, v1, RETAIN[0])
                i = np.array(sets["top50"])
                keep = g_primary["keep"][i]
                src = [int(x) for x in i[keep]]
                tgt = [int(g_primary["partner"][x]) for x in i[keep]]
                if src:
                    Za = collect_z(model, spec, seqs, am, saes[sa][0], mu, src, chunk)
                    Zb = collect_z(model, spec, seqs, am, saes[sb][0], mu, tgt, chunk)
                    pe, ja, degen = act_agree(Za, Zb)
                    peo, jao, _d2 = act_agree(Za, Zb, ord_mask)
                    sh = rng.permutation(Zb.shape[0])
                    pes, jas, _d3 = act_agree(Za, Zb[sh])
                    cell["activation_agreement_%d_%d" % (sa, sb)] = dict(
                        applied_retain_threshold=RETAIN[0],
                        n_pairs=len(src), n_top50=int(len(i)),
                        n_excluded_small_remainder=int((~keep).sum()),
                        excluded_feature_ids=[int(x) for x in i[~keep]],
                        n_zero_variance=degen, n_zero_variance_ordinary=_d2,
                        pearson_mean=float(np.nanmean(pe)),
                        pearson_median=float(np.nanmedian(pe)),
                        pearson_mean_ordinary_tokens=float(np.nanmean(peo)),
                        jaccard_mean=float(np.nanmean(ja)),
                        jaccard_mean_ordinary_tokens=float(np.nanmean(jao)),
                        null_token_permutation_pearson_mean=float(np.nanmean(pes)),
                        null_token_permutation_jaccard_mean=float(np.nanmean(jas)),
                        null_name="token_permutation",
                        null_note=("the TARGET activation matrix's token rows are permuted, "
                                   "which preserves each feature's overall firing rate and "
                                   "destroys temporal correspondence. It is NOT the "
                                   "firing-rate-stratified feature-pair shuffle an earlier "
                                   "docstring named."),
                        per_pair=[dict(src=int(x), tgt=int(y), pearson=_f(pe[k]),
                                       pearson_ordinary=_f(peo[k]), jaccard=_f(ja[k]),
                                       src_var=float(Za[:, k].var()),
                                       tgt_var=float(Zb[:, k].var()))
                                  for k, (x, y) in enumerate(zip(src, tgt))],
                        n_valid_tokens=int(Za.shape[0]),
                        note=("matches fixed by raw decoder geometry at the PRIMARY retain "
                              "threshold; correlations on valid tokens, and separately on "
                              "ordinary tokens because shared outlier positions can carry a "
                              "raw correlation"))
                    log("trkF2", "  beta=%g %d->%d act: pearson %.3f (ord %.3f) jaccard "
                                 "%.3f | shuffled null %.3f"
                        % (beta, sa, sb, np.nanmean(pe), np.nanmean(peo), np.nanmean(ja),
                           np.nanmean(pes)))
        res["cells"]["beta=%g" % beta] = cell
        log("trkF2", "beta=%g done (%.1f min)" % (beta, (time.time() - t0) / 60))
        _w(out / ("track_f2_%s.csv" % a.model), rows)
        write_json(out / ("track_f2_%s.json" % a.model), res)
        for s in list(saes):          # list() -- do not mutate while iterating
            del saes[s]
        torch.cuda.empty_cache()

    res["wall_minutes"] = (time.time() - t0) / 60
    _w(out / ("track_f2_%s.csv" % a.model), rows)
    write_json(out / ("track_f2_%s.json" % a.model), res)
    log("trkF2", "DONE %s: %d rows, %d missing" % (a.model, len(rows), len(res["missing"])))


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def _unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)


def _row_norms(model, spec, seqs, am, mu, chunk):
    import torch
    from v3.common.data import get_blocks
    from v3.common.registry import block_index
    dev = next(model.parameters()).device
    o = np.zeros(seqs.shape, np.float32)
    grab = {}
    blocks = get_blocks(model, spec.block_path)

    def h(m, i, out):
        hh = out[0] if isinstance(out, tuple) else out
        grab["n"] = (hh.float() - mu).norm(dim=-1).cpu().numpy()
    hd = blocks[block_index(spec)].register_forward_hook(h)
    with torch.no_grad():
        for i in range(0, seqs.shape[0], chunk):
            mb = None if am is None else am[i:i + chunk].to(dev)
            model(seqs[i:i + chunk].to(dev), attention_mask=mb)
            o[i:i + chunk] = grab["n"]
    hd.remove()
    return o


def _w(path, rows):
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
