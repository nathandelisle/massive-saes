"""Per-model prep: reproduce the SAEs' training geometry, gate it, profile every block.

1. STORE. Re-run the v3 harvest stream exactly (wikitext-103-v1 train, one paragraph per
   sequence, 128 tokens, batch 32, first 5,000,000 valid positions) but STREAMING: the 5M x d
   store is never written. We keep only what the SAEs depend on --
     mean   float64 over all 5M rows (v3: accumulated from the fp32 hook output)
     rows   the first 100,000 rows, cast to fp16 exactly as v3 stored them
            (PCA uses the first 50,000; the alive rule uses the first 100,000)
   plus unigram token counts over the same 5M tokens (A1's log-frequency target comes from a
   corpus separate from every evaluation partition) and the last train row consumed, which
   is asserted to precede every evaluation partition.
2. PCA. v3's rule: first 50k rows centered by the full mean, float64 covariance, eigh.
3. ALIVE. v3's rule per dictionary: firing rate > 1e-3 on the first 100k rows, bypass inputs
   PC1-projected first. Also firing rates on the `cal` partition (the new, independent pool).
4. GATE 3. PC1 variance share and every dictionary's alive count are compared with the
   archived A5 values. A model that fails is reported and its downstream jobs refuse to run.
5. A2 PROFILE. Every block on the `cal` partition: centred PC1 share, uncentred top-direction
   second-moment share, cos(PC1, mean), channel outliers, PC1 coefficient by position and its
   between-/within-position variance split, direction preservation vs the site's PC1, and each
   block's attention / MLP write along the site's PC1. Pre-rise / post-fall sites for the layer
   experiment are chosen HERE, from calibration geometry, before any SAE sharing is seen.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from . import common, modelio, partitions
from .common import log
from v3.common import registry

N_TOKENS = registry.TRAIN_TOKENS
KEEP = registry.ALIVE_SAMPLE          # 100,000
# The ARCHIVED alive counter (v3 io_utils.alive) iterated `range(0, n, 8192)` over the store
# and sliced acts[i:i+8192] without clamping to n, so its last batch ran to row 106,496 while
# the denominator stayed 100,000. Rates were inflated ~6.5% and ~+120 atoms per dictionary
# were called alive that the stated rule does not admit. We keep 106,496 rows so the archived
# rule can be replicated exactly (gate 3 checks it), and report the corrected rule beside it.
KEEP_ARCHIVED = ((registry.ALIVE_SAMPLE + 8191) // 8192) * 8192   # 106,496
PCA_N = registry.PCA_SAMPLE           # 50,000
TOPK_EVECS = 256


def stream_store(model, tok, spec, n_tokens=N_TOKENS, keep=KEEP_ARCHIVED, layer=None, batch_seqs=32,
                 max_rows_scan=None):
    """Exact replay of v3 data.tokenized_batches + harvest_activations, keeping row indices."""
    rows = partitions._rows()
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    d = spec.d_model
    mean = np.zeros(d, dtype=np.float64)
    kept = np.zeros((keep, d), dtype=np.float16)
    counts = np.zeros(len(tok), dtype=np.int64)
    written, produced, last_row, last_print = 0, 0, -1, 0
    max_tokens = int(n_tokens * 1.05) + partitions.SEQ_LEN
    b_ids, b_am, b_row = [], [], []
    t0 = time.time()

    def flush():
        nonlocal written, last_print
        ids = torch.tensor(b_ids)
        am = torch.tensor(b_am)
        h = modelio.prefix(model, spec, ids, am, layer=layer).float()
        keepm = am.bool().to(h.device)
        flat = h[keepm]
        take = min(flat.shape[0], n_tokens - written)
        chunk = flat[:take].cpu().numpy()
        if written < keep:
            k = min(keep - written, take)
            kept[written:written + k] = chunk[:k].astype(np.float16)
        mean[:] += chunk.astype(np.float64).sum(0)
        tk = ids[am.bool()][:take].numpy()
        np.add.at(counts, tk, 1)
        written += take

    for i, text in enumerate(rows):
        if not text or text.isspace():
            continue
        t = tok(text, add_special_tokens=True, truncation=True,
                max_length=partitions.SEQ_LEN)["input_ids"]
        if not t:
            continue
        b_ids.append(t + [pad] * (partitions.SEQ_LEN - len(t)))
        b_am.append([1] * len(t) + [0] * (partitions.SEQ_LEN - len(t)))
        b_row.append(i)
        if len(b_ids) == batch_seqs:
            flush()
            produced += int(sum(sum(m) for m in b_am))
            last_row = b_row[-1]
            b_ids, b_am, b_row = [], [], []
            if written - last_print >= 1_000_000 or written >= n_tokens:
                last_print = written
                log("store", "%s %d/%d tokens  %.0fs" % (spec.key, written, n_tokens,
                                                         time.time() - t0))
            if written >= n_tokens or produced >= max_tokens:
                break
        if max_rows_scan and i >= max_rows_scan:
            break
    if b_ids and written < n_tokens:
        flush()
        last_row = b_row[-1]
    mean /= max(written, 1)
    return dict(mean=mean, kept=kept[:min(keep, written)], counts=counts,
                n_written=int(written), last_row=int(last_row))


def pca(kept, mean, n=PCA_N):
    X = kept[:n].astype(np.float64) - mean[None, :]
    C = (X.T @ X) / (X.shape[0] - 1)
    ev, evec = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1]
    return ev[o], evec[:, o]


@torch.no_grad()
def firing_rates_rows(dic, rows, batch=8192):
    cnt = torch.zeros(dic.d_sae, device=dic.device)
    n = rows.shape[0]
    for i in range(0, n, batch):
        x = torch.as_tensor(rows[i:i + batch].astype(np.float32), device=dic.device)
        z, _ = dic.encode(x)
        cnt += (z != 0).float().sum(0)
    return (cnt / n).cpu().numpy()


@torch.no_grad()
def partition_rows(model, spec, part, max_tokens=16384):
    """Site residual rows (valid positions only) for a partition, computed ONCE, on CPU."""
    out = []
    for sel, ids, am in part.batches(max_tokens=max_tokens):
        h = modelio.prefix(model, spec, ids, am)
        out.append(h[am.bool().to(h.device)].cpu())
    return torch.cat(out)


@torch.no_grad()
def firing_rates_partition(dic, rows, batch=8192):
    cnt = torch.zeros(dic.d_sae, device=dic.device)
    for i in range(0, rows.shape[0], batch):
        z, _ = dic.encode(rows[i:i + batch].to(dic.device))
        cnt += (z != 0).float().sum(0)
    return (cnt / max(rows.shape[0], 1)).cpu().numpy(), int(rows.shape[0])


# ---------------------------------------------------------------------------------------
# A2: every-block geometry on the calibration partition
# ---------------------------------------------------------------------------------------
def _sublayers(block, family):
    """(attention-write module, mlp-write module) whose OUTPUT is added to the residual."""
    if family == "gpt2":
        return block.attn, block.mlp
    if family == "pythia":
        return block.attention, block.mlp
    if family == "gemma":           # Gemma-2 normalises each sublayer output before the add
        return block.post_attention_layernorm, block.post_feedforward_layernorm
    return block.self_attn, block.mlp


@torch.no_grad()
def layer_profile(model, tok, spec, cal, v1_site, mu_site, mem_gb=12.0):
    """Two passes over `cal`: (1) per-layer moments -> PC1 per layer; (2) coefficient,
    position, outlier and sublayer-write statistics against those PC1s."""
    blks = modelio.blocks(model, spec)
    L, d = len(blks), spec.d_model
    per_layer_bytes = d * d * 8 + d * 8 * 3
    if torch.cuda.is_available():
        # size the fp64 accumulator groups to what is actually free on the tightest device
        # (a 22 GB A10G holding a 16 GB model OOM'd at the fixed 12 GB budget)
        free = min(torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count()))
        mem_gb = min(mem_gb, max(0.3, 0.45 * free / 1e9) * torch.cuda.device_count())
    group = max(1, int(mem_gb * 1e9 // per_layer_bytes))
    log("profile", "%d blocks, accumulator groups of %d (%.1f GB budget)" % (L, group, mem_gb))
    layers = list(range(L))
    res = {}
    v1s = torch.as_tensor(v1_site, dtype=torch.float64)
    for g0 in range(0, L, group):
        grp = layers[g0:g0 + group]
        acc = {l: dict(S=None, SS=None, n=0, S0=None, n0=0) for l in grp}
        hooks = []

        def mk(l):
            def hk(m, i, o):
                h = (o[0] if isinstance(o, tuple) else o).detach()
                am_ = hk.am.to(h.device).bool()
                x = h[am_].double()
                a = acc[l]
                if a["S"] is None:
                    a["S"] = torch.zeros(d, dtype=torch.float64, device=h.device)
                    a["SS"] = torch.zeros(d, d, dtype=torch.float64, device=h.device)
                    a["S0"] = torch.zeros(d, dtype=torch.float64, device=h.device)
                a["S"] += x.sum(0)
                a["SS"] += x.T @ x
                a["n"] += x.shape[0]
                a["S0"] += h[:, 0].double().sum(0)
                a["n0"] += h.shape[0]
            return hk
        fns = {}
        for l in grp:
            fns[l] = mk(l)
            hooks.append(blks[l].register_forward_hook(fns[l]))
        for sel, ids, am in cal.batches(max_tokens=8192):
            for l in grp:
                fns[l].am = am
            model(ids.to(modelio.input_device(model)),
                  attention_mask=am.to(modelio.input_device(model)))
        for hd in hooks:
            hd.remove()
        for l in grp:
            a = acc[l]
            n = a["n"]
            m = a["S"] / n
            C = a["SS"] / n - torch.outer(m, m)
            ev, evec = torch.linalg.eigh(C)
            ev, evec = ev.flip(0), evec.flip(1)
            U = a["SS"] / n                                   # uncentred second moment
            eu = torch.linalg.eigvalsh(U).flip(0)
            v1 = evec[:, 0]
            mn = m / (m.norm() + 1e-30)
            res[l] = dict(
                block=l, hidden_state_index=l + 1, n_rows=int(n),
                pc1_var_share=float(ev[0] / ev.sum()), pc2_var_share=float(ev[1] / ev.sum()),
                uncentered_top_share=float(eu[0] / eu.sum()),
                cos_pc1_mean=float(abs(v1 @ mn)),
                cos_pc1_site_pc1=float(abs(v1.cpu() @ v1s)),
                mean_norm=float(m.norm()), pos0_mean_norm=float((a["S0"] / a["n0"]).norm()),
                _v1=v1.float().cpu().numpy(), _m=m.float().cpu().numpy())
        del acc
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # pass 2: coefficients along each layer's PC1 by position, channel outliers, writes
    coef_stats = {l: dict(by_pos_sum=np.zeros(partitions.SEQ_LEN),
                          by_pos_sq=np.zeros(partitions.SEQ_LEN),
                          by_pos_n=np.zeros(partitions.SEQ_LEN),
                          chan_absmax=np.zeros(d, np.float32), chan_abs_sum=np.zeros(d),
                          nrows=0) for l in range(L)}
    writes = {l: dict(attn_proj_sum=0.0, attn_proj_abs=0.0, mlp_proj_sum=0.0,
                      mlp_proj_abs=0.0, attn_pos0=0.0, mlp_pos0=0.0, n=0, n0=0)
              for l in range(L)}
    v1t = {l: torch.as_tensor(res[l]["_v1"]) for l in range(L)}
    mt = {l: torch.as_tensor(res[l]["_m"]) for l in range(L)}
    vsite = torch.as_tensor(v1_site, dtype=torch.float32)
    hooks = []
    cur = {}

    def mk2(l):
        def hk(m, i, o):
            h = (o[0] if isinstance(o, tuple) else o).detach().float()
            am_ = cur["am"].to(h.device)
            c = ((h - mt[l].to(h.device)) @ v1t[l].to(h.device))       # [B,T]
            cs = coef_stats[l]
            Tn = h.shape[1]
            msk = am_.float()
            cs["by_pos_sum"][:Tn] += (c * msk).sum(0).cpu().numpy()
            cs["by_pos_sq"][:Tn] += (c * c * msk).sum(0).cpu().numpy()
            cs["by_pos_n"][:Tn] += msk.sum(0).cpu().numpy()
            hv = h[am_.bool()]
            cs["chan_absmax"] = np.maximum(cs["chan_absmax"], hv.abs().amax(0).cpu().numpy())
            cs["chan_abs_sum"] += hv.abs().sum(0).double().cpu().numpy()
            cs["nrows"] += hv.shape[0]
        return hk

    def mkw(l, kind):
        def hk(m, i, o):
            w = (o[0] if isinstance(o, tuple) else o).detach().float()
            am_ = cur["am"].to(w.device).bool()
            p = w @ vsite.to(w.device)                                  # [B,T]
            ww = writes[l]
            ww[kind + "_proj_sum"] += float(p[am_].sum())
            ww[kind + "_proj_abs"] += float(p[am_].abs().sum())
            ww[kind + "_pos0"] += float(p[:, 0].sum())
            if kind == "attn":
                ww["n"] += int(am_.sum())
                ww["n0"] += int(p.shape[0])
        return hk
    for l in range(L):
        hooks.append(blks[l].register_forward_hook(mk2(l)))
        a_mod, m_mod = _sublayers(blks[l], spec.family)
        hooks.append(a_mod.register_forward_hook(mkw(l, "attn")))
        hooks.append(m_mod.register_forward_hook(mkw(l, "mlp")))
    with torch.no_grad():
        for sel, ids, am in cal.batches(max_tokens=8192):
            cur["am"] = am
            model(ids.to(modelio.input_device(model)),
                  attention_mask=am.to(modelio.input_device(model)))
    for hd in hooks:
        hd.remove()

    out = []
    for l in range(L):
        r = {k: v for k, v in res[l].items() if not k.startswith("_")}
        cs = coef_stats[l]
        n = np.maximum(cs["by_pos_n"], 1)
        mpos = cs["by_pos_sum"] / n
        vpos = cs["by_pos_sq"] / n - mpos ** 2
        tot_n = cs["by_pos_n"].sum()
        grand = cs["by_pos_sum"].sum() / max(tot_n, 1)
        between = float((cs["by_pos_n"] * (mpos - grand) ** 2).sum() / max(tot_n, 1))
        within = float((cs["by_pos_n"] * vpos).sum() / max(tot_n, 1))
        chan_mean_abs = cs["chan_abs_sum"] / max(cs["nrows"], 1)
        med = float(np.median(chan_mean_abs))
        r.update(coef_pos0_mean=float(mpos[0]), coef_pos0_sd=float(np.sqrt(max(vpos[0], 0))),
                 coef_rest_mean=float((cs["by_pos_sum"][1:].sum()) / max(cs["by_pos_n"][1:].sum(), 1)),
                 coef_between_position_var=between, coef_within_position_var=within,
                 coef_between_share=between / max(between + within, 1e-30),
                 chan_absmax_over_median_meanabs=float(cs["chan_absmax"].max() / max(med, 1e-30)),
                 top_outlier_channels=[int(c) for c in np.argsort(-cs["chan_absmax"])[:8]])
        w = writes[l]
        r.update(attn_write_on_site_pc1_mean=w["attn_proj_sum"] / max(w["n"], 1),
                 mlp_write_on_site_pc1_mean=w["mlp_proj_sum"] / max(w["n"], 1),
                 attn_write_on_site_pc1_absmean=w["attn_proj_abs"] / max(w["n"], 1),
                 mlp_write_on_site_pc1_absmean=w["mlp_proj_abs"] / max(w["n"], 1),
                 attn_write_pos0_mean=w["attn_pos0"] / max(w["n0"], 1),
                 mlp_write_pos0_mean=w["mlp_pos0"] / max(w["n0"], 1))
        out.append(r)
    return out


def choose_sites(profile, canonical_block, thr=0.5):
    """Pre-rise and post-fall blocks from CALIBRATION geometry, chosen before any sharing.

    pre-rise : the last block before the PC1 share first reaches `thr`
    post-fall: the first block after the last block at or above `thr`
    Falls back to prespecified early (L/4) / late (3L/4) blocks, and records that it did.
    """
    s = [p["pc1_var_share"] for p in profile]
    L = len(s)
    above = [i for i, v in enumerate(s) if v >= thr]
    rule = "calibration-geometry"
    if above and above[0] > 0 and above[0] <= canonical_block:
        pre = above[0] - 1
    else:
        pre, rule = max(0, L // 4 - 1), "fallback-early-late"
    last = above[-1] if above else None
    if last is not None and last < L - 1 and last >= canonical_block:
        post = last + 1
    else:
        post, rule = min(L - 1, (3 * L) // 4), "fallback-early-late"
    if pre == canonical_block:
        pre = max(0, canonical_block - 2)
    if post == canonical_block:
        post = min(L - 1, canonical_block + 2)
    return dict(pre_rise_block=int(pre), post_fall_block=int(post),
                canonical_block=int(canonical_block), rule=rule, threshold=thr,
                pc1_share_pre=s[pre], pc1_share_canonical=s[canonical_block],
                pc1_share_post=s[post])


# ---------------------------------------------------------------------------------------
def fetch_dictionaries(spec, inv, dest):
    """Every TopK std/bypass x 5 seeds dictionary for this model, from wherever it is in S3."""
    out = {}
    for r in inv["rows"]:
        if r["model"] != spec.key or r["arch"] != "topk":
            continue
        h = r["config_hash"]
        p = pathlib.Path(dest) / ("%s.pt" % h)
        if not p.exists():
            src = r.get("rx_prefix") or (r["prefixes"][0] if r.get("prefixes") else None)
            if src is None:
                raise RuntimeError("no S3 location for %s %s" % (spec.key, h))
            common.s3_get("%s/saes/%s.pt" % (src, h), p)
        out[(r["condition"], int(r["seed"]))] = p
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-tokens", type=int, default=N_TOKENS)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--archived-mean", default=None,
                    help="70B: archived mean.npy + pca npz replace the 5M stream; only the "
                         "first 100k rows are re-harvested (alive rule, PCA check)")
    ap.add_argument("--dtype", default=None)
    a = ap.parse_args(argv)
    spec = modelio.spec_of(a.model)
    out = common.RESULTS / "prep" / a.model
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = getattr(torch, a.dtype) if a.dtype else None
    model, tok = modelio.load(spec, device=dev, device_map=a.device_map, dtype=dt)
    rep = dict(model=a.model, env=common.env_info(), site_block=modelio.site_index(spec),
               hidden_state_index=modelio.site_index(spec) + 1)

    # 1-2. store and PCA
    if a.archived_mean:
        st = stream_store(model, tok, spec, n_tokens=KEEP_ARCHIVED, keep=KEEP_ARCHIVED)
        arch = np.load(pathlib.Path(a.archived_mean) / "pca_50000.npz")
        mean = np.load(pathlib.Path(a.archived_mean) / "mean.npy")
        ev, evec = pca(st["kept"], mean)
        rep["store"] = dict(mode="archived mean (5M) + re-harvested first %d rows" % KEEP,
                            n_written=st["n_written"], last_row=st["last_row"])
        rep["pca_vs_archived"] = dict(
            cos_v1=float(abs(evec[:, 0] @ arch["evecs"][:, 0])),
            pc1_share_archived=float(arch["evals"][0] / arch["evals"].sum()),
            pc1_share_now=float(ev[0] / ev.sum()))
    else:
        st = stream_store(model, tok, spec, n_tokens=a.n_tokens)
        mean = st["mean"]
        ev, evec = pca(st["kept"], mean)
        rep["store"] = dict(mode="full re-stream", n_written=st["n_written"],
                            last_row=st["last_row"])
    rep["store"]["precedes_partitions"] = bool(st["last_row"] < partitions.MIN_ROW)
    v1 = evec[:, 0]
    rep["pc1"] = dict(pc1_var_explained=float(ev[0] / ev.sum()),
                      pc2_var_explained=float(ev[1] / ev.sum()),
                      spectral_gap=float(ev[0] / max(ev[1], 1e-30)),
                      cos_pc1_mean=float(abs(v1 @ (mean / (np.linalg.norm(mean) + 1e-12)))))
    np.save(out / "mean.npy", mean)
    common.save_npz(out / "pca.npz", evals=ev, evecs=evec[:, :TOPK_EVECS].astype(np.float32),
                    v1=v1)
    np.save(out / "unigram_counts.npy", st["counts"])
    log("prep", "pc1 share %.5f" % rep["pc1"]["pc1_var_explained"])

    # 3. alive
    inv = json.loads((common.PKG / "inventory.json").read_text())
    paths = fetch_dictionaries(spec, inv, common.SAES)
    cal = partitions.Partition("cal", tok)
    cal_rows = partition_rows(model, spec, cal)
    alive, alive_corr, rates, rates_cal = {}, {}, {}, {}
    sdev = modelio.site_device(model, spec)
    for (cond, seed), p in sorted(paths.items()):
        dic, _ = modelio.load_dictionary(p, cond, mean, v1, sdev)
        r_corr = firing_rates_rows(dic, st["kept"][:KEEP])                # stated rule
        cnt = firing_rates_rows(dic, st["kept"][:KEEP_ARCHIVED]) * min(len(st["kept"]), KEEP_ARCHIVED)
        r_arch = cnt / KEEP                                                # archived counter
        rc, ncal = firing_rates_partition(dic, cal_rows)
        key = "%s_s%d" % (cond, seed)
        rates[key], rates_cal[key] = r_corr, rc
        alive[key] = r_arch > registry.ALIVE_FIRING_RATE        # PRIMARY: historical universe
        alive_corr[key] = r_corr > registry.ALIVE_FIRING_RATE
        del dic
    common.save_npz(out / "alive.npz", **{"alive_" + k: v for k, v in alive.items()},
                    **{"alivecorr_" + k: v for k, v in alive_corr.items()},
                    **{"rate_" + k: v for k, v in rates.items()},
                    **{"ratecal_" + k: v for k, v in rates_cal.items()})
    rep["alive_rules"] = dict(
        primary="archived counter: firing counts over the first 106,496 store rows / 100,000 "
                "(reproduces every archived ranking universe)",
        corrected="stated rule: firing rate over the first 100,000 rows > 1e-3",
        n_alive_corrected={k: int(v.sum()) for k, v in alive_corr.items()})
    rep["n_alive"] = {k: int(v.sum()) for k, v in alive.items()}
    rep["n_alive_cal"] = {k: int((v > registry.ALIVE_FIRING_RATE).sum())
                          for k, v in rates_cal.items()}
    rep["partition_cal"] = cal.summary()

    # 4. gate 3 against the archive
    ref = json.loads((common.PKG / "archive_ref.json").read_text()).get(a.model)
    g = dict(status="NO_ARCHIVE_REFERENCE")
    if ref:
        dpc = abs(rep["pc1"]["pc1_var_explained"] - ref["pc1_var_explained"])
        dal = {}
        for k, v in rep["n_alive"].items():
            if k in ref.get("n_alive", {}):
                dal[k] = (v, ref["n_alive"][k], abs(v - ref["n_alive"][k]) / max(ref["n_alive"][k], 1))
        worst = max((x[2] for x in dal.values()), default=0.0)
        ok = dpc < 2e-3 and worst < 0.005 and rep["store"]["precedes_partitions"]
        g = dict(status="PASS" if ok else "FAIL", pc1_share_now=rep["pc1"]["pc1_var_explained"],
                 pc1_share_archived=ref["pc1_var_explained"], abs_diff_pc1=dpc,
                 alive_now_vs_archived=dal, worst_alive_rel_diff=worst,
                 tolerance=dict(pc1_abs=2e-3, alive_rel=0.005),
                 note="alive counts under the ARCHIVED counter (106,496 rows / 100,000)")
    rep["gate3"] = g
    log("prep", "GATE 3 %s" % g["status"])

    # 5. A2 profile and site choice
    prof = layer_profile(model, tok, spec, cal, v1, mean)
    rep["sites"] = choose_sites(prof, modelio.site_index(spec))
    common.write_json(out / "layer_profile.json", prof)
    common.write_json(out / "prep.json", rep)
    common.s3_sync_up(out, "%s/%s/prep" % (common.SHARED, a.model))
    log("prep", "done; sites %s" % rep["sites"])
    return rep


if __name__ == "__main__":
    main()
