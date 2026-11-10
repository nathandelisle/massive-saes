"""Importance job for ONE dictionary (model, condition, seed): gates, then B1/B2.

Stages (each writes its own output and is skipped if that output exists, so a relaunched box
resumes):
  gates        1, 2, 3', 4, 5, 7, 9 on this dictionary, and an archive reproduction of the
               historical zero-ablation scores where an archived array exists for this source
  cal          calibration statistics (means, firing rates, donor samples) on `cal`
  rank_zero    zero ablation, every alive feature, ranking partition
  rank_grad    gradient x input (3 estimands) + integrated gradients with step doubling
  valid_zero   zero ablation on the independent validation partition
  valid_grad
  rank_sp      support-preserving resampling, `draws` keyed draws
  rank_mg      marginal resampling (zeros included), `draws` keyed draws
  rank_st      position-stratified support-preserving resampling (third operator)
  rank_mean    mean ablation
  valid_sp / valid_mg   only if the compute rule leaves room

THE COMPUTE RULE (declared here, before any result, and recorded in the output): the draw
count for the resampling operators is chosen from the MEASURED zero-ablation throughput of
this very box: 16 keyed draws always; the resampled universe is every alive feature if that fits, else the registered universe U; mean ablation /
validation-partition resampling run only if the remaining budget covers them. It depends on
throughput, never on any result.
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from . import common, engine as E, modelio as M, partitions as P
from .common import log
from v3.common import registry
from v3.common import splice as v3splice


def fetch_prep(model):
    d = common.WORK / "prep" / model
    for f in ("mean.npy", "pca.npz", "alive.npz", "prep.json"):
        common.s3_get("%s/%s/prep/%s" % (common.SHARED, model, f), d / f)
    prep = json.loads((d / "prep.json").read_text())
    mean = np.load(d / "mean.npy")
    pca = np.load(d / "pca.npz")
    alive = np.load(d / "alive.npz")
    return prep, mean, pca, alive


def fetch_sae(model, cond, seed):
    inv = json.loads((common.PKG / "inventory.json").read_text())
    r = next(r for r in inv["rows"] if r["model"] == model and r["arch"] == "topk"
             and r["condition"] == cond and int(r["seed"]) == int(seed))
    h = r["config_hash"]
    p = common.SAES / ("%s.pt" % h)
    src = r.get("rx_prefix") or r["prefixes"][0]
    common.s3_get("%s/saes/%s.pt" % (src, h), p)
    return p, h


# ---------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def run_gates(model, tok, spec, suf, dic, sae_path, mean, v1, alive, cond, seed, out):
    rep = {}
    dev = dic.device
    part = P.Partition("rank", tok, max_seqs=12)
    ids, am = part.ids, part.am
    T = int(part.lengths.max())
    ids, am = ids[:, :T], am[:, :T]
    hs, native = M.all_hidden(model, ids, am)
    b = M.site_index(spec)
    h = M.prefix(model, spec, ids, am)
    v = am.bool()
    # gate 1: capture point equals hidden_states[b+1] on exact token ids
    g1 = float((h.float() - hs[b + 1].float().to(h.device))[v.to(h.device)].abs().max())
    rep["gate1"] = dict(max_abs=g1, pass_=g1 == 0.0, what="prefix capture == hidden_states[b+1]")
    # gate 2: identity suffix and projection+restore identity reproduce native logits
    lg = suf.logits(h, am)
    d2 = float((lg.float() - native.float().to(lg.device))[v.to(lg.device)].abs().max())
    x = h.float() - dic.mu
    coef = x @ dic.v1
    rec = (x - coef[..., None] * dic.v1) + coef[..., None] * dic.v1 + dic.mu
    lg2 = suf.logits(rec.to(h.dtype), am)
    d2b = float((lg2.float() - native.float().to(lg2.device))[v.to(lg2.device)].abs().max())
    # dtype-appropriate, relative to the logit scale: half precision 1e-3, float32 1e-5 (the
    # projection+restore round trip on ~3e3-magnitude massive activations costs ~1 fp32 ulp)
    scale = float(native.float().abs().max())
    tol = (1e-3 if native.dtype != torch.float32 else 1e-5) * scale
    # projection+restore is an identity up to ONE rounding of the cast back to the model dtype;
    # check that in residual space (every element within 1 ulp of native), which is strict in
    # every dtype. The resulting logit gap (bf16: one-ulp flips can move a logit ~0.25) is
    # recorded as information, not gated on.
    rh = rec.to(h.dtype).float()
    hf = h.float()
    # forward-error bound of the fp32 ops (x - c v1 + c v1 + mu: 4 roundings, each relative to the
    # operands' magnitude, which for massive activations is |c v1| >> |h_i|) plus one rounding of
    # the final cast
    cv = (coef[..., None] * dic.v1).abs()
    e32 = torch.finfo(torch.float32).eps
    ulp = (torch.finfo(h.dtype).eps * hf.abs() + 4 * e32 * (x.abs() + cv + dic.mu.abs())
           + torch.finfo(h.dtype).tiny)
    vv = v.to(h.device)
    viol = int(((rh - hf).abs() > ulp)[vv].sum())
    rep["gate2"] = dict(identity_max_abs_logit=d2, proj_restore_max_abs_logit_info=d2b, tol=tol,
                        proj_restore_elements_beyond_rounding_bound=viol,
                        proj_restore_frac_elements_changed=float(((rh != hf)[vv]).float().mean()),
                        pass_=d2 <= tol and viol == 0)
    # gate 3': this module's splice == the archived v3 splice hook, same SAE file
    # Compared in RESIDUAL space: the v3 hook computes the PC1 coefficient on [B,T,d] and this
    # module on [N,d], so float32 reductions run in a different order; after the cast to a
    # half dtype that can flip single elements by one rounding step, which at massive-activation
    # magnitudes moves logits visibly. Pass = every element within one rounding step of the cast
    # plus the float32 forward-error bound of the splice; the logit gap is recorded as info.
    sae_v3, _ = registry_load(sae_path, dev)
    blk = M.blocks(model, spec)[b]
    cap = {}

    def grab(key):
        def f(m, i, o):
            cap[key] = (o[0] if isinstance(o, tuple) else o).detach().float().clone()
        return f
    with v3splice.splice(model, spec, sae=sae_v3, mu=mean, v1=v1, mode=cond):
        hd = blk.register_forward_hook(grab("v3"))
        ref = model(ids.to(M.input_device(model)), attention_mask=am.to(M.input_device(model))).logits
        hd.remove()
    with M.splice_hook(model, spec, dic):
        hd = blk.register_forward_hook(grab("mine"))
        mine = model(ids.to(M.input_device(model)), attention_mask=am.to(M.input_device(model))).logits
        hd.remove()
    vr = v.to(ref.device)
    d3 = max(float((ref[i].float() - mine[i].float())[vr[i]].abs().max()) for i in range(ref.shape[0]))
    del ref, mine
    a3, b3 = cap["v3"], cap["mine"]
    hv = h.float().to(a3.device)
    xr = (hv - dic.mu.to(a3.device))
    cmag = (xr @ dic.v1.to(a3.device)).abs()[..., None] * dic.v1.to(a3.device).abs()
    e32 = torch.finfo(torch.float32).eps
    bound = (torch.finfo(h.dtype).eps * torch.maximum(a3.abs(), b3.abs())
             + 8 * e32 * (a3.abs() + cmag + dic.mu.to(a3.device).abs() + xr.abs())
             + torch.finfo(h.dtype).tiny)
    vv = v.to(a3.device)
    viol = int(((a3 - b3).abs() > bound)[vv].sum())
    rep["gate3_splice_equals_v3"] = dict(
        residual_elements_beyond_rounding_bound=viol,
        residual_frac_elements_differing=float(((a3 != b3)[vv]).float().mean()),
        residual_max_abs=float((a3 - b3)[vv].abs().max()),
        max_abs_logit_info=d3, pass_=viol == 0)
    # PC1 leakage of the bypass reconstruction
    z, coefb = dic.encode(h.reshape(-1, h.shape[-1]))
    recon = (z @ dic.W_dec + dic.b_dec)
    leak = float(((recon @ dic.v1) ** 2).mean() / max(float((coefb ** 2).mean()) if coefb is not None else 1.0, 1e-30))
    rep["pc1_leakage"] = dict(mean_sq_recon_on_v1_over_mean_sq_coef=leak, mode=cond,
                              note="bypass: the SAE output's PC1 component relative to the routed one")
    # gate 4: L0 <= k and denominators
    nz = (z != 0).sum(1)
    m = v.to(nz.device).reshape(-1)
    k = getattr(dic.sae, "k", None)
    rep["gate4"] = dict(max_row_support=int(nz[m].max()), k=k,
                        l0_mean=float(nz[m].float().mean()),
                        n_targets=int((part.lengths[:] - 1).clip(0).sum()),
                        pass_=(k is None) or int(nz[m].max()) <= int(k))
    # gate 5: finite NLL, Gemma softcap + eager attention present
    nll, val = M.suffix_nll(suf, h, ids, am)
    fin = bool(torch.isfinite(nll).all())
    g5 = dict(finite=fin, dtype=str(h.dtype))
    if spec.family == "gemma":
        g5["final_logit_softcapping"] = getattr(model.config, "final_logit_softcapping", None)
        g5["attn_impl"] = getattr(model.config, "_attn_implementation", None)
        fin = fin and g5["final_logit_softcapping"] is not None and g5["attn_impl"] == "eager"
    g5["pass_"] = fin
    rep["gate5"] = g5
    common.write_json(out / "gates_partial.json", rep)
    return rep


def registry_load(path, dev):
    from v3.common import saes as v3saes
    return v3saes.load_sae(str(path), device=dev)


def gate7_engine(model, spec, suf, dic, tok, alive, out, n_seqs=16):
    """Engine (suffix, batched items) vs full-forward hook, for random / dense / rare features."""
    part = P.Partition("rank", tok, max_seqs=n_seqs)
    cache = E.Cache(model, spec, suf, dic, part, max_tokens=4096)
    fires = np.nonzero((cache.fire_nseq > 0) & np.asarray(alive, bool))[0]
    rng = np.random.default_rng(7)
    cos2 = ((dic.W_dec / dic.W_dec.norm(dim=1, keepdim=True)) @ dic.v1).pow(2).cpu().numpy()
    dense = fires[np.argsort(-cos2[fires])][:4]
    rare = fires[np.argsort(cache.fire_nseq[fires])][:4]
    rnd = rng.choice(fires, size=min(4, len(fires)), replace=False)
    feats = np.unique(np.concatenate([dense, rare, rnd]))
    S, F = E.items_firing(cache, feats)
    dl, _ = cache.run(S, F, lambda s, f, old, am: torch.zeros_like(old), max_tokens=4096)
    eng = {int(j): float(dl[F == j].sum()) for j in feats}
    def reference(per_seq):
        """Full-forward reference. per_seq=False: one padded batch of all sequences;
        per_seq=True: each sequence alone, trimmed. Same math, different kernel shapes, so the
        two differ only by floating-point rounding: their gap is the dtype's noise floor."""
        groups = ([np.array([i]) for i in range(len(part.lengths))] if per_seq
                  else [np.arange(len(part.lengths))])
        out = {int(j): 0.0 for j in feats}
        with torch.no_grad():
            for g in groups:
                T = int(part.lengths[g].max())
                ids, am = part.ids[g, :T], part.am[g, :T]
                go = lambda: model(ids.to(M.input_device(model)),
                                   attention_mask=am.to(M.input_device(model))).logits
                with M.splice_hook(model, spec, dic):
                    n0, v0 = M.token_nll(go(), ids, am)
                for j in feats:
                    def zt(zz, j=int(j)):
                        zz = zz.clone(); zz[..., j] = 0.0; return zz
                    with M.splice_hook(model, spec, dic, z_transform=zt):
                        n2, _ = M.token_nll(go(), ids, am)
                    out[int(j)] += float(((n2 - n0) * v0).sum())
        return out
    ref = reference(False)
    ref_b = reference(True)
    scale = max(1.0, max(abs(x) for x in ref.values()))
    worst = max(abs(eng[j] - ref[j]) for j in eng)
    noise = max(abs(ref[j] - ref_b[j]) for j in eng)
    # engine and reference are two equally valid roundings of the same quantity; the engine
    # must agree with the reference to within 3x the reference's own rounding noise floor
    # (plus 1e-3 of the logit scale). In float32 the floor is ~0 and this is the strict check.
    tol = 1e-3 * scale + 3 * noise
    rows = [dict(feature=j, engine=eng[j], full_forward=ref[j], full_forward_per_seq=ref_b[j],
                 cls=("dense" if j in dense else "rare" if j in rare else "random")) for j in feats]
    return dict(max_abs=worst, tol=tol, noise_floor=noise, pass_=worst <= tol, features=rows,
                dtype=str(next(M.blocks(model, spec)[0].parameters()).dtype))


def gate7_checked(model, spec, suf, dic, tok, alive, out):
    """Gate 7 is a LOGIC check. In half precision the engine and the reference are two different
    roundings whose gap (the measured noise floor) can reach several nats on dense features,
    which would make any tolerance loose. So when the weights fit twice over, the model is
    upcast to float32 for the check (bf16/fp16 -> fp32 -> bf16/fp16 is exact) and the strict
    tolerance applies; the half-precision run is kept as information on measurement noise."""
    p0 = next(model.parameters())
    half = gate7_engine(model, spec, suf, dic, tok, alive, out)
    if p0.dtype == torch.float32 or not torch.cuda.is_available():
        half["precision_of_check"] = str(p0.dtype)
        return half
    nbytes = sum(p.numel() for p in model.parameters()) * 4
    free = min(torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count()))
    ndev = len({p.device for p in model.parameters()})
    if ndev > 1 or nbytes / 2 > 0.7 * free:
        half["precision_of_check"] = str(p0.dtype) + " (fp32 upcast does not fit; weak check)"
        return half
    dt = p0.dtype
    model.float()
    try:
        suf32 = M.Suffix(model, spec)
        g = gate7_engine(model, spec, suf32, dic, tok, alive, out)
    finally:
        model.to(dt)
        torch.cuda.empty_cache()
    g["precision_of_check"] = "float32 (upcast for the logic check)"
    g["half_precision_info"] = {k: half[k] for k in ("max_abs", "noise_floor", "features")}
    return g


def gate9_fd(cache, n_checks=3, eps=1e-2):
    """Finite-difference check of dL/dz on latents, against autograd."""
    part = cache.part
    s = int(np.argmax(part.lengths))
    Tb = int(part.lengths[s])
    Sb = torch.as_tensor([s], device=cache.dic.device)
    idx = cache.Zidx[s, :Tb].long()
    Z = torch.zeros(1, Tb, cache.dic.d_sae, device=cache.dic.device)
    Z[0].scatter_(1, idx.clamp_min(0), torch.where(idx >= 0, cache.Zval[s, :Tb], 0.0))
    Zg = Z.clone().requires_grad_(True)
    loss, _, _ = E._loss_from_z(cache, Sb, Tb, Zg)
    loss.backward()
    g = Zg.grad[0]
    nzpos = torch.nonzero(Z[0])
    rng = np.random.default_rng(9)
    pick = nzpos[torch.as_tensor(rng.choice(len(nzpos), size=min(n_checks, len(nzpos)), replace=False))]
    out = []
    for t, j in pick.tolist():
        with torch.no_grad():
            Zp, Zm = Z.clone(), Z.clone()
            Zp[0, t, j] += eps
            Zm[0, t, j] -= eps
            lp, _, _ = E._loss_from_z(cache, Sb, Tb, Zp)
            lm, _, _ = E._loss_from_z(cache, Sb, Tb, Zm)
        fd = (float(lp) - float(lm)) / (2 * eps)
        out.append(dict(pos=t, feature=j, autograd=float(g[t, j]), finite_diff=fd))
    rel = max(abs(o["autograd"] - o["finite_diff"]) / max(abs(o["finite_diff"]), 1e-3) for o in out)
    return dict(checks=out, max_rel=rel, pass_=rel < 0.1,
                note="loose tolerance: half-precision suffix, central difference eps=%g" % eps)


def archive_repro(model, tok, spec, suf, dic, cond, alive, ref_key, n_feats=64):
    """Seed-42 sources only: our engine on the ARCHIVED protocol (64 validation sequences)
    against the archived imp_zero array, on a random alive sample of features."""
    from v3.common import data as v3data
    f = common.WORK / "archive" / pathlib.Path(ref_key).name
    if common.s3_get(ref_key, f, required=False) is None:
        return dict(status="no archived array")
    z = np.load(f, allow_pickle=True)
    arch = np.asarray(z["imp_zero"], float)
    ids, am = v3data.eval_sequences(tok, n_seqs=registry.EVAL_SEQS_IMPORTANCE)

    class _P:   # wrap the archived eval set as a Partition-like object
        pass
    p = _P()
    p.ids, p.am = ids, am
    p.lengths = am.sum(1).numpy()
    p.doc = np.arange(len(ids)); p.row = np.arange(len(ids))
    p.docs = np.arange(len(ids)); p.doc_index = np.arange(len(ids))
    p.name = "archived_eval64"

    def batches(max_tokens=8192, max_seqs=64):
        order = np.argsort(-p.lengths, kind="stable")
        i = 0
        while i < len(order):
            T = int(p.lengths[order[i]])
            n = max(1, min(max_seqs, max_tokens // max(T, 1)))
            sel = order[i:i + n]
            yield sel, ids[sel, :T], am[sel, :T]
            i += n
    p.batches = batches
    cache = E.Cache(model, spec, suf, dic, p, max_tokens=4096)
    al = np.asarray(z["alive"], bool)
    cand = np.nonzero(al & (cache.fire_nseq > 0) & np.isfinite(arch))[0]
    rng = np.random.default_rng(11)
    top = cand[np.argsort(-arch[cand])][:16]
    pick = np.unique(np.concatenate([top, rng.choice(cand, size=min(n_feats, len(cand)), replace=False)]))
    S, F = E.items_firing(cache, pick)
    dl, _ = cache.run(S, F, lambda s, f, old, am_: torch.zeros_like(old), max_tokens=4096)
    mine = np.array([dl[F == j].sum() for j in pick]) / cache.n_targets
    a = arch[pick]
    corr = float(np.corrcoef(mine, a)[0, 1]) if len(pick) > 2 else float("nan")
    return dict(status="compared", n_features=int(len(pick)), pearson=corr,
                max_abs_diff=float(np.max(np.abs(mine - a))),
                median_abs_diff=float(np.median(np.abs(mine - a))),
                archived_scale=float(np.max(np.abs(a))),
                top16_mine=[float(x) for x in mine[:16]], top16_archived=[float(x) for x in a[:16]],
                note="archive used v3 fastabl on the same 64 sequences; differences reflect GPU "
                     "numerics and our per-feature restriction to firing sequences")


UNIVERSE_RULE = ("U = top-256 alive by rank-partition zero ablation  UNION  top-256 by gradient x "
                 "input magnitude  UNION  top-256 by IG magnitude  UNION  the dense PC1 cluster "
                 "(cos^2 > 5 x alive median)  UNION  256 uniformly random alive features "
                 "(seeded by SAE seed); under the reduced (70B) protocol U is built before any "
                 "ablation, so the zero list is absent. Fixed before any resampling outcome exists. Features "
                 "outside U are NOT_EVALUATED (NaN), never zero. On the small models every alive "
                 "feature is resampled, which measures how much of the full resampling top-50 U "
                 "would have contained.")


def resample_universe(out, dic, alive, seed, n=256):
    al = np.asarray(alive, bool)
    idx = np.nonzero(al)[0]
    g = np.load(out / "rank_grad.npz")
    srcs = [g["gxi_magnitude"], g["ig_magnitude"]]
    if (out / "rank_zero.npz").exists():
        srcs.append(np.load(out / "rank_zero.npz")["score"])
    parts = []
    for s in srcs:
        s = np.asarray(s, np.float64)
        s = s.reshape(s.shape[0], -1).sum(1)            # per-document arrays -> corpus score
        s = np.where(al & np.isfinite(s), s, -np.inf)
        parts.append(np.argsort(-s)[:n])
    Wn = dic.W_dec / dic.W_dec.norm(dim=1, keepdim=True)
    cos2 = (Wn @ dic.v1).pow(2).cpu().numpy()
    parts.append(idx[cos2[idx] > 5 * np.median(cos2[idx])])
    rng = np.random.default_rng(1000 + seed)
    parts.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    u = np.unique(np.concatenate(parts))
    return u[al[u]]


# ---------------------------------------------------------------------------------------
def save_metric(out, name, res, cache, extra=None):
    if E.SHARD_MODE:
        return                       # helper boxes hold partial arrays: never write a metric file
    d = dict(eff=res["eff"], targets=res["targets"], status=res["status"],
             n_items=np.int64(res["n_items"]), n_targets=np.int64(cache.n_targets),
             base_ce=np.float64(cache.base_ce), l0=np.float64(cache.l0),
             docs=cache.part.docs)
    d["score"] = E.score_from_eff(res["eff"], cache.n_targets, res["status"])
    for k, v in (extra or {}).items():
        d[k] = v
    common.save_npz(out / ("%s.npz" % name), **d)
    common.pin(out, "%s.npz" % name, "imp")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hours", type=float, default=12.0, help="compute budget for this job")
    ap.add_argument("--max-draws", type=int, default=16)
    ap.add_argument("--min-draws", type=int, default=4)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--max-seqs", type=int, default=None, help="local smoke only")
    ap.add_argument("--universe", choices=["auto", "all", "U"], default="auto",
                    help="resampling universe, fixed by the job table (not by runtime): 'all' = "
                         "every alive feature (small models, seed 42: measures U's recall); 'U' = "
                         "the registered universe; 'auto' = legacy runtime rule")
    ap.add_argument("--lean", action="store_true",
                    help="budget protocol (non-reference seeds of 7-9B models): exhaustive zero "
                         "ablation and gradient metrics on the rank partition only")
    ap.add_argument("--reduced", action="store_true",
                    help="70B protocol: gradients exhaustive; zero, mean and resampling on the "
                         "registered universe U built from the gradient rankings")
    ap.add_argument("--nshard", type=int, default=0, help="total boxes working this dictionary (helpers + assembler)")
    ap.add_argument("--shard", type=int, default=None, help="this box's index; the assembler is nshard-1")
    ap.add_argument("--helper", action="store_true", help="compute own chunks of the main job's remaining "
                                                         "stages and publish them; never writes metric files")
    a = ap.parse_args(argv)
    t_start = time.time()
    main_job = "rx-imp-%s-%s-s%d" % (a.model, a.condition, a.seed)
    if a.nshard > 1:
        E.SHARD = (a.nshard - 1 if a.shard is None else a.shard, a.nshard)
        E.SHARED = "rx-shards/%s" % main_job
        E.SHARD_MODE = bool(a.helper)
    spec = M.spec_of(a.model)
    out = common.RESULTS / "imp" / a.model / ("%s_s%d" % (a.condition, a.seed))
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prep, mean, pca, alive_npz = fetch_prep(a.model)
    v1 = pca["v1"]
    key = "%s_s%d" % (a.condition, a.seed)
    alive = alive_npz["alive_" + key]
    sae_path, h = fetch_sae(a.model, a.condition, a.seed)
    model, tok = M.load(spec, device=dev, device_map=a.device_map,
                        dtype=getattr(torch, a.dtype) if a.dtype else None)
    if spec.family != "gemma":
        torch.backends.cuda.matmul.allow_tf32 = False
    suf = M.Suffix(model, spec)
    sdev = M.site_device(model, spec)
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(sdev).total_memory / 1e9
        if gb < 40 and a.max_tokens > 4096:
            a.max_tokens = 4096          # 24 GB cards (A10G/L4): halve the token batch
            log("mem", "%.0f GB GPU: max_tokens -> %d" % (gb, a.max_tokens))
    dic, extra = M.load_dictionary(sae_path, a.condition, mean, v1, sdev)
    meta = dict(model=a.model, condition=a.condition, seed=a.seed, config_hash=h,
                sae_sha256=common.sha256_file(sae_path), env=common.env_info(),
                prep_gate3=prep.get("gate3", {}).get("status"),
                site_block=M.site_index(spec), hidden_state_index=M.site_index(spec) + 1,
                n_alive_archived_rule=int(alive.sum()))
    common.write_json(out / "meta.json", meta)

    # ---------------- gates ----------------
    gfile = out / "gates.json"
    if not common.fresh(out, "gates.json", "gates"):
        g = run_gates(model, tok, spec, suf, dic, sae_path, mean, v1, alive, a.condition, a.seed, out)
        g["gate7"] = gate7_checked(model, spec, suf, dic, tok, alive, out)
        if a.seed == 42:
            ref = json.loads((common.PKG / "archive_ref.json").read_text()).get(a.model, {})
            k = (ref.get("a5_npz") or {}).get(a.condition)
            g["archive_repro"] = archive_repro(model, tok, spec, suf, dic, a.condition, alive, k) if k else dict(status="no archived array key")
        hard = ["gate1", "gate2", "gate3_splice_equals_v3", "gate4", "gate5", "gate7"]
        g["all_hard_pass"] = all(bool(g[x].get("pass_")) for x in hard)
        common.write_json(gfile, g)
        common.pin(out, "gates.json", "gates")
        log("gates", "all_hard_pass=%s  g7 max_abs=%.3g tol=%.3g" % (
            g["all_hard_pass"], g["gate7"]["max_abs"], g["gate7"]["tol"]))
        if not g["all_hard_pass"]:
            raise RuntimeError("correctness gate failed; measurements NOT recorded: %s" %
                               {x: g[x].get("pass_") for x in hard})
    else:
        g = json.loads(gfile.read_text())
        if "precision_of_check" not in g.get("gate7", {}):
            # written before gate 7 became a float32 logic check: redo that gate only
            g["gate7_superseded_half"] = g["gate7"]
            g["gate7"] = gate7_checked(model, spec, suf, dic, tok, alive, out)
            hard = ["gate1", "gate2", "gate3_splice_equals_v3", "gate4", "gate5", "gate7"]
            g["all_hard_pass"] = all(bool(g[x].get("pass_")) for x in hard)
            common.write_json(gfile, g)
            log("gates", "gate7 re-checked: %s" % g["gate7"]["pass_"])
        if not g.get("all_hard_pass"):
            raise RuntimeError("gates previously failed on this dictionary")

    feats = np.nonzero(alive)[0]
    cal = P.Partition("cal", tok, max_seqs=a.max_seqs)
    cs = E.CalStats(model, spec, dic, cal, cap=2048, max_tokens=a.max_tokens)
    common.write_json(out / "cal_report.json", cs.report(alive))
    plan = dict(rule="16 keyed draws per resampling operator; universe = all alive if the measured "
                     "cost fits 75% of the remaining budget, else U; mean and validation zero/grad "
                     "always; validation resampling only if the budget covers it",
                hours=a.hours, min_draws=a.min_draws, max_draws=a.max_draws)
    main_done = set()
    if a.helper:
        mk = "%s/results/imp/%s/%s_s%d" % (main_job, a.model, a.condition, a.seed)
        for f in ("plan.json", "resample_universe.npy", "resample_universe_small.npy"):
            common.s3_get("%s/%s" % (mk, f), out / f, required=False)
        r = common._aws(["s3", "ls", "s3://%s/%s/" % (common.BUCKET, mk)], check=False)
        main_done = {l.split()[-1] for l in (r.stdout or "").splitlines() if l.endswith(".npz")}
        log("shard", "helper %s/%d for %s; main already has %s" % (E.SHARD[0], E.SHARD[1], main_job, sorted(main_done)))
    if common.fresh(out, "plan.json", "imp"):   # resume: the plan was fixed on first run
        plan = json.loads((out / "plan.json").read_text())

    def prog(done, n, el):
        log("items", "%d/%d  %.0fs" % (done, n, el))

    if a.universe == "all" and plan.get("resample_universe") == "U":
        plan["resample_universe"] = "all_alive"
        plan["universe_source"] = "job table (switched from runtime rule)"
        common.write_json(out / "plan.json", plan)
        common.pin(out, "plan.json", "imp")
        for st in ("sp", "mg", "st"):
            for pn in ("rank", "valid"):
                f = out / ("%s_%s.npz" % (pn, st))
                if f.exists():
                    f.unlink()
                    log("plan", "discarded %s (computed on U; job table requires all)" % f.name)
        import shutil
        for pn in ("rank", "valid"):
            cd = out / ("chunks_%s" % pn)
            for c in (cd.glob("*resample*") if cd.exists() else []):
                shutil.rmtree(c) if c.is_dir() else c.unlink()
    for pname in (("rank",) if a.lean else ("rank", "valid")):
        part = P.Partition(pname, tok, max_seqs=a.max_seqs)
        cache = None
        chunks = out / ("chunks_%s" % pname)
        common.fresh(out, chunks.name, "imp")          # stale partial chunks are discarded
        common.pin(out, chunks.name, "imp")
        todo = ["zero", "grad"] + (["sp", "mg", "st", "mean"] if pname == "rank" else ["sp", "mg"])
        if a.reduced:
            todo = ["grad", "zero"] + todo[2:]
        if a.lean:
            todo = ["grad", "zero"] if a.reduced else ["zero", "grad"]
        for st in todo:
            f = out / ("%s_%s.npz" % (pname, st))
            if a.helper and (st in ("zero", "grad") and pname == "rank" or f.name in main_done or st == "grad"):
                continue                        # helpers only parallelise chunked stages the main still lacks
            if common.fresh(out, f.name, "imp"):
                continue
            if pname == "valid" and st in ("sp", "mg") and not plan.get("valid_resampling"):
                continue
            if st == "mean" and not plan.get("mean"):
                continue
            if a.reduced and st in ("sp", "mg", "st") and a.seed != 42:
                continue          # 70B: resampling on the reference seed only (registered)
            if cache is None:
                t0 = time.time()
                cache = E.Cache(model, spec, suf, dic, part, max_tokens=a.max_tokens)
                log("cache", "%s %s: %.0fs  L0 %.2f  base CE %.4f  targets %d" % (
                    pname, part.summary(), time.time() - t0, cache.l0, cache.base_ce, cache.n_targets))
            t0 = time.time()
            if st == "zero":
                zsel = feats
                if a.reduced:
                    uf = out / "resample_universe.npy"
                    if not common.fresh(out, uf.name, "imp"):
                        np.save(uf, resample_universe(out, dic, alive, a.seed))
                        common.pin(out, uf.name, "imp")
                    zsel = np.load(uf)
                r = E.run_metric(cache, cs, "zero", zsel, alive, max_tokens=a.max_tokens,
                                 progress=prog, chunk_dir=chunks)
                save_metric(out, "%s_zero" % pname, r, cache)
                if pname == "rank":
                    dt = time.time() - t0
                    plan["zero_rank_seconds"] = dt
                    plan["zero_rank_items"] = r["n_items"]
                    left = a.hours * 3600 - (time.time() - t_start)
                    # Registered rule (B2): 16 keyed draws for every resampling operator, always.
                    # Universe: every alive feature when that fits this box's budget, measured
                    # from THIS box's zero-ablation time (sp ~ st ~ 1 zero pass per draw, mg ~ 3);
                    # otherwise the predeclared universe U (see resample_universe). Mean
                    # replacement and the validation-partition zero/gradient passes are
                    # exhaustive and always run.
                    draws = 16
                    full_cost = draws * (dt + 3 * dt + dt)
                    fixed = 3 * dt + 2 * dt                       # mean (rank) + valid zero/grad
                    plan["draws"] = draws
                    plan["mean"] = True
                    plan["resample_universe"] = ("U" if a.reduced else
                                                 "all_alive" if a.universe == "all" else
                                                 "U" if a.universe == "U" else
                                                 "all_alive" if fixed + full_cost <= 0.75 * left else "U")
                    plan["universe_source"] = "job table" if a.universe != "auto" else "runtime rule"
                    plan["reduced_protocol"] = bool(a.reduced)
                    plan["lean_protocol"] = bool(a.lean)
                    frac = 1.0 if plan["resample_universe"] == "all_alive" else 0.15
                    plan["valid_resampling"] = bool(fixed + full_cost * frac * 2 <= 0.75 * left)
                    plan["universe_rule"] = UNIVERSE_RULE
                    common.write_json(out / "plan.json", plan)
                    common.pin(out, "plan.json", "imp")
                    log("plan", plan)
            elif st == "grad":
                # backward stores every suffix layer's activations: size the batch to the
                # model (70B on 2x96 GB with 80% weights leaves ~19 GB/GPU)
                dm = M.spec_of(a.model).d_model
                gm = E.gradient_metrics(cache, alive=alive,
                                        max_tokens=256 if dm >= 8192 else 1024 if dm >= 3584 else 2048)
                common.save_npz(out / ("%s_grad.npz" % pname),
                                gxi_magnitude=gm["gxi_magnitude"], gxi_signed=gm["gxi_signed"],
                                ig_signed=gm["ig_signed"], ig_magnitude=gm["ig_magnitude"],
                                ig_steps=np.int64(gm["ig_steps"]),
                                ig_converged=np.bool_(gm["ig_converged"]),
                                n_targets=np.int64(cache.n_targets), docs=cache.part.docs)
                common.pin(out, "%s_grad.npz" % pname, "imp")
                common.write_json(out / ("%s_ig_convergence.json" % pname), gm["ig_convergence"])
                if pname == "rank" and "gate9" not in g:
                    g["gate9"] = dict(fd=gate9_fd(cache), ig_convergence=gm["ig_convergence"],
                                      ig_converged=gm["ig_converged"])
                    common.write_json(gfile, g)
            else:
                plan = json.loads((out / "plan.json").read_text())
                draws = int(plan.get("draws", 16))
                uf = out / "resample_universe.npy"
                if st == "mean" and not a.reduced:
                    sel = feats
                elif plan.get("resample_universe") == "all_alive":
                    sel = feats
                elif a.reduced:
                    # 70B: top-50 by zero ablation within U, plus 50 random members of U
                    us = out / "resample_universe_small.npy"
                    if not common.fresh(out, us.name, "imp"):
                        U = np.load(out / "resample_universe.npy")
                        zs = np.load(out / "rank_zero.npz")["score"][U]
                        top = U[np.argsort(-np.where(np.isfinite(zs), zs, -np.inf))[:50]]
                        rest = np.setdiff1d(U, top)
                        rnd = np.random.default_rng(2000 + a.seed).choice(rest, min(50, len(rest)), replace=False)
                        np.save(us, np.unique(np.concatenate([top, rnd])))
                        common.pin(out, us.name, "imp")
                    sel = np.load(us)
                else:
                    if not common.fresh(out, uf.name, "imp"):
                        np.save(uf, resample_universe(out, dic, alive, a.seed))
                        common.pin(out, uf.name, "imp")
                    sel = np.load(uf)
                metric = {"sp": "sp_resample", "mg": "mg_resample", "st": "st_resample",
                          "mean": "mean"}[st]
                r = E.run_metric(cache, cs, metric, sel, alive,
                                 draws=(1 if st == "mean" else draws), max_tokens=a.max_tokens,
                                 progress=prog, chunk_dir=chunks)
                save_metric(out, "%s_%s" % (pname, st), r, cache,
                            extra=dict(draws=np.int64(1 if st == "mean" else draws),
                                       universe=np.asarray(sel, np.int64)))
            log(st, "%s done in %.0fs" % (pname, time.time() - t0))
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    common.write_json(out / "plan.json", plan)
    return dict(model=a.model, condition=a.condition, seed=a.seed,
                hours=(time.time() - t_start) / 3600)
