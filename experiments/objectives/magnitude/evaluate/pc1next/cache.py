"""Reusable held-out activations and SAE outputs; no logits are stored on disk."""
from pathlib import Path
import numpy as np
import torch
from .backend import absolute_ce, scores, valid_predictors
from .io import atomic_json, atomic_npz, completed, load_json, ratio


def blocks(data, n):
    for start in range(0, len(data["ids"]), n):
        yield start, slice(start, min(start+n, len(data["ids"])))


def cache_path(out, run, split, start):
    return Path(out) / "cache" / run["name"] / split / f"b{start:07d}.npz"


def make_cache(be, sae, run, split, data, out, pid, block_size):
    for start, sl in blocks(data, block_size):
        path = cache_path(out, run, split, start)
        if completed(path, pid): continue
        ids, mask = be.tensors(data, sl)
        original, x = be.forward(ids, mask)
        z, hat = be.encode(sae, x, run)
        spliced, _ = be.forward(ids, mask, hat)
        dce, kl, count = scores(original, spliced, ids, mask)
        if not torch.isfinite(z).all() or not torch.isfinite(hat).all(): raise FloatingPointError("Nonfinite SAE cache")
        atomic_npz(path, protocol_id=pid, start=start, x=x.float().cpu().numpy(), z=z.float().cpu().numpy(),
            hat=hat.float().cpu().numpy(), ce_model=absolute_ce(original, ids, mask).cpu().numpy(),
            dce_splice=dce.cpu().numpy(), kl_splice=kl.cpu().numpy(), count=count.cpu().numpy())
        print(f"[pc1-cache] {run['name']} {split} b={start} tokens={int(count.sum())}", flush=True)


def calibrate(be, sae, run, data, out, pid, block_size, cfg):
    target = Path(out) / "calibration" / (run["name"] + ".npz")
    if completed(target, pid):
        with np.load(target) as n: return {k:n[k] for k in n.files}
    make_cache(be, sae, run, "calibration", data, out, pid, block_size)
    d = sae.W_dec.detach().cpu().numpy().astype(np.float64)
    v = be.v.cpu().numpy().astype(np.float64); v /= np.linalg.norm(v)
    counts = np.zeros(len(d)); zsum = counts.copy(); z2 = counts.copy(); nt = 0
    for start, sl in blocks(data, block_size):
        with np.load(cache_path(out, run, "calibration", start)) as n:
            z = n["z"][data["mask"][sl]].astype(np.float64)
        counts += (z > 0).sum(0); zsum += z.sum(0); z2 += (z*z).sum(0); nt += len(z)
    norms = np.linalg.norm(d, axis=1)
    if np.any(norms <= 0): raise ValueError("Zero decoder row: matching undefined")
    fresh_alive = counts / nt > cfg.get("min_firing_rate", 0.0)
    alive = fresh_alive.copy()
    if run.get("frozen_ids_path"):
        old = load_json(run["frozen_ids_path"])
        if old.get("checkpoint_sha256") and old["checkpoint_sha256"] != cfg["checkpoint_hashes"][run["name"]]:
            raise ValueError("Frozen alive mask came from another checkpoint")
        frozen = np.asarray(old["calibration_alive_ids"], dtype=int)
        if len(np.unique(frozen)) != len(frozen) or np.any((frozen < 0) | (frozen >= len(d))): raise ValueError("Bad frozen IDs")
        alive[:] = False; alive[frozen] = True
    if not alive.any(): raise ValueError("No calibration-alive atoms")
    alignment = np.square(d @ v / norms)
    threshold = cfg.get("group_multiplier", 5.0) * np.median(alignment[alive])
    group = np.flatnonzero(alive & (alignment > threshold))
    if run.get("frozen_ids_path") and cfg.get("use_frozen_group", True):
        old = load_json(run["frozen_ids_path"])
        if "structural_group_ids" in old: group = np.asarray(old["structural_group_ids"], dtype=int)
    if len(group) and (len(np.unique(group))!=len(group) or np.any((group < 0) | (group >= len(d))) or not alive[group].all()): raise ValueError("Invalid structural group")
    arrays = dict(protocol_id=pid, decoder=d, pc1=v, alive=alive, fresh_alive=fresh_alive,
        group=group, firing_rate=counts/nt, mean_z=zsum/nt, mean_z2=z2/nt,
        positive_mean=np.divide(zsum, counts, out=np.zeros_like(zsum), where=counts>0),
        individual_energy=z2/nt*norms**2, alignment=alignment, n_tokens=nt)
    atomic_npz(target, **arrays)
    print(f"[pc1-calibration] {run['name']} alive={alive.sum()} fresh={fresh_alive.sum()} group={len(group)}", flush=True)
    return arrays


def fidelity(out, run, split, data, pid, block_size):
    target = Path(out) / "fidelity" / run["name"] / (split + ".json")
    if target.exists(): return load_json(target)
    with np.load(Path(out)/"calibration"/(run["name"]+".npz")) as n: v=n["pc1"]
    sx=np.zeros(len(v));sq=np.zeros(len(v)); xx=xxp=xxq=err=errp=errq=l0=total=ce=ce_s=kl=count=0.0
    per_article = {}
    for start, sl in blocks(data, block_size):
        with np.load(cache_path(out, run, split, start)) as n:
            x=n["x"][data["mask"][sl]].astype(np.float64)
            h=n["hat"][data["mask"][sl]].astype(np.float64)
            e=x-h; total+=len(x); sx+=x.sum(0); xx+=(x*x).sum(); xxp+=np.square(x@v).sum()
            err+=np.square(e).sum(); errp+=np.square(e@v).sum()
            xq=x-(x@v)[:,None]*v;eq=e-(e@v)[:,None]*v
            xxq+=np.square(xq).sum();sq+=xq.sum(0);errq+=np.square(eq).sum()
            l0+=(n["z"][data["mask"][sl]]>0).sum()
            ce+=n["ce_model"].sum(); ce_s+=n["dce_splice"].sum(); kl+=n["kl_splice"].sum(); count+=n["count"].sum()
            for i,a in enumerate(data["articles"][sl]):
                rec=per_article.setdefault(str(a),dict(count=0.,dce_sum=0.,kl_sum=0.))
                rec['count']+=float(n['count'][i]);rec['dce_sum']+=float(n['dce_splice'][i]);rec['kl_sum']+=float(n['kl_splice'][i])
    varp=xxp-(sx@v)**2/total; varq=xxq-sq@sq/total;var=varp+varq
    result=dict(protocol_id=pid,run=run['name'],split=split,n_tokens=int(total), l0=l0/total,
        fvu_total=ratio(err,var), fvu_pc1=ratio(errp,varp),fvu_perp=ratio(errq,varq),w1=ratio(varp,var),
        variance_decomposition_relative_residual=ratio(abs(var-(xx-sx@sx/total)),var),
        error_decomposition_relative_residual=ratio(abs(err-errp-errq),max(err,1.)),
        error_pc1_fraction=ratio(errp,err),ce_model=ce/count,splice_dce=ce_s/count,splice_kl=kl/count,articles=per_article)
    if min(varp,varq)>1e-20:
        result['fvu_identity_residual']=result['fvu_total']-(result['w1']*result['fvu_pc1']+(1-result['w1'])*result['fvu_perp'])
    atomic_json(target,result); return result
