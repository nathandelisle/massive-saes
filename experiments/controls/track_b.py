"""Compare normalization, norm filtering, whitening, and PC1 bypass."""
import argparse
import csv
import json
import pathlib
import time

import numpy as np

from rcommon import add_v3, config_hash, fetch_sae, log, paper_cfg, write_json
from xforms import BUILD

ARMS = {
    # name                  xform            mode       k     archived
    "raw_topk64":          ("raw",           "standard", 64,  True),
    "bypass_64p1":         ("raw",           "bypass",   64,  True),
    "bypass_63p1":         ("raw",           "bypass",   63,  False),
    "unitnorm_topk64":     ("unitnorm",      "standard", 64,  False),
    "gao_topk64":          ("gao",           "standard", 64,  False),
    "normfilter10x_topk64": ("normfilter10x", "standard", 64,  False),
    "whiten_topk64":       ("whiten",        "standard", 64,  False),
    "gao_bypass_63p1":     ("gao",           "bypass",   63,  False),
}


def _sae_path(model, arm, seed):
    return "saes_b/%s_%s_s%d.pt" % (model, arm, seed)


# ---------------------------------------------------------------------------

def make_batches(adir, mean, xf, batch_size, steps, seed, device, v1_norm=None):
    """Centered training batches, transformed, with the arm's training-row filter."""
    import torch
    from v3.common.data import open_activations
    acts, _ = open_activations(adir)
    n = acts.shape[0]
    rng = np.random.default_rng(seed)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    v1_t = None if v1_norm is None else torch.tensor(v1_norm, dtype=torch.float32,
                                                     device=device)
    for _ in range(steps):
        idx = np.sort(rng.integers(0, n, size=batch_size))
        x = torch.tensor(np.asarray(acts[idx], dtype=np.float32), device=device) - mean_t
        m = xf.train_mask(x)
        if m is not None and int(m.sum()) > 8:
            x = x[m]
        t, _aux = xf.fwd(x)
        if v1_t is not None:                       # bypass INSIDE the transformed space
            t = t - (t @ v1_t)[:, None] * v1_t[None, :]
        yield t


def splice_arm(model, spec, sae, mu, xf, v1_raw, mode, v1_xf=None, transform=None,
               layer=None):
    """Hook: raw -> centre -> (optional raw bypass) -> xform -> SAE -> inverse -> raw.

    Per-token inverse scalars come from the RECIPIENT activation and are frozen.
    """
    import contextlib
    import torch
    from v3.common.data import get_blocks
    from v3.common.registry import block_index
    from v3.common.splice import hook_device

    @contextlib.contextmanager
    def ctx():
        blocks = get_blocks(model, spec.block_path)
        dev = hook_device(model, spec, layer)
        p0 = next(sae.parameters(), None)
        if p0 is not None and p0.device != dev:
            sae.to(dev)
        mu_t = mu.to(dev)
        v1r = None if v1_raw is None else v1_raw.to(dev)
        v1n = None if v1_xf is None else torch.as_tensor(v1_xf, device=dev)

        def hook(module, inputs, output):
            is_t = isinstance(output, tuple)
            h = output[0] if is_t else output
            od = h.dtype
            x = h.float() - mu_t
            coef = None
            if mode == "bypass" and v1n is None:
                coef = x @ v1r
                x = x - coef[..., None] * v1r
            B, T, d = x.shape
            t, aux = xf.fwd(x.reshape(-1, d))
            ncoef = None
            if mode == "bypass" and v1n is not None:
                ncoef = t @ v1n
                t = t - ncoef[:, None] * v1n[None, :]
            z = sae.encode(t)
            if transform is not None:
                z = transform(z)
            th = sae.decode(z)
            if ncoef is not None:
                th = th + ncoef[:, None] * v1n[None, :]
            xh = xf.inv(th, aux).reshape(B, T, d)
            new = xh + mu_t
            if coef is not None:
                new = new + coef[..., None] * v1r
            new = new.to(od)
            return (new,) + output[1:] if is_t else new

        hd = blocks[block_index(spec, layer)].register_forward_hook(hook)
        try:
            yield
        finally:
            hd.remove()
    return ctx()


def eval_arm(model, spec, seqs, am, sae, mu, xf, v1_raw, mode, v1_xf, chunk,
             row_keep=None, v1_stats=None):
    """Pooled raw-space sufficient statistics plus splice CE, over one population.

    Accumulates sums, never per-document ratios:
        var_tot  = sum ||x||^2            err_tot  = sum ||x - xhat||^2
        var_par  = sum <x,v1>^2           err_par  = sum <x - xhat, v1>^2
    then FVU_total = err_tot/var_tot, FVU_content = (err_tot-err_par)/(var_tot-var_par),
    w1 = var_par/var_tot, epsilon = err_par/err_tot.

    `v1_raw` is the BYPASS direction and is None for a standard arm. `v1_stats` is the
    frozen raw PC1 used for the decomposition and must be supplied for EVERY arm: w1 and
    content FVU are properties of the activations, not of the arm, and the whole point of
    the comparison is content FVU on the standard arm against content FVU on the others.
    Overloading one argument for both silently reported w1 = 0 and content FVU = total FVU
    on every standard arm.
    """
    import torch
    import torch.nn.functional as F
    from v3.common.data import get_blocks
    from v3.common.registry import block_index
    from v3.common.splice import hook_device
    dev = hook_device(model, spec)
    acc = dict(var_tot=0.0, var_par=0.0, err_tot=0.0, err_par=0.0, rows=0,
               l0=0.0, l0_rows=0, own_var=0.0, own_err=0.0)
    fires = {"v": torch.zeros(sae.d_sae, device=dev, dtype=torch.float64)}
    mu_t, v1r = mu.to(dev), (None if v1_raw is None else v1_raw.to(dev))
    v1s = v1r if v1_stats is None else v1_stats.to(dev)
    v1n = None if v1_xf is None else torch.as_tensor(v1_xf, device=dev)
    blocks = get_blocks(model, spec.block_path)
    keep_t = {"m": None}

    def hook(module, inputs, output):
        is_t = isinstance(output, tuple)
        h = output[0] if is_t else output
        od = h.dtype
        x = h.float() - mu_t
        coef = None
        if mode == "bypass" and v1n is None:
            coef = x @ v1r
            xin = x - coef[..., None] * v1r
        else:
            xin = x
        B, T, d = xin.shape
        t, aux = xf.fwd(xin.reshape(-1, d))
        t_pre = t
        ncoef = None
        if mode == "bypass" and v1n is not None:
            ncoef = t @ v1n
            t = t - ncoef[:, None] * v1n[None, :]
        z = sae.encode(t)
        th = sae.decode(z)
        if ncoef is not None:
            th = th + ncoef[:, None] * v1n[None, :]
        xh = xf.inv(th, aux).reshape(B, T, d)
        new = xh + mu_t
        if coef is not None:
            new = new + coef[..., None] * v1r
        with torch.no_grad():
            m = keep_t["m"]
            xf_ = x.reshape(-1, d)
            xhf = (new - mu_t).reshape(-1, d)
            if m is not None:
                xf_, xhf = xf_[m], xhf[m]
                zz = z[m]
            else:
                zz = z
            e = xf_ - xhf
            acc["var_tot"] += float(xf_.pow(2).sum())
            acc["err_tot"] += float(e.pow(2).sum())
            if v1s is not None:
                acc["var_par"] += float((xf_ @ v1s).pow(2).sum())
                acc["err_par"] += float((e @ v1s).pow(2).sum())
            acc["rows"] += int(xf_.shape[0])
            oe = t_pre - th
            acc["own_var"] += float(t_pre[m].pow(2).sum() if m is not None
                                    else t_pre.pow(2).sum())
            acc["own_err"] += float((oe[m] if m is not None else oe).pow(2).sum())
            acc["l0"] += float((zz != 0).float().sum())
            acc["l0_rows"] += int(zz.shape[0])
            fires["v"] += (zz != 0).sum(0).double()
        return (new.to(od),) + output[1:] if is_t else new.to(od)

    hd = blocks[block_index(spec)].register_forward_hook(hook)
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, seqs.shape[0], chunk):
            batch = seqs[i:i + chunk].to(dev)
            mb = None if am is None else am[i:i + chunk].to(dev)
            k = None
            if row_keep is not None:
                k = torch.as_tensor(row_keep[i:i + chunk].reshape(-1), device=dev)
            if mb is not None:
                mk = mb.reshape(-1).bool()
                k = mk if k is None else (k & mk)
            keep_t["m"] = k
            logits = model(batch, attention_mask=mb).logits.float()
            tgt = batch[:, 1:].reshape(-1).clone()
            if mb is not None:
                tgt[mb[:, 1:].reshape(-1) == 0] = -100
            tot += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt,
                                         reduction="sum", ignore_index=-100))
            cnt += int((tgt != -100).sum())
            del logits
    hd.remove()
    vt, vp, et, ep = acc["var_tot"], acc["var_par"], acc["err_tot"], acc["err_par"]
    rate = (fires["v"] / max(acc["l0_rows"], 1)).cpu().numpy()
    return dict(splice_ce=tot / max(cnt, 1), n_ce_tokens=cnt, rows=acc["rows"],
                fvu_total=et / max(vt, 1e-12),
                fvu_content=(et - ep) / max(vt - vp, 1e-12) if vt > vp else None,
                w1=vp / max(vt, 1e-12), epsilon=ep / max(et, 1e-12) if et > 0 else None,
                own_space_fvu=acc["own_err"] / max(acc["own_var"], 1e-12),
                l0=acc["l0"] / max(acc["l0_rows"], 1),
                n_alive=int((rate > 1e-3).sum()),
                n_any_positive=int((rate > 0).sum()),
                sufficient_stats=dict(var_tot=vt, var_par=vp, err_tot=et, err_par=ep,
                                      rows=acc["rows"]))


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 0])
    p.add_argument("--out", default="results/track_b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--eval-seqs", type=int, default=64)
    p.add_argument("--rank-top", type=int, default=400,
                   help="exact-ablate this many screened features per arm for the "
                        "raw-space importance ranking")
    p.add_argument("--hours", type=float, default=10.0)
    a = p.parse_args()

    import torch
    add_v3()
    from v3.common import registry
    from v3.common.data import load_model_and_tokenizer, eval_sequences, open_activations
    from v3.common.io_utils import ensure_acts
    from v3.common.pca import estimate_pcs
    from v3.common.saes import train_sae, save_sae, load_sae
    from v3.common.fastabl import tuned_chunk, screen_candidates
    from v3.common.matching import max_cos

    spec = registry.MODELS[a.model]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False
    t_start = time.time()

    adir = ensure_acts(spec, device=dev)
    _e, evecs, mean = estimate_pcs(adir)
    v1_np = evecs[:, 0]
    acts, _ = open_activations(adir)
    fit_rows = np.asarray(acts[:200_000], dtype=np.float32)
    torch.cuda.empty_cache()

    model, tok = load_model_and_tokenizer(spec, device=dev)
    model.eval()
    seqs, am = eval_sequences(tok, n_seqs=a.eval_seqs)
    chunk = tuned_chunk(spec)
    mu = torch.tensor(mean, dtype=torch.float32, device=dev)
    v1 = torch.tensor(v1_np, dtype=torch.float32, device=dev)

    # the frozen shared retained population (same filter for every arm)
    nf = BUILD["normfilter10x"]().fit(fit_rows, mean)
    row_norm = _row_norms(model, spec, seqs, am, mu, chunk)
    shared_keep = row_norm <= nf.cutoff
    log("trkB", "%s: shared retained population keeps %.4f of eval rows (cutoff %.2f)"
        % (a.model, float(shared_keep.mean()), nf.cutoff))

    res = dict(model=a.model, hf_id=spec.hf_id, layer=spec.target_layer,
               steps=a.steps, batch=a.batch, eval_seqs=a.eval_seqs, chunk=chunk,
               shared_filter=nf.meta(), pc1_fit="training rows only, first 50k, float64",
               populations=dict(
                   full="every non-padded eval row",
                   shared="rows within the frozen 10x training-median norm cutoff",
                   own="the arm's own training space, intended population"),
               arms={}, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    rows, ranks = [], {}

    for arm in a.arms:
        if arm not in ARMS:
            raise SystemExit("unknown arm %s" % arm)
        xname, mode, k, archived = ARMS[arm]
        res["arms"][arm] = dict(xform=xname, mode=mode, k=k, archived=archived,
                                total_coefficients=k + (1 if mode == "bypass" else 0),
                                cells={})
        for seed in a.seeds:
            if time.time() - t_start > a.hours * 3600:
                res["arms"][arm]["cells"][str(seed)] = dict(status="unmeasured_budget")
                log("trkB", "BUDGET EXHAUSTED before %s/s%d" % (arm, seed))
                continue
            t0 = time.time()
            xf = BUILD[xname]()
            xf.fit(fit_rows, mean)
            if hasattr(xf, "to"):
                xf.to(dev)
            res["arms"][arm]["xform_meta"] = xf.meta()

            v1_xf = None
            if mode == "bypass" and xname != "raw":
                # the direction is estimated IN NORMALIZED TRAINING SPACE, which is the
                # point of this arm: does the remaining covariance concentration matter
                # after normalization?
                Xt, _ = xf.fwd(torch.tensor(fit_rows[:50_000] - mean[None, :],
                                            dtype=torch.float32, device=dev))
                Xt = Xt.double()
                C = (Xt.T @ Xt / max(Xt.shape[0] - 1, 1)).cpu().numpy()
                lam, V = np.linalg.eigh(C)
                v1_xf = V[:, int(np.argmax(lam))].astype(np.float32)
                res["arms"][arm]["v1_in_transformed_space_var_share"] = float(
                    lam.max() / lam.sum())
                del Xt

            path = pathlib.Path(_sae_path(a.model, arm, seed))
            if archived:
                h = config_hash(paper_cfg(spec, seed, mode, k=k))
                sae, _x = load_sae(str(fetch_sae(h, "saes/%s.pt" % h)), device=dev)
                src = "archived:%s" % h
            elif path.exists():
                sae, _x = load_sae(str(path), device=dev)
                src = "cached"
            else:
                bs = make_batches(adir, mean, xf, a.batch, a.steps, seed, dev,
                                  v1_norm=(v1_xf if mode == "bypass" and xname != "raw"
                                           else None))
                if mode == "bypass" and xname == "raw":
                    bs = _bypass_raw(bs, v1)
                sae, tlog = train_sae(bs, spec.d_model, spec.d_sae, arch="topk", k=k,
                                      seed=seed, device=dev, total_steps=a.steps,
                                      tag="%s/%s/s%d" % (a.model, arm, seed))
                path.parent.mkdir(parents=True, exist_ok=True)
                save_sae(sae, str(path), extra=dict(arm=arm, xform=xf.meta(), k=k,
                                                    mode=mode, log=tlog[-3:]))
                src = "trained"
            sae.eval()

            v1r = v1 if (mode == "bypass" and xname == "raw") else None
            # v1_stats is the frozen raw PC1 and is passed for EVERY arm, including the
            # standard ones, so w1 and content FVU exist on the row the others are compared
            # against.
            ev_full = eval_arm(model, spec, seqs, am, sae, mu, xf, v1r, mode, v1_xf, chunk,
                               v1_stats=v1)
            ev_shared = eval_arm(model, spec, seqs, am, sae, mu, xf, v1r, mode, v1_xf,
                                 chunk, row_keep=shared_keep, v1_stats=v1)

            # raw-space decoder geometry and the raw-space importance ranking
            D = sae.W_dec.detach().float()
            Draw = xf.raw_direction(D)
            Dn = Draw / Draw.norm(dim=1, keepdim=True).clamp_min(1e-8)
            c2 = (Dn @ v1).pow(2).cpu().numpy()
            imp, base = _rank_arm(model, spec, seqs, am, sae, mu, xf, v1r, mode, v1_xf,
                                  chunk, a.rank_top, screen_candidates)
            ids = np.array(sorted(imp, key=lambda j: -imp[j]), dtype=int)
            ranks[(arm, seed)] = dict(ids=ids, dec=Dn.cpu().numpy(),
                                      alive=np.array([imp.get(int(j)) is not None
                                                      for j in range(spec.d_sae)]))
            cell = dict(status="measured", source=src, seconds=time.time() - t0,
                        full=ev_full, shared=ev_shared,
                        top50_mean_pc1_cos2=float(np.mean(c2[ids[:50]])),
                        frac_top50_above_0p01=float((c2[ids[:50]] >= 0.01).mean()),
                        frac_alive_above_0p01=float((c2 >= 0.01).mean()),
                        baseline_spliced_ce=base,
                        top50=[int(x) for x in ids[:50]])
            res["arms"][arm]["cells"][str(seed)] = cell
            rows.append(dict(model=a.model, arm=arm, seed=seed, xform=xname, mode=mode,
                             k=k, total_coefficients=k + (1 if mode == "bypass" else 0),
                             per_token_scalars=xf.per_token_scalars, source=src,
                             population="full", **{kk: vv for kk, vv in ev_full.items()
                                                   if kk != "sufficient_stats"},
                             top50_mean_pc1_cos2=cell["top50_mean_pc1_cos2"]))
            rows.append(dict(model=a.model, arm=arm, seed=seed, xform=xname, mode=mode,
                             k=k, total_coefficients=k + (1 if mode == "bypass" else 0),
                             per_token_scalars=xf.per_token_scalars, source=src,
                             population="shared_retained",
                             **{kk: vv for kk, vv in ev_shared.items()
                                if kk != "sufficient_stats"},
                             top50_mean_pc1_cos2=cell["top50_mean_pc1_cos2"]))
            log("trkB", "  %s/s%d [%s] %.1f min | raw FVU %.4f content %.4f w1 %.4f "
                        "CE %.4f L0 %.1f alive %d"
                % (arm, seed, src, (time.time() - t0) / 60, ev_full["fvu_total"],
                   ev_full["fvu_content"] or float("nan"), ev_full["w1"],
                   ev_full["splice_ce"], ev_full["l0"], ev_full["n_alive"]))
            write_json(out / ("track_b_%s.json" % a.model), res)
            _wrows(out / ("track_b_rows_%s.csv" % a.model), rows)
            del sae
            torch.cuda.empty_cache()

    # --- recurrence within each arm, plus the dictionary-wide baseline -------------
    rec = []
    for arm in a.arms:
        ks = [(arm, s) for s in a.seeds if (arm, s) in ranks]
        for ka in ks:
            for kb in ks:
                if ka == kb:
                    continue
                A, B = ranks[ka], ranks[kb]
                mc = max_cos(A["dec"][A["ids"][:50]], B["dec"])
                dw = max_cos(A["dec"], B["dec"])
                rec.append(dict(model=a.model, arm=arm, source_seed=ka[1],
                                target_seed=kb[1], top_n=50,
                                top50_shared_0p7=float((np.abs(mc) >= 0.7).mean()),
                                dictionary_wide_shared_0p7=float((np.abs(dw) >= 0.7).mean()),
                                top50_minus_dictionary_wide=float(
                                    (np.abs(mc) >= 0.7).mean() - (np.abs(dw) >= 0.7).mean()),
                                median_cos_top50=float(np.median(np.abs(mc))),
                                **{("shared_%.1f" % t).replace(".", "p"):
                                   float((np.abs(mc) >= t).mean())
                                   for t in (0.5, 0.6, 0.7, 0.8, 0.9)}))
    _wrows(out / ("track_b_recurrence_%s.csv" % a.model), rec)
    res["wall_minutes"] = (time.time() - t_start) / 60
    write_json(out / ("track_b_%s.json" % a.model), res)
    _wrows(out / ("track_b_rows_%s.csv" % a.model), rows)
    log("trkB", "DONE %s in %.1f min" % (a.model, res["wall_minutes"]))


def _bypass_raw(batches, v1_t):
    for x in batches:
        yield x - (x @ v1_t)[:, None] * v1_t[None, :]


def _row_norms(model, spec, seqs, am, mu, chunk):
    import torch
    from v3.common.data import get_blocks
    from v3.common.registry import block_index
    dev = next(model.parameters()).device
    out = np.zeros(seqs.shape, dtype=np.float32)
    grab = {}
    blocks = get_blocks(model, spec.block_path)

    def h(m, i, o):
        hh = o[0] if isinstance(o, tuple) else o
        grab["n"] = (hh.float() - mu).norm(dim=-1).cpu().numpy()
    hd = blocks[block_index(spec)].register_forward_hook(h)
    with torch.no_grad():
        for i in range(0, seqs.shape[0], chunk):
            mb = None if am is None else am[i:i + chunk].to(dev)
            model(seqs[i:i + chunk].to(dev), attention_mask=mb)
            out[i:i + chunk] = grab["n"]
    hd.remove()
    return out


def _rank_arm(model, spec, seqs, am, sae, mu, xf, v1r, mode, v1_xf, chunk, n_top,
              screen_fn):
    """Raw-space zero-ablation delta-CE over an E[z^2] screen, inside this arm's splice."""
    import torch
    dev = next(model.parameters()).device
    # E[z^2] screen from one pass
    s2 = torch.zeros(sae.d_sae, device=dev, dtype=torch.float64)
    fires = torch.zeros(sae.d_sae, device=dev, dtype=torch.float64)
    ntok = 0
    grab = {}

    def cap(z):
        grab["z"] = z
        return z
    with splice_arm(model, spec, sae, mu, xf, v1r, mode, v1_xf, transform=cap):
        with torch.no_grad():
            for i in range(0, seqs.shape[0], chunk):
                mb = None if am is None else am[i:i + chunk].to(dev)
                model(seqs[i:i + chunk].to(dev), attention_mask=mb)
                z = grab["z"]
                if mb is not None:
                    z = z[mb.reshape(-1).bool()]
                s2 += z.pow(2).sum(0).double()
                fires += (z != 0).sum(0).double()
                ntok += int(z.shape[0])
    ez2 = (s2 / max(ntok, 1)).cpu().numpy()
    rate = (fires / max(ntok, 1)).cpu().numpy()
    cand = screen_fn(ez2, rate > 1e-3, n_top)
    base = _ce(model, spec, seqs, am, sae, mu, xf, v1r, mode, v1_xf, chunk, None)
    imp = {}
    for j in cand:
        j = int(j)

        def t(z, j=j):
            z = z.clone()
            z[..., j] = 0.0
            return z
        imp[j] = _ce(model, spec, seqs, am, sae, mu, xf, v1r, mode, v1_xf, chunk, t) - base
    return imp, base


def _ce(model, spec, seqs, am, sae, mu, xf, v1r, mode, v1_xf, chunk, transform):
    import torch
    import torch.nn.functional as F
    dev = next(model.parameters()).device
    tot, cnt = 0.0, 0
    with splice_arm(model, spec, sae, mu, xf, v1r, mode, v1_xf, transform=transform):
        with torch.no_grad():
            for i in range(0, seqs.shape[0], chunk):
                b = seqs[i:i + chunk].to(dev)
                mb = None if am is None else am[i:i + chunk].to(dev)
                lg = model(b, attention_mask=mb).logits.float()
                tgt = b[:, 1:].reshape(-1).clone()
                if mb is not None:
                    tgt[mb[:, 1:].reshape(-1) == 0] = -100
                tot += float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), tgt,
                                             reduction="sum", ignore_index=-100))
                cnt += int((tgt != -100).sum())
                del lg
    return tot / max(cnt, 1)


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
