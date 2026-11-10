"""Gradient-validation diagnosis for the 2026-10-08 audit (item 2). CPU only; not in any protocol hash.

Question: the production finite-difference gate (evaljob.gate9_fd: central difference, eps = 0.01, model in
fp16/bf16, latents fp32, summed-NLL loss in fp32) failed on 85/85 dictionaries. Is that the derivative or
the check? This module reruns the SAME loss on the SAME latents, mask and splice
(engine._loss_from_z == splice -> suffix -> summed valid NLL), with only the arithmetic precision changed:

  fp64: model.double(); splice in float64 (W_dec, b_dec, mu, v1, coef cast to float64); logits kept in
        float64 (a float64 copy of modelio.suffix_nll whose only change is not casting logits to fp32).
  fp32: model.float(); modelio.suffix_nll / dic.splice_fp32 unchanged.

For each (model, condition), at the exact (position, feature) pairs stored in production gates.json plus 17
more nonzero latents of the same sequence:
  - autograd in fp64 and fp32; production autograd value (GPU, registry dtype) from gates.json
  - central differences for eps in EPS, in fp64 and fp32 -> error vs fp64 autograd
Plus, in fp32 over the whole rank partition:
  - gradient x input (|.| and signed) per feature -> compared with the production rank_grad.npz (GPU)
  - batch invariance (gate 8): the same with max_tokens 512 vs 2048
  - IG (optional): steps ladder, both ranking definitions, signed and cancellation-resistant completeness
"""
import argparse
import json
import time

import numpy as np
import torch

from . import common, engine as E, modelio as M, partitions as P
from . import evaljob as EJ
from .common import log

EPS = [1.0, 0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6]


def suffix_nll_hp(suf, h, ids, am, chunk_rows=2048):
    """modelio.suffix_nll with logits kept in h's dtype (float64 path); otherwise identical."""
    import contextlib
    model = suf.model
    base = M._base_model(model)
    head = model.get_output_embeddings()
    cap = getattr(model.config, "final_logit_softcapping", None)
    B, T = h.shape[:2]
    dummy = torch.zeros(B, T, suf.emb_dim, dtype=suf.emb_dtype, device=M.input_device(model))
    if len(suf.tail) == 0:
        hid = M._final_norm(base)(h)
    else:
        with suf._truncated(h):
            hid = base(inputs_embeds=dummy, attention_mask=am.to(M.input_device(model)),
                       use_cache=False).last_hidden_state
    hid = hid[:, :-1].reshape(-1, hid.shape[-1])
    tgt = ids[:, 1:].reshape(-1).to(hid.device)
    out = []
    for i in range(0, hid.shape[0], chunk_rows):
        lg = head(hid[i:i + chunk_rows]).to(h.dtype)
        if cap:
            lg = torch.tanh(lg / cap) * cap
        lp = torch.log_softmax(lg, dim=-1)
        out.append(-lp.gather(-1, tgt[i:i + chunk_rows].unsqueeze(-1)).squeeze(-1))
    nll = torch.cat(out).reshape(B, T - 1)
    valid = am[:, 1:] != 0
    nll = torch.where(valid, nll, torch.zeros_like(nll))
    return nll.sum(), valid


def make_loss(cache, dic, prec):
    """Summed valid NLL from dense latents Z [1,Tb,d_sae] -- engine._loss_from_z at precision `prec`."""
    if prec == "fp32":
        return lambda Sb, Tb, Z: E._loss_from_z(cache, Sb, Tb, Z)[0]
    W, b, mu, v1 = (t.double() for t in (dic.W_dec, dic.b_dec, dic.mu, dic.v1))

    def f(Sb, Tb, Z):
        r = Z @ W + b + mu
        if dic.mode == "bypass":
            r = r + cache.coef[Sb, :Tb].double()[..., None] * v1
        s, _ = suffix_nll_hp(cache.suf, r, cache.ids[Sb, :Tb], cache.am[Sb, :Tb])
        return s
    return f


def dense_z(cache, s, Tb, dtype):
    idx = cache.Zidx[s, :Tb].long()
    Z = torch.zeros(1, Tb, cache.dic.d_sae, dtype=dtype, device=cache.dic.device)
    Z[0].scatter_(1, idx.clamp_min(0), torch.where(idx >= 0, cache.Zval[s, :Tb], 0.0).to(dtype))
    return Z


def fd_sweep(cache, dic, prec, picks):
    dt = torch.float64 if prec == "fp64" else torch.float32
    f = make_loss(cache, dic, prec)
    s = int(np.argmax(cache.part.lengths))
    Tb = int(cache.part.lengths[s])
    Sb = torch.as_tensor([s], device=dic.device)
    Z = dense_z(cache, s, Tb, dt)
    Zg = Z.clone().requires_grad_(True)
    L = f(Sb, Tb, Zg)
    L.backward()
    g = Zg.grad[0]
    out = []
    for t, j in picks:
        row = dict(pos=int(t), feature=int(j), z=float(Z[0, t, j]), autograd=float(g[t, j]), fd={})
        for eps in EPS:
            with torch.no_grad():
                Zp, Zm = Z.clone(), Z.clone()
                Zp[0, t, j] += eps
                Zm[0, t, j] -= eps
                lp, lm = f(Sb, Tb, Zp), f(Sb, Tb, Zm)
            row["fd"][str(eps)] = (float(lp) - float(lm)) / (2 * eps)
        out.append(row)
    return dict(seq=s, length=Tb, loss=float(L), checks=out)


def gxi_totals(cache, max_tokens):
    """Per-feature totals of |z dL/dz| and z dL/dz (engine.gradient_metrics' gxi), fp32, any batching."""
    part, dev, d_sae = cache.part, cache.dic.device, cache.dic.d_sae
    order = np.argsort(-part.lengths, kind="stable")
    mag = torch.zeros(d_sae, dtype=torch.float64)
    sgn = torch.zeros(d_sae, dtype=torch.float64)
    i = 0
    while i < len(order):
        Tb = int(part.lengths[order[i]])
        nb = max(1, max_tokens // max(Tb, 1))
        sel = order[i:i + nb]
        i += nb
        Sb = torch.as_tensor(sel, device=dev)
        idx = cache.Zidx[Sb, :Tb].long()
        Z = torch.zeros(len(Sb), Tb, d_sae, device=dev)
        Z.scatter_(2, idx.clamp_min(0), torch.where(idx >= 0, cache.Zval[Sb, :Tb], 0.0))
        Z.requires_grad_(True)
        loss, _, _ = E._loss_from_z(cache, Sb, Tb, Z)
        loss.backward()
        prod = (Z.detach() * Z.grad) * cache.am[Sb, :Tb, None].float()
        mag += prod.abs().sum((0, 1)).double()
        sgn += prod.sum((0, 1)).double()
    N = cache.n_targets
    return (mag / N).numpy(), (sgn / N).numpy()


def ig_diag(cache, alive, steps_ladder):
    """engine.gradient_metrics' IG loop, recording both ranking definitions and three completeness forms."""
    part, dev, d_sae = cache.part, cache.dic.device, cache.dic.d_sae
    order = np.argsort(-part.lengths, kind="stable")
    batches, i = [], 0
    while i < len(order):
        Tb = int(part.lengths[order[i]])
        nb = max(1, 2048 // max(Tb, 1))
        batches.append((order[i:i + nb], Tb))
        i += nb
    hist = []
    for m in steps_ladder:
        acc = torch.zeros(d_sae, dtype=torch.float64)
        acc_abs = torch.zeros(d_sae, dtype=torch.float64)
        r_num, r_den, r_batch_abs, r_seq_abs, d_seq_abs = 0.0, 0.0, 0.0, 0.0, 0.0
        for sel, Tb in batches:
            Sb = torch.as_tensor(sel, device=dev)
            idx = cache.Zidx[Sb, :Tb].long()
            Z0 = torch.zeros(len(Sb), Tb, d_sae, device=dev)
            Z0.scatter_(2, idx.clamp_min(0), torch.where(idx >= 0, cache.Zval[Sb, :Tb], 0.0))
            with torch.no_grad():
                _, n1, v1 = E._loss_from_z(cache, Sb, Tb, Z0)
                _, n0, v0 = E._loss_from_z(cache, Sb, Tb, torch.zeros_like(Z0))
                L1s = (n1 * v1.to(n1.device)).sum(1).double()
                L0s = (n0 * v0.to(n0.device)).sum(1).double()
            g_acc = torch.zeros_like(Z0)
            for k in range(m):
                Zp = ((k + 0.5) / m * Z0).detach().requires_grad_(True)
                loss, _, _ = E._loss_from_z(cache, Sb, Tb, Zp)
                loss.backward()
                g_acc += Zp.grad
            ig = Z0 * g_acc / m * cache.am[Sb, :Tb, None].float()
            acc += ig.sum((0, 1)).double()
            acc_abs += ig.abs().sum((0, 1)).double()
            ig_seq = ig.sum((1, 2)).double()
            dl_seq = L1s - L0s
            r_num += float(ig_seq.sum() - dl_seq.sum())
            r_den += abs(float(dl_seq.sum()))
            r_batch_abs += abs(float(ig_seq.sum() - dl_seq.sum()))
            r_seq_abs += float((ig_seq - dl_seq).abs().sum())
            d_seq_abs += float(dl_seq.abs().sum())
        sig = acc.numpy()
        mag = acc_abs.numpy()
        top_engine = set(np.argsort(-np.where(alive, np.abs(sig), -np.inf))[:50].tolist())
        top_magnitude = set(np.argsort(-np.where(alive, mag, -np.inf))[:50].tolist())
        h = dict(steps=m, completeness_signed_engine=abs(r_num) / max(r_den, 1e-30),
                 completeness_batch_abs=r_batch_abs / max(r_den, 1e-30),
                 completeness_per_sequence_abs=r_seq_abs / max(d_seq_abs, 1e-30),
                 top_engine=sorted(top_engine), top_magnitude=sorted(top_magnitude))
        if hist:
            h["overlap_prev_engine_def"] = len(top_engine & set(hist[-1]["top_engine"])) / 50
            h["overlap_prev_magnitude_def"] = len(top_magnitude & set(hist[-1]["top_magnitude"])) / 50
        h["engine_vs_magnitude_top50_overlap"] = len(top_engine & top_magnitude) / 50
        log("ig", json.dumps({k: v for k, v in h.items() if not k.startswith("top_")}))
        hist.append(h)
    return hist, mag


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ig", default="", help="comma list of IG step counts, e.g. 16,32,64")
    ap.add_argument("--extra-picks", type=int, default=17)
    a = ap.parse_args(argv)
    t0 = time.time()
    out = common.RESULTS / "gradcheck"
    out.mkdir(parents=True, exist_ok=True)
    tag = "%s_%s_s%d" % (a.model, a.condition, a.seed)
    spec = M.spec_of(a.model)
    prep, mean, pca, alive_npz = EJ.fetch_prep(a.model)
    alive = alive_npz["alive_%s_s%d" % (a.condition, a.seed)]
    sae_path, h = EJ.fetch_sae(a.model, a.condition, a.seed)
    job = "rx-imp-%s-%s-s%d" % (a.model, a.condition, a.seed)
    base = "%s/results/imp/%s/%s_s%d" % (job, a.model, a.condition, a.seed)
    gf = out / ("%s_gates.json" % tag)
    common.s3_get(base + "/gates.json", gf)
    gates = json.loads(gf.read_text())
    prod_fd = (gates.get("gate9") or {}).get("fd") or {}
    picks = [(c["pos"], c["feature"]) for c in prod_fd.get("checks", [])]
    torch.set_num_threads(max(1, torch.get_num_threads()))
    res = dict(model=a.model, condition=a.condition, seed=a.seed, production_gate9=prod_fd,
               registry_dtype=spec.dtype, device="cpu")
    # ---------------- fp32 ----------------
    model, tok = M.load(spec, device="cpu", dtype=torch.float32)
    suf = M.Suffix(model, spec)
    dic, _ = M.load_dictionary(sae_path, a.condition, mean, pca["v1"], "cpu")
    part = P.Partition("rank", tok)
    cache = E.Cache(model, spec, suf, dic, part, max_tokens=4096)
    log("gradcheck", "%s cache %.0fs targets %d" % (tag, time.time() - t0, cache.n_targets))
    s = int(np.argmax(part.lengths))
    Tb = int(part.lengths[s])
    nz = torch.nonzero(dense_z(cache, s, Tb, torch.float32)[0]).tolist()
    rng = np.random.default_rng(10)
    have = set(map(tuple, picks))
    for i in rng.permutation(len(nz)):
        if len(picks) >= len(have) + a.extra_picks:
            break
        if tuple(nz[i]) not in have:
            picks.append(tuple(nz[i]))
    res["picks"] = picks
    res["fp32"] = fd_sweep(cache, dic, "fp32", picks)
    mag2048, sgn2048 = gxi_totals(cache, 2048)
    mag512, sgn512 = gxi_totals(cache, 512)
    res["batch_invariance_gxi"] = dict(max_abs_diff_magnitude=float(np.abs(mag2048 - mag512).max()),
                                       max_rel_diff_magnitude=float((np.abs(mag2048 - mag512) / np.maximum(np.abs(mag2048), 1e-12)).max()),
                                       max_abs_diff_signed=float(np.abs(sgn2048 - sgn512).max()))
    pg = out / ("%s_rank_grad.npz" % tag)
    common.s3_get(base + "/rank_grad.npz", pg)
    zprod = np.load(pg)
    prod_mag = np.asarray(zprod["gxi_magnitude"], float)
    prod_mag = prod_mag.reshape(prod_mag.shape[0], -1).sum(1) if prod_mag.ndim > 1 else prod_mag
    from scipy.stats import spearmanr
    al = np.asarray(alive, bool)
    t_cpu = set(np.argsort(-np.where(al, mag2048, -np.inf))[:50].tolist())
    t_prod = set(np.argsort(-np.where(al, prod_mag, -np.inf))[:50].tolist())
    res["gxi_cpu_fp32_vs_production"] = dict(
        spearman_alive=float(spearmanr(mag2048[al], prod_mag[al]).correlation),
        top50_overlap=len(t_cpu & t_prod) / 50,
        rel_l2=float(np.linalg.norm(mag2048[al] - prod_mag[al]) / np.linalg.norm(prod_mag[al])),
        top50_cpu=sorted(t_cpu), top50_production=sorted(t_prod))
    np.savez(out / ("%s_gxi_cpu_fp32.npz" % tag), magnitude=mag2048, signed=sgn2048)
    if a.ig:
        hist, ig_mag = ig_diag(cache, al, [int(x) for x in a.ig.split(",")])
        res["ig"] = hist
        pim = np.asarray(zprod["ig_magnitude"], float)
        pim = pim.reshape(pim.shape[0], -1).sum(1) if pim.ndim > 1 else pim
        t_ig = set(np.argsort(-np.where(al, ig_mag, -np.inf))[:50].tolist())
        t_pig = set(np.argsort(-np.where(al, pim, -np.inf))[:50].tolist())
        res["ig_cpu_vs_production_magnitude_top50_overlap"] = len(t_ig & t_pig) / 50
    common.write_json(out / ("%s.json" % tag), res)
    # ---------------- fp64 (same latents, mask, splice) ----------------
    model.double()
    cache.suf = M.Suffix(model, spec)
    cache.dtype = torch.float64
    res["fp64"] = fd_sweep(cache, dic, "fp64", picks)
    res["seconds"] = time.time() - t0
    common.write_json(out / ("%s.json" % tag), res)
    log("gradcheck", "%s done %.0fs" % (tag, time.time() - t0))
    return dict(tag=tag)


if __name__ == "__main__":
    main()
