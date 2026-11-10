"""Shared helpers for the Gated investigation (2026-09-18). The v3 package is imported, never edited.

Definitions used throughout
  exact L0     : mean nonzero codes per row over rows [0, 100000) of the training store, final batch clamped
  archived L0  : io.alive rates.sum() -- encodes 13 x 8192 = 106,496 rows and divides by 100,000 (+6.496%)
  alive        : firing rate > 1e-3 over exactly 100,000 rows (the v2 definition, without the overrun)
  FVU          : sum||x - xhat||^2 / sum||x - mean||^2 ; content = same on the complement of the store PC1
  held-out     : WikiText-103 TEST split through the same model/block/tokenization (one document per 128-token
                 sequence, padding excluded), centered with the TRAINING store mean
Gated (v3/common/saes.py GatedSAE):
  pre = (x - b_dec) W_enc^T ; pi = pre + b_gate ; mag = ReLU(pre * exp(r_mag) + b_enc) ; z = 1[pi > 0] * mag
  loss = ||z W_dec + b_dec - x||^2 + l1 * sum ReLU(pi) + ||ReLU(pi) W_dec.detach() + b_dec.detach() - x||^2
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "objectives" / "gated"))
from v3.common import registry, io_utils as io                        # noqa: E402
from v3.common import saes as S                                       # noqa: E402
from v3.common.data import (get_blocks, tokenized_batches, _CaptureHook,  # noqa: E402
                            open_activations, load_model_and_tokenizer)
from v3.common.registry import block_index                            # noqa: E402
from v3.common.matching import max_cos                                # noqa: E402

BUCKET = os.environ.get("S3_BUCKET", "")
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
N_EXACT = 100_000
THR = 0.7


class GatedMagL1SAE(S.GatedSAE):
    """Gated SAE plus a direct L1 price on the reconstruction coefficients z (the controlled test).

    Everything else is the original GatedSAE: same init (BaseSAE with the same seed), same gate L1 on ReLU(pi),
    same auxiliary loss through the detached decoder. beta = 0 reproduces GatedSAE exactly.
    """
    arch = "gated_magl1"

    def __init__(self, d_model, d_sae, l1_coef=1e-3, beta=0.0, seed=0):
        super().__init__(d_model, d_sae, l1_coef=l1_coef, seed=seed)
        self.beta = float(beta)

    def sparsity_loss(self, z, pre):
        return super().sparsity_loss(z, pre) + self.beta * z.abs().sum(-1).mean()


S.ARCHS["gated_magl1"] = GatedMagL1SAE


def s3():
    import boto3
    if not BUCKET:
        raise RuntimeError("Set S3_BUCKET to the bucket containing the archived inputs.")
    return boto3.client("s3")


def fetch(key, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        s3().download_file(BUCKET, key, dst)
    return dst


def s3_exists(key):
    try:
        s3().head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cache_name(spec, seed, arch, **kw):
    return os.path.basename(io.sae_cache_path(spec, seed, arch=arch, condition="standard", **kw))


def load_ckpt(path, device="cuda"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sae, extra = S.load_sae(path, device=device)
    return sae, extra, blob["cfg"]


def dump(obj, name):
    p = io.path("results", "gated_investigation", name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"  -> {p}", flush=True)
    return p


# ------------------------------------------------------------------ activations
@torch.no_grad()
def harvest_split(spec, split="test", n_tokens=200_000, model=None, tok=None, device="cuda"):
    """Held-out activations through the same block and tokenization as the training store (float32 [n, d])."""
    own = model is None
    if own:
        model, tok = load_model_and_tokenizer(spec, device=device)
    blocks = get_blocks(model, spec.block_path)
    hook = _CaptureHook()
    h = blocks[block_index(spec)].register_forward_hook(hook)
    out, n = [], 0
    for ids, am in tokenized_batches(tok, corpus="wikitext", split=split, batch_seqs=32, max_tokens=n_tokens + 4096):
        ids, am = ids.to(device), am.to(device)
        model(ids, attention_mask=am)
        flat = hook.buf.float()[am.bool()]
        out.append(flat.cpu().numpy())
        n += flat.shape[0]
        if n >= n_tokens:
            break
    h.remove()
    if own:
        del model
        torch.cuda.empty_cache()
    return np.concatenate(out)[:n_tokens].astype(np.float32)


def store(spec, device="cuda"):
    from v3.common.pca import estimate_pcs
    adir = io.ensure_acts(spec, device=device)
    evals, evecs, _ = estimate_pcs(adir)
    acts, mean = open_activations(adir)
    return dict(adir=adir, acts=acts, mean=mean, v1=evecs[:, 0], w1=float(evals[0] / evals.sum()), evecs=evecs)


def _rows(st, n=N_EXACT, batch=8192):
    mean_t = torch.tensor(st["mean"], dtype=torch.float32, device="cuda")
    for i in range(0, n, batch):
        j = min(i + batch, n)
        yield torch.tensor(np.asarray(st["acts"][i:j], dtype=np.float32), device="cuda") - mean_t


# ------------------------------------------------------------------ measurements
@torch.no_grad()
def fidelity(sae, X_iter, v1):
    """Exact L0, firing counts, total and content FVU over the rows yielded by X_iter (already centered)."""
    v = torch.tensor(v1, dtype=torch.float64, device="cuda")
    tot = rows = 0
    fire = None
    err = var = perr = pvar = 0.0
    for x in X_iter:
        hat, z = sae(x)
        nz = z != 0
        tot += int(nz.sum()); rows += x.shape[0]
        fire = nz.double().sum(0) if fire is None else fire + nz.double().sum(0)
        e = (hat - x).double(); xd = x.double()
        err += float(e.square().sum()); var += float(xd.square().sum())
        ep = e - (e @ v)[:, None] * v[None, :]; xp = xd - (xd @ v)[:, None] * v[None, :]
        perr += float(ep.square().sum()); pvar += float(xp.square().sum())
    rate = (fire / rows).cpu().numpy()
    return dict(l0_exact=tot / rows, rows=rows, n_alive=int((rate > registry.ALIVE_FIRING_RATE).sum()),
                n_never_fired=int((rate == 0).sum()), fvu_total=err / var, fvu_content=perr / pvar,
                w1_rows=1 - pvar / var), rate


def store_fidelity(sae, st):
    return fidelity(sae, _rows(st), st["v1"])


def heldout_fidelity(sae, X, mean, v1, batch=8192):
    mean_t = torch.tensor(mean, dtype=torch.float32, device="cuda")

    def it():
        for i in range(0, len(X), batch):
            yield torch.tensor(X[i:i + batch], device="cuda") - mean_t
    r, _ = fidelity(sae, it(), v1)
    return {f"heldout_{k}": v for k, v in r.items()}


def archived_l0(sae, st):
    rates, alive = io.alive(sae, st["adir"], condition="standard", v1=st["v1"], device="cuda")
    return float(rates.sum()), int(alive.sum())


@torch.no_grad()
def gate_stats(sae, st, n=N_EXACT):
    """Per-feature gate vs magnitude statistics for a GatedSAE over n store rows.

    For every feature f: fire rate; mean reconstruction coefficient z and mean penalized gate value ReLU(pi) on the
    tokens where the gate is open; their ratio (reconstruction per unit of penalized quantity); the implied L1 price
    per unit of reconstruction coefficient, l1 * sum ReLU(pi) / sum z; and the share of the reconstructed PC1
    coefficient the feature carries. Also the three loss terms per token.
    """
    v1 = torch.tensor(st["v1"], dtype=torch.float32, device="cuda")
    D = sae.W_dec.float()
    dv = D @ v1
    d = sae.d_sae
    cnt = torch.zeros(d, device="cuda", dtype=torch.float64)
    sz = torch.zeros_like(cnt); sg = torch.zeros_like(cnt); spc1 = torch.zeros_like(cnt); sz2 = torch.zeros_like(cnt)
    mse = pen = aux = zl1 = tot_pc1 = 0.0
    rows = 0
    for x in _rows(st, n):
        pre = (x - sae.b_dec) @ sae.W_enc.T
        pi = pre + sae.b_gate
        mag = torch.relu(pre * sae.r_mag.exp() + sae.b_enc)
        g = (pi > 0).float()
        z = g * mag
        gate = torch.relu(pi)
        hat = z @ sae.W_dec + sae.b_dec
        mse += float((hat - x).double().square().sum())
        aux += float(((gate @ sae.W_dec + sae.b_dec) - x).double().square().sum())
        pen += float(gate.double().sum()) * sae.l1_coef
        zl1 += float(z.double().sum())
        cnt += g.double().sum(0); sz += z.double().sum(0); sg += gate.double().sum(0); sz2 += z.double().square().sum(0)
        c = (z * dv[None, :]).double()
        spc1 += c.sum(0); tot_pc1 += float(c.sum())
        rows += x.shape[0]
    cnt_np, sz_np, sg_np = cnt.cpu().numpy(), sz.cpu().numpy(), sg.cpu().numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        per = dict(fire_rate=cnt_np / rows,
                   mean_z_active=np.where(cnt_np > 0, sz_np / np.maximum(cnt_np, 1), np.nan),
                   mean_gate_active=np.where(cnt_np > 0, sg_np / np.maximum(cnt_np, 1), np.nan),
                   z_per_gate=np.where(sg_np > 0, sz_np / sg_np, np.nan),
                   price_per_unit_z=np.where(sz_np > 0, sae.l1_coef * sg_np / sz_np, np.nan),
                   pc1_share=spc1.cpu().numpy() / tot_pc1 if tot_pc1 else np.full(d, np.nan),
                   exp_r_mag=sae.r_mag.exp().float().cpu().numpy(),
                   b_enc=sae.b_enc.float().cpu().numpy(), b_gate=sae.b_gate.float().cpu().numpy(),
                   rms_z_active=np.where(cnt_np > 0, np.sqrt(sz2.cpu().numpy() / np.maximum(cnt_np, 1)), np.nan))
    totals = dict(mse_per_token=mse / rows, gate_penalty_per_token=pen / rows, aux_mse_per_token=aux / rows,
                  z_l1_per_token=zl1 / rows, gate_l1_per_token=float(sg.sum()) / rows,
                  penalty_over_mse=(pen / rows) / (mse / rows), aggregate_price_per_unit_z=pen / max(zl1, 1e-30))
    return per, totals


@torch.no_grad()
def l1_price_relu(sae, st, n=N_EXACT):
    """For ReLU+l1 the penalized quantity IS z: price per unit coefficient = l1 exactly. Report the loss split."""
    mse = zl1 = 0.0
    rows = 0
    for x in _rows(st, n):
        hat, z = sae(x)
        mse += float((hat - x).double().square().sum()); zl1 += float(z.double().sum()); rows += x.shape[0]
    return dict(mse_per_token=mse / rows, penalty_per_token=sae.l1_coef * zl1 / rows, z_l1_per_token=zl1 / rows,
                penalty_over_mse=(sae.l1_coef * zl1) / mse, aggregate_price_per_unit_z=float(sae.l1_coef))


def grad_flow(sae, st, batch=4096):
    """Which parameters each loss term actually moves (gradient norms on one batch, training mode semantics)."""
    x = next(_rows(st, batch, batch))
    out = {}
    terms = {"reconstruction_mse": lambda z, hat: (hat - x).square().sum(-1).mean(),
             "gate_l1_penalty": lambda z, hat: sae.sparsity_loss(z, None)}
    if isinstance(sae, S.GatedSAE):
        terms["aux_gate_reconstruction"] = lambda z, hat: sae.aux_loss(x)
    for name, fn in terms.items():
        sae.zero_grad(set_to_none=True)
        with torch.enable_grad():
            hat, z = sae(x)
            fn(z, hat).backward()
        out[name] = {pn: (float(p.grad.norm()) if p.grad is not None else 0.0) for pn, p in sae.named_parameters()}
    sae.zero_grad(set_to_none=True)
    return out


def classes(cos2):
    return np.where(cos2 >= 0.3, "canonical", np.where(cos2 >= 0.01, "band", "content"))


def summarize_per_feature(per, cos2, alive, top50=None):
    """Median of each per-feature statistic by PC1-alignment class (alive features), and for the top-50 set."""
    cl = classes(cos2)
    groups = {c: alive & (cl == c) for c in ("canonical", "band", "content")}
    if top50 is not None:
        m = np.zeros_like(alive); m[top50] = True
        groups["top50"] = m
        for c in ("canonical", "band", "content"):
            groups[f"top50_{c}"] = m & (cl == c)
    out = {}
    for gname, gm in groups.items():
        rec = dict(n=int(gm.sum()))
        for k, v in per.items():
            vv = v[gm]; vv = vv[np.isfinite(vv)]
            rec[f"median_{k}"] = float(np.median(vv)) if len(vv) else None
            if k in ("pc1_share",):
                rec[f"sum_{k}"] = float(np.nansum(v[gm]))
        out[gname] = rec
    return out


def init_similarity(sae, seed):
    """Cosine of each final decoder/encoder row with its own initialization (BaseSAE: tied randn rows from
    Generator(seed), row-normalized). Rows near 1 were barely moved by training."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(sae.d_sae, sae.d_model, generator=g)
    W = (W / W.norm(dim=1, keepdim=True)).numpy()
    out = {}
    for name, M in (("decoder", sae.W_dec.detach().float().cpu().numpy()), ("encoder", sae.W_enc.detach().float().cpu().numpy())):
        c = (M * W).sum(1) / np.maximum(np.linalg.norm(M, axis=1), 1e-30)
        out[name] = dict(median_cos_with_init=float(np.median(c)), frac_rows_cos_gt_0p9=float((c > 0.9).mean()),
                         frac_rows_cos_gt_0p5=float((c > 0.5).mean()))
    return out, None


def decoder_norms(sae):
    n = sae.W_dec.detach().float().norm(dim=1).cpu().numpy()
    return dict(min=float(n.min()), max=float(n.max()), max_abs_dev_from_1=float(np.abs(n - 1).max()))


def sharing_pair(dec_a, alive_a, imp_a, dec_b, k=50):
    """Signed max cosine of a's atoms into ALL of b's decoder rows (matching.sharing_stats convention)."""
    mc = max_cos(dec_a, dec_b)
    shared = mc >= THR
    imp = np.asarray(imp_a, float).copy(); imp[~alive_a] = -np.inf
    top = np.argsort(imp)[::-1][:k]
    return dict(top50=float(shared[top].mean()), dictwide=float(shared[alive_a].mean())), mc, top


def dec_np(sae):
    return S.decoder_np(sae)


def log_tail(extra):
    return (extra or {}).get("log")


def now():
    return time.strftime("%H:%M:%S")
