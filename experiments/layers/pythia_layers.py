"""
Pythia layer comparisons on Modal.

    modal run --detach pythia_layers.py                       # full run (~14-20 h, 1x H100)
    modal run pythia_layers.py --smoke                        # ~10 min end-to-end sanity pass
    modal run --detach pythia_layers.py --stages probe,figs   # subset
    modal run --detach pythia_layers.py --force --stages l1   # redo a stage

Reviewer ask -> stage map (all trained from scratch on Pythia-1.4B; no
dependence on external checkpoints):

  s2Py Q1  importance-metric battery ......................... imp + metrics
           (zero-ablation, mean-ablation, gradient x input, integrated
            gradients; top-50 sharing under EACH metric + rank agreement)
  s2Py Q2  non-TopK architecture ............................. l1
           (ReLU+L1 with AUTOMATIC lambda calibration to L0 ~= k=64 via
            pilot runs; bad existing checkpoints are detected and retrained)
  s2Py Q3  auto-interp before/after bypass ................... autointerp
           (top-100 by importance + 50 random-alive controls per condition,
            explanation + detection, 10 pos / 10 neg held-out per feature)
  s2Py Q4 / 6K2t weakness  downstream evals .................. probing
           (sparse probing, 4 tasks: SST-2, AG News, Amazon polarity,
            Emotion; 5 resampling reps, mean +/- sd, k up to 100;
            raw-activation ceiling + majority baselines)
  AiXx score-raise ask  what does PC1 encode ................. probe
           (entropy / surprisal / log-frequency / position / attention-sink
            mass from PC1 projection vs residual vs random-direction
            controls; sequence-level train/test split, block-bootstrap CIs)
  AiXx figs  decoder distributions / cosine histograms ....... figs
  AiXx + 6K2t mechanism  (why dense / not an artifact) ....... cloud + layers
           (cloud: dense-cluster cancellation diagnostic — off-axis energy,
            cancellation ratio, effective rank, std vs byp;
            layers: negative control at L3 (low PC1 -> predicts NO
            inversion) + dose point at L23, dose-response figure)

Stages, in order:
  collect topk l1 imp metrics cloud layers figs probe probing autointerp report

State persists in Modal volume 'neurips-21722-rebuttal'; completed work is
skipped on re-run (per-checkpoint granularity). Upgraded stages write
versioned outputs (probe_v2.json, sparse_probing_v2.json, autointerp2_*) so
they recompute over a volume populated by v1. Fetch results:

    modal volume get neurips-21722-rebuttal results ./results

autointerp needs ANTHROPIC_API_KEY (read from your local env and forwarded,
or pass --anthropic-key); without it, contexts are cached and LLM scoring is
skipped — rerun just that stage later with the key.

Optional: upload extra layer results from other runs to be merged into the
dose-response figure as {"L6": {"pc1_pct": ..., "conditions": {"standard":
{"dict_wide": ..., "top50": ...}, "bypass": {...}}}}:

    modal volume put neurips-21722-rebuttal external_layers.json results/external_layers.json
"""
import json
import math
import os
import re
import shutil
import time
from types import SimpleNamespace

import modal

VOL_NAME = "neurips-21722-rebuttal"
VOL_PATH = "/vol"
STAGE_ORDER = ["collect", "topk", "l1", "imp", "metrics", "cloud", "layers",
               "figs", "probe", "probing", "autointerp", "report"]
DEFAULT_STAGES = ",".join(STAGE_ORDER)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=5.0,<6",     # verified against 5.14 (dtype=, tensor hook outputs)
        "datasets>=3.0",            # verified against 5.0 (hub parquet names)
        "numpy<3",
        "scipy>=1.11",
        "scikit-learn>=1.4",
        "matplotlib>=3.8",
        "anthropic>=0.40",
        "hf-transfer",
    )
    .env({"HF_HOME": f"{VOL_PATH}/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
app = modal.App("neurips-21722-rebuttal")
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

with image.imports():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from sklearn.feature_selection import f_classif
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------------ config

def get_cfg(smoke: bool):
    if not smoke:
        return dict(
            model_name="EleutherAI/pythia-1.4b",
            layer_idx=12,            # block whose OUTPUT is spliced ("layer 12")
            d_model=2048, d_sae=8192, k=64, seq_len=128,
            collect_tokens=5_000_000, collect_batch=32,
            train_steps=20_000, train_bs=4096, lr=3e-4,
            l1_coef=None,            # None -> auto-calibrate; float -> use as-is
            l1_target_l0=64, l1_pilot_steps=3000, l1_max_pilots=8,
            l1_accept=(0.75, 1.35),  # accepted L0/target band
            seeds=(42, 137),
            eval_seqs=64,            # arXiv importance protocol
            alive_rate=1e-3, alive_sample=500_000, stat_tokens=1_000_000,
            importance_cap=None,
            metrics_ig_steps=8, metrics_batch=16,
            cloud_cos=0.5,
            extra_layers=(3, 23),    # negative control + dose point
            probe_tokens=400_000, probe_freq_tokens=2_000_000,
            probe_attn_tokens=100_000, probe_batch=32,
            probe_boot=200, probe_ridge_rows=250_000,
            sp_tasks=("sst2", "ag_news", "amazon_polarity", "emotion"),
            sp_train_pool=5_000, sp_train=4_000, sp_test=1_000,
            sp_ks=(1, 2, 5, 10, 20, 50, 100), sp_reps=5,
            ai_top_n=100, ai_rand_n=50, ai_pool_tokens=1_000_000,
            ai_det_pos=10, ai_det_neg=10,
        )
    return dict(  # tiny end-to-end pass; same code paths, minutes not hours
        model_name="EleutherAI/pythia-70m",
        layer_idx=3, d_model=512, d_sae=1024, k=16, seq_len=64,
        collect_tokens=65_536, collect_batch=16,
        train_steps=40, train_bs=512, lr=3e-4,
        l1_coef=None, l1_target_l0=16, l1_pilot_steps=25, l1_max_pilots=3,
        l1_accept=(0.5, 2.0),
        seeds=(42, 137),
        eval_seqs=4, alive_rate=1e-3, alive_sample=4_000, stat_tokens=16_000,
        importance_cap=10,
        metrics_ig_steps=2, metrics_batch=4,
        cloud_cos=0.5,
        extra_layers=(1,),
        probe_tokens=4_096, probe_freq_tokens=100_000,
        probe_attn_tokens=256, probe_batch=8,
        probe_boot=25, probe_ridge_rows=50_000,
        sp_tasks=("sst2", "emotion"),
        sp_train_pool=60, sp_train=48, sp_test=24,
        sp_ks=(1, 2, 5), sp_reps=2,
        ai_top_n=4, ai_rand_n=2, ai_pool_tokens=4_096,
        ai_det_pos=3, ai_det_neg=3,
    )


def mk_paths(root, tag=""):
    P = SimpleNamespace(root=root, tag=tag)
    P.acts_dir = f"{root}/acts{tag}"
    P.ckpt = f"{root}/ckpts"
    P.res = f"{root}/results"
    P.acts_vol = f"{P.acts_dir}/acts.dat"
    P.meta = f"{P.acts_dir}/meta.json"
    P.pc1 = f"{P.acts_dir}/pc1.npz"
    P.acts_local = f"/tmp/acts_local{tag}.dat"
    for d in (P.acts_dir, P.ckpt, P.res):
        os.makedirs(d, exist_ok=True)
    return P


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] {path}", flush=True)


# ------------------------------------------------------------ model / data

def get_model(cfg, eager=False):
    key = "_model_eager" if eager else "_model"
    if key not in cfg:
        kw = dict(dtype=torch.float32)  # transformers 5.x: 'dtype', not 'torch_dtype'
        if eager:
            kw["attn_implementation"] = "eager"
        tok = AutoTokenizer.from_pretrained(cfg["model_name"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **kw)
        m.to(cfg["device"]).eval()
        cfg[key] = (m, tok)
    return cfg[key]


def blocks(model):
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("unknown architecture")


def load_tokens(cfg, tok, n_tokens, splits=("train",)):
    """WikiText-103 -> LongTensor [n_seqs, seq_len]. Deterministic."""
    if isinstance(splits, str):
        splits = (splits,)
    T = cfg["seq_len"]
    ids, need = [], n_tokens + T
    for split in splits:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
        buf = []
        for row in ds:
            if row["text"].strip():
                buf.append(row["text"])
            if len(buf) >= 1000:
                ids.extend(tok("\n".join(buf), add_special_tokens=False)["input_ids"])
                buf = []
                if len(ids) >= need:
                    break
        if buf and len(ids) < need:
            ids.extend(tok("\n".join(buf), add_special_tokens=False)["input_ids"])
        if len(ids) >= need:
            break
    n_seqs = min(len(ids), n_tokens) // T
    arr = np.array(ids[: n_seqs * T], dtype=np.int64).reshape(n_seqs, T)
    return torch.from_numpy(arr)


class Grabber:
    """Records a block's output (transformers 5.x: plain tensor; older: tuple)."""

    def __init__(self, model, layer_idx):
        self.value = None

        def _hook(mod, inp, out):
            self.value = (out[0] if isinstance(out, tuple) else out).detach()
        self.h = blocks(model)[layer_idx].register_forward_hook(_hook)

    def remove(self):
        self.h.remove()


@torch.no_grad()
def collect_block_outputs(cfg, model, tokens, with_targets=False, memmap=None,
                          batch_size=None):
    bs = batch_size or cfg["collect_batch"]
    grab = Grabber(model, cfg["layer_idx"])
    acts, ents, surs, tids = [], [], [], []
    row, s64 = 0, None
    try:
        for i in range(0, len(tokens), bs):
            b = tokens[i:i + bs].to(cfg["device"])
            out = model(b)
            x = grab.value
            if with_targets:
                logp = F.log_softmax(out.logits.float(), dim=-1)
                ent = -(logp.exp() * logp).sum(-1)
                sur = -logp[:, :-1].gather(-1, b[:, 1:, None]).squeeze(-1)
                x = x[:, :-1]
                ents.append(ent[:, :-1].reshape(-1).cpu())
                surs.append(sur.reshape(-1).cpu())
                tids.append(b[:, :-1].reshape(-1).cpu())
            flat = x.reshape(-1, x.shape[-1])
            if memmap is not None:
                memmap[row:row + len(flat)] = flat.to(torch.float16).cpu().numpy()
                if s64 is None:
                    s64 = torch.zeros(flat.shape[-1], dtype=torch.float64)
                s64 += flat.double().sum(0).cpu()
                row += len(flat)
            else:
                acts.append(flat.to(torch.float16).cpu())
            if (i // bs) % 100 == 0:
                print(f"  collect {i}/{len(tokens)} seqs", flush=True)
    finally:
        grab.remove()
    res = {}
    if memmap is not None:
        memmap.flush()
        res.update(rows=row, sum64=s64.numpy())
    else:
        res["acts"] = torch.cat(acts)
    if with_targets:
        res["entropy"] = torch.cat(ents).numpy()
        res["surprisal"] = torch.cat(surs).numpy()
        res["token_ids"] = torch.cat(tids).numpy()
    return res


# ------------------------------------------------------------------- PC1

def compute_pc1(acts, n_sample=50_000, seed=0, mu=None):
    n, d = len(acts), acts.shape[1]
    if mu is None:
        s, c = np.zeros(d, np.float64), 0
        for i in range(0, n, 100_000):
            chunk = np.asarray(acts[i:i + 100_000], dtype=np.float64)
            s += chunk.sum(0)
            c += len(chunk)
        mu = s / c
    else:
        mu = np.asarray(mu, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=min(n_sample, n), replace=False))
    X = np.asarray(acts[idx], dtype=np.float64) - mu
    C = X.T @ X / (len(X) - 1)
    lam, V = np.linalg.eigh(C)
    v1, var = V[:, -1], float(lam[-1] / lam.sum())
    print(f"[pc1] variance explained = {100 * var:.1f}%", flush=True)
    return (torch.tensor(mu, dtype=torch.float32),
            torch.tensor(v1, dtype=torch.float32), var)


def load_pc1(P):
    z = np.load(P.pc1)
    return (torch.tensor(z["mu"], dtype=torch.float32),
            torch.tensor(z["v1"], dtype=torch.float32), float(z["var"]))


def ensure_acts(cfg, P):
    if not os.path.exists(P.acts_local):
        if not os.path.exists(P.acts_vol):
            raise RuntimeError(f"activation store {P.acts_vol} missing — run "
                               "'collect' (or 'layers') first")
        print(f"[acts{P.tag}] copying store to local disk ...", flush=True)
        shutil.copyfile(P.acts_vol, P.acts_local)
    meta = json.load(open(P.meta))
    return np.memmap(P.acts_local, dtype=np.float16, mode="r",
                     shape=(meta["n"], meta["d"]))


# ------------------------------------------------------------------- SAEs

_sae_cls = {}


def sae_classes():
    if _sae_cls:
        return _sae_cls["topk"], _sae_cls["relu"]

    class TopKSAE(nn.Module):
        """TopK SAE (Gao et al.): tied init, unit-norm decoder rows."""

        def __init__(self, d_model, d_sae, k):
            super().__init__()
            self.k = k
            self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
            self.b_enc = nn.Parameter(torch.zeros(d_sae))
            self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
            self.b_dec = nn.Parameter(torch.zeros(d_model))
            nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=1)
                self.W_enc.data = self.W_dec.data.t().clone()

        def encode(self, x):
            pre = F.relu(x @ self.W_enc + self.b_enc)
            if self.k is not None and self.k < pre.shape[-1]:
                topv, topi = pre.topk(self.k, dim=-1)
                z = torch.zeros_like(pre)
                z.scatter_(-1, topi, topv)
                return z
            return pre

        def decode(self, z):
            return z @ self.W_dec + self.b_dec

        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z

        @torch.no_grad()
        def normalize_decoder(self):
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

        def decoder_dirs(self):
            return F.normalize(self.W_dec.detach(), dim=1)

    class ReLUSAE(TopKSAE):
        def __init__(self, d_model, d_sae):
            super().__init__(d_model, d_sae, k=None)

    _sae_cls.update(topk=TopKSAE, relu=ReLUSAE)
    return TopKSAE, ReLUSAE


def new_sae(cfg, arch):
    TopKSAE, ReLUSAE = sae_classes()
    if arch == "topk":
        return TopKSAE(cfg["d_model"], cfg["d_sae"], cfg["k"])
    return ReLUSAE(cfg["d_model"], cfg["d_sae"])


def ckpt_path(P, arch, cond, seed):
    return f"{P.ckpt}/sae_{arch}_{cond}_{seed}{P.tag}.pt"


def load_sae(cfg, path, arch):
    sae = new_sae(cfg, arch)
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    sae.load_state_dict(sd)
    return sae.to(cfg["device"]).eval()


# -------------------------------------------------- splice / importance

class SAESplice:
    """Replaces block layer_idx's output with the SAE splice.
    standard: x_hat = decode(encode(x - mu)) + mu
    bypass  : p = (x - mu)@v1 ; x_hat = decode(encode(x - mu - p v1)) + p v1 + mu
    ablate_feature: zero (or set to ablate_value) one latent before decode."""

    def __init__(self, cfg, model, sae, mu, v1, mode):
        assert mode in ("standard", "bypass")
        self.cfg, self.model, self.sae, self.mode = cfg, model, sae, mode
        self.mu = mu.to(cfg["device"])
        self.v1 = v1.to(cfg["device"])
        self.ablate_feature = None
        self.ablate_value = None
        self._h = None

    def _hook(self, mod, inp, out):
        tup = isinstance(out, tuple)
        x = out[0] if tup else out
        shp = x.shape
        c = x.reshape(-1, shp[-1]).float() - self.mu
        if self.mode == "bypass":
            p = c @ self.v1
            c = c - p.unsqueeze(-1) * self.v1
        z = self.sae.encode(c)
        if self.ablate_feature is not None:
            z[:, self.ablate_feature] = (self.ablate_value
                                         if self.ablate_value is not None else 0.0)
        r = self.sae.decode(z)
        if self.mode == "bypass":
            r = r + p.unsqueeze(-1) * self.v1
        r = (r + self.mu).to(x.dtype).reshape(shp)
        return (r,) + tuple(out[1:]) if tup else r

    def __enter__(self):
        self._h = blocks(self.model)[self.cfg["layer_idx"]].register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        self._h.remove()


class GradSplice(SAESplice):
    """Splice that exposes the latent as a grad leaf (z scaled by .alpha) for
    gradient-x-input / integrated-gradients importance. Earlier layers are
    detached, so backward touches only layers after the splice."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.alpha = 1.0
        self.z0 = None
        self.zl = None

    def _hook(self, mod, inp, out):
        tup = isinstance(out, tuple)
        x = out[0] if tup else out
        shp = x.shape
        c = (x.reshape(-1, shp[-1]).float() - self.mu).detach()
        if self.mode == "bypass":
            p = c @ self.v1
            c = c - p.unsqueeze(-1) * self.v1
        z = self.sae.encode(c)
        self.z0 = z.detach()
        zl = (self.alpha * z).detach().requires_grad_(True)
        self.zl = zl
        r = self.sae.decode(zl)
        if self.mode == "bypass":
            r = r + p.unsqueeze(-1) * self.v1
        r = (r + self.mu).to(x.dtype).reshape(shp)
        return (r,) + tuple(out[1:]) if tup else r


@torch.no_grad()
def ce_loss(cfg, model, tokens, bs=64):
    total, count = 0.0, 0
    for i in range(0, len(tokens), bs):
        b = tokens[i:i + bs].to(cfg["device"])
        logits = model(b).logits
        total += F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            b[:, 1:].reshape(-1), reduction="sum").item()
        count += b[:, 1:].numel()
    return total / count


@torch.no_grad()
def feature_stats(cfg, sae, acts, mu, v1, bypass, n_rows, bs=8192, seed=0):
    """Firing rate and mean activation (incl. zeros) per feature."""
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(acts), min(n_rows, len(acts)), replace=False))
    mu_d, v1_d = mu.to(cfg["device"]), v1.to(cfg["device"])
    fired = torch.zeros(cfg["d_sae"], device=cfg["device"])
    zsum = torch.zeros(cfg["d_sae"], device=cfg["device"])
    n = 0
    for i in range(0, len(sub), bs):
        x = torch.as_tensor(np.asarray(acts[sub[i:i + bs]], dtype=np.float32),
                            device=cfg["device"])
        c = x - mu_d
        if bypass:
            p = c @ v1_d
            c = c - p.unsqueeze(-1) * v1_d
        z = sae.encode(c)
        fired += (z > 0).sum(0)
        zsum += z.sum(0)
        n += len(x)
    return (fired / n).cpu(), (zsum / n).cpu()


def ensure_stats(cfg, P, arch, cond):
    path = f"{P.res}/stats_{arch}_{cond}{P.tag}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return torch.tensor(z["rates"]), torch.tensor(z["mean_z"])
    sae = load_sae(cfg, ckpt_path(P, arch, cond, cfg["seeds"][0]), arch)
    mm = ensure_acts(cfg, P)
    mu, v1, _ = load_pc1(P)
    rates, mean_z = feature_stats(cfg, sae, mm, mu, v1, cond == "byp",
                                  cfg["stat_tokens"])
    np.savez(path, rates=rates.numpy(), mean_z=mean_z.numpy())
    return rates, mean_z


@torch.no_grad()
def ablation_importance(cfg, model, sae, mu, v1, mode, eval_tokens, fids,
                        values=None):
    """CE increase when one latent is zeroed (values=None) or set to its mean
    (values=tensor) inside the splice. Paper's zero-ablation protocol."""
    imps = {}
    with SAESplice(cfg, model, sae, mu, v1, mode) as sp:
        base = ce_loss(cfg, model, eval_tokens)
        print(f"[imp] spliced base CE = {base:.4f} ({len(fids)} features)", flush=True)
        for j, f in enumerate(fids):
            sp.ablate_feature = int(f)
            sp.ablate_value = None if values is None else float(values[int(f)])
            imps[int(f)] = ce_loss(cfg, model, eval_tokens) - base
            if (j + 1) % 250 == 0:
                print(f"  {j + 1}/{len(fids)}", flush=True)
        sp.ablate_feature = None
        sp.ablate_value = None
    return base, imps


def grad_importance(cfg, model, sae, mu, v1, mode, eval_tokens, ig_steps=1):
    """Per-feature mean |z * dCE/dz|; ig_steps>1 -> integrated gradients
    (mean grad along the 0->z path, times z, abs)."""
    for prm in model.parameters():
        prm.requires_grad_(False)
    dev = cfg["device"]
    total = torch.zeros(cfg["d_sae"], device=dev)
    ntok = 0
    bs = cfg["metrics_batch"]
    alphas = [1.0] if ig_steps <= 1 else [(j + 1) / ig_steps for j in range(ig_steps)]
    sp = GradSplice(cfg, model, sae, mu, v1, mode)
    with sp:
        for i in range(0, len(eval_tokens), bs):
            b = eval_tokens[i:i + bs].to(dev)
            z0, acc = None, None
            for a in alphas:
                sp.alpha = a
                logits = model(b).logits
                ce = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                    b[:, 1:].reshape(-1))
                ce.backward()
                g = sp.zl.grad.detach()
                if z0 is None:
                    z0 = sp.z0
                acc = g / len(alphas) if acc is None else acc + g / len(alphas)
            attr = (z0 * acc).abs()
            total += attr.sum(0)
            ntok += attr.shape[0]
    return (total / ntok).cpu()


def cross_seed_max_cos(sae_a, sae_b):
    Da, Db = sae_a.decoder_dirs(), sae_b.decoder_dirs()
    return (Da @ Db.t()).max(dim=1).values.cpu()


def top_n_recovery(importance, maxcos, alive_mask, top_n=50, thresh=0.7):
    order = [f for f in sorted(importance, key=importance.get, reverse=True)
             if bool(alive_mask[f])]
    top = order[:top_n]
    return float(np.mean([maxcos[f].item() > thresh for f in top])), top


def dict_wide_share(maxcos, alive_mask, thresh=0.7):
    return float((maxcos[alive_mask] > thresh).float().mean())


# ----------------------------------------------------------------- stages

def stage_collect(cfg, P):
    model, tok = get_model(cfg)
    toks = load_tokens(cfg, tok, cfg["collect_tokens"], "train")
    n = toks.numel()
    mm = np.memmap(P.acts_local, dtype=np.float16, mode="w+",
                   shape=(n, cfg["d_model"]))
    res = collect_block_outputs(cfg, model, toks, memmap=mm)
    mu, v1, var = compute_pc1(mm, mu=res["sum64"] / res["rows"])
    np.savez(P.pc1, mu=mu.numpy(), v1=v1.numpy(), var=var)
    save_json({"n": int(res["rows"]), "d": cfg["d_model"],
               "model": cfg["model_name"], "layer": cfg["layer_idx"],
               "seq_len": cfg["seq_len"]}, P.meta)
    print("[collect] copying store to volume ...", flush=True)
    shutil.copyfile(P.acts_local, P.acts_vol)


def train_one(cfg, P, arch, cond, seed, l1_coef=None, steps=None, save=True,
              quiet=False, mark_final=True):
    """Trains one SAE; returns (final_L0, sae). For relu, an existing
    checkpoint whose stored L0 is far off target is deleted and retrained."""
    out = ckpt_path(P, arch, cond, seed)
    if save and os.path.exists(out):
        if arch == "relu":
            old = torch.load(out, map_location="cpu", weights_only=False)
            l0_old = float(old.get("final_L0", float("nan")))
            lo, hi = cfg["l1_accept"]
            tgt = cfg["l1_target_l0"]
            if (not old.get("final", False)) and \
                    not (0.5 * lo * tgt <= l0_old <= 2.0 * hi * tgt):
                print(f"[recal] {out} has L0={l0_old:.0f} (target ~{tgt}); "
                      "retraining with calibrated lambda", flush=True)
                os.remove(out)
                for stale in (f"{P.res}/importance_relu_{cond}{P.tag}.json",
                              f"{P.res}/stats_relu_{cond}{P.tag}.npz",
                              f"{P.res}/sharing_summary.json"):
                    if os.path.exists(stale):
                        os.remove(stale)
            else:
                print(f"[skip] {out} exists (L0={l0_old:.0f})")
                return l0_old, None
        else:
            print(f"[skip] {out} exists")
            return None, None
    torch.manual_seed(seed)
    mm = ensure_acts(cfg, P)
    mu, v1, _ = load_pc1(P)
    mu_d, v1_d = mu.to(cfg["device"]), v1.to(cfg["device"])
    bypass = cond == "byp"
    sae = new_sae(cfg, arch).to(cfg["device"])
    opt = torch.optim.Adam(sae.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(seed)
    n = len(mm)
    steps = steps or cfg["train_steps"]

    samp = torch.as_tensor(np.asarray(
        mm[np.sort(rng.choice(n, min(20_000, n), replace=False))],
        dtype=np.float32), device=cfg["device"]) - mu_d
    if bypass:
        pp = samp @ v1_d
        samp = samp - pp.unsqueeze(-1) * v1_d
    sig_var = samp.pow(2).sum(-1).mean().item()
    del samp

    l0_hist = []
    for step in range(steps):
        idx = np.sort(rng.integers(0, n, size=cfg["train_bs"]))
        x = torch.as_tensor(np.asarray(mm[idx], dtype=np.float32),
                            device=cfg["device"])
        c = x - mu_d
        if bypass:
            p = c @ v1_d
            c = c - p.unsqueeze(-1) * v1_d
        recon, z = sae(c)
        mse = (recon - c).pow(2).sum(-1).mean()
        loss = mse
        if arch == "relu":
            loss = loss + l1_coef * z.abs().sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        if step % 100 == 0 or step == steps - 1:
            l0_hist.append((z > 0).float().sum(-1).mean().item())
        if not quiet and (step % 500 == 0 or step == steps - 1):
            print(f"  {arch}/{cond}/s{seed} step {step:6d}  "
                  f"FVU {mse.item() / sig_var:.3f}  L0 {l0_hist[-1]:7.1f}",
                  flush=True)
    l0 = float(np.mean(l0_hist[-3:]))
    if save:
        torch.save({"state_dict": sae.state_dict(), "arch": arch, "cond": cond,
                    "seed": seed, "final_L0": l0, "l1_coef": l1_coef,
                    "final": mark_final}, out)
    return l0, sae


def calibrate_l1(cfg, P, cond):
    """Pilot-run search for lambda giving final L0 ~= k. Multiplicative
    updates on log-lambda; returns lambda and the calibration trace."""
    tgt = cfg["l1_target_l0"]
    lo, hi = cfg["l1_accept"]
    lam, trace = 1.0, []
    best = (float("inf"), lam)
    for it in range(cfg["l1_max_pilots"]):
        l0, _ = train_one(cfg, P, "relu", cond, seed=cfg["seeds"][0],
                          l1_coef=lam, steps=cfg["l1_pilot_steps"],
                          save=False, quiet=True)
        l0 = max(l0, 1e-2)
        trace.append({"lambda": lam, "L0": l0})
        print(f"[cal/{cond}] pilot {it}: lambda={lam:.4g} -> L0={l0:.1f} "
              f"(target {tgt})", flush=True)
        dist = abs(math.log(l0 / tgt))
        if dist < best[0]:
            best = (dist, lam)
        if lo * tgt <= l0 <= hi * tgt:
            return lam, trace
        lam = float(np.clip(lam * (l0 / tgt) ** 0.7, lam / 30, lam * 30))
    print(f"[cal/{cond}] band not hit in {cfg['l1_max_pilots']} pilots; "
          f"using best lambda={best[1]:.4g}", flush=True)
    return best[1], trace


def stage_train(cfg, P, arch):
    cal = {}
    for cond in ("std", "byp"):
        lam = None
        if arch == "relu":
            lam = cfg["l1_coef"]
            if lam is None:
                lam, trace = calibrate_l1(cfg, P, cond)
                cal[cond] = {"lambda": lam, "trace": trace}
            # full-length correction: pilots can underestimate final L0
            # (ReLU SAEs densify over training), so verify on seed A and
            # re-adjust lambda from the ACTUAL final L0 if needed.
            tgt, (lo, hi) = cfg["l1_target_l0"], cfg["l1_accept"]
            for corr in range(3):
                l0, _ = train_one(cfg, P, "relu", cond, cfg["seeds"][0],
                                  l1_coef=lam, mark_final=False)
                if l0 is None or lo * tgt <= l0 <= hi * tgt or corr == 2:
                    break
                new_lam = float(np.clip(lam * (l0 / tgt) ** 0.7,
                                        lam / 30, lam * 30))
                print(f"[recal/{cond}] full-run L0={l0:.1f} off target {tgt}; "
                      f"lambda {lam:.4g} -> {new_lam:.4g}, retraining seed "
                      f"{cfg['seeds'][0]}", flush=True)
                lam = new_lam
                os.remove(ckpt_path(P, "relu", cond, cfg["seeds"][0]))
                cal.setdefault(cond, {}).setdefault(
                    "full_run_corrections", []).append(
                    {"L0": l0, "new_lambda": lam})
            cal.setdefault(cond, {})["lambda"] = lam
            fin = ckpt_path(P, "relu", cond, cfg["seeds"][0])
            if os.path.exists(fin):
                d = torch.load(fin, map_location="cpu", weights_only=False)
                if not d.get("final", False):
                    d["final"] = True
                    torch.save(d, fin)
        for seed in cfg["seeds"]:
            l0, _ = train_one(cfg, P, arch, cond, seed, l1_coef=lam)
            if arch == "relu" and l0 is not None:
                cal.setdefault(cond, {}).setdefault("final_L0", {})[str(seed)] = l0
    if arch == "relu" and cal:
        save_json(cal, f"{P.res}/l1_calibration{P.tag}.json")


def stage_imp(cfg, P):
    model, tok = get_model(cfg)
    mu, v1, _ = load_pc1(P)
    etoks = load_tokens(cfg, tok, cfg["eval_seqs"] * cfg["seq_len"], "validation")
    for arch in ("topk", "relu"):
        for cond in ("std", "byp"):
            out = f"{P.res}/importance_{arch}_{cond}{P.tag}.json"
            if os.path.exists(out):
                print(f"[skip] {out} exists")
                continue
            sae = load_sae(cfg, ckpt_path(P, arch, cond, cfg["seeds"][0]), arch)
            rates, _ = ensure_stats(cfg, P, arch, cond)
            alive = (rates > cfg["alive_rate"]).nonzero().squeeze(-1).tolist()
            fids = alive
            if cfg["importance_cap"]:
                fids = sorted(alive, key=lambda f: -rates[f])[: cfg["importance_cap"]]
            print(f"[imp] {arch}/{cond}: {len(alive)} alive, scoring {len(fids)}")
            base, imps = ablation_importance(
                cfg, model, sae, mu, v1,
                "bypass" if cond == "byp" else "standard", etoks, fids)
            save_json({"alive": alive, "base_ce": base,
                       "importance": {str(k): v for k, v in imps.items()}}, out)


def load_imp(P, arch, cond, d_sae):
    with open(f"{P.res}/importance_{arch}_{cond}{P.tag}.json") as f:
        d = json.load(f)
    alive = torch.zeros(d_sae, dtype=torch.bool)
    alive[d["alive"]] = True
    return alive, {int(k): float(v) for k, v in d["importance"].items()}, d["base_ce"]


def stage_metrics(cfg, P):
    """s2Py Q1: does the inversion depend on the importance metric? Top-50
    sharing under zero-ablation, mean-ablation, gradient x input, and
    integrated gradients, plus rank agreement between metrics."""
    model, tok = get_model(cfg)
    mu, v1, _ = load_pc1(P)
    etoks = load_tokens(cfg, tok, cfg["eval_seqs"] * cfg["seq_len"], "validation")
    out_all = {}
    for cond in ("std", "byp"):
        mode = "bypass" if cond == "byp" else "standard"
        sae_a = load_sae(cfg, ckpt_path(P, "topk", cond, cfg["seeds"][0]), "topk")
        sae_b = load_sae(cfg, ckpt_path(P, "topk", cond, cfg["seeds"][1]), "topk")
        alive, zero_imp, _ = load_imp(P, "topk", cond, cfg["d_sae"])
        _, mean_z = ensure_stats(cfg, P, "topk", cond)
        fids = list(zero_imp.keys())
        mc = cross_seed_max_cos(sae_a, sae_b)

        rankings = {"zero_ablation": zero_imp}
        mpath = f"{P.res}/importance_meanabl_{cond}{P.tag}.json"
        if os.path.exists(mpath):
            rankings["mean_ablation"] = {int(k): float(v) for k, v in
                                         json.load(open(mpath)).items()}
        else:
            print(f"[metrics] mean-ablation ({cond})", flush=True)
            _, mimp = ablation_importance(cfg, model, sae_a, mu, v1, mode,
                                          etoks, fids, values=mean_z)
            save_json({str(k): v for k, v in mimp.items()}, mpath)
            rankings["mean_ablation"] = mimp
        print(f"[metrics] gradient x input ({cond})", flush=True)
        g = grad_importance(cfg, model, sae_a, mu, v1, mode, etoks, ig_steps=1)
        rankings["grad_x_input"] = {f: g[f].item() for f in fids}
        print(f"[metrics] integrated gradients ({cond}, "
              f"m={cfg['metrics_ig_steps']})", flush=True)
        ig = grad_importance(cfg, model, sae_a, mu, v1, mode, etoks,
                             ig_steps=cfg["metrics_ig_steps"])
        rankings["integrated_grads"] = {f: ig[f].item() for f in fids}

        res = {"top50_share": {}, "top50_jaccard_vs_zero": {},
               "spearman_vs_zero": {}}
        zr = rankings["zero_ablation"]
        ztop = set(top_n_recovery(zr, mc, alive)[1])
        common = sorted(zr.keys())
        zvals = np.array([zr[f] for f in common])
        for mname, r in rankings.items():
            share, top = top_n_recovery(r, mc, alive)
            res["top50_share"][mname] = share
            res["top50_jaccard_vs_zero"][mname] = (
                len(ztop & set(top)) / max(1, len(ztop | set(top))))
            res["spearman_vs_zero"][mname] = float(
                spearmanr(zvals, np.array([r[f] for f in common])).statistic)
            print(f"  [{cond}] {mname:17s} top-50 share "
                  f"{100 * share:5.1f}%  jacc {res['top50_jaccard_vs_zero'][mname]:.2f}",
                  flush=True)
        out_all[cond] = res
    save_json(out_all, f"{P.res}/metrics_summary.json")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    mnames = list(out_all["std"]["top50_share"].keys())
    xs = np.arange(len(mnames))
    for j, (cond, color, lbl) in enumerate(
            [("std", "#b23b3b", "standard"), ("byp", "#2f7d5d", "bypass")]):
        ax.bar(xs + (j - 0.5) * 0.35,
               [100 * out_all[cond]["top50_share"][m] for m in mnames],
               0.35, color=color, label=lbl)
    ax.set_xticks(xs)
    ax.set_xticklabels([m.replace("_", "\n") for m in mnames], fontsize=8)
    ax.set_ylabel("top-50 cross-seed share (%)")
    ax.set_title("Inversion holds under every importance metric")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{P.res}/fig_metrics.png", dpi=200)
    fig.savefig(f"{P.res}/fig_metrics.pdf")


def stage_cloud(cfg, P):
    """Dense-cluster cancellation diagnostic: each PC1-aligned feature's
    activation-weighted contribution c_i = E[z_i] d_i; individual vs
    aggregate off-axis energy, cancellation ratio, effective rank."""
    mu, v1, _ = load_pc1(P)
    out = {}
    for cond in ("std", "byp"):
        sae = load_sae(cfg, ckpt_path(P, "topk", cond, cfg["seeds"][0]), "topk")
        alive, _, _ = load_imp(P, "topk", cond, cfg["d_sae"])
        rates, mean_z = ensure_stats(cfg, P, "topk", cond)
        D = sae.decoder_dirs().cpu()
        cos = (D @ v1)
        dense = (cos.abs() > cfg["cloud_cos"]) & alive
        n_dense = int(dense.sum())
        rec = {"n_alive": int(alive.sum()), "n_dense": n_dense,
               "cos_threshold": cfg["cloud_cos"]}
        if n_dense >= 2:
            C = mean_z[dense].unsqueeze(-1) * D[dense]          # [nd, d]
            proj = C @ v1
            par = proj.unsqueeze(-1) * v1
            off = C - par
            indiv = float(off.pow(2).sum())
            aggr = float(off.sum(0).pow(2).sum())
            gram_eigs = torch.linalg.eigvalsh(C @ C.t()).clamp(min=0)
            s = gram_eigs.sum()
            rec.update(
                pc1_fraction_of_cloud_energy=float(proj.pow(2).sum() /
                                                   C.pow(2).sum()),
                individual_offaxis_energy=indiv,
                aggregate_offaxis_energy=aggr,
                cancellation_ratio=float(indiv / max(aggr, 1e-9)),
                cov_effective_rank=float(s ** 2 / gram_eigs.pow(2).sum()),
                cov_top_eig_share=float(gram_eigs.max() / s))
        out[cond] = rec
        print(f"[cloud] {cond}: {rec}", flush=True)
    save_json(out, f"{P.res}/cloud_diagnostic_pipeline.json")


def _share_numbers(cfg, P, arch, cond):
    sae_a = load_sae(cfg, ckpt_path(P, arch, cond, cfg["seeds"][0]), arch)
    sae_b = load_sae(cfg, ckpt_path(P, arch, cond, cfg["seeds"][1]), arch)
    alive, imp, base_ce = load_imp(P, arch, cond, cfg["d_sae"])
    mc = cross_seed_max_cos(sae_a, sae_b)
    t50, top = top_n_recovery(imp, mc, alive)
    return {"n_alive": int(alive.sum()), "dict_wide_share": dict_wide_share(mc, alive),
            "top50_share": t50, "base_ce": base_ce}, (sae_a, alive, imp, mc, top)


def stage_layers(cfg, P):
    """Creation-layer negative control + dose-response. Trains std/byp pairs
    at each extra layer, evaluates sharing, and plots inversion vs PC1%."""
    get_model(cfg)  # warm the shared model cache
    for L in cfg["extra_layers"]:
        out = f"{P.res}/layer_L{L}.json"
        if os.path.exists(out):
            print(f"[skip] {out} exists")
            continue
        cfg_L = dict(cfg)
        cfg_L["layer_idx"] = L
        P_L = mk_paths(P.root, tag=f"_L{L}")
        if not os.path.exists(P_L.pc1):
            stage_collect(cfg_L, P_L)
        for cond in ("std", "byp"):
            for seed in cfg["seeds"]:
                train_one(cfg_L, P_L, "topk", cond, seed)
        model, tok = get_model(cfg_L)
        mu, v1, var = load_pc1(P_L)
        etoks = load_tokens(cfg_L, tok, cfg["eval_seqs"] * cfg["seq_len"],
                            "validation")
        rec = {"pc1_pct": 100 * var, "conditions": {}}
        for cond in ("std", "byp"):
            ipath = f"{P_L.res}/importance_topk_{cond}{P_L.tag}.json"
            if not os.path.exists(ipath):
                sae = load_sae(cfg_L, ckpt_path(P_L, "topk", cond,
                                                cfg["seeds"][0]), "topk")
                rates, _ = ensure_stats(cfg_L, P_L, "topk", cond)
                alive = (rates > cfg["alive_rate"]).nonzero().squeeze(-1).tolist()
                fids = alive
                if cfg["importance_cap"]:
                    fids = sorted(alive, key=lambda f: -rates[f])[: cfg["importance_cap"]]
                base, imps = ablation_importance(
                    cfg_L, model, sae, mu, v1,
                    "bypass" if cond == "byp" else "standard", etoks, fids)
                save_json({"alive": alive, "base_ce": base,
                           "importance": {str(k): v for k, v in imps.items()}},
                          ipath)
            nums, _ = _share_numbers(cfg_L, P_L, "topk", cond)
            rec["conditions"]["standard" if cond == "std" else "bypass"] = nums
            print(f"[layers] L{L}/{cond}: {nums}", flush=True)
        save_json(rec, out)
        if os.path.exists(P_L.acts_local):
            os.remove(P_L.acts_local)

    # dose-response figure: this run's layers + main layer + optional external
    sweep = {}
    for L in cfg["extra_layers"]:
        p = f"{P.res}/layer_L{L}.json"
        if os.path.exists(p):
            sweep[f"L{L}"] = json.load(open(p))
    main_ok = all(os.path.exists(f"{P.res}/importance_topk_{c}.json")
                  for c in ("std", "byp"))
    if main_ok:
        _, _, var = load_pc1(mk_paths(P.root))
        rec = {"pc1_pct": 100 * var, "conditions": {}}
        for cond in ("std", "byp"):
            nums, _ = _share_numbers(cfg, mk_paths(P.root), "topk", cond)
            rec["conditions"]["standard" if cond == "std" else "bypass"] = nums
        sweep[f"L{cfg['layer_idx']}"] = rec
    ext = f"{P.res}/external_layers.json"
    if os.path.exists(ext):
        for kk, vv in json.load(open(ext)).items():
            sweep.setdefault(kk, vv)
    save_json(sweep, f"{P.res}/layer_sweep.json")
    if len(sweep) >= 2:
        pts = sorted(sweep.items(), key=lambda kv: kv[1]["pc1_pct"])
        x = [v["pc1_pct"] for _, v in pts]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for key, color, lbl in (("standard", "#b23b3b", "standard"),
                                ("bypass", "#2f7d5d", "bypass")):
            y = [100 * v["conditions"][key]["top50_share"] for _, v in pts]
            ax.plot(x, y, "o-", color=color, label=lbl)
        for (name, v), xi in zip(pts, x):
            ax.annotate(name, (xi, 100 * v["conditions"]["standard"]["top50_share"]),
                        fontsize=7, textcoords="offset points", xytext=(3, -9))
        ax.set_xlabel("PC1 variance explained (%)")
        ax.set_ylabel("top-50 cross-seed share (%)")
        ax.set_title("Inversion severity tracks PC1 dominance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{P.res}/fig_dose_response.png", dpi=200)
        fig.savefig(f"{P.res}/fig_dose_response.pdf")


def stage_figs(cfg, P):
    mu, v1, _ = load_pc1(P)
    summary, cache = {}, {}
    for arch in ("topk", "relu"):
        for cond in ("std", "byp"):
            nums, extra = _share_numbers(cfg, P, arch, cond)
            summary[f"{arch}_{cond}"] = nums
            cache[f"{arch}_{cond}"] = extra

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, cond, title, color in zip(
            axes, ("std", "byp"), ("Standard SAE", "After PC1 bypass"),
            ("#b23b3b", "#2f7d5d")):
        a, alive, *_ = cache[f"topk_{cond}"]
        cosv1 = (a.decoder_dirs().cpu() @ v1).abs().numpy()[alive.numpy()]
        ax.hist(cosv1, bins=60, range=(0, 1), color=color, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel(r"$|\cos(d_i,\ v_1)|$")
        ax.set_title(f"{title}\n{100 * float((cosv1 > 0.5).mean()):.1f}% "
                     r"of alive features $|\cos|>0.5$", fontsize=9)
    axes[0].set_ylabel("features (log)")
    fig.tight_layout()
    fig.savefig(f"{P.res}/figR1_decoder_pc1_alignment.png", dpi=200)
    fig.savefig(f"{P.res}/figR1_decoder_pc1_alignment.pdf")

    fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for ax, cond, title, color in zip(
            axes2, ("std", "byp"), ("Standard SAE", "After PC1 bypass"),
            ("#b23b3b", "#2f7d5d")):
        _, alive, imp, mc, top = cache[f"topk_{cond}"]
        s = summary[f"topk_{cond}"]
        ax.hist(mc[alive].numpy(), bins=50, range=(0, 1), color="0.7",
                alpha=0.8, density=True, label="all alive")
        ax.hist(mc[top].numpy(), bins=25, range=(0, 1), color=color,
                alpha=0.85, density=True, label=f"top-{len(top)} by importance")
        ax.axvline(0.7, ls="--", c="k", lw=1)
        ax.set_title(f"{title}\ntop-50 shared {100 * s['top50_share']:.0f}%  "
                     f"(dict-wide {100 * s['dict_wide_share']:.0f}%)", fontsize=9)
        ax.set_xlabel("cross-seed max decoder cosine")
        ax.legend(fontsize=7)
    axes2[0].set_ylabel("density")
    fig2.tight_layout()
    fig2.savefig(f"{P.res}/figR2_crossseed_cosine.png", dpi=200)
    fig2.savefig(f"{P.res}/figR2_crossseed_cosine.pdf")
    save_json(summary, f"{P.res}/sharing_summary.json")


def fit_r2(Xtr, ytr, Xte, yte, alphas=(1.0, 10.0, 100.0), max_rows=None,
           return_pred=False):
    if max_rows and len(Xtr) > max_rows:
        sel = np.random.default_rng(0).choice(len(Xtr), max_rows, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    if Xtr.shape[1] == 1:
        m = LinearRegression().fit(Xtr, ytr)
    else:
        cut = int(0.8 * len(Xtr))
        best, best_a = -np.inf, alphas[0]
        for a in alphas:
            mm_ = Ridge(alpha=a).fit(Xtr[:cut], ytr[:cut])
            v = r2_score(ytr[cut:], mm_.predict(Xtr[cut:]))
            if v > best:
                best, best_a = v, a
        m = Ridge(alpha=best_a).fit(Xtr, ytr)
    pred = m.predict(Xte)
    r2 = float(r2_score(yte, pred))
    return (r2, pred) if return_pred else r2


def seq_boot_ci(yte, pred, seq_te, n_boot, seed=0):
    """Block bootstrap over held-out sequences -> 95% CI on R^2."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(seq_te)
    idx_by_seq = {s: np.where(seq_te == s)[0] for s in uniq}
    r2s = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        rows = np.concatenate([idx_by_seq[s] for s in pick])
        r2s.append(r2_score(yte[rows], pred[rows]))
    return [float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))]


def stage_probe(cfg, P):
    """AiXx's experiment, hardened: sequence-level split (no within-sequence
    leakage), block-bootstrap CIs, entropy/surprisal/log-freq/position
    targets, attention-sink-mass target from eager attention."""
    model, tok = get_model(cfg)
    mu, v1, var = load_pc1(P)
    toks = load_tokens(cfg, tok, cfg["probe_tokens"], ("validation", "test"))
    data = collect_block_outputs(cfg, model, toks, with_targets=True,
                                 batch_size=cfg["probe_batch"])
    acts = data["acts"].float()
    ent, sur, tids = data["entropy"], data["surprisal"], data["token_ids"]
    Tm1 = cfg["seq_len"] - 1
    n_seqs = len(acts) // Tm1
    seq_id = np.repeat(np.arange(n_seqs), Tm1)
    pos = np.tile(np.arange(Tm1), n_seqs)

    ftoks = load_tokens(cfg, tok, cfg["probe_freq_tokens"], "train").reshape(-1).numpy()
    ml = int(max(tids.max(), ftoks.max())) + 1
    counts = np.bincount(ftoks, minlength=ml)
    logfreq = np.log1p(counts[tids]).astype(np.float64)

    centered = acts - mu
    p = (centered @ v1).numpy().astype(np.float64)[:, None]
    residual = (centered - torch.as_tensor(p, dtype=torch.float32) * v1).numpy()
    normf = acts.norm(dim=-1).numpy().astype(np.float64)[:, None]
    centered = centered.numpy()

    rng = np.random.default_rng(0)
    seq_perm = rng.permutation(n_seqs)
    tr_seqs = set(seq_perm[: int(0.8 * n_seqs)].tolist())
    tr = np.array([i for i in range(len(p)) if seq_id[i] in tr_seqs])
    te = np.array([i for i in range(len(p)) if seq_id[i] not in tr_seqs])

    targets = {"entropy": ent.astype(np.float64),
               "surprisal": sur.astype(np.float64),
               "log_freq": logfreq,
               "log_position": np.log1p(pos).astype(np.float64)}
    preds = {"pc1_proj_1d": p, "act_norm_1d": normf,
             "residual_ridge": residual, "full_act_ridge": centered}
    results = {"pc1_variance_explained": var, "n_positions": int(len(p)),
               "n_seqs": int(n_seqs), "split": "by_sequence",
               "r2": {}, "r2_ci95": {}, "spearman_pc1": {}}
    for tname, y in targets.items():
        results["r2"][tname] = {}
        results["r2_ci95"][tname] = {}
        for pname, X in preds.items():
            r2, pred = fit_r2(X[tr], y[tr], X[te], y[te],
                              max_rows=cfg["probe_ridge_rows"], return_pred=True)
            results["r2"][tname][pname] = r2
            if pname in ("pc1_proj_1d", "residual_ridge"):
                results["r2_ci95"][tname][pname] = seq_boot_ci(
                    y[te], pred, seq_id[te], cfg["probe_boot"])
            print(f"  R2[{tname:12s} <- {pname:15s}] = {r2:+.3f}", flush=True)
        r2s = []
        for _ in range(10):
            u = rng.standard_normal(centered.shape[1])
            u /= np.linalg.norm(u)
            Xr = (centered @ u)[:, None]
            r2s.append(fit_r2(Xr[tr], y[tr], Xr[te], y[te]))
        results["r2"][tname]["rand_dir_1d_mean"] = float(np.mean(r2s))
        results["r2"][tname]["rand_dir_1d_max"] = float(np.max(r2s))
        results["spearman_pc1"][tname] = float(spearmanr(p[:, 0], y).statistic)

    if cfg["probe_attn_tokens"] > 0:
        model_a, _ = get_model(cfg, eager=True)
        atoks = toks[: max(2, cfg["probe_attn_tokens"] // cfg["seq_len"])]
        grab = Grabber(model_a, cfg["layer_idx"])
        Xa, ya = [], []
        with torch.no_grad():
            for i in range(0, len(atoks), 8):
                b = atoks[i:i + 8].to(cfg["device"])
                out = model_a(b, output_attentions=True)
                li = min(cfg["layer_idx"] + 1, len(out.attentions) - 1)
                sink = out.attentions[li][:, :, :, 0].mean(1)
                Xa.append(grab.value[:, :-1].reshape(-1, cfg["d_model"]).float().cpu())
                ya.append(sink[:, :-1].reshape(-1).float().cpu())
        grab.remove()
        Xa = torch.cat(Xa) - mu
        ya = torch.cat(ya).numpy().astype(np.float64)
        na_seqs = len(Xa) // Tm1
        aseq = np.repeat(np.arange(na_seqs), Tm1)
        pa = (Xa @ v1).numpy().astype(np.float64)[:, None]
        ra = (Xa - torch.as_tensor(pa, dtype=torch.float32) * v1).numpy()
        ap = rng.permutation(na_seqs)
        atr_s = set(ap[: int(0.8 * na_seqs)].tolist())
        atr = np.array([i for i in range(len(pa)) if aseq[i] in atr_s])
        ate = np.array([i for i in range(len(pa)) if aseq[i] not in atr_s])
        results["r2"]["attn_sink_mass"] = {}
        results["r2_ci95"]["attn_sink_mass"] = {}
        for pname, X in (("pc1_proj_1d", pa), ("residual_ridge", ra)):
            r2, pred = fit_r2(X[atr], ya[atr], X[ate], ya[ate],
                              max_rows=cfg["probe_ridge_rows"], return_pred=True)
            results["r2"]["attn_sink_mass"][pname] = r2
            results["r2_ci95"]["attn_sink_mass"][pname] = seq_boot_ci(
                ya[ate], pred, aseq[ate], cfg["probe_boot"])
        print(f"  R2[sink <- pc1] = "
              f"{results['r2']['attn_sink_mass']['pc1_proj_1d']:+.3f}  "
              f"[sink <- residual] = "
              f"{results['r2']['attn_sink_mass']['residual_ridge']:+.3f}", flush=True)

    save_json(results, f"{P.res}/probe_v2.json")

    tnames = [t for t in results["r2"] if t != "attn_sink_mass"]
    pnames = ["pc1_proj_1d", "rand_dir_1d_mean", "residual_ridge", "full_act_ridge"]
    labels = ["PC1 proj (1-d)", "random dir (1-d)", "residual (ridge)",
              "full act (ridge)"]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    xs = np.arange(len(tnames))
    for j, (pn, lb) in enumerate(zip(pnames, labels)):
        ax.bar(xs + (j - 1.5) * 0.2, [max(results["r2"][t][pn], 0) for t in tnames],
               0.2, label=lb)
    ax.set_xticks(xs)
    ax.set_xticklabels(tnames)
    ax.set_ylabel(r"held-out $R^2$")
    ax.set_title(f"What does PC1 encode? ({cfg['model_name'].split('/')[-1]}, "
                 f"layer {cfg['layer_idx']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{P.res}/fig_probe_r2.png", dpi=200)
    fig.savefig(f"{P.res}/fig_probe_r2.pdf")

    fig2, ax2 = plt.subplots(figsize=(4.2, 3.4))
    hb = ax2.hexbin(p[te, 0], targets["entropy"][te], gridsize=60,
                    bins="log", cmap="viridis")
    fig2.colorbar(hb, label="log count")
    ax2.set_xlabel("PC1 projection")
    ax2.set_ylabel("next-token entropy (nats)")
    fig2.tight_layout()
    fig2.savefig(f"{P.res}/fig_pc1_vs_entropy.png", dpi=200)


SP_TASKS = {
    "sst2": dict(path=("nyu-mll/glue", "sst2"), text="sentence", label="label",
                 train="train", test="validation"),
    "ag_news": dict(path=("fancyzhx/ag_news",), text="text", label="label",
                    train="train", test="test"),
    "amazon_polarity": dict(path=("fancyzhx/amazon_polarity",), text="content",
                            label="label", train="train", test="test"),
    "emotion": dict(path=("dair-ai/emotion",), text="text", label="label",
                    train="train", test="test"),
}


@torch.no_grad()
def sp_embed(cfg, model, tok, sae, mu, v1, bypass, texts, bs=32):
    mu_d, v1_d = mu.to(cfg["device"]), v1.to(cfg["device"])
    grab = Grabber(model, cfg["layer_idx"])
    Z, A = [], []
    for i in range(0, len(texts), bs):
        enc = tok(list(texts[i:i + bs]), padding="max_length", truncation=True,
                  max_length=cfg["seq_len"], return_tensors="pt")
        ids = enc["input_ids"].to(cfg["device"])
        mask = enc["attention_mask"].to(cfg["device"]).float()
        model(ids, attention_mask=mask)
        x = grab.value.float()
        c = x - mu_d
        if bypass:
            p = c @ v1_d
            c = c - p.unsqueeze(-1) * v1_d
        z = sae.encode(c.reshape(-1, cfg["d_model"])).reshape(
            len(ids), -1, cfg["d_sae"])
        denom = mask.sum(-1, keepdim=True).clamp(min=1)
        Z.append(((z * mask.unsqueeze(-1)).sum(1) / denom).cpu())
        A.append(((x * mask.unsqueeze(-1)).sum(1) / denom).cpu())
    grab.remove()
    return torch.cat(Z).numpy(), torch.cat(A).numpy()


def stage_probing(cfg, P):
    """Sparse probing with resampling: embed a fixed pool once per
    task/condition, then rep-resample the training subset for mean +/- sd."""
    model, tok = get_model(cfg)
    mu, v1, _ = load_pc1(P)
    combined = {}
    for cond in ("std", "byp"):
        out = f"{P.res}/sparse_probing_v2_{cond}.json"
        if os.path.exists(out):
            combined[cond] = json.load(open(out))
            print(f"[skip] {out} exists")
            continue
        sae = load_sae(cfg, ckpt_path(P, "topk", cond, cfg["seeds"][0]), "topk")
        res = {"condition": cond, "tasks": {}}
        for tname in cfg["sp_tasks"]:
            spec = SP_TASKS[tname]
            ds_tr = load_dataset(*spec["path"], split=spec["train"]) \
                .shuffle(seed=0).select(range(cfg["sp_train_pool"]))
            ds_te = load_dataset(*spec["path"], split=spec["test"]).shuffle(seed=0)
            ds_te = ds_te.select(range(min(cfg["sp_test"], len(ds_te))))
            ytr_pool = np.array(ds_tr[spec["label"]])
            yte = np.array(ds_te[spec["label"]])
            print(f"[probing/{cond}] {tname}: embedding "
                  f"{len(ds_tr)}/{len(ds_te)}", flush=True)
            Zp, Ap = sp_embed(cfg, model, tok, sae, mu, v1, cond == "byp",
                              ds_tr[spec["text"]])
            Zte, Ate = sp_embed(cfg, model, tok, sae, mu, v1, cond == "byp",
                                ds_te[spec["text"]])
            task = {"majority": float(np.mean(yte == np.bincount(ytr_pool).argmax())),
                    "sae_topk_acc": {str(k): [] for k in cfg["sp_ks"]},
                    "raw_act_probe_acc": []}
            for rep in range(cfg["sp_reps"]):
                rr = np.random.default_rng(rep)
                rows = rr.choice(len(Zp), min(cfg["sp_train"], len(Zp)),
                                 replace=False)
                Ztr, ytr = Zp[rows], ytr_pool[rows]
                Fsc, _ = f_classif(Ztr, ytr)
                Fsc = np.nan_to_num(Fsc, nan=0.0)
                for kk in cfg["sp_ks"]:
                    sel = np.argsort(-Fsc)[:kk]
                    clf = LogisticRegression(max_iter=2000).fit(Ztr[:, sel], ytr)
                    task["sae_topk_acc"][str(kk)].append(
                        float(clf.score(Zte[:, sel], yte)))
                sc = StandardScaler().fit(Ap[rows])
                clf = LogisticRegression(max_iter=2000).fit(
                    sc.transform(Ap[rows]), ytr)
                task["raw_act_probe_acc"].append(
                    float(clf.score(sc.transform(Ate), yte)))
            task["sae_topk_acc_mean"] = {k: float(np.mean(v)) for k, v in
                                         task["sae_topk_acc"].items()}
            task["sae_topk_acc_sd"] = {k: float(np.std(v)) for k, v in
                                       task["sae_topk_acc"].items()}
            task["raw_mean"] = float(np.mean(task["raw_act_probe_acc"]))
            print(f"  mean accs: {task['sae_topk_acc_mean']}  "
                  f"raw={task['raw_mean']:.3f}", flush=True)
            res["tasks"][tname] = task
        save_json(res, out)
        combined[cond] = res
    ks = [str(k) for k in cfg["sp_ks"]]
    lines = ["| task | cond | " + " | ".join(f"k={k}" for k in ks) +
             " | raw acts | majority |",
             "|---|---|" + "---|" * (len(ks) + 2)]
    for cond, res in combined.items():
        for tname, task in res["tasks"].items():
            row = [tname, cond]
            row += [f"{task['sae_topk_acc_mean'][k]:.3f}±"
                    f"{task['sae_topk_acc_sd'][k]:.3f}" for k in ks]
            row += [f"{task['raw_mean']:.3f}", f"{task['majority']:.3f}"]
            lines.append("| " + " | ".join(row) + " |")
    with open(f"{P.res}/sparse_probing_v2_table.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


EXPLAIN_PROMPT = """You are analyzing one feature (latent) of a sparse \
autoencoder trained on a language model. Below are text snippets on which \
the feature activates strongly. In each snippet, the single token with the \
strongest activation is marked with << >>.

{snippets}

In at most 20 words, state the pattern this feature responds to. If the \
examples share no discernible pattern, reply exactly: NO PATTERN. Reply \
with the description only."""

DETECT_PROMPT = """A sparse autoencoder feature has this description: "{desc}"

For each numbered snippet below, decide whether this feature would activate \
somewhere in it. Reply with only a JSON array of {n} integers in order \
(1 = activates, 0 = does not).

{snippets}"""


def call_llm(client, model, prompt, max_tokens=300, retries=3):
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model, max_tokens=max_tokens, temperature=0.0,
                messages=[{"role": "user", "content": prompt}])
            return resp.content[0].text.strip()
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"  [api] {type(e).__name__}: {e} — retry in {wait}s", flush=True)
            time.sleep(wait)
    return None


def render_snip(tok, ids, mark=None):
    if mark is None:
        return tok.decode(ids)
    return (tok.decode(ids[:mark]) + "<<" + tok.decode(ids[mark:mark + 1]) +
            ">>" + tok.decode(ids[mark + 1:]))


@torch.no_grad()
def gather_contexts(cfg, model, tok, sae, mu, v1, bypass, feats,
                    window=15, n_pos=40, n_neg=30, bs=32):
    toks = load_tokens(cfg, tok, cfg["ai_pool_tokens"], ("validation", "test"))
    mu_d, v1_d = mu.to(cfg["device"]), v1.to(cfg["device"])
    fidx = torch.tensor(feats, device=cfg["device"])
    pos = {f: [] for f in feats}
    neg = {f: [] for f in feats}
    rng = np.random.default_rng(0)
    grab = Grabber(model, cfg["layer_idx"])
    T = toks.shape[1]
    for i in range(0, len(toks), bs):
        b = toks[i:i + bs].to(cfg["device"])
        model(b)
        c = grab.value.reshape(-1, cfg["d_model"]).float() - mu_d
        if bypass:
            p = c @ v1_d
            c = c - p.unsqueeze(-1) * v1_d
        z = sae.encode(c)[:, fidx].reshape(len(b), T, len(feats))
        for fi, f in enumerate(feats):
            zf = z[:, :, fi]
            k = min(4, zf.numel())
            topv, flat = zf.reshape(-1).topk(k)
            for v, fl in zip(topv.tolist(), flat.tolist()):
                if v > 0:
                    pos[f].append((v, i + fl // T, fl % T))
            if len(neg[f]) < n_neg:
                zeros = (zf == 0).nonzero()
                if len(zeros):
                    r, p_ = zeros[rng.integers(len(zeros))].tolist()
                    neg[f].append((i + r, p_))
        if (i // bs) % 200 == 199:  # keep candidate lists bounded
            for f in feats:
                pos[f].sort(key=lambda t: -t[0])
                del pos[f][n_pos * 3:]
    grab.remove()
    ctx = {}
    for f in feats:
        pos[f].sort(key=lambda t: -t[0])
        P_, N_ = [], []
        for v, row, p_ in pos[f][:n_pos]:
            lo, hi = max(0, p_ - window), min(T, p_ + window + 1)
            P_.append({"act": v,
                       "text": render_snip(tok, toks[row, lo:hi].tolist(), p_ - lo)})
        for row, p_ in neg[f]:
            lo, hi = max(0, p_ - window), min(T, p_ + window + 1)
            N_.append({"text": render_snip(tok, toks[row, lo:hi].tolist())})
        ctx[f] = {"pos": P_, "neg": N_}
    return ctx


def stage_autointerp(cfg, P):
    """Two-phase auto-interp over top-N by importance PLUS a random-alive
    control group per condition; detection on 10 pos / 10 neg held-out."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    llm_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    mu, v1, _ = load_pc1(P)
    model, tok = get_model(cfg)
    npos_det, nneg_det = cfg["ai_det_pos"], cfg["ai_det_neg"]
    for cond in ("std", "byp"):
        outdir = f"{P.res}/autointerp2_{cond}"
        os.makedirs(outdir, exist_ok=True)
        if os.path.exists(f"{outdir}/summary.json"):
            print(f"[skip] {outdir}/summary.json exists")
            continue
        alive, imp, _ = load_imp(P, "topk", cond, cfg["d_sae"])
        ranked = [f for f in sorted(imp, key=imp.get, reverse=True) if alive[f]]
        top = ranked[: cfg["ai_top_n"]]
        pool_rest = [int(f) for f in alive.nonzero().squeeze(-1).tolist()
                     if f not in set(top)]
        rand = np.random.default_rng(1).choice(
            pool_rest, min(cfg["ai_rand_n"], len(pool_rest)),
            replace=False).tolist()
        feats = top + rand
        group = {**{f: "top" for f in top}, **{f: "random" for f in rand}}
        cpath = f"{outdir}/contexts.json"
        if os.path.exists(cpath):
            ctx = {int(k): v for k, v in json.load(open(cpath)).items()}
        else:
            sae = load_sae(cfg, ckpt_path(P, "topk", cond, cfg["seeds"][0]), "topk")
            ctx = gather_contexts(cfg, model, tok, sae, mu, v1,
                                  cond == "byp", feats)
            save_json({str(k): v for k, v in ctx.items()}, cpath)
        if not key:
            print(f"[autointerp/{cond}] contexts cached; no ANTHROPIC_API_KEY — "
                  "skipping LLM scoring (rerun this stage with the key set)")
            continue
        import anthropic
        client = anthropic.Anthropic()
        rpath = f"{outdir}/results.json"
        results = ({int(k): v for k, v in json.load(open(rpath)).items()}
                   if os.path.exists(rpath) else {})
        for f in feats:
            if f in results:
                continue
            recs = ctx.get(f, {"pos": [], "neg": []})
            need_pos = 12 + npos_det
            if len(recs["pos"]) < need_pos or len(recs["neg"]) < nneg_det:
                results[f] = {"group": group[f], "skip": "too few contexts",
                              "n_pos": len(recs["pos"])}
                continue
            snips = "\n".join(f"- {r['text']}" for r in recs["pos"][:12])
            desc = call_llm(client, llm_model,
                            EXPLAIN_PROMPT.format(snippets=snips))
            if desc is None:
                results[f] = {"group": group[f], "skip": "api failure"}
                continue
            no_pat = desc.strip().upper().startswith("NO PATTERN")
            held = recs["pos"][12:12 + npos_det]
            items = ([(r["text"], 1) for r in held] +
                     [(r["text"], 0) for r in recs["neg"][:nneg_det]])
            order = np.random.default_rng(f).permutation(len(items))
            items = [items[j] for j in order]
            if no_pat:
                score = 0.5
            else:
                det = "\n".join(f"{j + 1}. {t}" for j, (t, _) in enumerate(items))
                ans = call_llm(client, llm_model, DETECT_PROMPT.format(
                    desc=desc, n=len(items), snippets=det))
                m = re.search(r"\[.*?\]", ans or "", re.S)
                try:
                    pr = np.array(json.loads(m.group(0)))
                    assert len(pr) == len(items)
                except Exception:
                    results[f] = {"group": group[f],
                                  "skip": "unparseable detection", "raw": ans}
                    continue
                lab = np.array([l for _, l in items])
                tpr = pr[lab == 1].mean() if (lab == 1).any() else 0.0
                tnr = (1 - pr[lab == 0]).mean() if (lab == 0).any() else 0.0
                score = float((tpr + tnr) / 2)
            results[f] = {"group": group[f], "description": desc,
                          "no_pattern": bool(no_pat),
                          "detection_balanced_acc": score,
                          "importance": imp.get(f)}
            print(f"  [{cond}] {group[f]:6s} feat {f:5d}  det {score:.2f}  "
                  f"{desc[:55]}", flush=True)
            save_json({str(k): v for k, v in results.items()}, rpath)
        summary = {"condition": cond, "llm": llm_model}
        for g in ("top", "random"):
            sc = [r["detection_balanced_acc"] for r in results.values()
                  if r.get("group") == g and "detection_balanced_acc" in r]
            summary[g] = {
                "n_scored": len(sc),
                "n_no_pattern": int(sum(r.get("no_pattern", False)
                                        for r in results.values()
                                        if r.get("group") == g)),
                "mean_detection_balanced_acc":
                    float(np.mean(sc)) if sc else None}
        save_json(summary, f"{outdir}/summary.json")


def stage_report(cfg, P):
    print("\n" + "=" * 64 + "\nSUMMARY\n" + "=" * 64)
    digest = {}
    for name in ("sharing_summary.json", "metrics_summary.json",
                 "cloud_diagnostic_pipeline.json", "layer_sweep.json",
                 "probe_v2.json", "l1_calibration.json",
                 "autointerp2_std/summary.json", "autointerp2_byp/summary.json"):
        path = f"{P.res}/{name}"
        if os.path.exists(path):
            digest[name] = json.load(open(path))
            print(f"\n--- {name}")
            print(json.dumps(digest[name], indent=2)[:2200])
    for name in ("sparse_probing_v2_table.md",):
        path = f"{P.res}/{name}"
        if os.path.exists(path):
            print(f"\n--- {name}")
            print(open(path).read())
    save_json(digest, f"{P.res}/rebuttal_numbers.json")
    print(f"\nAll artifacts: modal volume get {VOL_NAME} results ./results")


# ------------------------------------------------------------------ driver

def _run(stages_csv, cfg, root, commit=None, force=False):
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    print(f"[run] device={cfg['device']}  model={cfg['model_name']}  "
          f"layer={cfg['layer_idx']}  d_sae={cfg['d_sae']}  k={cfg['k']}")
    P = mk_paths(root)
    req = [s.strip() for s in stages_csv.split(",") if s.strip()]
    bad = set(req) - set(STAGE_ORDER)
    if bad:
        raise ValueError(f"unknown stage(s) {sorted(bad)}; valid: {STAGE_ORDER}")

    def done(*files):
        return all(os.path.exists(f) for f in files)

    plan = [
        ("collect", [P.meta, P.pc1], lambda: stage_collect(cfg, P)),
        ("topk", [ckpt_path(P, "topk", c, s) for c in ("std", "byp")
                  for s in cfg["seeds"]], lambda: stage_train(cfg, P, "topk")),
        ("l1", [], lambda: stage_train(cfg, P, "relu")),   # self-skipping/self-healing
        ("imp", [f"{P.res}/importance_{a}_{c}.json" for a in ("topk", "relu")
                 for c in ("std", "byp")], lambda: stage_imp(cfg, P)),
        ("metrics", [f"{P.res}/metrics_summary.json"],
         lambda: stage_metrics(cfg, P)),
        ("cloud", [f"{P.res}/cloud_diagnostic_pipeline.json"],
         lambda: stage_cloud(cfg, P)),
        ("layers", [f"{P.res}/layer_sweep.json"], lambda: stage_layers(cfg, P)),
        ("figs", [f"{P.res}/sharing_summary.json"], lambda: stage_figs(cfg, P)),
        ("probe", [f"{P.res}/probe_v2.json"], lambda: stage_probe(cfg, P)),
        ("probing", [f"{P.res}/sparse_probing_v2_table.md"],
         lambda: stage_probing(cfg, P)),
        ("autointerp", [f"{P.res}/autointerp2_std/summary.json",
                        f"{P.res}/autointerp2_byp/summary.json"],
         lambda: stage_autointerp(cfg, P)),
        ("report", [], lambda: stage_report(cfg, P)),
    ]
    for name, outputs, fn in plan:
        if name not in req:
            continue
        if name != "report" and not force and outputs and done(*outputs):
            print(f"[skip] stage '{name}' (outputs exist)")
            continue
        t0 = time.time()
        print(f"\n[stage] {name} ...", flush=True)
        fn()
        print(f"[stage] {name} done in {(time.time() - t0) / 60:.1f} min",
              flush=True)
        if commit:
            commit()


@app.function(image=image, gpu="H100", timeout=24 * 60 * 60,
              volumes={VOL_PATH: vol}, memory=131072, cpu=16)
def run_pipeline(stages: str = DEFAULT_STAGES, smoke: bool = False,
                 l1_coef: float = 0.0, force: bool = False,
                 anthropic_key: str = ""):
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    cfg = get_cfg(smoke)
    if l1_coef > 0:
        cfg["l1_coef"] = l1_coef
    _run(stages, cfg, VOL_PATH, commit=vol.commit, force=force)


@app.local_entrypoint()
def main(stages: str = DEFAULT_STAGES, smoke: bool = False,
         l1_coef: float = 0.0, force: bool = False, anthropic_key: str = ""):
    key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("[note] no ANTHROPIC_API_KEY found — autointerp will cache "
              "contexts and skip LLM scoring")
    run_pipeline.remote(stages=stages, smoke=smoke, l1_coef=l1_coef,
                        force=force, anthropic_key=key)
