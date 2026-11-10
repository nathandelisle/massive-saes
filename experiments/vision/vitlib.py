"""Shared pieces for the corrected vision comparison (v2).

What changed from the archived `vit_sae_job.py`, and why:

1. PATCH EXTRACTION. The archive took `h[:, 1:]`, which drops CLS but keeps the four learned register
   tokens of `facebook/dinov2-with-registers-base` inside the "patch" population. Registers are exactly
   where that model is trained to park high-norm activations, so the archived with-registers geometry and
   SAE are contaminated. Here the token layout is read from the model config and verified per batch:
   patches are the last `grid*grid` positions, and every image must contribute exactly that many rows.
2. MATCHING. The archive's `reproducibility()` documented both directions but looped `i < j` and computed
   only max_j cos(D_i, D_j). Here every ordered pair is computed, over alive features and over all
   features, with the nearest-match distribution kept.
3. IMAGE SPLIT. The archive fit PCA/SAE and evaluated on the same activation pool (a random subset of the
   same images) and silently fell back to Food101. Here images are split before activations are taken, the
   dataset/split/indices/hashes are recorded, and evaluation uses held-out images only.

The SAE class and training loop are copied verbatim from the archive so the objective is unchanged:
BatchTopK (threshold from the flattened batch, ties kept), Adam, same init, same FVU denominator.
"""
import hashlib
import io
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ images
def build_image_manifest(dataset, config, split, n_images, seed=0, fallbacks=()):
    """Deterministic image list with content hashes.

    A fallback chain is allowed HERE ONLY, and the dataset actually used is recorded in the manifest; every
    other arm loads this manifest, so all arms are guaranteed to share one dataset and one image set. The
    archived pilot instead let each run fall back independently (it landed on Food101 without saying so).
    """
    from datasets import load_dataset
    tried, ds, used = [], None, None
    for nm, cf, sp in [(dataset, config, split), *fallbacks]:
        try:
            ds = load_dataset(nm, cf, split=sp) if cf else load_dataset(nm, split=sp)
            used = (nm, cf, sp)
            break
        except Exception as e:
            tried.append({"dataset": nm, "config": cf, "split": sp, "error": str(e)[:200]})
    if ds is None:
        raise RuntimeError(f"no usable dataset; tried {json.dumps(tried)}")
    dataset, config, split = used
    n = min(n_images, len(ds))
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(ds), size=n, replace=False).tolist())
    recs = []
    for i in idx:
        row = ds[int(i)]
        img = row.get("image") or row.get("img")
        im = img.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        recs.append(dict(index=int(i), sha256=hashlib.sha256(buf.getvalue()).hexdigest()[:32],
                         size=list(im.size)))
    return dict(dataset=dataset, config=config, split=split, n_images=len(recs),
                selection_seed=seed, fallbacks_tried=tried, images=recs)


def load_images(manifest):
    from datasets import load_dataset
    ds = (load_dataset(manifest["dataset"], manifest["config"], split=manifest["split"])
          if manifest["config"] else load_dataset(manifest["dataset"], split=manifest["split"]))
    out = []
    for r in manifest["images"]:
        row = ds[int(r["index"])]
        im = (row.get("image") or row.get("img")).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        got = hashlib.sha256(buf.getvalue()).hexdigest()[:32]
        if got != r["sha256"]:
            raise RuntimeError(f"image {r['index']} hash mismatch: {got} != {r['sha256']}")
        out.append(im)
    return out


def split_images(manifest, n_train):
    """Split by image, before any activation is taken."""
    tr = dict(manifest, images=manifest["images"][:n_train], role="train")
    te = dict(manifest, images=manifest["images"][n_train:], role="heldout")
    return tr, te


# ------------------------------------------------------------------ token layout
def token_layout(model, pixel_values):
    """(n_prefix, grid_tokens) for an HF DINOv2 model, read from config, then verified."""
    cfg = model.config
    n_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)
    patch = int(cfg.patch_size)
    h, w = pixel_values.shape[-2:]
    grid = (h // patch) * (w // patch)
    return 1 + n_reg, grid, n_reg          # CLS + registers, then patches


@torch.no_grad()
def collect_patch_activations(model, processor, images, layers, batch=16, extra_trailing=0,
                              forward_fn=None, log=print):
    """Residual-stream activations at `layers`, image-patch tokens only.

    extra_trailing: tokens appended at the END by an intervention (Jiang's test-time registers), excluded
    here and reported separately by the caller. Learned registers are excluded via the config layout.
    """
    out = {L: [] for L in layers}
    prefix_out = {L: [] for L in layers}
    trailing_out = {L: [] for L in layers}
    per_image_rows = []
    for i in range(0, len(images), batch):
        px = processor(images=images[i:i + batch], return_tensors="pt")["pixel_values"].to(DEV)
        hs = (forward_fn(px) if forward_fn is not None
              else model(pixel_values=px, output_hidden_states=True).hidden_states)
        n_prefix, grid, n_reg = token_layout(model, px)
        for L in layers:
            h = hs[L].float()
            total = h.shape[1]
            if total != n_prefix + grid + extra_trailing:
                raise RuntimeError(f"token count {total} != {n_prefix}+{grid}+{extra_trailing} at layer {L}")
            end = total - extra_trailing
            out[L].append(h[:, n_prefix:end].reshape(-1, h.shape[-1]).cpu())
            prefix_out[L].append(h[:, :n_prefix].reshape(-1, h.shape[-1]).cpu())
            if extra_trailing:
                trailing_out[L].append(h[:, end:].reshape(-1, h.shape[-1]).cpu())
        per_image_rows.append(grid)
        if i == 0:
            log(f"[tokens] prefix={n_prefix} (CLS+{n_reg} learned registers) patches={grid} "
                f"trailing={extra_trailing} total={total}")
    rows = {L: torch.cat(v) for L, v in out.items()}
    for L in layers:
        assert len(rows[L]) == len(images) * per_image_rows[0], "patch rows per image are not constant"
    return (rows, {L: torch.cat(v) for L, v in prefix_out.items()},
            {L: torch.cat(v) for L, v in trailing_out.items()} if extra_trailing else {},
            per_image_rows[0])


# ------------------------------------------------------------------ geometry
def geometry(A, v1=None):
    X = A.double()
    mu = X.mean(0)
    C = X - mu
    nrm = C.norm(dim=1)
    med = float(nrm.median())
    if v1 is None:
        # power iteration on the covariance, float64
        v = torch.randn(X.shape[1], generator=torch.Generator().manual_seed(1729), dtype=torch.float64)
        v /= v.norm()
        for _ in range(200):
            w = C.T @ (C @ v) / len(C)
            nw = w.norm()
            if float(nw) == 0:
                break
            w = w / nw
            if float((w - v).norm().min()) < 1e-9:
                v = w
                break
            v = w
        v1 = v
    p = C @ v1
    return dict(median_patch_norm=med, max_over_median=float(nrm.max() / max(med, 1e-9)),
                frac_norm_gt_3x=float((nrm > 3 * med).float().mean()),
                frac_norm_gt_10x=float((nrm > 10 * med).float().mean()),
                pc1_var_share=float((p * p).sum() / (C * C).sum()),
                outlier_pc1_energy_frac=(float((p[nrm > 3 * med] ** 2).sum() / max(float((p ** 2).sum()), 1e-30))
                                         if (nrm > 3 * med).any() else None),
                n_rows=int(len(A))), v1.float(), mu.float()


# ------------------------------------------------------------------ SAE (verbatim from the archive)
class SAE(torch.nn.Module):
    def __init__(self, d, m, mu, v=None, mean_init_bd=False):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("v", v if v is not None else torch.zeros(d))
        self.bypass = v is not None
        self.W_e = torch.nn.Parameter(torch.empty(d, m))
        self.b_e = torch.nn.Parameter(torch.zeros(m))
        self.W_d = torch.nn.Parameter(torch.empty(m, d))
        self.b_d = torch.nn.Parameter(torch.zeros(d))
        torch.nn.init.kaiming_uniform_(self.W_d, a=5 ** 0.5)
        with torch.no_grad():
            self.W_d.data = F.normalize(self.W_d.data, dim=1)
            self.W_e.data = self.W_d.data.t().contiguous().clone()
            if mean_init_bd:
                self.b_d.data = torch.zeros(d)

    def split(self, x):
        c = x - self.mu
        if self.bypass:
            par = (c @ self.v).unsqueeze(-1) * self.v
            return c - par, par
        return c, torch.zeros_like(c)

    def forward(self, x, k):
        c, par = self.split(x)
        pre = F.relu(c @ self.W_e + self.b_e)
        B, m = pre.shape
        flat = pre.reshape(-1)
        kk = min(k * B, flat.numel())
        thr = torch.topk(flat.detach(), kk, sorted=False).values.min()   # BatchTopK, ties kept
        z = pre * (pre >= thr)
        rec = z @ self.W_d + self.b_d
        xh = rec + par + self.mu
        return xh, z, float(thr)


def train_sae(A, mu, v, seed, cfg, tag, checkpoints=(3000,), log=print):
    """Verbatim archive trainer + intermediate checkpoints (state dicts) at `checkpoints`."""
    torch.manual_seed(seed)
    d = A.shape[-1]
    m = d * cfg["d_sae_mult"]
    sae = SAE(d, m, mu.to(DEV), None if v is None else v.to(DEV)).to(DEV)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg["lr"])
    g = torch.Generator().manual_seed(seed + 999)
    n = len(A)
    hist, snaps = [], {}
    for step in range(cfg["steps"]):
        idx = torch.randint(0, n, (cfg["sae_bs"],), generator=g)
        x = A[idx].to(DEV)
        xh, z, thr = sae(x, cfg["k"])
        loss = (xh - x).pow(2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == cfg["steps"] - 1:
            with torch.no_grad():
                fvu = float(loss / (x - sae.mu).pow(2).sum(-1).mean())
            hist.append({"step": step, "fvu": fvu})
            if step % 1000 == 0:
                log(f"  [sae {tag} s{seed}] {step}/{cfg['steps']} fvu={fvu:.4f}")
        if (step + 1) in checkpoints:
            snaps[step + 1] = {k: t.detach().clone() for k, t in sae.state_dict().items()}
    return sae, hist, snaps


@torch.no_grad()
def eval_sae(sae, A, cfg, v1=None):
    """Held-out FVU, content FVU against a fixed PC1 direction, achieved L0, alive count."""
    tot_e = tot_v = l0 = 0.0
    perp_e = perp_v = 0.0
    fires = None
    vv = None if v1 is None else v1.to(DEV).double()
    for i in range(0, len(A), cfg["sae_bs"]):
        x = A[i:i + cfg["sae_bs"]].to(DEV)
        xh, z, _ = sae(x, cfg["k"])
        e = (xh - x).double()
        c = (x - sae.mu).double()
        tot_e += float(e.pow(2).sum()); tot_v += float(c.pow(2).sum())
        if vv is not None:
            ep = e - (e @ vv).unsqueeze(-1) * vv
            cp = c - (c @ vv).unsqueeze(-1) * vv
            perp_e += float(ep.pow(2).sum()); perp_v += float(cp.pow(2).sum())
        l0 += float((z > 0).float().sum())
        f = (z > 0).float().sum(0)
        fires = f if fires is None else fires + f
    return dict(fvu=tot_e / max(tot_v, 1e-9),
                content_fvu=(perp_e / max(perp_v, 1e-9)) if vv is not None else None,
                l0=l0 / len(A), n_alive=int((fires > 0).sum()), n_features=int(fires.numel()),
                dead_frac=float((fires == 0).float().mean()), n_eval_rows=int(len(A)))


@torch.no_grad()
def matching(saes, alive_masks=None, thr=0.7):
    """Every ORDERED pair, both directions. Over alive features and over all features."""
    def pair(Di, Dj, mask):
        Dn, Dm = F.normalize(Di, dim=1), F.normalize(Dj, dim=1)
        best = []
        for c in range(0, len(Dn), 4096):
            best.append((Dn[c:c + 4096].to(DEV) @ Dm.to(DEV).t()).max(dim=1).values.cpu())
        b = torch.cat(best)
        if mask is not None:
            b = b[mask]
        q = torch.quantile(b, torch.tensor([.1, .25, .5, .75, .9]))
        return dict(n=int(b.numel()), median=float(b.median()), mean=float(b.mean()),
                    frac_gt_0p7=float((b > thr).float().mean()), frac_gt_0p5=float((b > 0.5).float().mean()),
                    quantiles={k: float(x) for k, x in zip(["p10", "p25", "p50", "p75", "p90"], q)},
                    hist=[int(x) for x in torch.histc(b, bins=20, min=-1, max=1)])
    out = {"alive": {}, "all": {}}
    for i in range(len(saes)):
        for j in range(len(saes)):
            if i == j:
                continue
            Di, Dj = saes[i].W_d.detach(), saes[j].W_d.detach()
            out["all"][f"{i}->{j}"] = pair(Di, Dj, None)
            if alive_masks is not None:
                out["alive"][f"{i}->{j}"] = pair(Di, Dj, alive_masks[i])
    for key in ("alive", "all"):
        vals = out[key]
        if vals:
            out[key + "_summary"] = dict(
                mean_median=float(np.mean([v["median"] for v in vals.values()])),
                mean_frac_gt_0p7=float(np.mean([v["frac_gt_0p7"] for v in vals.values()])),
                n_ordered_pairs=len(vals),
                note="ordered pairs from the same 3 seeds are not independent replicates")
    return out


def dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: (o.tolist() if hasattr(o, "tolist") else str(o)))
    print("  ->", path, flush=True)
