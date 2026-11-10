"""Cache layout and the train-if-missing entry point.

Everything lands under $PC1_V3_ROOT (default ./workdir):
  workdir/acts/{model}_{corpus}[_bosfree][_L{layer}]/     activation memmaps
  workdir/saes/{config_hash}.pt                           trained SAEs
  workdir/results/{exp_id}/*.json                         per-experiment outputs

The config hash covers everything that defines a training run, so exp_a3
(seed completion), exp_a5 (metric battery), exp_b1/b2 (architecture law) and
exp_c2 (FVU decomposition) all share SAEs instead of retraining. Sync the
workdir to S3 between instances: `aws s3 sync workdir s3://<bucket>/pc1-v3/`.
"""

import hashlib
import json
import os

from . import registry
from .data import harvest_activations, activation_batches
from .pca import estimate_pcs, project_out
from .saes import train_sae, save_sae, load_sae

ROOT = os.environ.get("PC1_V3_ROOT", os.path.abspath("workdir"))


def path(*parts):
    p = os.path.join(ROOT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def act_dir(spec, corpus="wikitext", bos_free=False, layer=None):
    tag = f"{spec.key}_{corpus}"
    if bos_free:
        tag += "_bosfree"
    # `layer is not None`, NOT truthiness: layer 0 is a legitimate target and
    # `if layer` would silently route it to the default-layer store
    if layer is not None and layer != spec.target_layer:
        tag += f"_L{layer}"
    return os.path.join(ROOT, "acts", tag)


def config_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def ensure_acts(spec, corpus="wikitext", bos_free=False, layer=None,
                n_tokens=registry.TRAIN_TOKENS, device="cuda", device_map=None):
    d = act_dir(spec, corpus, bos_free, layer)
    harvest_activations(spec, d, corpus=corpus, n_tokens=n_tokens,
                        device=device, exclude_first_position=bos_free,
                        device_map=device_map, layer=layer)
    return d


class _BypassBatches:
    """Wrap centered batches, projecting out PC1 on the fly (bypass condition)."""

    def __init__(self, batches, v1, device):
        import torch
        self.batches = batches
        self.v1 = torch.tensor(v1, dtype=torch.float32, device=device)

    def __iter__(self):
        for x in self.batches:
            coef = x @ self.v1
            yield x - coef[:, None] * self.v1[None, :]


def get_sae(spec, seed, arch="topk", condition="standard", corpus="wikitext",
            bos_free=False, layer=None, steps=registry.STEPS,
            batch_size=registry.BATCH_SIZE, device="cuda", retrain=False,
            device_map=None, **arch_kwargs):
    """Train-if-missing. condition in {standard, bypass}. Returns (sae, v1, mean,
    act_dir, meta).

    `device_map` is forwarded to the harvest. Without it a model too large for
    one card is loaded with `.to(device)` inside ensure_acts and OOMs -- and it
    does so while the caller's own copy of the same model is still resident, so
    the failure is a 2x-model allocation, not a 1x one. Only matters when the
    activation store is missing; a cache hit never loads a model at all.
    """
    import numpy as np
    adir = ensure_acts(spec, corpus=corpus, bos_free=bos_free, layer=layer,
                       device=device, device_map=device_map)
    evals, evecs, mean = estimate_pcs(adir)
    v1 = evecs[:, 0]
    # single source of truth for the cache key -- see sae_cache_path()
    sae_path = sae_cache_path(spec, seed, arch=arch, condition=condition,
                              corpus=corpus, bos_free=bos_free, layer=layer,
                              steps=steps, batch_size=batch_size, **arch_kwargs)
    os.makedirs(os.path.dirname(sae_path), exist_ok=True)
    cfg = dict(model=spec.key, seed=seed, arch=arch, condition=condition)
    if os.path.exists(sae_path) and not retrain:
        sae, extra = load_sae(sae_path, device=device)
        return sae, v1, mean, adir, extra
    batches = activation_batches(adir, batch_size, steps, seed, device=device)
    if condition == "bypass":
        batches = _BypassBatches(batches, v1, device)
    sae, log = train_sae(batches, spec.d_model, spec.d_sae, arch=arch,
                         seed=seed, device=device,
                         tag=f"{spec.key}/{arch}/{condition}/s{seed}",
                         total_steps=steps,
                         **arch_kwargs)
    extra = dict(cfg=cfg, log=log[-3:])
    save_sae(sae, sae_path, extra=extra)
    return sae, v1, mean, adir, extra


def sae_cache_path(spec, seed, arch="topk", condition="standard",
                   corpus="wikitext", bos_free=False, layer=None,
                   steps=registry.STEPS, batch_size=registry.BATCH_SIZE,
                   **arch_kwargs):
    """Where get_sae would store this exact config. Kept in lockstep with
    get_sae's cfg construction -- if that changes, change this too."""
    from .saes import ARCHS
    cfg = dict(model=spec.key, seed=seed, arch=arch, condition=condition,
               corpus=corpus, bos_free=bos_free,
               layer=layer if layer is not None else spec.target_layer, steps=steps,
               batch=batch_size, d_sae=spec.d_sae,
               **{k: (float(v) if isinstance(v, (int, float)) else v)
                  for k, v in arch_kwargs.items()})
    if hasattr(ARCHS.get(arch), "recalibrate"):
        cfg["jr_sched"] = "recal100_full_v2"
    return os.path.join(ROOT, "saes", f"{config_hash(cfg)}.pt")


def load_cached_sae(spec, seed, device="cpu", **kw):
    """Load a trained SAE WITHOUT touching the activation store.

    `get_sae` calls ensure_acts() before it checks the SAE cache, so on a fresh
    box an analysis-only script that just wants decoders triggers a full 5M-token
    harvest -- which is what timed out exp_a6 after 60 minutes even though all 38
    checkpoints were sitting in S3. Returns None if not cached.
    """
    p = sae_cache_path(spec, seed, **kw)
    if not os.path.exists(p):
        return None
    sae, _extra = load_sae(p, device=device)
    return sae


def alive(sae, adir, condition="standard", v1=None, device="cuda",
          n_sample=registry.ALIVE_SAMPLE, batch=8192):
    """Firing rates + alive mask (rate > 0.1%, v2 definition). For bypass SAEs
    the inputs are PC1-projected first, matching what the SAE saw in training."""
    import numpy as np
    import torch
    from .data import open_activations
    acts, mean = open_activations(adir)
    n = min(n_sample, acts.shape[0])
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    v1_t = (torch.tensor(v1, dtype=torch.float32, device=device)
            if condition == "bypass" else None)
    counts = torch.zeros(sae.d_sae, device=device)
    with torch.no_grad():
        for i in range(0, n, batch):
            x = torch.tensor(np.asarray(acts[i:i + batch], dtype=np.float32),
                             device=device) - mean_t
            if v1_t is not None:
                x = x - (x @ v1_t)[:, None] * v1_t[None, :]
            z = sae.encode(x)
            counts += (z != 0).float().sum(0)
    rates = (counts / n).cpu().numpy()
    return rates, rates > registry.ALIVE_FIRING_RATE


def write_json(exp_id, name, obj):
    p = path("results", exp_id, f"{name}.json")
    json.dump(obj, open(p, "w"), indent=2, default=float)
    print(f"  -> {p}")
    return p


def read_json(exp_id, name):
    p = os.path.join(ROOT, "results", exp_id, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None
