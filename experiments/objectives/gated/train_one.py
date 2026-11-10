"""Train one Gated SAE with the ORIGINAL trainer (io.get_sae path, same cache key) but save atomically, so a
sibling process polling for the checkpoint never sees a partial file. usage: train_one.py <model> <arch> <seed> <l1_coef>"""
import os
import sys

import torch

import gi_common as G
from gi_common import io, registry, S
from v3.common.data import activation_batches

model, arch, seed, coef = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
spec = registry.MODELS[model]
final = io.sae_cache_path(spec, seed, arch=arch, condition="standard", l1_coef=coef)
if os.path.exists(final):
    print("already present", final)
    sys.exit(0)
adir = io.ensure_acts(spec, device="cuda")
batches = activation_batches(adir, registry.BATCH_SIZE, registry.STEPS, seed, device="cuda")
sae, log = S.train_sae(batches, spec.d_model, spec.d_sae, arch=arch, seed=seed, device="cuda",
                       tag=f"{spec.key}/{arch}/standard/s{seed}", total_steps=registry.STEPS, l1_coef=coef)
tmp = final + ".tmp"
S.save_sae(sae, tmp, extra=dict(cfg=dict(model=spec.key, seed=seed, arch=arch, condition="standard"), log=log[-3:]))
os.replace(tmp, final)
print("saved", final, flush=True)
