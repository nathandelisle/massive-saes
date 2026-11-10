"""Seed helper for a long train cell (2026-10-08 speed-up): train + rank ONE (condition, seed)
dictionary exactly as trainjob.cell does, into this box's own results prefix.

Logistics only. trainjob.py is NOT modified (it is part of the 'train' protocol hash): the prologue
below repeats trainjob.main's lines for the given cell, and the per-seed work is trainjob.cell itself
with trainjob.SEEDS narrowed to the one seed. The npz it writes is pinned under the same 'train'
protocol, so after rx_merge_seeds.py copies it (plus the SAE checkpoint and the PROVENANCE record)
into the main job's prefix, the relaunched main job reuses it unchanged through common.fresh.
The cell-level aggregate over a single seed is meaningless and is discarded here; the main job
recomputes the real cell over all five seeds.
"""
import argparse
import json

import torch

from . import common, trainjob
from .common import log


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["a4", "a2"], required=True)
    ap.add_argument("--arch", default="topk")
    ap.add_argument("--site", choices=["pre", "post"], default=None)
    ap.add_argument("--screen", type=int, default=0)
    ap.add_argument("--cond", choices=["standard", "bypass"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    assert a.seed in trainjob.SEEDS, a.seed
    v = trainjob._v3()
    spec = v.registry.MODELS[a.model]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # ---- verbatim from trainjob.main ----
    if a.kind == "a2":
        pf = common.WORK / "prep" / a.model / "prep.json"
        common.s3_get("%s/%s/prep/prep.json" % (common.SHARED, a.model), pf)
        prep = json.loads(pf.read_text())
        sites = prep["sites"]
        layer = int(sites["pre_rise_block"] if a.site == "pre" else sites["post_fall_block"])
        tag = "a2_%s_L%d" % (a.site, layer)
    else:
        layer, tag = None, "a4_%s" % a.arch
    out = common.RESULTS / "train" / a.model / tag
    out.mkdir(parents=True, exist_ok=True)
    big = spec.min_gpu_gb > 100
    adir = v.io.ensure_acts(spec, layer=layer, device=dev, device_map="auto" if big else None)
    torch.cuda.empty_cache()
    model, tok = trainjob.M.load(spec, device=dev, device_map="auto" if big else None,
                                 headroom=0.75 if big else 0.80)
    seqs, am = v.eval_sequences(tok, n_seqs=v.registry.EVAL_SEQS_IMPORTANCE)
    chunk = v.tuned_chunk(spec, vocab_size=int(model.config.vocab_size))
    arch = "topk" if a.kind == "a2" else a.arch
    cond = a.cond
    if arch == "relu_l1":
        evals, evecs, _ = v.estimate_pcs(adir)
        v1 = evecs[:, 0]
        coef, ok = trainjob.calibrate_relu(v, spec, adir, cond, v1, dev, out)
        akw = v.b1._akw("relu_l1", coef)
    else:
        akw, ok = dict(k=v.registry.TOPK_K), True
    # ---- one seed of trainjob.cell ----
    trainjob.SEEDS = [a.seed]
    got = trainjob.fetch_archived_saes(v, spec, arch, layer, akw)
    log("trainseed", "%s/%s/%s s%d: reused %d archived checkpoints"
        % (a.model, tag, cond, a.seed, len(got)))
    name = "%s_%s_s%d.npz" % (arch, cond, a.seed)
    try:
        trainjob.cell(v, model, spec, seqs, am, arch, cond, akw, layer, a.screen, chunk, out, dev)
    except Exception as e:                      # only the single-seed aggregate may fail here
        if not common.fresh(out, name, "train"):
            raise
        log("trainseed", "single-seed aggregate skipped (%r); per-seed npz is pinned" % (e,))
    if not common.fresh(out, name, "train"):
        raise RuntimeError("per-seed output %s missing after cell()" % name)
    sae_path = v.io.sae_cache_path(spec, a.seed, arch=arch, condition=cond, layer=layer, **akw)
    rec = common._prov(out)[name]
    common.write_json(out / ("seed_%s_s%d.json" % (cond, a.seed)),
                      dict(model=a.model, tag=tag, cond=cond, seed=a.seed, npz=name,
                           sae=sae_path.replace("\\", "/").rsplit("/", 1)[1],
                           protocol=rec["protocol"], layer=layer))
    return dict(tag=tag, cond=cond, seed=a.seed, npz=name, protocol=rec["protocol"])
