"""Workstream D job, per model: sparse probing for all five seeds of one condition at a time
(standard, then bypass), historical reproduction + corrected protocol + baselines.

Two stages so a GPU is rented only for the forward passes:
  --stage cache   (GPU box)  pooled features for every task/split/pooling, saved and uploaded to
                             s3://BUCKET/rx-shared/<model>/probe_cache/<cond>/
  --stage fit     (CPU box)  downloads that cache and fits every probe; no model is loaded
  --stage all                both on one box (local tests)
"""
import argparse

import torch

from . import common, modelio as M, probing
from .evaljob import fetch_prep, fetch_sae

SEEDS = [42, 137, 0, 1, 2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--conditions", nargs="+", default=["standard", "bypass"])
    ap.add_argument("--stage", choices=["cache", "fit", "all"], default="all")
    a = ap.parse_args(argv)
    spec = M.spec_of(a.model)
    prep, mean, pca, alive_npz = fetch_prep(a.model)
    model = tok = None
    if a.stage in ("cache", "all"):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, tok = M.load(spec, device=dev, device_map=a.device_map,
                            dtype=getattr(torch, a.dtype) if a.dtype else None)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        sdev = M.site_device(model, spec)
    done = {}
    for cond in a.conditions:
        out = common.RESULTS / "probe" / a.model / cond
        shared = "%s/%s/probe_cache/%s" % (common.SHARED, a.model, cond)
        common.write_json(out / ("meta_%s.json" % a.stage),
                          dict(model=a.model, condition=cond, seeds=SEEDS, stage=a.stage,
                               env=common.env_info()))
        names = ["%s_s%d" % (cond, s) for s in SEEDS]
        alive = {n: alive_npz["alive_" + n] for n in names}
        dics = {}
        if model is not None:
            for s in SEEDS:
                pth, h = fetch_sae(a.model, cond, s)
                dics["%s_s%d" % (cond, s)], _ = M.load_dictionary(pth, cond, mean, pca["v1"], sdev)
        if a.stage == "fit":
            common.s3_sync_down(shared, out / "cache")
        probing.run(model, tok, spec, dics, alive, out, conditions=[cond],
                    stage=("cache" if a.stage == "cache" else "all"), names=names,
                    v1=pca["v1"].astype("float64"), mode=cond)
        if a.stage == "cache":
            common.s3_sync_up(out / "cache", shared)
        done[cond] = a.stage
        del dics
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return done
