"""Diagnose Qwen-2.5-7B standard s42 group_dce.top50_full = NaN (audit item 2). Not in any protocol hash.

Repeats mechanism.run's whole-group ablation for the stored top-50 group (and the dense group as a control)
with the same dictionary, rank partition and Cache.run_group, in the registry dtype (fp16, as produced) and
in bfloat16. Records every value and whether it is finite.
"""
import json
import time

import numpy as np
import torch

from . import common, engine as E, mechanism, modelio as M, partitions as P
from .evaljob import fetch_prep, fetch_sae
from .common import log


def main(argv=None):
    model_name, cond, seed = "qwen-2.5-7b", "standard", 42
    out = common.RESULTS / "qwennan"
    out.mkdir(parents=True, exist_ok=True)
    spec = M.spec_of(model_name)
    prep, mean, pca, alive_npz = fetch_prep(model_name)
    sae_path, h = fetch_sae(model_name, cond, seed)
    mj = out / "mechanism_prod.json"
    common.s3_get("rx-wsx-%s-%s-s%d/results/wsx/%s/%s_s%d/mechanism.json" % (model_name, cond, seed, model_name, cond, seed), mj)
    groups = json.loads(mj.read_text())["groups"]
    res = dict(production_group_dce=json.loads(mj.read_text())["group_dce"], runs={})
    for dt in ("float16", "bfloat16"):
        t0 = time.time()
        model, tok = M.load(spec, device="cuda", dtype=getattr(torch, dt))
        sdev = M.site_device(model, spec)
        dic, _ = M.load_dictionary(sae_path, cond, mean, pca["v1"], sdev)
        suf = M.Suffix(model, spec)
        part = P.Partition("rank", tok)
        cache = E.Cache(model, spec, suf, dic, part)
        dirs = mechanism.directions(dic)
        r = dict(base_ce=cache.base_ce, n_targets=cache.n_targets)
        for g in ("top50", "dense"):
            for comp, W in dirs.items():
                v = cache.run_group(np.asarray(groups[g]), W) / cache.n_targets
                r["%s_%s" % (g, comp)] = float(v)
                r["%s_%s_finite" % (g, comp)] = bool(np.isfinite(v))
                log("qwennan", "%s %s_%s = %r" % (dt, g, comp, v))
        r["seconds"] = time.time() - t0
        res["runs"][dt] = r
        common.write_json(out / "qwen_group_nan_diagnosis.json", res)
        del model, cache, dic, suf
        torch.cuda.empty_cache()
    return dict(done=True)
