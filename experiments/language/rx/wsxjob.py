"""Per-dictionary workstream job (model, condition, seed): C contexts, A4 mechanism, B3 patching.

Needs the dictionary's importance job (for the independent zero-ablation ranking that selects
the interpretation features, the top-50 mechanism set, and the generic patching pool).
Each stage writes its own output and is skipped on resume.
"""
import argparse
import json
import time

import numpy as np
import torch

from . import common, contexts, engine as E, mechanism, modelio as M, partitions as P, patching
from .common import log
from .evaljob import fetch_prep, fetch_sae


def fetch_zero_score(model, cond, seed):
    key = "rx-imp-%s-%s-s%d/results/imp/%s/%s_s%d/rank_zero.npz" % (model, cond, seed, model, cond, seed)
    f = common.WORK / "imp" / ("%s_%s_s%d_rank_zero.npz" % (model, cond, seed))
    common.s3_get(key, f)
    z = np.load(f)
    return np.asarray(z["score"], np.float64), z


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--max-seqs", type=int, default=None, help="local smoke only")
    ap.add_argument("--stages", nargs="+", default=["contexts", "mechanism", "patching"])
    a = ap.parse_args(argv)
    t0 = time.time()
    spec = M.spec_of(a.model)
    out = common.RESULTS / "wsx" / a.model / ("%s_s%d" % (a.condition, a.seed))
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prep, mean, pca, alive_npz = fetch_prep(a.model)
    v1 = pca["v1"]
    alive = alive_npz["alive_%s_s%d" % (a.condition, a.seed)]
    zero_score, _ = fetch_zero_score(a.model, a.condition, a.seed)
    sae_path, h = fetch_sae(a.model, a.condition, a.seed)
    model, tok = M.load(spec, device=dev, device_map=a.device_map,
                        dtype=getattr(torch, a.dtype) if a.dtype else None)
    sdev = M.site_device(model, spec)
    dic, _ = M.load_dictionary(sae_path, a.condition, mean, v1, sdev)
    common.write_json(out / "meta.json", dict(model=a.model, condition=a.condition, seed=a.seed,
                                              config_hash=h, env=common.env_info()))
    done = {}
    if "contexts" in a.stages:
        feats, member, top, rnd = contexts.select_features(alive, zero_score, a.model,
                                                           a.condition, a.seed)
        ctx = contexts.build(model, tok, spec, dic, feats, member, out / "contexts",
                             a.condition, a.seed, max_seqs=a.max_seqs)
        common.s3_put(out / "contexts" / "contexts.json", "%s/%s/contexts/%s_s%d.json"
                      % (common.SHARED, a.model, a.condition, a.seed))
        done["contexts"] = len(ctx["features"])
    if "mechanism" in a.stages and not common.fresh(out, "mechanism.json", "wsx"):
        suf = M.Suffix(model, spec)
        cal = P.Partition("cal", tok, max_seqs=a.max_seqs)
        cs = E.CalStats(model, spec, dic, cal)
        part = P.Partition("rank", tok, max_seqs=a.max_seqs)
        cache = E.Cache(model, spec, suf, dic, part, keep_err=True)
        mechanism.run(model, spec, dic, cache, cache.E, alive, zero_score, cs, out)
        common.pin(out, "mechanism.json", "wsx")
        del cache
        done["mechanism"] = True
    if "patching" in a.stages:
        Wn = dic.W_dec / dic.W_dec.norm(dim=1, keepdim=True)
        cos2 = (Wn @ dic.v1).pow(2).cpu().numpy()
        patching.run(model, tok, spec, dic, alive, zero_score, cos2, a.condition,
                     out / "patching", rng_seed=a.seed)
        done["patching"] = True
    return dict(stages=done, hours=(time.time() - t0) / 3600)
