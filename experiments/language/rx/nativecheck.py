"""Native-margin check for the eligibility discrepancies (audit item 1B). CPU; not in any protocol hash.

Patching eligibility uses the NATIVE model only: donor margin > 0 and recipient margin < 0, computed in the
registry dtype on GPU. The eligible sets differ by GPU type (L4 / A10G / L40S). For every item that is eligible on
some boxes but not others ("flipped"), and for 300 items eligible everywhere ("common"), this recomputes the native
donor / recipient / foil margins in float32 on CPU, selecting test items exactly as patching.run does
(same task builders, same alignment filter, rng 20261005 permutations in the same split order).
Expected if the discrepancy is hardware numerics: flipped items sit at near-zero margins.
"""
import json

import numpy as np
import torch

from . import common, modelio as M, patching as PT
from .common import log


def main(argv=None):
    items = json.loads((common.PKG / "rx" / "native_items.json").read_text())
    out = common.RESULTS / "nativecheck"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for key, it in items.items():
        model_name, task = key.split("|")
        spec = M.spec_of(model_name)
        model, tok = M.load(spec, device="cpu", dtype=torch.float32)
        tasks = PT.build_tasks(tok)
        splits = {k: PT._aligned(tok, v) for k, v in tasks[task][0].items()}
        rng = np.random.default_rng(20261005)
        sel = {}
        for k, n in (("calibration", 512), ("train", 2048), ("validation", 1024), ("test", 4096)):
            rows = splits.get(k, [])
            ix = rng.permutation(len(rows))[:n]
            sel[k] = [rows[i] for i in ix]
        enc = PT.Encoded(tok, sel["test"])
        r = {}
        for grp in ("flipped", "common_sample"):
            idx = np.asarray(it[grp], int)
            nat = PT.native_margins(model, enc, idx)
            don, rec, foil = nat["don"][idx], nat["rec"][idx], nat["foil"][idx]
            dist = np.minimum(np.abs(don), np.abs(rec))       # distance of the eligibility decision from flipping
            r[grp] = dict(n=len(idx), items=idx.tolist(), don=don.tolist(), rec=rec.tolist(), foil=foil.tolist(),
                          eligible_fp32=((don > 0) & (rec < 0)).tolist(),
                          min_abs_margin_median=float(np.median(dist)),
                          min_abs_margin_q90=float(np.quantile(dist, 0.9)),
                          min_abs_margin_max=float(dist.max()))
        res[key] = r
        log("native", "%s flipped median min|margin| %.4f (q90 %.4f, max %.4f); common median %.4f" % (
            key, r["flipped"]["min_abs_margin_median"], r["flipped"]["min_abs_margin_q90"],
            r["flipped"]["min_abs_margin_max"], r["common_sample"]["min_abs_margin_median"]))
        common.write_json(out / "native_margin_check.json", res)
        del model
    return dict(done=len(res))
