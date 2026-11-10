"""Convention probes for checkpoints whose reconstruction looks wrong, before anything is reported.

Two questions, both about whether we are feeding the released SAE what it expects:

  hook offset  Is the release's "layer N" our HF block N? Evaluate the same SAE against cached activations
               from several neighbouring blocks. A release is only mis-mapped if a neighbour is dramatically
               better; a small difference is not evidence.
  input scale  Several releases (Llama Scope, Qwen Scope) normalise activations dataset-wise during training
               and record the factor. If the converter folded it into the weights, feeding raw activations is
               correct and rescaling should make things worse. If it did not, the released scale should be
               dramatically better. Only the factor the release documents (and its inverse) is tried --
               no fitted or searched scale, so this cannot tune FVU.

Reports total and content FVU for every (layer, scale) cell on a fixed slice of the evaluation cache. Diagnostic
only: nothing here feeds the audit tables.

usage: probe.py --group qwen3_8b --job-id qwen3_8b-xxxx --layers 5 6 7 8 9 --rows 20000
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np


def fvu(x, h, v):
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    mu = x.mean(0)
    xc, hc = x - mu, h - mu
    err = xc - hc
    tot = float((err ** 2).sum() / (xc ** 2).sum())
    ep = err - np.outer(err @ v, v)
    xp = xc - np.outer(xc @ v, v)
    con = float((ep ** 2).sum() / (xp ** 2).sum())
    w1 = float(1 - (xp ** 2).sum() / (xc ** 2).sum())
    return tot, con, w1


def main():
    import torch
    from adapters import Adapter, fit_axis_gpu
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="manifest.json")
    p.add_argument("--group", required=True)
    p.add_argument("--job-id")
    p.add_argument("--layers", nargs="+", type=int, required=True)
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--cache", default="probe_cache")
    p.add_argument("--out", default="results/probes")
    p.add_argument("--rows", type=int, default=20000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sqrt-d-scale", action="store_true", help="also try the sibling releases' documented constant")
    a = p.parse_args()
    m = json.loads(pathlib.Path(a.manifest).read_text())
    jobs = [j for j in m["jobs"] if j["group"] == a.group and (not a.job_id or j["job_id"] == a.job_id)]
    job = jobs[0]

    # one harvest pass over the probe layers, same conventions as the audit (context, bos, dtype, corpus)
    probe_manifest = dict(m, calibration_rows=a.rows, evaluation_rows=a.rows,
                          protocol=m["protocol"] + "_PROBE_NOT_FINAL",
                          jobs=[dict(job, layer=L, job_id=f"{job['job_id']}-probeL{L}") for L in a.layers])
    pm = pathlib.Path(a.cache) / "probe_manifest.json"
    pm.parent.mkdir(parents=True, exist_ok=True)
    pm.write_text(json.dumps(probe_manifest, indent=1))
    subprocess.run([sys.executable, "-u", "harvest.py", "--manifest", str(pm), "--group", a.group,
                    "--corpus", a.corpus, "--cache", a.cache, "--device", a.device], check=True)
    root = pathlib.Path(a.cache) / a.group / f"ctx{m['context_length']}"

    scales = {"as_released_raw": 1.0}
    reg = job.get("registry_entry") or {}
    nsf = reg.get("norm_scaling_factor")
    if nsf:
        scales["times_documented_norm_scaling_factor"] = float(nsf)
        scales["divided_by_documented_norm_scaling_factor"] = 1.0 / float(nsf)
    native = job.get("native_config_fields") or {}
    adapter = Adapter(job, a.device)
    # Sibling releases from the same lab family (Llama Scope, Gemma Scope) normalise activations by a single
    # constant so that the MEAN norm is sqrt(d_model) / unit mean squared norm. If a release documents no
    # normalisation but its converter folded none in either, feeding raw activations can be badly out of scale.
    # This tries that documented constant (computed from the calibration split, never fitted to the eval error).
    if a.sqrt_d_scale:
        xtr0 = np.load(root / "train" / f"layer_{job['layer']}.npy", mmap_mode="r")
        mean_norm = float(np.linalg.norm(np.asarray(xtr0[:20000]), axis=1).mean())
        scales["documented_sibling_convention_mean_norm_to_sqrt_d"] = float(np.sqrt(xtr0.shape[1]) / mean_norm)
        out_extra = dict(calibration_mean_norm=mean_norm, sqrt_d=float(np.sqrt(xtr0.shape[1])))
    else:
        out_extra = {}
    pos = np.load(root / "test" / "pos.npy")
    keep = pos > 0                                            # the population every release can be compared on
    out = dict(job_id=job["job_id"], model=job["model"], sae_id=job["sae_id"], declared_layer=job["layer"],
               probe_layers=a.layers, scales=scales, rows=int(keep.sum()),
               normalization_reported_by_loader=adapter.norm, cells=[], **out_extra)
    for L in a.layers:
        xte = np.load(root / "test" / f"layer_{L}.npy", mmap_mode="r")
        x = np.asarray(xte[: len(keep)])[keep]
        xtr = np.load(root / "train" / f"layer_{L}.npy", mmap_mode="r")
        origin, v, info = fit_axis_gpu(np.asarray(xtr), a.device)
        for name, s in scales.items():
            xs = (x * s).astype(np.float32)
            rec = np.concatenate([adapter.reconstruct(xs[i:i + 4096])["raw"][1] for i in range(0, len(xs), 4096)])
            tot, con, w1 = fvu(xs, rec, v)
            cell = dict(layer=L, scale=name, scale_value=s, fvu_total=tot, fvu_content=con,
                        content_over_total=(con / tot if tot > 0 else None), w1=w1, declared=bool(L == job["layer"]))
            out["cells"].append(cell)
            print(json.dumps(cell), flush=True)
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    best = min(out["cells"], key=lambda c: c["fvu_total"])
    declared = [c for c in out["cells"] if c["declared"] and c["scale"] == "as_released_raw"][0]
    out["verdict"] = dict(best=best, declared=declared,
                          ratio_declared_over_best=declared["fvu_total"] / max(best["fvu_total"], 1e-30),
                          note=("A neighbouring layer or a documented scale is dramatically better than the declared "
                                "convention only if this ratio is large; report it as a convention finding, never as a fit."))
    pathlib.Path(a.out, f"probe_{job['job_id']}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["verdict"], indent=1), flush=True)


if __name__ == "__main__":
    main()
