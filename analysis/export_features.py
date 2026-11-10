"""Export checkpoint decoders and importance scores for pairwise feature matching.

Uses cached files or an explicitly supplied S3_BUCKET. Large checkpoint files are not bundled."""
import hashlib
import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

B = os.environ.get("S3_BUCKET", "")
OUT = Path(__file__).resolve().parent / "features"
PKG = Path(__file__).resolve().parents[1] / "experiments" / "language"
GAPS = Path(__file__).resolve().parents[1] / "experiments" / "geometry"
CACHE = Path(os.environ.get("PC1_SAE_CACHE", str(OUT.parent / "_sae_cache")))
SEEDS = [42, 137, 0, 1, 2]
sys.path.insert(0, str(PKG / "pc1_v3"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def s3get(key, dst):
    dst = Path(dst)
    if not dst.exists():
        if not B:
            raise FileNotFoundError(f"Missing {dst}. Supply the cached file or set S3_BUCKET to its archive bucket.")
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["aws", "s3", "cp", "s3://%s/%s" % (B, key), str(dst), "--quiet"])
        if r.returncode != 0 or not dst.exists():
            return None
    return dst


def export_model(model, conds=("standard", "bypass")):
    from v3.common import registry
    spec = registry.MODELS[model]
    inv = json.loads((PKG / "inventory.json").read_text())
    prep = OUT / model / "prep"
    for f in ("mean.npy", "pca.npz", "alive.npz", "prep.json"):
        s3get("rx-shared/%s/prep/%s" % (model, f), prep / f)
    mean = np.load(prep / "mean.npy").astype(np.float64)
    pca = np.load(prep / "pca.npz")
    v1 = pca["v1"].astype(np.float64)
    v1 /= np.linalg.norm(v1)
    block = int(json.loads((prep / "prep.json").read_text())["site_block"])
    geo = OUT / model / "geometry.npz"
    np.savez(geo, mean=mean, v1=v1, evecs=pca["evecs"].astype(np.float64), block_index=np.int64(block),
             model_id=np.array(model))
    alive_npz = np.load(prep / "alive.npz")
    made = []
    for c in conds:
        for s in SEEDS:
            r = next((r for r in inv["rows"] if r["model"] == model and r["arch"] == "topk"
                      and r["condition"] == c and int(r["seed"]) == s), None)
            if r is None:
                continue
            src = r.get("rx_prefix") or r["prefixes"][0]
            pt = s3get("%s/saes/%s.pt" % (src, r["config_hash"]), CACHE / model / ("%s.pt" % r["config_hash"]))
            if pt is None:
                print("MISSING checkpoint", model, c, s)
                continue
            sd = torch.load(pt, map_location="cpu", weights_only=False)
            state = sd["state"] if "state" in sd else sd
            dec = state["W_dec"].float().numpy()
            if dec.shape[0] != spec.d_sae:            # orientation: rows must be features
                dec = dec.T
            alive = alive_npz["alive_%s_s%d" % (c, s)].astype(bool)
            zkey = "rx-imp-%s-%s-s%d/results/imp/%s/%s_s%d/rank_zero.npz" % (model, c, s, model, c, s)
            zf = s3get(zkey, OUT / model / ("rank_zero_%s_s%d.npz" % (c, s)))
            if zf is not None:
                z = np.load(zf)
                imp = np.asarray(z["score"], np.float64)
                measured = (np.asarray(z["status"]) != 0)
                imp = np.where(measured, imp, np.nan)
                proto = "rx_zero imp-m1; summed dCE/valid targets; rank partition; status!=NOT_EVALUATED"
            else:
                imp = np.full(spec.d_sae, np.nan)
                measured = np.zeros(spec.d_sae, bool)
                proto = "none (RX zero-ablation not available)"
            f = OUT / model / ("%s_s%d.npz" % (c, s))
            np.savez(f, decoder=dec, alive=alive, importance=imp, importance_measured=measured)
            meta = dict(cohort_id="v3core_rxzero", model_id=model, block_index=block, condition=c, seed=s,
                        checkpoint_sha256=sha(pt), checkpoint_s3="s3://%s/%s/saes/%s.pt" % (B, src, r["config_hash"]),
                        config_hash=r["config_hash"], mean_sha256=sha(prep / "mean.npy"),
                        pc1_sha256=sha(prep / "pca.npz"), geometry_sha256=sha(geo),
                        ranking_data_id="RX rank partition, partitions.json sha256 %s" % sha(PKG / "partitions.json"),
                        importance_protocol_id=proto,
                        alive_rule="v3 archived counter: fires on (rows over first 106,496 store rows)/100,000 > 1e-3",
                        n_importance_measured=int(measured.sum()), n_alive=int(alive.sum()),
                        rank_zero_s3=("s3://%s/%s" % (B, zkey)) if zf is not None else None)
            (OUT / model / ("%s_s%d.json" % (c, s))).write_text(json.dumps(meta, indent=1))
            made.append((c, s, zf is not None))
            pt.unlink()                                   # keep disk small: hash recorded above
    return made


def run_pairs(model, all_pairs=False):
    res = []
    d = OUT / model
    for c in ("standard", "bypass"):
        srcs = SEEDS if all_pairs else [42]
        for a in srcs:
            for b in SEEDS:
                if a == b:
                    continue
                fa, fb = d / ("%s_s%d.npz" % (c, a)), d / ("%s_s%d.npz" % (c, b))
                if not (fa.exists() and fb.exists()):
                    continue
                ma = json.loads((d / ("%s_s%d.json" % (c, a))).read_text())
                if ma["n_importance_measured"] == 0:
                    continue
                out = d / "pairs" / ("%s_s%d_vs_s%d.json" % (c, a, b))
                if out.exists():
                    try:                                  # a killed write leaves a zero-filled file
                        json.loads(out.read_text())
                        res.append(out); continue
                    except ValueError:
                        print("MALFORMED, recomputing", out, flush=True)
                        out.unlink()
                out.parent.mkdir(exist_ok=True)
                tmp = out.with_suffix(".json.tmp")         # atomic: write tmp, validate, rename
                r = subprocess.run([sys.executable, str(GAPS / "recalculate_features.py"), "--a", str(fa), "--b", str(fb),
                                    "--a-meta", str(d / ("%s_s%d.json" % (c, a))), "--b-meta", str(d / ("%s_s%d.json" % (c, b))),
                                    "--geometry", str(d / "geometry.npz"), "--out", str(tmp)],
                                   capture_output=True, text=True, cwd=str(GAPS))
                if r.returncode != 0:
                    print("FAIL", model, c, a, b, r.stderr[-800:], flush=True)
                    tmp.unlink(missing_ok=True)
                else:
                    json.loads(tmp.read_text())
                    tmp.replace(out)
                    res.append(out)
    return res


if __name__ == "__main__":
    models = sys.argv[1:] or ["gpt2", "pythia-70m-deduped", "pythia-160m", "pythia-1.4b", "pythia-6.9b",
                              "gemma-2-2b", "gemma-2-9b", "llama-3.1-8b", "qwen-2.5-7b"]
    for m in models:
        d = OUT / m
        if all((d / ("%s_s%d.json" % (c, s))).exists() for c in ("standard", "bypass") for s in SEEDS):
            made = [(None, None, True)] * 10           # already exported: only (re)run missing pairs
        else:
            made = export_model(m)
        print(m, "exported", len(made), "dictionaries; with RX zero:", sum(x[2] for x in made), flush=True)
        r = run_pairs(m, all_pairs=True)
        print(m, "pairs", len(r), flush=True)
