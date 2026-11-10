"""Per-atom cross-seed sharing for the seed-42 source dictionary, precomputed so rx_analysis.py can
run where the 11 GB of exported decoders are not present (e.g. on the AWS pack box).

shared[j] = fraction of the target seeds (137/0/1/2) containing an atom with cosine >= 0.7 to source
atom j — exactly rx_analysis.importance_table's definition (float64, per-row unit norm), computed
in row/column chunks so peak memory stays well under 1 GB for the 16k x 4096 dictionaries.
Writes features/<model>/shared_<cond>_s42.npz (shared, alive, n_targets).
"""
import sys
from pathlib import Path

import numpy as np

FEAT = Path(__file__).resolve().parent / "features"
SEEDS = [42, 137, 0, 1, 2]
THR = 0.7


def unit64(d):
    d = d.astype(np.float64)
    return d / np.linalg.norm(d, axis=1, keepdims=True).clip(1e-12)


def maxcos_chunked(src32, tgt32, rb=1024, cb=4096):
    out = np.full(len(src32), -np.inf)
    for j in range(0, len(tgt32), cb):
        t = unit64(tgt32[j:j + cb])
        for i in range(0, len(src32), rb):
            out[i:i + rb] = np.maximum(out[i:i + rb], (unit64(src32[i:i + rb]) @ t.T).max(1))
        del t
    return out


def main(models):
    for m in models:
        for c in ("standard", "bypass"):
            src = FEAT / m / ("%s_s42.npz" % c)
            dst = FEAT / m / ("shared_%s_s42.npz" % c)
            if not src.exists() or dst.exists():
                continue
            A = np.load(src)
            da, alive = A["decoder"], A["alive"].astype(bool)
            hits = []
            for s in SEEDS[1:]:
                f = FEAT / m / ("%s_s%d.npz" % (c, s))
                if f.exists():
                    hits.append(maxcos_chunked(da, np.load(f)["decoder"]) >= THR)
            shared = np.mean(hits, axis=0)
            np.savez(dst, shared=shared, alive=alive, n_targets=len(hits))
            print(m, c, "targets", len(hits), "base", round(float(shared[alive].mean()), 4), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or [p.name for p in sorted(FEAT.iterdir()) if p.is_dir()])
