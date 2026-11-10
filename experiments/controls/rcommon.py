"""Shared helpers for the 22 Sep reviewer-response campaign.

Nothing in `v3/common` or `pc1night` is edited; everything here is a wrapper that
imports the archived implementation read-only, so a cell measured by this campaign
stays comparable with the archived cells.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

V3 = os.environ.get("PC1_V3", str(pathlib.Path(__file__).resolve().parents[1] / "language" / "pc1_v3"))
BUCKET = os.environ.get("S3_BUCKET", "")
HERE = pathlib.Path(__file__).resolve().parent


def log(tag, *a):
    print("[%s %s]" % (tag, time.strftime("%H:%M:%S")), *a, flush=True)


def add_v3():
    if V3 not in sys.path:
        sys.path.insert(0, V3)
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))


def config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def paper_cfg(spec, seed, condition, k=64.0, arch="topk"):
    """The exact cfg dict whose sha1 keys the archived SAE cache.

    `k` is a FLOAT in the stored keys because every campaign passed `k=64` through
    `**arch_kwargs` where the dataclass default is a float. Passing an int here
    produces `"k": 64` and a different hash that matches nothing.
    """
    from v3.common import registry
    return dict(model=spec.key, seed=seed, arch=arch, condition=condition,
                corpus="wikitext", bos_free=False, layer=spec.target_layer,
                steps=registry.STEPS, batch=registry.BATCH_SIZE,
                d_sae=spec.d_sae, k=float(k))


def fetch_sae(h, dest, index=None):
    """Pull workdir/saes/{hash}.pt from whichever archived prefix holds it."""
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    idx = json.loads(pathlib.Path(index or os.environ.get(
        "SAE_INDEX", str(HERE / "inventory.json"))).read_text())
    row = next((r for r in idx["rows"] if r["config_hash"] == h and r["present"]), None)
    if row is None:
        raise RuntimeError("SAE %s is not present in S3 per the inventory" % h)
    if not BUCKET:
        raise FileNotFoundError("Supply the cached checkpoint, or set S3_BUCKET to its archive bucket.")
    key = "s3://%s/%s/saes/%s.pt" % (BUCKET, row["prefixes"][0], h)
    subprocess.run(["aws", "s3", "cp", key, str(dest), "--only-show-errors"],
                   check=True, timeout=3600)
    return dest


def load_pool_npz(model, condition, seed, pooldir=None):
    """The archived per-(model, condition, seed) ranking from the causal-v2 campaign.

    Carries ranked_ids, delta_ce, ez2, rate, alive, pc1_cos2, screened. Returns None
    if this cell was never ranked (llama-3.1-8b), so the caller can rank it itself.
    """
    p = pathlib.Path(pooldir or (HERE / "pools"))
    f = p / ("rank_%s_%s_s%d.npz" % (model, condition, seed))
    return np.load(f, allow_pickle=True) if f.exists() else None


# ---------------------------------------------------------------------------
# position strata
# ---------------------------------------------------------------------------

def position_strata(seqs, am, tok, acts_norm=None, massive_thresh=None):
    """Boolean masks over [N, T] positions.

    `first`      position 0 of every sequence (BOS where the tokenizer adds one)
    `special`    any other special-token id
    `massive`    row L2 norm above a threshold FIT ON TRAINING ROWS and frozen
    `ordinary`   everything else that is a real (non-padded) token

    The massive threshold is passed in, never fitted here on evaluation rows: fitting
    it on the evaluation median is the legacy rule and is reported separately.
    """
    N, T = seqs.shape
    real = np.ones((N, T), bool) if am is None else (np.asarray(am) != 0)
    first = np.zeros((N, T), bool)
    first[:, 0] = True
    sp_ids = set(int(i) for i in (tok.all_special_ids or []))
    special = np.isin(np.asarray(seqs), list(sp_ids)) if sp_ids else np.zeros((N, T), bool)
    special = special & ~first
    massive = np.zeros((N, T), bool)
    if acts_norm is not None and massive_thresh is not None:
        massive = np.asarray(acts_norm) > float(massive_thresh)
    ordinary = real & ~first & ~special & ~massive
    return dict(all=real, first=first & real, special=special & real,
                massive=massive & real, ordinary=ordinary)


def alignment_bands(pc1_cos2, alive, ranked_ids, n=50, seed=0,
                    low_hi=0.001, mid_lo=0.01, mid_hi=0.1):
    """Low- and intermediate-alignment strata, selected WITHOUT any test outcome.

    Bands are on cos^2(decoder_j, v1). `low` is drawn from alive features below
    `low_hi`; `mid` from the band [mid_lo, mid_hi). Both exclude the frozen top-50,
    and both are drawn by a fixed RNG, not by importance, so they are strata and not
    a second ranking.
    """
    rng = np.random.default_rng(seed)
    top = set(int(x) for x in ranked_ids[:50])
    c2 = np.asarray(pc1_cos2, float)
    al = np.asarray(alive, bool)
    out = {}
    for name, lo, hi in (("low_align", -1.0, low_hi), ("mid_align", mid_lo, mid_hi)):
        pool = [int(j) for j in np.where(al & (c2 >= lo) & (c2 < hi))[0] if int(j) not in top]
        if len(pool) > n:
            pool = [int(x) for x in rng.choice(pool, size=n, replace=False)]
        out[name] = sorted(pool)
    return out


def write_json(path, obj):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(p)
    return p
