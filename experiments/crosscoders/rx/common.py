"""Paths, S3, logging, atomic writes, completion markers, provenance."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
V3 = PKG / "pc1_v3"
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

BUCKET = os.environ.get("S3_BUCKET", "")
WORK = pathlib.Path(os.environ.get("PC1_V3_ROOT", str(PKG / "workdir")))
RESULTS = WORK / "results"
SAES = WORK / "saes"
SHARED = "rx-shared"           # cross-job artifacts: s3://BUCKET/rx-shared/<model>/...
LOCAL = os.environ.get("RX_LOCAL", "1") == "1"   # local CPU smoke: never touch S3


def log(tag, *a):
    print("[%s %s]" % (tag, time.strftime("%H:%M:%S")), *a, flush=True)


def write_json(path, obj):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_default))
    tmp.replace(p)
    return p


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_npz(path, **arrays):
    import numpy as np
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(p)
    return p


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_hash():
    """Hash of every .py in this package and the vendored v3 code: goes into every result."""
    h = hashlib.sha256()
    for p in sorted(list(HERE.rglob("*.py")) + list((V3 / "v3").rglob("*.py"))):
        h.update(p.relative_to(PKG).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def env_info():
    import platform
    out = dict(python=platform.python_version(), host=platform.node(), code_hash=code_hash())
    try:
        import torch, transformers, numpy, datasets
        out.update(torch=torch.__version__, transformers=transformers.__version__,
                   numpy=numpy.__version__, datasets=datasets.__version__,
                   cuda=torch.version.cuda,
                   gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                   n_gpu=torch.cuda.device_count())
    except Exception as e:
        out["env_error"] = repr(e)
    return out


# ---------------------------------------------------------------------------------------
# Optional S3 transfers.
# ---------------------------------------------------------------------------------------
def _aws(args, timeout=3600, check=True):
    return subprocess.run(["aws"] + args, capture_output=True, text=True, timeout=timeout,
                          check=check)


def s3_get(key, dest, required=True):
    dest = pathlib.Path(dest)
    if dest.exists():
        return dest
    if LOCAL:
        if required:
            raise FileNotFoundError("local mode: %s not present at %s" % (key, dest))
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = _aws(["s3", "cp", "s3://%s/%s" % (BUCKET, key), str(dest), "--only-show-errors"],
             check=False)
    if r.returncode != 0:
        if required:
            raise RuntimeError("s3 get failed %s: %s" % (key, r.stderr[-400:]))
        return None
    return dest


def s3_put(src, key):
    if LOCAL:
        return
    _aws(["s3", "cp", str(src), "s3://%s/%s" % (BUCKET, key), "--only-show-errors"])


def s3_sync_up(src_dir, prefix):
    if LOCAL:
        return
    _aws(["s3", "sync", str(src_dir), "s3://%s/%s" % (BUCKET, prefix), "--only-show-errors"],
         check=False)


def s3_sync_down(prefix, dest_dir):
    if LOCAL:
        return
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)
    _aws(["s3", "sync", "s3://%s/%s" % (BUCKET, prefix), str(dest_dir), "--only-show-errors"],
         check=False)


def s3_exists(key):
    if LOCAL:
        return False
    r = _aws(["s3", "ls", "s3://%s/%s" % (BUCKET, key)], check=False)
    return r.returncode == 0 and bool(r.stdout.strip())


def mark_done(job, payload=None):
    """Completion marker the supervisor reads. Written LAST, after outputs are synced."""
    p = RESULTS / "DONE.json"
    kinds = {k: protocol_hash(k) for k in PROTOCOL_FILES}
    write_json(p, dict(job=job, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       env=env_info(), protocol_hashes=kinds, **(payload or {})))
    s3_put(p, "rx-status/%s/DONE.json" % job)


def mark_failed(job, reason):
    p = RESULTS / "FAILED.json"
    write_json(p, dict(job=job, reason=str(reason)[-4000:],
                       utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    s3_put(p, "rx-status/%s/FAILED.json" % job)


def cell_done(path):
    """Per-cell resume marker inside a job: a cell whose output exists is skipped."""
    return pathlib.Path(path).exists()


# ---------------------------------------------------------------------------------------
# provenance pinning
# ---------------------------------------------------------------------------------------
# Which source files define each job's measurements. An output is reused on resume only when
# the protocol hash it was written under equals the current one; otherwise it is recomputed.
# Outputs written before pinning existed carry no hash: they are kept, and listed as
# "unpinned" in PROVENANCE.json so analysis can treat them separately.
PROTOCOL_FILES = {
    # measurement code + an explicit semantic version of evaljob's measurement logic (plan rule,
    # universes, stage definitions): editing gates or docs in evaljob must not discard results
    "imp": ["rx/engine.py", "rx/modelio.py", "rx/partitions.py", "partitions.json", "inventory.json",
            "@IMP_MEASUREMENT_VERSION"],
    # correctness gates: any change to evaljob/engine/modelio re-runs them (cheap)
    "gates": ["rx/evaljob.py", "rx/engine.py", "rx/modelio.py"],
    "wsx": ["rx/engine.py", "rx/modelio.py", "rx/partitions.py", "partitions.json", "rx/contexts.py",
            "rx/mechanism.py", "rx/patching.py", "rx/tasks_sva.py", "rx/tasks_ioi.py", "rx/wsxjob.py"],
    "aux": ["rx/auxjob.py", "rx/modelio.py", "rx/partitions.py", "partitions.json", "rx/prep.py"],
    "probe": ["rx/probing.py", "rx/probejob.py", "rx/modelio.py"],
    "train": ["rx/trainjob.py", "pc1_v3/v3/common/saes.py", "pc1_v3/v3/common/io_utils.py",
              "pc1_v3/v3/common/data.py", "pc1_v3/v3/common/fastabl.py",
              "pc1_v3/scripts/exp_b1_architecture_law.py"],
    "judge": ["rx/judge.py"],
}


IMP_MEASUREMENT_VERSION = "imp-m1"   # bump when evaljob's measurement semantics change

# Compatible hashes for earlier versions of the same measurements.
COMPATIBLE = {"imp": {"dc0e4b1a87cef50f": "pre-split hash; gate-only edits since",
                      "bfdb828aa4f86e50": "pre-sharding hash; 2026-10-08 sharding change is logistics only "
                                          "(chunk fingerprints + shared S3 chunks), numerics unchanged"}}


def protocol_hash(kind):
    h = hashlib.sha256()
    for rel in PROTOCOL_FILES[kind]:
        h.update(rel.encode())
        h.update(IMP_MEASUREMENT_VERSION.encode() if rel.startswith("@") else (PKG / rel).read_bytes())
    return h.hexdigest()[:16]


def _prov_path(out):
    return pathlib.Path(out) / "PROVENANCE.json"


def _prov(out):
    p = _prov_path(out)
    return json.loads(p.read_text()) if p.exists() else {}


def fresh(out, name, kind):
    """True iff <out>/<name> exists and may be reused under the current protocol. A stale file
    (written under a different protocol hash) is deleted so the caller recomputes it."""
    f = pathlib.Path(out) / name
    if not f.exists():
        return False
    rec = _prov(out).get(name)
    cur = protocol_hash(kind)
    if rec is None and kind == "gates":
        log("prov", "%s has no provenance record: re-running gates" % name)
        f.unlink()
        return False
    if rec is None:
        pin(out, name, kind, unpinned=True)
        return True
    if rec.get("protocol") == cur or (rec.get("unpinned") and kind != "gates"):
        return True
    if rec.get("protocol") in COMPATIBLE.get(kind, {}):
        return True
    log("prov", "%s written under protocol %s != %s: recomputing" % (name, rec.get("protocol"), cur))
    if f.is_dir():
        import shutil
        shutil.rmtree(f)
    else:
        f.unlink()
    return False


def pin(out, name, kind, unpinned=False):
    d = _prov(out)
    d[name] = dict(kind=kind, protocol=None if unpinned else protocol_hash(kind),
                   code=code_hash(), unpinned=bool(unpinned),
                   utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json(_prov_path(out), d)
