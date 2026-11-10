"""Immutable protocols, atomic units, explicit coverage, and stable randomness."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def key(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def rng_for(*parts):
    return np.random.default_rng(int(key(parts)[:16], 16))


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_json(path):
    return json.loads(Path(path).read_text())


def seal(out, config):
    """Moving the same package is allowed; changed code, inputs or settings are not."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    code = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    files = {k: digest(config[k]) for k in ("mean_path", "pc1_path")}
    files.update({"tokens/" + k: digest(v) for k, v in config["tokens"].items()})
    for r in config["runs"]:
        files["checkpoint/" + r["name"]] = digest(r["checkpoint"])
        if r.get("frozen_ids_path"):
            files["frozen_ids/" + r["name"]] = digest(r["frozen_ids_path"])
        if r.get("checkpoint_sha256") and files["checkpoint/" + r["name"]] != r["checkpoint_sha256"]:
            raise ValueError("Checkpoint does not match grid evidence: " + r["name"])
    if config.get("v3_path"):
        files["v3/saes.py"] = digest(Path(config["v3_path"]) / "v3/common/saes.py")
    portable = json.loads(json.dumps(config))
    for k in ("out", "mean_path", "pc1_path", "v3_path"): portable.pop(k, None)
    portable["tokens"] = sorted(config["tokens"])
    for r in portable["runs"]:
        r.pop("checkpoint", None); r.pop("frozen_ids_path", None)
    manifest = dict(schema=1, code=code, inputs=files, settings=portable)
    protocol_id = key(manifest); target = out / "protocol.json"
    if target.exists():
        if load_json(target)["protocol_id"] != protocol_id:
            raise ValueError("Protocol changed: use a NEW output directory; never mix results.")
    else:
        atomic_json(target, dict(protocol_id=protocol_id, **manifest))
    return protocol_id


def completed(path, protocol_id):
    path = Path(path)
    if not path.exists(): return False
    with np.load(path, allow_pickle=False) as n:
        if str(n["protocol_id"]) != protocol_id:
            raise ValueError("Stale unit: " + str(path))
        # Force zip CRC checks, not just a successful metadata read.
        for name in n.files: np.asarray(n[name])
    return True


def assigned(unit, shard, nshards):
    if not 0 <= shard < nshards: raise ValueError("Require 0 <= shard < nshards")
    return int(key(unit)[:16], 16) % nshards == shard


def ratio(a, b):
    return float(a / b) if b > 1e-25 else None
