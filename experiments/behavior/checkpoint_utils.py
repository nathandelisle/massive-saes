"""Find and load checkpoints for behavioral experiments."""
import hashlib
import json
import os
import pathlib
import subprocess

BUCKET = os.environ.get("S3_BUCKET", "")


def config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def fetch_sae(h, dest):
    """Return a cached checkpoint or download it from the indexed S3 path."""
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    idx = json.loads(pathlib.Path(os.environ.get("SAE_INDEX", str(pathlib.Path(__file__).resolve().parent / "inventory.json"))).read_text())
    row = next((r for r in idx["rows"] if r["config_hash"] == h and r["present"]), None)
    if row is None:
        raise RuntimeError("SAE %s is not present in S3 per inventory.json" % h)
    if not BUCKET:
        raise FileNotFoundError("Supply the cached checkpoint, or set S3_BUCKET to its archive bucket.")
    key = "s3://%s/%s/saes/%s.pt" % (BUCKET, row["prefixes"][0], h)
    subprocess.run(["aws", "s3", "cp", key, str(dest), "--only-show-errors"], check=True, timeout=3600)
    return dest
