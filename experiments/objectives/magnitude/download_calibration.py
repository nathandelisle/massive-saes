"""Download the 40 saved magnitude-sweep calibration files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE.parents[2] / "results/additional/objectives/five_seed_beta/inputs/bcal")
    parser.add_argument("--run", help="Download one run, for example standard_k64_s42_beta0.")
    args = parser.parse_args()
    manifest = json.loads((HERE / "calibration_files.json").read_text())
    files = [f for f in manifest["files"] if args.run is None or Path(f["path"]).stem == args.run]
    if not files:
        parser.error("Run not found in calibration_files.json")
    args.out.mkdir(parents=True, exist_ok=True)
    for entry in files:
        target = args.out / Path(entry["path"]).name
        if target.exists() and target.stat().st_size == entry["bytes"] and sha256(target) == entry["sha256"]:
            print(f"Already verified: {target.name}", flush=True)
            continue
        url = f"https://huggingface.co/datasets/{manifest['repo']}/resolve/{manifest['revision']}/{entry['path']}"
        temporary = target.with_suffix(".npz.part")
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            if temporary.stat().st_size != entry["bytes"] or sha256(temporary) != entry["sha256"]:
                raise ValueError(f"Size or SHA-256 mismatch: {target.name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Verified: {target.name}", flush=True)


if __name__ == "__main__":
    main()
