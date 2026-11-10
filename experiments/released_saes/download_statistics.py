"""Download the per-checkpoint statistics used by the released-SAE collectors."""
import argparse
from pathlib import Path

REPOSITORY = "nathandelisle/massive-saes-results"
REVISION = "c512c9a270940b5f24cd5574dc403d51bbf615d7"
RESULTS = Path(__file__).resolve().parents[2] / "results/additional"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--all-rounds", action="store_true",
                        help="Also download the older audit's statistics.")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    rounds = "*" if args.all_rounds else "round2_results"
    snapshot_download(repo_id=REPOSITORY, repo_type="dataset", revision=args.revision,
                      allow_patterns=[f"released_saes/{rounds}/**/*.npz"],
                      local_dir=str(RESULTS))

if __name__ == "__main__":
    main()
