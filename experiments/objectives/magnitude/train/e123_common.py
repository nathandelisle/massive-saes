"""Shared helpers for the E0/E1/E2/E3/E5 pipeline (pc1_e123). Small and explicit."""
import hashlib, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE / 'pc1_v3'
sys.path.insert(0, str(SUITE)); sys.path.insert(0, str(HERE / 'experiments'))

HELDOUT_STATEMENT = (
    "Every legacy SAE (all models, seeds, conditions) was trained on activations harvested from "
    "Salesforce/wikitext config wikitext-103-v1 split=train: ONE DOCUMENT (dataset row) PER SEQUENCE, "
    "tok(text, add_special_tokens=True, truncation=True, max_length=128), padded to 128, first 5,000,000 "
    "valid (non-pad) tokens in dataset order (v3.common.data.tokenized_batches / harvest_activations). "
    "WikiText-103 validation and test splits are article-disjoint from train, so no validation/test row was "
    "ever in any legacy training stream. calibration = first n_cal non-empty validation rows, "
    "selection = the next n_sel validation rows, test = first n_test non-empty test-split rows; rows that are "
    "empty/whitespace, tokenize to <2 tokens, or exactly duplicate an earlier kept row (sha256 of text) are "
    "skipped so the three document-hash sets are disjoint. The legacy 'last 100k of the training pool' tail is "
    "NOT held out and is not used here.")


def spec_for(key):
    from v3.common import registry
    return registry.MODELS[key]


def root():
    return Path(os.environ.get('PC1_V3_ROOT', str(HERE / 'workdir'))).resolve()


def text_hash(t):
    return hashlib.sha256(t.encode('utf-8')).hexdigest()


def manifest_complete(d):
    p = Path(d) / 'manifest.json'
    if not p.is_file():
        return False
    try:
        return json.loads(p.read_text()).get('status') == 'complete'
    except Exception:
        return False


def hf_commit(repo_id):
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo_id).sha
    except Exception as e:  # offline or gated: record the failure, do not fail the run
        return None


def pick_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'
