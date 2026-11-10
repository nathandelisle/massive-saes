"""Build token inputs for the Pythia geometry experiments from the saved article partitions. Downloads model files and requires the separate training mean and PC1 files."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "language"))
os.environ.setdefault("RX_LOCAL", "1")
from rx import partitions as P  # noqa: E402
from huggingface_hub import snapshot_download, model_info  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def export(model_id, part, tag):
    rev = model_info(model_id).sha
    snap = snapshot_download(model_id, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = AutoTokenizer.from_pretrained(snap)
    pt = P.Partition(part, tok)
    d = OUT / "inputs" / tag
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "tokens.npz", input_ids=pt.ids.numpy().astype(np.int64),
             attention_mask=pt.am.numpy().astype(np.int64), doc_id=pt.doc.astype(np.int64))
    meta = dict(model_id=model_id, model_revision=rev, tokenizer_id=model_id, tokenizer_revision=rev,
                corpus="wikitext-103-v1 train (HF Salesforce/wikitext)",
                partition="RX %s partition (pkg_rx/partitions.json sha256 %s)" % (part, sha(P.PATH)),
                bos_rule="tokenizer(text, add_special_tokens=True); the Pythia/GPT-NeoX tokenizer adds no BOS token, "
                         "so position 0 is the first text token",
                train_eval_overlap="RX partitions use WikiText-103 train articles starting after row 400,000; the SAE "
                                   "training store (first 5M tokens) ends before that row (asserted in rx/prep.py), so "
                                   "they are article-disjoint from SAE training. Overlap with the model's pretraining "
                                   "corpus (the Pile) is not controlled.",
                n_sequences=int(pt.ids.shape[0]), n_tokens=int(pt.am.sum()), n_articles=int(len(pt.docs)),
                snapshot=str(snap))
    (d / "input-meta.json").write_text(json.dumps(meta, indent=1))
    print(tag, meta["n_sequences"], meta["n_tokens"], meta["n_articles"], rev)
    return snap, d


if __name__ == "__main__":
    snap160, _ = export("EleutherAI/pythia-160m", "cal", "pythia160m_cal")
    snap14, d14 = export("EleutherAI/pythia-1.4b", "a1test", "pythia14b_a1test")
    prep = OUT / "rx_prep" / "pythia-1.4b"
    mean = np.load(prep / "mean.npy")
    pca = np.load(prep / "pca.npz")
    v1 = pca["v1"].astype(np.float64)
    v1 = v1 / np.linalg.norm(v1)
    pj = json.loads((prep / "prep.json").read_text())
    np.savez(d14 / "geometry.npz", mean=mean.astype(np.float64), v1=v1,
             block_index=np.int64(pj["site_block"]), model_id=np.array("EleutherAI/pythia-1.4b"))
    (d14 / "geometry_provenance.json").write_text(json.dumps(dict(
        source="RX prep (rx-shared/pythia-1.4b/prep): mean.npy + pca.npz['v1'], recomputed by exact replay of the "
               "v3 5M-token store; gate 3 matched the archived v3 PC1 to |dpc1| ~ 1e-15",
        label="RX-reconstructed training geometry (not the original archived file)",
        site_block_zero_based=int(pj["site_block"]), hidden_state_index=int(pj["site_block"]) + 1,
        mean_sha256=sha(prep / "mean.npy"), pca_sha256=sha(prep / "pca.npz"),
        unigram_counts_sha256=sha(prep / "unigram_counts.npy")), indent=1))
    print("geometry block", pj["site_block"])
