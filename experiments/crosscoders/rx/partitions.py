"""Document-disjoint evaluation partitions, frozen once and shipped with the package.

Protocol kept from v2/v3: wikitext-103-v1 (non-raw), ONE PARAGRAPH PER SEQUENCE, truncated and
right-padded to 128, padding masked out of every loss. What changes is the DATA SPLIT:

  - documents are ARTICLES (a row " = Title = " opens one), not paragraphs, so two windows of
    one article never land in different partitions;
  - every partition is drawn from train-split articles that start AFTER the region the SAEs
    were trained on (the 5M-token store), which prep.py re-checks per tokenizer;
  - partitions are disjoint by article and fixed by a seeded shuffle, so every model and every
    box sees the same documents (token counts differ only by tokenizer, and are reported).

Partitions (target sizes in GPT-2 tokens; actual per-model counts are recorded):
  cal       65,536   calibration: latent means, donor reservoirs, alive-on-pool, thresholds,
                     A1 probe training
  rank      32,768   importance ranking
  valid     32,768   independent validation of every ranking
  a1test    65,536   A1 outer test
  explain  500,000   interpretation: explanation contexts
  ieval    500,000   interpretation: held-out evaluation contexts
"""
import json
import pathlib
import re

import numpy as np
import torch

from . import common

PATH = common.PKG / "partitions.json"
TARGETS = dict(cal=65_536, rank=32_768, valid=32_768, a1test=65_536,
               explain=500_000, ieval=500_000)
ORDER = ("rank", "valid", "cal", "a1test", "explain", "ieval")
# rows taken per article: small partitions must span many documents, or a document-level
# bootstrap has ~10 clusters to resample
ROW_CAP = dict(cal=10, rank=6, valid=6, a1test=10, explain=40, ieval=40)
MIN_ROW = 400_000        # first train row eligible; prep.py asserts the store ends before it
SEED = 20261005
SEQ_LEN = 128
_HEAD = re.compile(r"^ = [^=].* = \n?$")


def _rows():
    from datasets import load_dataset
    try:
        return load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")["text"]
    except Exception:
        return load_dataset("wikitext", "wikitext-103-v1", split="train")["text"]


def build(out=PATH):
    """Run ONCE (locally) to freeze the partitions. Needs only the GPT-2 tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    rows = _rows()
    arts, cur = [], None
    for i, t in enumerate(rows):
        if _HEAD.match(t):
            if cur:
                arts.append(cur)
            cur = dict(start=i, rows=[])
        if cur is not None and t and not t.isspace():
            cur["rows"].append(i)
    if cur:
        arts.append(cur)
    elig = [a for a in arts if a["start"] >= MIN_ROW and a["rows"]]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(elig))
    parts, k = {}, 0
    for name in ORDER:
        got, toks = [], 0
        while toks < TARGETS[name] and k < len(order):
            a = elig[int(order[k])]
            k += 1
            rr = a["rows"][:ROW_CAP[name]]
            n = sum(min(len(tok(rows[r])["input_ids"]), SEQ_LEN) for r in rr)
            got.append(dict(start=a["start"], rows=rr, gpt2_tokens=n))
            toks += n
        parts[name] = dict(articles=got, gpt2_tokens=toks, n_articles=len(got),
                           n_rows=sum(len(a["rows"]) for a in got))
    meta = dict(corpus="Salesforce/wikitext wikitext-103-v1 train", seed=SEED, min_row=MIN_ROW,
                n_articles_total=len(arts), n_articles_eligible=len(elig), seq_len=SEQ_LEN,
                row_cap_per_article=ROW_CAP, article_rule="a row matching ^ = [^=].* = opens an article",
                one_paragraph_per_sequence=True)
    pathlib.Path(out).write_text(json.dumps(dict(meta=meta, partitions=parts)))
    for n in ORDER:
        print("%-8s %4d articles %6d rows %8d gpt2 tokens" % (
            n, parts[n]["n_articles"], parts[n]["n_rows"], parts[n]["gpt2_tokens"]))
    # disjointness, asserted rather than assumed
    seen = {}
    for n in ORDER:
        for a in parts[n]["articles"]:
            assert a["start"] not in seen, (n, seen.get(a["start"]))
            seen[a["start"]] = n
    print("disjoint by article: OK (%d articles)" % len(seen))


class Partition:
    """Tokenised sequences for one partition, with document (article) ids per sequence."""

    def __init__(self, name, tok, rows=None, max_seqs=None):
        spec = json.loads(PATH.read_text())["partitions"][name]
        rows = rows if rows is not None else _rows()
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids, am, doc, row_idx = [], [], [], []
        for a in spec["articles"]:
            for r in a["rows"]:
                t = tok(rows[r], add_special_tokens=True, truncation=True,
                        max_length=SEQ_LEN)["input_ids"]
                if not t:
                    continue
                ids.append(t + [pad] * (SEQ_LEN - len(t)))
                am.append([1] * len(t) + [0] * (SEQ_LEN - len(t)))
                doc.append(a["start"])
                row_idx.append(r)
                if max_seqs and len(ids) >= max_seqs:
                    break
            if max_seqs and len(ids) >= max_seqs:
                break
        self.name = name
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.am = torch.tensor(am, dtype=torch.long)
        self.doc = np.asarray(doc, dtype=np.int64)          # article start row = document id
        self.row = np.asarray(row_idx, dtype=np.int64)
        self.lengths = self.am.sum(1).numpy()
        self.docs = np.unique(self.doc)
        self.doc_index = np.searchsorted(self.docs, self.doc)  # 0..n_docs-1 per sequence

    @property
    def n_targets(self):
        """Valid next-token targets: every real position except each sequence's last."""
        return int(np.clip(self.lengths - 1, 0, None).sum())

    def batches(self, max_tokens=8192, max_seqs=64):
        """Length-sorted batches trimmed to the batch's longest sequence (no padding waste).
        Yields (seq_indices, ids[B,Tb], am[B,Tb])."""
        order = np.argsort(-self.lengths, kind="stable")
        i = 0
        while i < len(order):
            T = int(self.lengths[order[i]])
            n = max(1, min(max_seqs, max_tokens // max(T, 1)))
            sel = order[i:i + n]
            yield sel, self.ids[sel, :T], self.am[sel, :T]
            i += n

    def summary(self):
        return dict(name=self.name, n_seqs=int(len(self.ids)), n_docs=int(len(self.docs)),
                    n_tokens=int(self.lengths.sum()), n_targets=self.n_targets)


if __name__ == "__main__":
    build()
