"""Article-disjoint data preparation. No row hashes masquerade as article IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .io import atomic_json, atomic_npz, key


def read_tokens(path, allow_legacy=False):
    with np.load(path, allow_pickle=False) as n:
        ids = n["input_ids"].astype(np.int64)
        mask = n["attention_mask"].astype(bool)
        if "article_ids" in n:
            articles = n["article_ids"].astype(str); scope = "article"
        elif allow_legacy and "document_hashes" in n:
            articles = n["document_hashes"].astype(str); scope = "legacy_row_NOT_article"
        else:
            raise ValueError("Need actual article_ids; legacy row hashes require split_policy=legacy explicitly")
        seq = n["sequence_ids"].astype(str) if "sequence_ids" in n else np.array([
            key([str(articles[i]), ids[i].tolist(), i]) for i in range(len(ids))])
    if ids.ndim != 2 or ids.shape != mask.shape or articles.shape != (len(ids),):
        raise ValueError("Bad token shapes")
    if not (mask.sum(1) >= 2).all() or len(set(seq)) != len(seq):
        raise ValueError("Short sequences or duplicate sequence IDs")
    if not (mask[:,:-1]&mask[:,1:]).any(1).all():raise ValueError('Each sequence needs at least one valid next-token prediction')
    return dict(ids=ids, mask=mask, articles=articles, seq=seq, scope=scope)


def check_splits(cfg):
    legacy = cfg.get("split_policy", "article") == "legacy"
    data = {k: read_tokens(p, legacy) for k, p in cfg["tokens"].items()}
    if set(data) != {"calibration", "selection", "test"}:
        raise ValueError("Exactly calibration, selection, test token caches required")
    overlap = []
    for a, b in (("calibration", "selection"), ("calibration", "test"), ("selection", "test")):
        shared = set(data[a]["articles"]) & set(data[b]["articles"])
        if shared:
            overlap.append(dict(a=a, b=b, n=len(shared)))
            if not legacy: raise ValueError(f"Article leakage: {a}/{b}, {len(shared)} shared")
    if legacy and not cfg.get("legacy_reason"):
        raise ValueError("Legacy mode requires legacy_reason; it cannot support new article-level inference")
    return data, dict(policy=cfg.get("split_policy", "article"), overlaps=overlap,
        splits={k: dict(rows=len(v["ids"]), valid_tokens=int(v["mask"].sum()),
                       clusters=len(set(v["articles"])), cluster_type=v["scope"])
                for k, v in data.items()})


def wikitext_articles(dataset, source):
    """A top-level WikiText title has exactly one '=' at each end; subsections stay inside."""
    title = None; rows = []
    for row in dataset:
        text = row["text"].strip()
        if not text: continue
        top = text.startswith("=") and text.endswith("=") and not text[1:].lstrip().startswith("=") and not text[:-1].rstrip().endswith("=")
        if top:
            if title is not None:
                yield dict(article_id=key([source, title, rows]), title=title, rows=rows, source=source)
            title = text; rows = []
        elif title is not None:
            rows.append(text)
    if title is not None:
        yield dict(article_id=key([source, title, rows]), title=title, rows=rows, source=source)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--articles", help="JSONL of {article_id,text} or {article_id,rows}; otherwise fetch WikiText validation+test")
    p.add_argument("--dataset-revision", help="Pin HF dataset commit when fetching")
    p.add_argument("--tokenizer", required=True); p.add_argument("--revision")
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--articles-per-split", type=int, default=0, help="0 uses up to 120 per split from the available articles; never invents extra WikiText articles")
    p.add_argument("--rows-per-article", type=int, default=8)
    args = p.parse_args(); out = Path(args.out)
    if (out / "manifest.json").exists(): raise FileExistsError("Data are frozen; use a new output directory")
    if args.articles:
        docs = [json.loads(s) for s in Path(args.articles).read_text().splitlines() if s.strip()]
    else:
        if not args.dataset_revision: p.error("Pin --dataset-revision for a reproducible fetch")
        from datasets import load_dataset
        docs = []
        for split in ("validation", "test"):
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split=split, revision=args.dataset_revision)
            docs += list(wikitext_articles(ds, "wikitext-103-v1/" + split))
    # Content duplicates cannot land in different folds under different identifiers.
    by_content = {}; original_document_count=len(docs)
    for d in docs:
        rows = d.get("rows", [d.get("text", "")])
        content = "\n".join(rows).strip()
        if content:
            h = hashlib.sha256(content.encode()).hexdigest()
            by_content.setdefault(h, dict(**{k:v for k,v in d.items() if k != "rows"}, rows=rows))
    docs = sorted(by_content.values(), key=lambda d: key([args.seed, str(d["article_id"])]))
    if len({str(d['article_id']) for d in docs}) != len(docs): raise ValueError("Duplicate article IDs")
    per_split=args.articles_per_split or min(120,len(docs)//3)
    if per_split<2:p.error('Need at least six usable articles')
    required = 3 * per_split
    if len(docs) < required:
        p.error(f"Need {required} distinct articles, found {len(docs)}; choose a recorded smaller design or another corpus")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    if tok.pad_token_id is None: raise ValueError("Tokenizer has no pad/eos token")
    manifest = dict(settings=vars(args),actual_articles_per_split=per_split,
        duplicates_removed=original_document_count-len(docs),splits={})
    # Allocate whole articles BEFORE tokenization. Do not backfill articles based on outcomes.
    for k, split in enumerate(("calibration", "selection", "test")):
        chosen = docs[k * per_split:(k + 1) * per_split]
        ids=[]; masks=[]; articles=[]; seq=[]; row_hashes=[]
        for d in chosen:
            candidates = sorted(enumerate(d["rows"]), key=lambda it: key([args.seed, d["article_id"], it[0]]))
            accepted = 0
            for j, text in candidates:
                enc = tok(text, truncation=True, max_length=args.ctx, padding="max_length", add_special_tokens=True)
                if sum(enc["attention_mask"]) < 2: continue
                ids.append(enc["input_ids"]); masks.append(enc["attention_mask"])
                articles.append(str(d["article_id"])); seq.append(key([d["article_id"], j]))
                row_hashes.append(hashlib.sha256(text.encode()).hexdigest())
                accepted += 1
                if accepted >= args.rows_per_article: break
        if not ids: raise ValueError("Empty split")
        atomic_npz(out / f"{split}.npz", input_ids=np.asarray(ids, dtype=np.int64), attention_mask=np.asarray(masks, dtype=bool),
                   article_ids=np.asarray(articles), sequence_ids=np.asarray(seq), document_hashes=np.asarray(row_hashes))
        manifest["splits"][split] = dict(assigned_articles=[str(d["article_id"]) for d in chosen],
            observed_articles=len(set(articles)), rows=len(ids), tokens=int(np.sum(masks)))
    atomic_json(out / "manifest.json", manifest)
    print("[pc1-data]", {k:(v['observed_articles'],v['tokens']) for k,v in manifest['splits'].items()}, flush=True)


if __name__ == "__main__": main()
