"""Stage B/C activation caches in the cached_coders format (code/ext_reference/README.txt).

Splits (article-disjoint, doc id = article start row):
  train      = "trainx": eligible wikitext-103 train articles (start row >= partitions MIN_ROW) that are in NO frozen
               partition, seeded order (SEED+1), <=40 rows/article, until --train-tokens model tokens. Larger than cal
               because a 8192-feature coder cannot be fitted on 66k tokens (DECISIONS D8).
  validation = rank, test = valid (the frozen partitions, as in Stage A / B0).
Same text protocol as every RX measurement: one paragraph per sequence, truncated to 128, padding masked.
Activations stored raw, uncentered, float16 (all cached models run in fp16; no cast loss), finiteness asserted.

Kinds
  mlp      : one cache per layer; x = tensor passed to block.mlp, y = the MLP's contribution as added to the residual
             (hooks as surveyB0, verified per family).
  resid    : per-layer residual streams (output of block L) saved as single-stream arrays; crosscoder pair caches
             are assembled from them by `pair` (y = [stream A | stream B]).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from rx import common, modelio as M, partitions as P

SPLITS = (("train", "trainx"), ("validation", "rank"), ("test", "valid"))


class TrainX(P.Partition):
    def __init__(self, tok, target_tokens, rows=None):
        rows = rows if rows is not None else P._rows()
        frozen = json.loads(P.PATH.read_text())
        used = {a["start"] for p in frozen["partitions"].values() for a in p["articles"]}
        arts, cur = [], None
        for i, t in enumerate(rows):
            if P._HEAD.match(t):
                if cur:
                    arts.append(cur)
                cur = dict(start=i, rows=[])
            if cur is not None and t and not t.isspace():
                cur["rows"].append(i)
        if cur:
            arts.append(cur)
        elig = [a for a in arts if a["start"] >= P.MIN_ROW and a["rows"] and a["start"] not in used]
        order = np.random.default_rng(P.SEED + 1).permutation(len(elig))
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids, am, doc, row_idx, n = [], [], [], [], 0
        for k in order:
            a = elig[int(k)]
            for r in a["rows"][:40]:
                t = tok(rows[r], add_special_tokens=True, truncation=True, max_length=P.SEQ_LEN)["input_ids"]
                if not t:
                    continue
                ids.append(t + [pad] * (P.SEQ_LEN - len(t)))
                am.append([1] * len(t) + [0] * (P.SEQ_LEN - len(t)))
                doc.append(a["start"]); row_idx.append(r); n += len(t)
            if n >= target_tokens:
                break
        self.name = "trainx"
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.am = torch.tensor(am, dtype=torch.long)
        self.doc = np.asarray(doc, dtype=np.int64)
        self.row = np.asarray(row_idx, dtype=np.int64)
        self.lengths = self.am.sum(1).numpy()
        self.docs = np.unique(self.doc)
        self.doc_index = np.searchsorted(self.docs, self.doc)
        assert not (set(self.docs.tolist()) & used)


def mlp_hooks(model, spec, layers, store):
    hs = []
    bl = M.blocks(model, spec)
    for l in layers:
        b = bl[l]
        hs.append(b.mlp.register_forward_pre_hook(lambda m, a, l=l: store.__setitem__(("x", l), a[0].detach())))
        tgt = b.post_feedforward_layernorm if spec.family == "gemma" else b.mlp
        hs.append(tgt.register_forward_hook(
            lambda m, i, o, l=l: store.__setitem__(("y", l), (o[0] if isinstance(o, tuple) else o).detach())))
    return hs


def resid_hooks(model, spec, layers, store):
    bl = M.blocks(model, spec)
    return [bl[l].register_forward_hook(
        lambda m, i, o, l=l: store.__setitem__(("r", l), (o[0] if isinstance(o, tuple) else o).detach()))
        for l in layers]


def write_split(model, spec, part, layers, kind, root, split, max_tokens=8192):
    n = int(part.lengths.sum())
    keys = [("x", l) for l in layers] + [("y", l) for l in layers] if kind == "mlp" else [("r", l) for l in layers]
    arrs = {}
    for k, l in keys:
        f = root / ("%s%d" % ("mlp" if kind == "mlp" else "resid", l)) / split
        f.mkdir(parents=True, exist_ok=True)
        name = "%s.npy" % k if kind == "mlp" else "r.npy"
        arrs[(k, l)] = (f / name, None)
    store, docs, pos, off = {}, [], [], 0
    hs = (mlp_hooks if kind == "mlp" else resid_hooks)(model, spec, layers, store)
    last = max(layers)
    try:
        with torch.no_grad():
            for sel, ids, am in part.batches(max_tokens=max_tokens):
                store.clear()
                M.prefix(model, spec, ids, am, layer=last)
                m = am.reshape(-1).bool()
                cnt = int(m.sum())
                for key in keys:
                    t = store[key]
                    v = t.reshape(-1, t.shape[-1])[m.to(t.device)].to(torch.float16)
                    assert torch.isfinite(v).all(), ("nonfinite", key)
                    path, mm = arrs[key]
                    if mm is None:
                        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, v.shape[1]))
                        arrs[key] = (path, mm)
                    mm[off:off + cnt] = v.cpu().numpy()
                T = ids.shape[1]
                docs.append(np.repeat(part.doc[sel], T)[m.numpy()])
                pos.append(np.tile(np.arange(T), len(sel))[m.numpy()])
                off += cnt
    finally:
        for h in hs:
            h.remove()
    assert off == n, (off, n)
    for (k, l), (path, mm) in arrs.items():
        mm.flush()
        np.save(path.parent / "doc_id.npy", np.concatenate(docs))
        np.save(path.parent / "position.npy", np.concatenate(pos).astype(np.int32))
    return n


def build(model, spec, tok, kind, layers, root, train_tokens, source):
    parts = {}
    for split, name in SPLITS:
        parts[split] = TrainX(tok, train_tokens) if name == "trainx" else P.Partition(name, tok)
    toks = {}
    for split, _ in SPLITS:
        toks[split] = write_split(model, spec, parts[split], layers, kind, root, split)
        common.log("cache", "%s %s %s: %d tokens" % (source, kind, split, toks[split]))
    for l in layers:
        d = root / ("%s%d" % ("mlp" if kind == "mlp" else "resid", l))
        meta = dict(task="transcoder" if kind == "mlp" else "stream", coordinates="raw_uncentered",
                    output_widths=[spec.d_model],
                    cache_id="%s-%s%d-%s" % (source, kind, l, hashlib.sha1(os.urandom(8)).hexdigest()[:10]),
                    source=dict(model=source, base_model=spec.hf_id, layer=l, dtype_stored="float16",
                                hooks=dict(x="block.mlp forward pre-hook",
                                           y=("block.post_feedforward_layernorm output" if spec.family == "gemma"
                                              else "block.mlp output")) if kind == "mlp" else "output of block %d" % l,
                                splits=dict(SPLITS), train_rule="trainx (ext/cache.py)", n_tokens=toks))
        (d / "metadata.json").write_text(json.dumps(meta, indent=1))
    return toks


def pair(a, b, out, transform=None, label=""):
    """Crosscoder cache y = [A | B] from two single-stream residual caches (same token order required)."""
    a, b, out = Path(a), Path(b), Path(out)
    ma, mb = json.loads((a / "metadata.json").read_text()), json.loads((b / "metadata.json").read_text())
    for split, _ in SPLITS:
        ra = np.load(a / split / "r.npy", mmap_mode="r")
        rb = np.load(b / split / "r.npy", mmap_mode="r")
        da, db = np.load(a / split / "doc_id.npy"), np.load(b / split / "doc_id.npy")
        assert ra.shape[0] == rb.shape[0] and (da == db).all(), "streams not token-aligned"
        assert (np.load(a / split / "position.npy") == np.load(b / split / "position.npy")).all()
        (out / split).mkdir(parents=True, exist_ok=True)
        y = np.lib.format.open_memmap(out / split / "y.npy", mode="w+", dtype=np.float16,
                                      shape=(ra.shape[0], ra.shape[1] + rb.shape[1]))
        for i in range(0, len(ra), 65536):
            A = np.asarray(ra[i:i + 65536], np.float32)
            B = np.asarray(rb[i:i + 65536], np.float32)
            if transform is not None:
                B = transform(B)
            y[i:i + 65536, :ra.shape[1]] = A
            y[i:i + 65536, ra.shape[1]:] = B
        y.flush()
        np.save(out / split / "doc_id.npy", da)
    meta = dict(task="crosscoder", coordinates="raw_uncentered", output_widths=[ra.shape[1], rb.shape[1]],
                cache_id="pair-%s-%s" % (label, hashlib.sha1(os.urandom(8)).hexdigest()[:10]),
                source=dict(stream_a=ma["cache_id"], stream_b=mb["cache_id"], label=label,
                            a=ma["source"], b=mb["source"]))
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["mlp", "resid"], required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--train-tokens", type=int, default=3_000_000)
    ap.add_argument("--revision", default=None)
    a = ap.parse_args(argv)
    spec = M.spec_of(a.model)
    layers = [int(x) for x in a.layers.split(",")]
    if a.revision:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=a.revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, revision=a.revision,
                                                     torch_dtype=getattr(torch, spec.dtype) if torch.cuda.is_available() else torch.float32).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        source = "%s@%s" % (a.model, a.revision)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model, tok = M.load(spec, device=dev, dtype=None if dev == "cuda" else torch.float32)
        source = a.model
    root = Path(a.root)
    return build(model, spec, tok, a.kind, layers, root, a.train_tokens, source)
