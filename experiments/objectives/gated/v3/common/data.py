"""Corpus loading and residual-stream activation harvesting.

Activations are taken from the OUTPUT of the transformer block at the target
layer (post-attention, post-MLP, pre-LayerNorm-of-next-layer) — v2 Section 4.
Stored as float16 memmaps of shape [n_tokens, d_model] plus a float64 running
mean, so PCA (float64, first 50k centered rows) and SAE training stream from
disk instead of holding 5M x 4096 in RAM.
"""

import json
import os

import numpy as np
import torch

from .registry import SEQ_LEN, TRAIN_TOKENS, block_index


def get_blocks(model, block_path: str):
    obj = model
    for part in block_path.split("."):
        obj = getattr(obj, part)
    return obj


def load_model_and_tokenizer(spec, device="cuda", dtype=None, device_map=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = dtype or getattr(torch, spec.dtype)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(torch_dtype=dtype)
    if device_map is not None:
        kwargs["device_map"] = device_map          # "auto" for llama-70b
    if spec.family == "gemma":
        # gemma-2's logit soft-capping is only applied by the eager attention
        # path; sdpa/flash silently skip it and change the output distribution
        kwargs["attn_implementation"] = "eager"
    try:
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
    except TypeError:                              # transformers>=5 renamed it
        kwargs["dtype"] = kwargs.pop("torch_dtype")
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
    if device_map is None:
        model = model.to(device)
    model.eval()
    return model, tok


def corpus_texts(corpus: str, split: str = "train"):
    """corpus in {wikitext, openwebtext}. Yields raw text strings.

    v2 used `wikitext-103-v1` (the NON-raw config, <unk>-substituted). Keeping
    the raw config would change the token stream and therefore every activation
    statistic, so v3 matches v2 and exposes the raw variant as `wikitext-raw`.
    """
    from datasets import load_dataset
    if corpus in ("wikitext", "wikitext-raw"):
        cfg = "wikitext-103-raw-v1" if corpus == "wikitext-raw" else "wikitext-103-v1"
        try:
            ds = load_dataset("Salesforce/wikitext", cfg, split=split)
        except Exception:
            ds = load_dataset("wikitext", cfg, split=split)
        for row in ds:
            t = row["text"]
            if t and not t.isspace():
                yield t
    elif corpus == "openwebtext":
        # eval MUST NOT come from the training stream; OpenWebText ships only a
        # train split, so hold out a deterministic tail slice for evaluation
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        for i, row in enumerate(ds):
            is_eval_slice = (i % 100) == 99
            if (split == "train") == is_eval_slice:
                continue
            yield row["text"]
    else:
        raise ValueError(corpus)


def tokenized_batches(tok, corpus="wikitext", split="train", seq_len=SEQ_LEN,
                      batch_seqs=32, max_tokens=TRAIN_TOKENS, pack=False,
                      min_chars=1):
    """Yield (ids, attention_mask), both LongTensor [B, seq_len].

    v2 fidelity (core/final.py): ONE DOCUMENT PER SEQUENCE, truncated to
    seq_len and padded to seq_len, with padding positions dropped from the
    activation store and masked out of cross-entropy. Packing instead would
    (a) put BOS at arbitrary positions, wrecking the App-P BOS-free condition
    and the Llama "mass at position 0" analysis, and (b) change the token
    distribution the SAE sees. `pack=True` is available for ablations but is
    NOT the protocol.
    """
    batch_ids, batch_am, produced = [], [], 0
    if pack:
        buf = []
        for text in corpus_texts(corpus, split):
            buf.extend(tok(text, add_special_tokens=True)["input_ids"])
            while len(buf) >= seq_len:
                batch_ids.append(buf[:seq_len])
                batch_am.append([1] * seq_len)
                buf = buf[seq_len:]
                if len(batch_ids) == batch_seqs:
                    yield (torch.tensor(batch_ids), torch.tensor(batch_am))
                    produced += batch_seqs * seq_len
                    batch_ids, batch_am = [], []
                    if produced >= max_tokens:
                        return
    else:
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        for text in corpus_texts(corpus, split):
            if len(text) < min_chars:
                continue
            ids = tok(text, add_special_tokens=True, truncation=True,
                      max_length=seq_len)["input_ids"]
            if not ids:
                continue
            n = len(ids)
            batch_ids.append(ids + [pad_id] * (seq_len - n))
            batch_am.append([1] * n + [0] * (seq_len - n))
            if len(batch_ids) == batch_seqs:
                yield (torch.tensor(batch_ids), torch.tensor(batch_am))
                produced += int(sum(sum(m) for m in batch_am))
                batch_ids, batch_am = [], []
                if produced >= max_tokens:
                    return
    if batch_ids:
        yield (torch.tensor(batch_ids), torch.tensor(batch_am))


class _CaptureHook:
    def __init__(self):
        self.buf = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        self.buf = hs.detach()
        return output


@torch.no_grad()
def harvest_activations(spec, out_dir, corpus="wikitext", n_tokens=TRAIN_TOKENS,
                        device="cuda", batch_seqs=32, exclude_first_position=False,
                        device_map=None, layer=None):
    """Write acts.memmap [n, d] float16, meta.json, mean.npy (float64) to out_dir.

    exclude_first_position: drop the first non-padding position of every
    sequence (BOS-exclusion condition, App P).
    layer: override target layer (used by the layer-profile experiments).
    Idempotent: returns immediately if meta.json exists with matching config.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    cfg = dict(model=spec.key, corpus=corpus, n_tokens=n_tokens,
               layer=layer or spec.target_layer,
               exclude_first_position=exclude_first_position)
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        if {k: old.get(k) for k in cfg} == cfg and old.get("complete"):
            return out_dir
    model, tok = load_model_and_tokenizer(spec, device=device, device_map=device_map)
    blocks = get_blocks(model, spec.block_path)
    li = block_index(spec, layer)
    hook = _CaptureHook()
    h = blocks[li].register_forward_hook(hook)

    d = spec.d_model
    mm = np.lib.format.open_memmap(os.path.join(out_dir, "acts.npy"), mode="w+",
                                   dtype=np.float16, shape=(n_tokens, d))
    mean = np.zeros(d, dtype=np.float64)
    written = 0
    for ids, am in tokenized_batches(tok, corpus=corpus, batch_seqs=batch_seqs,
                                     max_tokens=int(n_tokens * 1.05) + SEQ_LEN):
        dev = device if device_map is None else model.device
        ids, am = ids.to(dev), am.to(dev)
        # the mask matters twice: the model must not attend to padding, and
        # padded positions must never enter the activation store
        model(ids, attention_mask=am)
        acts = hook.buf.float()                     # [B, T, d]
        keep = am.bool()
        if exclude_first_position:
            # drop the FIRST REAL token of each sequence (BOS for
            # BOS-prepending tokenizers) — with one doc per sequence that is
            # unambiguously position 0
            keep = keep.clone()
            keep[:, 0] = False
        flat = acts[keep]                           # [n_valid, d]
        take = min(flat.shape[0], n_tokens - written)
        chunk = flat[:take].cpu().numpy()
        mm[written:written + take] = chunk.astype(np.float16)
        mean += chunk.astype(np.float64).sum(axis=0)
        prev, written = written, written + take
        # progress every ~1M tokens: the harvest is otherwise several silent
        # minutes in an unattended queue
        if written // 1_000_000 != prev // 1_000_000:
            print(f"      [harvest {spec.key} L{cfg['layer']}] "
                  f"{written:,}/{n_tokens:,} tokens", flush=True)
        if written >= n_tokens:
            break
    h.remove()
    mm.flush()
    mean /= max(written, 1)
    np.save(os.path.join(out_dir, "mean.npy"), mean)
    cfg.update(complete=True, n_written=int(written))
    json.dump(cfg, open(meta_path, "w"), indent=2)
    del model
    torch.cuda.empty_cache()
    return out_dir


def open_activations(act_dir):
    acts = np.load(os.path.join(act_dir, "acts.npy"), mmap_mode="r")
    mean = np.load(os.path.join(act_dir, "mean.npy"))
    return acts, mean


def activation_batches(act_dir, batch_size, steps, seed, device="cuda"):
    """Uniformly sampled centered activation batches for SAE training (v2 App G)."""
    acts, mean = open_activations(act_dir)
    n = acts.shape[0]
    rng = np.random.default_rng(seed)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    for _ in range(steps):
        idx = np.sort(rng.integers(0, n, size=batch_size))
        x = torch.tensor(np.asarray(acts[idx], dtype=np.float32), device=device)
        yield x - mean_t


def eval_sequences(tok, corpus="wikitext", n_seqs=64, seq_len=SEQ_LEN, seed=0,
                   min_chars=1):
    """Held-out sequences for importance/CE evaluation.

    Returns (ids, attention_mask). v2 (core/final.py) did exactly:
        eval_texts = [t["text"] for t in eval_dataset if t["text"].strip()][:64]
    i.e. the wikitext VALIDATION split, non-empty texts only, one document per
    sequence, truncated and padded to 128, with padding masked out of the loss.

    `min_chars=1` reproduces that. Raising it drops the short heading lines
    ("= Homarus gammarus =") that make much of v2's eval set padding, which
    yields more real tokens per sequence — a better eval set, but NOT v2's, so
    it must not be the default while we are checking reproduction. Report it as
    a separate robustness row if used.
    """
    split = "validation" if corpus.startswith("wikitext") else "eval"
    ids, ams = [], []
    for b_ids, b_am in tokenized_batches(tok, corpus=corpus, split=split,
                                         batch_seqs=8, min_chars=min_chars,
                                         max_tokens=n_seqs * seq_len * 8):
        for r, m in zip(b_ids, b_am):
            ids.append(r)
            ams.append(m)
            if len(ids) == n_seqs:
                return torch.stack(ids), torch.stack(ams)
    return torch.stack(ids), torch.stack(ams)
