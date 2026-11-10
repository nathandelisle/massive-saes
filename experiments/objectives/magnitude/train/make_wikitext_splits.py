"""Build calibration / selection / test token files from WikiText-103 validation+test,
tokenized EXACTLY like the legacy SAE training stream (see e123_common.HELDOUT_STATEMENT).
Output format = the research package's token cache: input_ids, attention_mask, document_hashes."""
import argparse, json
from pathlib import Path
import numpy as np
from e123_common import spec_for, text_hash, HELDOUT_STATEMENT, hf_commit
from common import start_run, finish_run, write_json


def iter_docs(split):
    from datasets import load_dataset
    try:
        ds = load_dataset('Salesforce/wikitext', 'wikitext-103-v1', split=split)
    except Exception:
        ds = load_dataset('wikitext', 'wikitext-103-v1', split=split)
    for row in ds:
        t = row['text']
        if t and not t.isspace():
            yield t


def take(tok, docs, n, seen, seq_len=128):
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids_all, mask_all, hashes, texts = [], [], [], []
    skipped = dict(short=0, duplicate=0)
    for t in docs:
        h = text_hash(t)
        if h in seen:
            skipped['duplicate'] += 1; continue
        ids = tok(t, add_special_tokens=True, truncation=True, max_length=seq_len)['input_ids']
        if len(ids) < 2:  # runners require >=2 valid tokens per sequence
            skipped['short'] += 1; continue
        seen.add(h); k = len(ids)
        ids_all.append(ids + [pad] * (seq_len - k)); mask_all.append([1] * k + [0] * (seq_len - k))
        hashes.append(h); texts.append(t)
        if len(hashes) == n:
            break
    if len(hashes) != n:
        raise ValueError(f'Only {len(hashes)} documents available, wanted {n}')
    return np.array(ids_all, np.int64), np.array(mask_all, np.int8), np.array(hashes), skipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True); p.add_argument('--n-cal', type=int, default=256)
    p.add_argument('--n-sel', type=int, default=256); p.add_argument('--n-test', type=int, default=512)
    p.add_argument('--out', required=True); a = p.parse_args()
    spec = spec_for(a.model)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out, meta = start_run(a.out, vars(a))
    seen = set(); val = iter_docs('validation'); summary = {}
    for name, n, docs in [('calibration', a.n_cal, val), ('selection', a.n_sel, val), ('test', a.n_test, iter_docs('test'))]:
        ids, mask, hashes, skipped = take(tok, docs, n, seen)
        np.savez(out / f'{name}.npz', input_ids=ids, attention_mask=mask, document_hashes=hashes)
        summary[name] = dict(n_sequences=int(len(ids)), n_tokens=int(mask.sum()), skipped=skipped,
                             mean_tokens=float(mask.sum(1).mean()), source_split='test' if name == 'test' else 'validation')
    sets = [set(np.load(out / f'{n}.npz')['document_hashes'].astype(str)) for n in ('calibration', 'selection', 'test')]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2]), 'hash sets overlap'
    write_json(out / 'splits.json', dict(model=a.model, hf_id=spec.hf_id, tokenizer_commit=hf_commit(spec.hf_id),
               seq_len=128, add_special_tokens=True, pad_token_id=int(tok.pad_token_id),
               corpus='Salesforce/wikitext', config='wikitext-103-v1', splits=summary, heldout_statement=HELDOUT_STATEMENT))
    finish_run(out, meta)
    print(json.dumps({k: (v['n_sequences'], v['n_tokens']) for k, v in summary.items()}))

if __name__ == '__main__':
    main()
