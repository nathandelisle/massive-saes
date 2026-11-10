"""A clearly separated, release-style evaluation of the Pythia checkpoint, after the encoder fix.

The main audit chunks documents independently at context 128 with no BOS. The release-era sparsify input
pipeline (sparsify@c259f271 sae/data.py) instead CONCATENATES documents with an EOS separator and cuts the
stream into fixed 2049-token sequences, so most sequences straddle document boundaries and almost no row sits
at a document start. That is a different token population, and this script measures whether the corrected
Pythia numbers depend on it.

This is a SEPARATE view. It does not replace the standardized context-128 protocol, and it is not a claim to
have reproduced EleutherAI's evaluation: the corpus is still our held-out WikiText-103 sample, and the SAE is
still evaluated with our own PC1 decomposition.

Document bootstrap here resamples the SOURCE documents that contributed rows; a packed sequence may draw on
several, so each row is attributed to the document its token came from.

usage: pythia_release_style.py --corpus corpus --out results/pythia_release_style --context 2049
"""
import argparse
import json
import pathlib
import time

import numpy as np

from harvest import get_blocks
from metrics import article_bootstrap, merge_stats, sufficient_stats, summarize


def packed_batches(tok, path, context, eos_id, batch_size, limit_rows, doc_index):
    """Concatenate documents with an EOS separator and cut the stream into fixed-length sequences."""
    stream, owners, seqs, owns, produced = [], [], [], [], 0
    for line in pathlib.Path(path).open(encoding='utf-8'):
        if not line.strip():
            continue
        d = json.loads(line)
        n = doc_index.setdefault(d['doc_id'], len(doc_index))
        ids = tok.encode(d['text'], add_special_tokens=False)
        stream.extend(ids + [eos_id])
        owners.extend([n] * (len(ids) + 1))
        while len(stream) >= context:
            seqs.append(stream[:context])
            owns.append(owners[:context])
            del stream[:context]
            del owners[:context]
            if len(seqs) == batch_size:
                yield seqs, owns
                produced += batch_size * context
                seqs, owns = [], []
                if produced >= limit_rows:
                    return
    if seqs:
        yield seqs, owns


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='manifest.json')
    p.add_argument('--corpus', default='corpus')
    p.add_argument('--out', default='results/pythia_release_style')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--context', type=int, default=2049)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--calibration-rows', type=int, default=50000)
    p.add_argument('--evaluation-rows', type=int, default=100000)
    p.add_argument('--sae-batch', type=int, default=512)
    p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    from adapters import Adapter, fit_axis_gpu
    torch.backends.cuda.matmul.allow_tf32 = False
    m = json.loads(pathlib.Path(a.manifest).read_text())
    job = next(j for j in m['jobs'] if j['group'] == 'pythia160m')
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(*x):
        print('[pythia-release]', *x, flush=True)

    tok = AutoTokenizer.from_pretrained(job['model'], revision=job['model_revision'])
    model = AutoModel.from_pretrained(job['model'], revision=job['model_revision'], torch_dtype=torch.float32,
                                      low_cpu_mem_usage=True).to(a.device).eval()
    blocks = get_blocks(model)
    cap = {}
    h = blocks[job['layer']].register_forward_hook(
        lambda mod, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))
    eos = tok.eos_token_id
    log('packing with eos_id=%s at context %d' % (eos, a.context))

    doc_index = {}

    def harvest(split, target):
        X = np.empty((target, model.config.hidden_size), dtype=np.float32)
        D = np.empty(target, dtype=np.int64)
        T = np.empty(target, dtype=np.int64)
        n = 0
        for seqs, owns in packed_batches(tok, pathlib.Path(a.corpus) / (split + '.jsonl'),
                                         a.context, eos, a.batch_size, target, doc_index):
            ids = torch.tensor(seqs, dtype=torch.long, device=a.device)
            cap.clear()
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)
            acts = cap['h'].reshape(-1, cap['h'].shape[-1]).float().cpu().numpy()
            take = min(len(acts), target - n)
            if not np.isfinite(acts[:take]).all():
                raise RuntimeError('Nonfinite activations')
            X[n:n + take] = acts[:take]
            D[n:n + take] = np.array(owns).reshape(-1)[:take]
            T[n:n + take] = ids.reshape(-1).cpu().numpy()[:take]
            n += take
            if n >= target:
                break
        if n < target:
            raise RuntimeError('%s: only %d/%d rows' % (split, n, target))
        return X, D, T

    started = time.time()
    xtr, dtr, ttr = harvest('train', a.calibration_rows)
    xte, dte, tte = harvest('test', a.evaluation_rows)
    h.remove()
    del model
    torch.cuda.empty_cache()
    log('harvested %d/%d rows in %.0fs; distinct eval source documents %d'
        % (len(xtr), len(xte), time.time() - started, len(np.unique(dte))))

    sep_tr, sep_te = (ttr == eos), (tte == eos)
    pops = {'all_packed': np.ones(len(xte), dtype=bool), 'no_separator': ~sep_te}
    pops_tr = {'all_packed': np.ones(len(xtr), dtype=bool), 'no_separator': ~sep_tr}

    adapter = Adapter(job, a.device)
    rows = []
    for pop in pops:
        idx = np.flatnonzero(pops_tr[pop])
        origin, v, info = fit_axis_gpu(xtr[idx], a.device)
        if not info['converged']:
            raise RuntimeError('PC1 convergence requires review: %s' % info)
        groups = {}
        l0 = [0.0, 0]
        for s in range(0, len(xte), a.sae_batch):
            e = min(s + a.sae_batch, len(xte))
            rec = adapter.reconstruct(xte[s:e])
            x, hh = rec['raw']
            keep = pops[pop][s:e]
            if not keep.any():
                continue
            l0[0] += float(rec['l0'][keep].sum())
            l0[1] += int(keep.sum())
            labs = dte[s:e]
            for d in np.unique(labs[keep]):
                ix = keep & (labs == d)
                st = sufficient_stats(x[ix], hh[ix], origin, v)
                prev = groups.get(int(d))
                groups[int(d)] = st if prev is None else merge_stats([prev, st], v)
        ids = sorted(groups)
        g = np.stack([groups[d] for d in ids])
        vals = summarize(merge_stats(g, v), v)
        if vals['identity_error'] is not None and vals['identity_error'] > 1e-6:
            raise RuntimeError('Energy partition identity failed for ' + pop)
        ci = article_bootstrap(g, v, draws=a.bootstrap)
        np.savez_compressed(out / ('pythia_release_style__%s.npz' % pop),
                            group_stats=g, axis=v, origin=origin, document_indices=ids)
        row = dict(vals, population=pop, space='raw', n_documents=len(ids), pc1_fit=info,
                   l0=l0[0] / max(l0[1], 1), separator_rows_in_eval=int(sep_te.sum()))
        for k in ('fvu_total', 'fvu_content', 'content_over_total', 'w1_eval'):
            if k in ci:
                row[k + '_lo95'], row[k + '_hi95'] = ci[k]
        rows.append(row)
        log('  %-14s n=%d docs=%d fvu_total=%.6f fvu_content=%.6f ratio=%.4f w1=%.6f'
            % (pop, row['n_rows'], row['n_documents'], row['fvu_total'], row['fvu_content'],
               row['content_over_total'], row['w1_eval']))

    result = dict(status='ok', kind='pythia_release_style_packed_evaluation', job=job,
                  context_length=a.context, packing='documents concatenated with an EOS separator, '
                  'cut into fixed-length sequences (sparsify@c259f271 sae/data.py); no BOS',
                  interpretation='A separate view on OUR held-out WikiText-103 sample, NOT a reproduction '
                                 'of the release evaluation; it tests whether the corrected Pythia ratio '
                                 'depends on independent context-128 chunking.',
                  calibration_rows=a.calibration_rows, evaluation_rows=a.evaluation_rows,
                  eos_token_id=int(eos), rows=rows, adapter_smoke=adapter.smoke(xte[:128]),
                  seconds=time.time() - started, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (out / 'pythia_release_style.json').write_text(json.dumps(result, indent=2, default=str))
    log('done in %.0fs' % (time.time() - started))


if __name__ == '__main__':
    main()
