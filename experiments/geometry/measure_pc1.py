"""E20: PC1 / norm / token-frequency association, from cached rows or scalar observations.

Uses a FROZEN training mean and PC1, never silently refits them on evaluation data.
Confidence intervals resample documents, conditional on this fixed direction and model.
See README.md for the input contract and the deliberately limited interpretation.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from science_common import unit, safe_corr, quantile_interval, write_json, provenance


def observations(x, mean, v1, batch=4096):
    v1 = unit(v1)
    mean = np.asarray(mean, dtype=np.float64)
    if x.ndim != 2 or x.shape[1:] != mean.shape or mean.shape != v1.shape or not np.isfinite(mean).all():
        raise ValueError('Activation, mean and PC1 dimensions must agree.')
    score, norm = np.empty(len(x)), np.empty(len(x))
    for i in range(0, len(x), batch):
        y = np.asarray(x[i:i+batch], dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite activations.')
        score[i:i+batch] = (y - mean) @ v1
        norm[i:i+batch] = np.linalg.norm(y, axis=1)
    return score, norm


def assoc(score, norm, freq):
    r = safe_corr(score, norm)
    rho = safe_corr(rankdata(abs(score)), rankdata(freq))
    return dict(norm_pearson_abs_r=abs(r) if r is not None else None,
                norm_squared_pearson_r=r*r if r is not None else None,
                abs_centered_pc1_vs_train_token_frequency_spearman=rho)


def analyze(score, norm, token_id, doc_id, position, train_counts, mean, v1,
            bootstrap=200, seed=20261007):
    v1 = unit(v1)
    score, norm = np.asarray(score, float), np.asarray(norm, float)
    arrays = [score, norm, token_id, doc_id, position]
    n = len(score)
    if n < 3 or any(np.asarray(a).shape != (n,) for a in arrays):
        raise ValueError('All observation arrays must have the same nontrivial length.')
    if not np.isfinite(score).all() or not np.isfinite(norm).all() or np.any(norm < 0):
        raise ValueError('Invalid scores / norms.')
    for name, a in [('token_id', token_id), ('doc_id', doc_id), ('position', position)]:
        if not np.issubdtype(np.asarray(a).dtype, np.integer) or np.any(np.asarray(a) < 0):
            raise ValueError(name + ' must be nonnegative integers.')
    counts = np.asarray(train_counts)
    if counts.ndim != 1 or np.any(counts < 0) or not np.isfinite(counts).all() or counts.sum() <= 0:
        raise ValueError('Training frequency counts are invalid.')
    if max(token_id) >= len(counts):
        raise ValueError('Token ids do not match the supplied training-count vocabulary.')
    freq = counts[token_id]
    mean = np.asarray(mean, dtype=float)
    if mean.shape != v1.shape or not np.isfinite(mean).all():
        raise ValueError('Mean / PC1 mismatch.')
    mu_norm = np.linalg.norm(mean)
    strata = dict(all=np.ones(n, bool), first_position=np.asarray(position) == 0,
                  nonfirst_position=np.asarray(position) > 0)
    result = dict(n_tokens=n, n_documents=int(len(np.unique(doc_id))),
                  training_count_total=int(counts.sum()),
                  evaluation_tokens_absent_from_training=int((freq == 0).sum()),
                  abs_cos_pc1_training_mean=float(abs(v1 @ mean) / mu_norm) if mu_norm else None,
                  strata={})
    for name, mask in strata.items():
        metrics = assoc(score[mask], norm[mask], freq[mask])
        row = dict(n_tokens=int(mask.sum()), n_documents=int(len(np.unique(np.asarray(doc_id)[mask]))),
                   metrics=metrics)
        docs = np.unique(np.asarray(doc_id)[mask])
        if bootstrap and len(docs) >= 2:
            # Resample whole articles. Recompute ranks with their bootstrap multiplicities.
            groups = [np.flatnonzero(mask & (np.asarray(doc_id) == d)) for d in docs]
            rng = np.random.default_rng(seed)
            draws = {k: [] for k in metrics}
            for _ in range(bootstrap):
                ix = np.concatenate([groups[i] for i in rng.integers(0, len(docs), len(docs))])
                vals = assoc(score[ix], norm[ix], freq[ix])
                for k, v in vals.items():
                    draws[k].append(v)
            row['document_bootstrap_95pct'] = {k: quantile_interval(v, bootstrap) for k, v in draws.items()}
        else:
            row['document_bootstrap_95pct'] = None
        result['strata'][name] = row
    result['interpretation'] = (
        'Descriptive association, token-weighted within the stated population. Squared Pearson '
        'correlation is NOT an outer-test regression R2 and does not establish absence of semantic '
        'information. CIs condition on the fitted PC1; first/nonfirst strata use the same PC1. '
        'Frequency is measured in the supplied training corpus, never estimated from test tokens.')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--activations', help='N x d .npy, raw uncentered valid rows')
    mode.add_argument('--observations', help='.npz with score, norm and token metadata from collect_pythia.py')
    ap.add_argument('--row-meta', help='.npz token_id/doc_id/position; required with activations')
    ap.add_argument('--geometry', required=True, help='.npz mean and v1, canonical TRAINING geometry')
    ap.add_argument('--train-counts', required=True, help='vocabulary-length .npy')
    ap.add_argument('--population', required=True, help='Exact corpus/partition, layer, model and cohort description')
    ap.add_argument('--bootstrap', type=int, default=200)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.bootstrap < 0:
        ap.error('bootstrap must be nonnegative')
    geom = np.load(a.geometry, allow_pickle=False)
    mean, v1 = geom['mean'], geom['v1']
    paths = [a.geometry, a.train_counts]
    if a.activations:
        if not a.row_meta:
            ap.error('--row-meta is required with --activations')
        x = np.load(a.activations, mmap_mode='r', allow_pickle=False)
        score, norm = observations(x, mean, v1)
        meta = np.load(a.row_meta, allow_pickle=False)
        paths += [a.activations, a.row_meta]
    else:
        meta = np.load(a.observations, allow_pickle=False)
        score, norm = meta['score'], meta['norm']
        if str(meta['geometry_sha256'].item()) != provenance([a.geometry])['files'][0]['sha256']:
            raise ValueError('Observation cache was made with a different geometry file.')
        paths += [a.observations]
    result = analyze(score, norm, meta['token_id'], meta['doc_id'], meta['position'],
                     np.load(a.train_counts, allow_pickle=False), mean, v1, a.bootstrap)
    result.update(population=a.population, provenance=provenance(paths), experiment='E20_remeasurement')
    write_json(a.out, result)
    print('Wrote', a.out)


if __name__ == '__main__':
    main()
