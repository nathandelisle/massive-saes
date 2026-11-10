"""CPU recalculations for E09/E23/E26: sharing, post-hoc selection, dense budgets, subspaces.

Inputs are explicit .npz exports of existing dictionaries/results, not guessed .pt formats.
Decoder rows are features; importance is FULL LENGTH with an explicit measured mask.
Pair metadata must agree on cohort, model, layer, preprocessing, data and estimator.
Results remain per ordered seed pair; this script does not pretend pairs are independent.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from science_common import unit, write_json, provenance, sha256


def normalized_rows(d):
    d = np.asarray(d, dtype=np.float64)
    if d.ndim != 2 or not np.isfinite(d).all():
        raise ValueError('decoder must be a finite [features, hidden] matrix')
    n = np.linalg.norm(d, axis=1)
    if np.any(n == 0):
        raise ValueError('Zero decoder atom: resolve its alive status, do not silently match it.')
    return d / n[:, None]


def max_cosines(a, b, chunk=256):
    a, b = normalized_rows(a), normalized_rows(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError('Decoder hidden widths differ.')
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        out[i:i+chunk] = np.max(a[i:i+chunk] @ b.T, axis=1)
    return np.clip(out, -1, 1)


def basis(rows, rtol=1e-8):
    rows = np.asarray(rows, dtype=np.float64)
    if not 0 < rtol < 1:
        raise ValueError('rank_rtol must lie in (0,1)')
    if not len(rows):
        return np.zeros((rows.shape[1], 0)), np.empty(0)
    _, s, vt = np.linalg.svd(rows, full_matrices=False)
    rank = int(np.sum(s > s[0] * rtol)) if s[0] else 0
    return vt[:rank].T, s


def overlap(rows_a, rows_b, rtol=1e-8):
    if rows_a.shape[1] != rows_b.shape[1]:
        raise ValueError('Subspaces must have the same ambient dimension.')
    qa, sa = basis(rows_a, rtol)
    qb, sb = basis(rows_b, rtol)
    ra, rb, d = qa.shape[1], qb.shape[1], qa.shape[0]
    if min(ra, rb) == 0:
        return dict(rank_a=ra, rank_b=rb, status='empty_subspace')
    cs2 = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1)**2
    trace = float(cs2.sum())
    null_mean = max(ra, rb) / d
    return dict(status='measured', rank_a=ra, rank_b=rb, ambient_dimension=d,
                mean_cos2=float(cs2.mean()), principal_cos2=cs2,
                fraction_a_captured_by_b=trace/ra, fraction_b_captured_by_a=trace/rb,
                haar_random_mean_cos2_expectation=null_mean,
                excess_above_random_expectation=float(cs2.mean()-null_mean),
                guaranteed_intersection_dimension=max(0, ra+rb-d), rank_rtol=rtol,
                singular_values_a=sa, singular_values_b=sb,
                note='Exact Haar-null expectation, NOT a simulated quantile or p-value. '
                     'Mean cos2=1 can mean containment when ranks differ; it need not mean identical subspaces.')


def top_ids(importance, allowed, k):
    ids = np.flatnonzero(allowed)
    # Deterministic tie break by integer feature id.
    return ids[np.lexsort((ids, -importance[ids]))][:k]


def sharing(importance, alive, measured, cosines, alignment, threshold=.7,
            dense_cut=.01, top_ks=(10,50,100,200,500), dense_inclusive=True):
    imp = np.asarray(importance, float)
    alive, measured = np.asarray(alive), np.asarray(measured)
    if alive.dtype != bool or measured.dtype != bool:
        raise ValueError('alive / importance_measured must be boolean masks, not id lists.')
    if any(np.asarray(x).shape != imp.shape for x in [alive, measured, cosines, alignment]) or imp.ndim != 1:
        raise ValueError('Per-feature arrays must have the full dictionary length.')
    if not np.isfinite(imp[measured]).all() or not np.isfinite(cosines).all():
        raise ValueError('Nonfinite measured importance or cosine.')
    if np.any((alignment < 0) | (alignment > 1+1e-6)):
        raise ValueError('Invalid cos-squared alignment')
    if not alive.any():
        raise ValueError('No alive features.')
    shared = cosines >= threshold
    scope = alive & measured
    baseline = float(shared[alive].mean())
    result = dict(n_alive=int(alive.sum()), n_measured_alive=int(scope.sum()),
                  exhaustive_alive_ranking=bool(measured[alive].all()),
                  matching_threshold=threshold, comparison='signed cosine >= threshold, all target atoms',
                  dictionary_sharing=baseline,
                  measured_universe_sharing=float(shared[scope].mean()) if scope.any() else None,
                  top={}, posthoc={})
    retain = alignment < dense_cut if dense_inclusive else alignment <= dense_cut
    for name, allowed in [('top', scope), ('posthoc', scope & retain)]:
        for k in top_ks:
            ix = top_ids(imp, allowed, k)
            if len(ix) != k:
                result[name][str(k)] = dict(status='insufficient_features', available=int(allowed.sum()))
                continue
            rate = float(shared[ix].mean())
            result[name][str(k)] = dict(status='measured', feature_ids=ix, sharing=rate,
                enrichment_over_alive_baseline=rate/baseline if baseline > 0 else None,
                cutoff_tie_count=int(np.sum(imp[allowed] == imp[ix[-1]])))
    result['posthoc_exclusion'] = dict(cutoff=dense_cut, comparison='>=' if dense_inclusive else '>')
    return result


def dense_budget(importance, alive, measured, alignment, cutoff, d):
    alive, measured = np.asarray(alive, bool), np.asarray(measured, bool)
    selected = alive & (alignment > cutoff)
    valid = alive & measured
    n = int(alive.sum())
    if n == 0:
        raise ValueError('No alive features')
    imp = np.asarray(importance, float)
    if not np.isfinite(imp[valid]).all():
        raise ValueError('Nonfinite measured importance')
    signed_total = float(imp[valid].sum())
    pos_total = float(np.maximum(imp[valid], 0).sum())
    signed_selected = float(imp[selected & measured].sum())
    positive_selected = float(np.maximum(imp[selected & measured], 0).sum())
    fraction = int(selected.sum()) / n
    return dict(n_alive=n, n_selected=int(selected.sum()), fraction_alive=fraction,
        strict_alignment_cutoff=cutoff, exhaustive_importance=bool(measured[alive].all()),
        measured_alive_count=int(valid.sum()), selected_measured_count=int((selected & measured).sum()),
        signed_importance_total=signed_total, signed_importance_selected=signed_selected,
        signed_importance_share=signed_selected/signed_total if signed_total > 0 else None,
        positive_part_importance_share=positive_selected/pos_total if pos_total > 0 else None,
        signed_share_over_dictionary_fraction=(signed_selected/signed_total)/fraction if signed_total>0 and fraction>0 else None,
        random_unit_direction_exceedance_at_observed_cutoff=float(beta.sf(cutoff,.5,(d-1)/2)),
        note='Importance shares use only measured features unless exhaustive_importance is true. '
             'Positive-part share is a separately labelled sensitivity, not a substitute for signed CE importance. '
             'The spherical random-direction reference is theoretical and does not model alive-selection bias.')


PAIR_KEYS = ['cohort_id','model_id','block_index','condition','mean_sha256','pc1_sha256','geometry_sha256',
             'ranking_data_id','importance_protocol_id','alive_rule']


def check_metadata(ma, mb):
    for k in PAIR_KEYS + ['seed','checkpoint_sha256']:
        if k not in ma or k not in mb or ma[k] is None or mb[k] is None:
            raise ValueError('Both sidecars require '+k)
    for k in PAIR_KEYS:
        if ma[k] != mb[k]:
            raise ValueError('Do not combine mismatched cohorts: '+k)
    if ma['seed'] == mb['seed']:
        raise ValueError('Cross-seed statistics require distinct seeds')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--a', required=True, help='Source .npz decoder/alive/importance/importance_measured')
    ap.add_argument('--b', required=True, help='Target .npz decoder/alive')
    ap.add_argument('--a-meta', required=True)
    ap.add_argument('--b-meta', required=True)
    ap.add_argument('--geometry', required=True, help='.npz v1; optional full evecs for PCA-subspace comparison')
    ap.add_argument('--threshold', type=float, default=.7)
    ap.add_argument('--rank-rtol', type=float, default=1e-8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if not -1 <= a.threshold <= 1:
        ap.error('Cosine threshold outside [-1,1]')
    ma, mb = [json.loads(Path(p).read_text()) for p in [a.a_meta, a.b_meta]]
    check_metadata(ma, mb)
    if ma['geometry_sha256'] != sha256(a.geometry):
        raise ValueError('Sidecar geometry hash does not match --geometry.')
    za, zb, g = [np.load(p, allow_pickle=False) for p in [a.a, a.b, a.geometry]]
    da, db = normalized_rows(za['decoder']), normalized_rows(zb['decoder'])
    v1 = unit(g['v1'])
    aa, ab = np.clip((da @ v1)**2,0,1), np.clip((db @ v1)**2,0,1)
    alive_a, alive_b = za['alive'], zb['alive']
    if alive_b.dtype != bool or alive_b.shape != (len(db),) or not alive_b.any():
        raise ValueError('Invalid target alive mask')
    cos = max_cosines(da, db)
    report = sharing(za['importance'], alive_a, za['importance_measured'], cos, aa, a.threshold)
    cut_a, cut_b = 5*float(np.median(aa[alive_a])), 5*float(np.median(ab[alive_b]))
    ca, cb = alive_a & (aa>cut_a), alive_b & (ab>cut_b)
    report['subspace_5x_median'] = overlap(da[ca], db[cb], a.rank_rtol)
    report['subspace_5x_median'].update(n_features_a=int(ca.sum()), n_features_b=int(cb.sum()),
                                       threshold_a=cut_a, threshold_b=cut_b)
    report['dense_budget_5x_median'] = dense_budget(za['importance'],alive_a,za['importance_measured'],aa,cut_a,len(v1))
    report['posthoc_5x_median'] = sharing(za['importance'], alive_a, za['importance_measured'], cos, aa,
                                          a.threshold, dense_cut=cut_a, dense_inclusive=False)['posthoc']
    # Historical 3x..50x sensitivity, calculated on precisely this same dictionary pair.
    report['posthoc_threshold_sweep'] = {}
    for mult in [3,5,10,15,20,50]:
        cut = mult*float(np.median(aa[alive_a]))
        report['posthoc_threshold_sweep'][str(mult)] = sharing(za['importance'], alive_a,
            za['importance_measured'], cos, aa, a.threshold, dense_cut=cut, top_ks=(50,), dense_inclusive=False)['posthoc']['50']
    report['rank_tolerance_sensitivity'] = {str(t): overlap(da[ca],db[cb],t)
                                           for t in [1e-6,1e-4]}
    if 'evecs' in g:
        e = np.asarray(g['evecs'], float)
        if e.ndim != 2 or e.shape[0] != len(v1) or not np.allclose(e.T@e,np.eye(e.shape[1]),atol=1e-4):
            raise ValueError('evecs must have orthonormal columns, sorted in descending eigenvalue order')
        report['pca_subspace'] = {}
        for name, rows in [('a',da[ca]),('b',db[cb])]:
            rank = basis(rows,a.rank_rtol)[0].shape[1]
            report['pca_subspace'][name] = (overlap(rows,e[:,:rank].T,a.rank_rtol)
                if rank and e.shape[1]>=rank else dict(status='insufficient_saved_eigenvectors', required=rank, available=e.shape[1]))
    report.update(source_meta=ma, target_meta=mb, provenance=provenance([a.a,a.b,a.a_meta,a.b_meta,a.geometry]),
                  interpretation='Geometric overlap is not a causal identifiability proof; feature sets and ranks are explicit.')
    write_json(a.out, report)
    print('Wrote',a.out)


if __name__ == '__main__':
    main()
