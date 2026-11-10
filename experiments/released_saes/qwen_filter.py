"""Evaluate Qwen Scope at 5x, 10x, and 20x the median calibration activation norm.

Qwen Scope describes filtering large norms without specifying the multiplier
or batch definition. These experiments use fixed calibration thresholds.
PC1 is fitted separately on the calibration activations retained at each cutoff.

Error and variance are also reported for first positions, other rejected
positions, and retained positions, using one common calibration PC1.

Usage: qwen_filter.py --group qwen3_8b --cache cache --out results/qwen_filter
"""
import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

from metrics import article_bootstrap, merge_stats, sufficient_stats, summarize

MULTIPLES = (5.0, 10.0, 20.0)


def atomic_json(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str))
    tmp.replace(path)


def row_norms(X, batch=8192):
    out = np.empty(len(X), dtype=np.float64)
    for s in range(0, len(X), batch):
        out[s:s + batch] = np.linalg.norm(np.asarray(X[s:s + batch], dtype=np.float64), axis=1)
    return out


def build_masks(folder, norms, thresholds):
    special = np.load(folder / 'special.npy')
    pos = np.load(folder / 'pos.npy')
    valid = ~special
    pops = {'all': np.ones(len(pos), dtype=bool), 'no_special': valid, 'after_first': pos > 0}
    for m, thr in thresholds.items():
        keep = valid & (norms <= thr)
        pops['norm%g' % m] = keep
        pops['norm%g_after_first' % m] = keep & (pos > 0)
    return pops, special, pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='manifest.json')
    p.add_argument('--group', default='qwen3_8b')
    p.add_argument('--job-id')
    p.add_argument('--cache', default='cache')
    p.add_argument('--out', default='results/qwen_filter')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--context', type=int)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--deadline-unix', type=float)
    a = p.parse_args()
    import torch
    from adapters import Adapter, fit_axis_gpu
    torch.backends.cuda.matmul.allow_tf32 = False
    m = json.loads(pathlib.Path(a.manifest).read_text())
    jobs = [j for j in m['jobs'] if j['group'] == a.group and (not a.job_id or j['job_id'] == a.job_id)]
    ctx = a.context or m['context_length']
    root = pathlib.Path(a.cache) / a.group / ('ctx%d' % ctx)
    meta = json.loads((root / 'metadata.json').read_text())
    if not meta['complete']:
        raise RuntimeError('Activation cache incomplete')
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    code_sha = hashlib.sha256((pathlib.Path(__file__).read_bytes()
                               + (pathlib.Path(__file__).parent / 'metrics.py').read_bytes())).hexdigest()

    def log(*x):
        print('[qwenfilter]', *x, flush=True)

    norms_cache = {}
    for job in jobs:
        if a.deadline_unix and time.time() > a.deadline_unix:
            log('deadline reached; stopping')
            break
        started = time.time()
        dest = out / ('%s__ctx%d.json' % (job['job_id'], ctx))
        if dest.exists():
            log('already done', dest.name)
            continue
        layer = job['layer']
        xtr = np.load(root / 'train' / ('layer_%d.npy' % layer), mmap_mode='r')
        xte = np.load(root / 'test' / ('layer_%d.npy' % layer), mmap_mode='r')
        if layer not in norms_cache:
            norms_cache[layer] = (row_norms(xtr), row_norms(xte))
        ntr, nte = norms_cache[layer]

        # ---- thresholds: frozen from CALIBRATION rows only, before any test-side evaluation
        tr_valid = ~np.load(root / 'train' / 'special.npy')
        med = float(np.median(ntr[tr_valid]))
        thresholds = {mm: mm * med for mm in MULTIPLES}
        log('%s layer %d: calibration median raw L2 norm %.4f -> thresholds %s'
            % (job['job_id'], layer, med, {k: round(v, 3) for k, v in thresholds.items()}))

        tr_pops, tr_special, tr_pos = build_masks(root / 'train', ntr, thresholds)
        te_pops, te_special, te_pos = build_masks(root / 'test', nte, thresholds)
        doc = np.load(root / 'test' / 'doc.npy')

        adapter = Adapter(job, a.device)
        if xtr.shape[1] != adapter.sae.W_dec.shape[1]:
            raise RuntimeError('Hook dimension mismatch')
        if adapter.norm != 'none':
            raise RuntimeError('Qwen Scope is expected to need no runtime normalization; got ' + adapter.norm)

        axes = {}
        retention = {}
        for pop, mask in tr_pops.items():
            idx = np.flatnonzero(mask)
            if len(idx) < 1000:
                raise RuntimeError('Population %s retains only %d calibration rows' % (pop, len(idx)))
            origin, v, info = fit_axis_gpu(np.asarray(xtr[idx]), a.device)
            if not info['converged']:
                raise RuntimeError('PC1 convergence requires review: %s' % info)
            axes[pop] = (origin, v, info)
            retention[pop] = dict(
                calibration_rows=int(len(idx)), calibration_fraction=float(len(idx) / len(mask)),
                evaluation_rows=int(te_pops[pop].sum()),
                evaluation_fraction=float(te_pops[pop].sum() / len(te_pops[pop])),
                evaluation_documents=int(len(np.unique(doc[te_pops[pop]]))))
        log('  fitted %d calibration axes' % len(axes))

        # ---- common-axis attribution classes (defined on the 10x rule, the package's documented default)
        thr10 = thresholds[10.0]
        cls = np.where(te_special, 'special',
                       np.where(te_pos == 0, 'first_position',
                                np.where(nte > thr10, 'other_rejected', 'retained')))
        common_origin, common_v, common_info = axes['no_special']
        attrib = {c: np.zeros(5) for c in ('special', 'first_position', 'other_rejected', 'retained')}

        groups = {pop: {} for pop in te_pops}
        l0s = {pop: [0.0, 0] for pop in te_pops}
        alive = np.zeros(adapter.n_features, dtype=bool)
        for start in range(0, len(xte), a.batch_size):
            if a.deadline_unix and time.time() > a.deadline_unix:
                raise TimeoutError('deadline reached mid-checkpoint')
            stop = min(start + a.batch_size, len(xte))
            recon = adapter.reconstruct(xte[start:stop])
            alive |= recon['alive']
            x, h = recon['raw']
            labs = doc[start:stop]
            for pop, mask in te_pops.items():
                keep = mask[start:stop]
                if not keep.any():
                    continue
                l0s[pop][0] += float(recon['l0'][keep].sum())
                l0s[pop][1] += int(keep.sum())
                origin, v, _ = axes[pop]
                for d in np.unique(labs[keep]):
                    ix = keep & (labs == d)
                    st = sufficient_stats(x[ix], h[ix], origin, v)
                    prev = groups[pop].get(int(d))
                    groups[pop][int(d)] = st if prev is None else merge_stats([prev, st], v)
            # attribution under ONE fixed axis, so the classes live on a common scale
            c = x - common_origin
            pp = c @ common_v
            uu = c - pp[:, None] * common_v
            r = x - h
            rp = r @ common_v
            ru = r - rp[:, None] * common_v
            for name in attrib:
                ix = cls[start:stop] == name
                if not ix.any():
                    continue
                attrib[name] += np.array([ix.sum(), float(pp[ix] @ pp[ix]), float(np.sum(uu[ix] * uu[ix])),
                                          float(rp[ix] @ rp[ix]), float(np.sum(ru[ix] * ru[ix]))])

        rows = []
        for pop in te_pops:
            origin, v, info = axes[pop]
            ids = sorted(groups[pop])
            g = np.stack([groups[pop][d] for d in ids])
            st = merge_stats(g, v)
            values = summarize(st, v)
            if values['identity_error'] is not None and values['identity_error'] > 1e-6:
                raise RuntimeError('Energy partition identity failed for ' + pop)
            ci = article_bootstrap(g, v, draws=a.bootstrap)
            npz = out / ('%s__ctx%d__%s.npz' % (job['job_id'], ctx, pop))
            np.savez_compressed(npz, group_stats=g, axis=v, origin=origin, document_indices=ids)
            row = dict(values, population=pop, space='raw', n_documents=len(ids), pc1_fit=info,
                       l0=l0s[pop][0] / max(l0s[pop][1], 1), statistics_file=npz.name, **retention[pop])
            row['norm_threshold_absolute'] = float(thresholds[float(pop[4:].split('_')[0])]) if pop.startswith('norm') else None
            row['norm_threshold_multiple'] = float(pop[4:].split('_')[0]) if pop.startswith('norm') else None
            for k in ('fvu_total', 'fvu_content', 'content_over_total', 'w1_eval'):
                if k in ci:
                    row[k + '_lo95'], row[k + '_hi95'] = ci[k]
            rows.append(row)
            log('  %-22s n=%7d ret=%.4f fvu_total=%.4f fvu_content=%.4f ratio=%s w1=%.4f'
                % (pop, row['n_rows'], row['evaluation_fraction'], row['fvu_total'], row['fvu_content'],
                   ('%.3f' % row['content_over_total']) if row['content_over_total'] else 'na', row['w1_eval']))

        total_var = sum(attrib[c][1] + attrib[c][2] for c in attrib)
        total_err = sum(attrib[c][3] + attrib[c][4] for c in attrib)
        attribution = {c: dict(n_rows=int(attrib[c][0]),
                               variance_pc1=float(attrib[c][1]), variance_content=float(attrib[c][2]),
                               error_pc1=float(attrib[c][3]), error_content=float(attrib[c][4]),
                               share_of_total_variance=float((attrib[c][1] + attrib[c][2]) / total_var) if total_var else None,
                               share_of_total_error=float((attrib[c][3] + attrib[c][4]) / total_err) if total_err else None)
                       for c in attrib}
        result = dict(status='ok', kind='qwen_norm_filter_sensitivity', job=job, context_length=ctx,
                      cache_metadata=meta, code_sha256=code_sha,
                      calibration_median_raw_l2_norm=med, thresholds_absolute={str(k): v for k, v in thresholds.items()},
                      threshold_rule=('ABSOLUTE, frozen from calibration rows before any filtered evaluation; '
                                      'a standardized approximation, NOT the release batch-relative rule'),
                      exact_release_filter='Release multiplier and batch definition unspecified.',
                      rows=rows, attribution_common_axis=attribution,
                      attribution_axis=dict(population='no_special', **common_info),
                      n_alive_any_eval_token=int(alive.sum()), n_features=adapter.n_features,
                      adapter_smoke=adapter.smoke(np.asarray(xte[:128])),
                      seconds=time.time() - started,
                      utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        atomic_json(dest, result)
        log('wrote %s in %.0fs' % (dest.name, time.time() - started))
        del adapter
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
