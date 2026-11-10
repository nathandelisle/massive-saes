"""Paired initialization-level summaries; never count seed pairs as replicates.
Reads a JSON registry of E2/E3 analysis summaries or E4 evaluation summaries.
Reports all conditions and missing seeds, without pooling different model sites.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from common import write_json, start_run, finish_run


def paired_summary(a, b, expected, rng, bootstrap=2000):
    ids = sorted(set(a) & set(b)); missing = sorted(set(expected)-set(ids))
    delta = np.array([b[s]-a[s] for s in ids], float)
    result = dict(paired_seeds=ids, missing_seeds=missing, n_pairs=len(ids),
                  complete=not missing, differences=delta.tolist(),
                  mean_difference=float(delta.mean()) if len(delta) else None,
                  ci95=None, interpretation='B minus A; seed uncertainty conditional on this evaluation corpus')
    if len(delta) >= 3:
        draws = delta[rng.integers(len(delta), size=(bootstrap, len(delta)))].mean(1)
        result['ci95'] = np.quantile(draws, [.025, .975]).tolist()
    return result


def dyadic_summary(pairs, metric):
    """Average directed edges, then jackknife vertices (initialization seeds)."""
    edges = {}
    for a, b, value in pairs:
        if a == b or value is None or not np.isfinite(value): continue
        edges.setdefault(tuple(sorted((a,b))), []).append(float(value))
    edges = {k:float(np.mean(v)) for k,v in edges.items()}
    seeds = sorted(set(itertools.chain.from_iterable(edges)))
    missing = [list(p) for p in itertools.combinations(seeds, 2) if p not in edges]
    mean = float(np.mean(list(edges.values()))) if edges else None
    se = None
    if len(seeds) >= 3 and not missing:
        leave = np.array([np.mean([v for (a,b),v in edges.items() if s not in (a,b)]) for s in seeds])
        se = float(np.sqrt((len(seeds)-1)/len(seeds)*((leave-leave.mean())**2).sum()))
    return dict(metric=metric, mean=mean, n_initializations=len(seeds),
                n_undirected_pairs=len(edges), missing_pairs=missing, seed_jackknife_se=se,
                note='Directed pairs share endpoints. Their count is not the independent sample size.')


def extract_metrics(r):
    if 'fvu' in r:
        return dict(fvu_total=r['fvu']['fvu_total'], fvu_perp=r['fvu']['fvu_perp'], l0=r['l0'])
    result = {k:r[k] for k in ['fvu_total','fvu_perp','l0']}
    if r.get('truth'):
        result['truth_recall'] = r['truth']['one_to_one_recall_cos07']
        for label, values in r['truth']['coefficient_fits'].items():
            result['truth_r2_'+label] = values['mean_r2']
    return result


def valid_number(x):
    return x is not None and np.isfinite(x)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registry', required=True)
    p.add_argument('--out', required=True); p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args(); registry = json.loads(Path(a.registry).read_text())
    inputs = [a.registry]+[c['summary'] for c in registry['cases']]
    out, meta = start_run(a.out, vars(a), inputs); rng = np.random.default_rng(1729); reports = []
    seen = set()
    for case in registry['cases']:
        identifier = (case['site'], case.get('replicate', 'main'))
        if identifier in seen: raise ValueError('Repeated site/replicate would duplicate evidence')
        seen.add(identifier); data = json.loads(Path(case['summary']).read_text())
        expected = case.get('expected_seeds', registry.get('expected_seeds', [42,137,0,1,2]))
        buckets = {}; names = {}
        for r in data['runs']:
            key = (r['condition'], r.get('k', 'fixed'), r.get('beta', 0.))
            if r['seed'] in buckets.get(key, {}): raise ValueError('Duplicate initialization within condition')
            buckets.setdefault(key, {})[r['seed']] = extract_metrics(r)
            names[r['name']] = r
        conditions = []
        for key, runs in sorted(buckets.items(), key=str):
            conditions.append(dict(condition=key[0], k=key[1], beta=key[2], seeds=sorted(runs),
                missing_seeds=sorted(set(expected)-set(runs)),
                metrics={m:dict(mean=None,
                    per_seed={str(s):r[m] for s,r in runs.items()}) for m in next(iter(runs.values()))}))
            for m, report in conditions[-1]['metrics'].items():
                valid=[r[m] for r in runs.values() if valid_number(r[m])]
                report['mean']=float(np.mean(valid)) if valid else None
                report['invalid_seeds']=[s for s,r in runs.items() if not valid_number(r[m])]
        contrasts = []
        for key, b in sorted(buckets.items(), key=str):
            candidates = []
            if key[0] == 'bypass': candidates.append(('bypass_vs_standard', ('standard',key[1],key[2])))
            if key[0]=='bypass' and isinstance(key[1],int) and ('standard',key[1]+1,key[2]) in buckets:
                candidates.append(('bypass_vs_standard_equal_total_budget',('standard',key[1]+1,key[2])))
            if key[2] != 0: candidates.append(('penalty_vs_zero', (key[0],key[1],0.)))
            for label, baseline in candidates:
                ref = buckets.get(baseline, {})
                metrics = set.intersection(*(set(r) for r in [*b.values(), *ref.values()])) if ref else set(next(iter(b.values())))
                contrasts.append(dict(comparison=label, A=baseline, B=key, metrics={m:paired_summary(
                    {s:r[m] for s,r in ref.items() if valid_number(r[m])}, {s:r[m] for s,r in b.items() if valid_number(r[m])}, expected, rng, a.bootstrap)
                    for m in sorted(metrics)}))
                raw={(r['condition'],r.get('k','fixed'),r.get('beta',0.),r['seed']):r for r in data['runs']}
                eligibility=[]
                for seed in expected:
                    ra=raw.get((*baseline,seed),{});rb=raw.get((*key,seed),{})
                    fields=[ra.get('calibration_l0'),rb.get('calibration_l0'),ra.get('calibration_fvu_perp'),rb.get('calibration_fvu_perp')]
                    if not all(valid_number(v) for v in fields):
                        eligibility.append(dict(seed=seed,matched=None,reason='missing calibration metrics'));continue
                    la,lb,fa,fb=fields
                    if label=='bypass_vs_standard_equal_total_budget':
                        la+=int(baseline[0]=='bypass');lb+=int(key[0]=='bypass')
                    eligibility.append(dict(seed=seed,matched=abs(la-lb)<=.05*max(la,1e-12) and abs(fa-fb)<=.05*max(fa,1e-12),
                        noninferior=lb<=1.05*la and fb<=1.05*fa))
                contrasts[-1]['calibration_tradeoff_check']=eligibility
        pair_reports = []
        for pair_key in sorted(buckets,key=str):
            for metric in ['group_normalized_mse','group_matched_atom_nmse_mean','aggregate_to_atom_error_ratio','raw_sharing','group_raw_sharing','projected_sharing','group_projected_sharing','median_activation_corr']:
                pairs = [(names[p['reference']]['seed'], names[p['comparison']]['seed'], p.get(metric))
                    for p in data.get('pairs',[]) if (names[p['reference']]['condition'],
                        names[p['reference']].get('k','fixed'),names[p['reference']].get('beta',0.)) == pair_key]
                if pairs: pair_reports.append(dict(condition=pair_key[0],k=pair_key[1],beta=pair_key[2],**dyadic_summary(pairs, metric)))
        reports.append(dict(site=case['site'], replicate=case.get('replicate','main'),
            expected_seeds=expected, conditions=conditions, contrasts=contrasts, pairs=pair_reports))
        print(f'[pc1-replication] {identifier} conditions={len(conditions)} contrasts={len(contrasts)}', flush=True)
    write_json(out/'summary.json', dict(cases=reports,
        scope='Model sites and generated corpora remain separate. These intervals quantify initialization variation, not independent model-family evidence.'))
    finish_run(out, meta)


if __name__ == '__main__': main()
