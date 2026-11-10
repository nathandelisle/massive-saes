"""GPT-2 B1 companion geometry: PC1 alignment of calibration-alive atoms and cross-seed sharing by alignment band.

Inputs: the 40 bridge calibration files (decoder rows, frozen calibration-alive mask, alignment = cos^2(decoder, train PC1))
and BRIDGE_RESULTS per-seed top-50 selected ids (test split, model baseline, dCE).
Bands follow exp_b2_l1_phase.py (the Pythia sweep's own definitions): canonical cos^2 >= 0.3, tiling band 0.01 <= cos^2 < 0.3,
content cos^2 < 0.01. Sharing threshold 0.7 on the signed max cosine against another seed's calibration-alive atoms (as in the bridge).
"""
import json, itertools
import numpy as np
from pathlib import Path

S = Path(__file__).resolve().parents[3] / "results/additional/objectives/five_seed_beta/inputs"
BR = json.load(open(S / 'BRIDGE_RESULTS.json'))
sel = {r['run']: r['selected_ids'] for r in BR['per_seed']
       if r['split'] == 'test' and r['baseline'] == 'model' and r['metric'] == 'dce_sum' and r['top_n'] == 50 and r['threshold'] == 0.7}
seeds = [42, 137, 0, 1, 2]
out = dict(definition=__doc__.strip(), rows=[], aggregate=[])
for cond in ('standard', 'bypass'):
    for beta in (0.0, 0.03, 0.1, 0.3):
        cal = {}
        for s in seeds:
            name = f'{cond}_k64_s{s}_beta{beta:g}'
            with np.load(S / 'bcal' / f'{name}.npz') as n:
                d = n['decoder']; d = d / np.linalg.norm(d, axis=1, keepdims=True)
                cal[s] = dict(name=name, d=d, alive=n['alive'].copy(), align=n['alignment'].copy())
        best = {}
        for a, b in itertools.permutations(seeds, 2):
            A, Bc = cal[a], cal[b]
            ia = np.flatnonzero(A['alive']); ib = np.flatnonzero(Bc['alive'])
            c = A['d'][ia] @ Bc['d'][ib].T
            m = np.full(len(A['d']), np.nan); m[ia] = c.max(1)
            best[(a, b)] = m
        agg = dict(condition=cond, beta=beta)
        per = []
        for a in seeds:
            A = cal[a]; al = A['alive']; c2 = A['align']
            groups = dict(canonical=al & (c2 >= 0.3), band=al & (c2 >= 0.01) & (c2 < 0.3), content=al & (c2 < 0.01))
            others = [b for b in seeds if b != a]
            row = dict(condition=cond, beta=beta, seed=a, run=A['name'], n_alive=int(al.sum()),
                       max_cos2_pc1=float(c2[al].max()))
            for g, mask in groups.items():
                row[f'n_{g}'] = int(mask.sum())
                row[f'sharing_{g}'] = (float(np.mean([best[(a, b)][mask] >= 0.7 for b in others])) if mask.any() else None)
            top = np.asarray(sel[A['name']], int)
            row['top50_frac_canonical'] = float((c2[top] >= 0.3).mean())
            row['top50_frac_band'] = float(((c2[top] >= 0.01) & (c2[top] < 0.3)).mean())
            row['top50_frac_content'] = float((c2[top] < 0.01).mean())
            row['top50_mean_cos2_pc1'] = float(c2[top].mean())
            per.append(row)
        out['rows'] += per
        for k in per[0]:
            if k in ('condition', 'beta', 'seed', 'run'):
                continue
            vals = [r[k] for r in per if r[k] is not None]
            agg[k] = dict(mean=float(np.mean(vals)) if vals else None, sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None, n=len(vals))
        out['aggregate'].append(agg)
json.dump(out, open(S / 'gpt2_b1_geometry.json', 'w'), indent=1)
print('cond beta | alive | canonical n (sharing) | band n (sharing) | content n (sharing) | top50: frac canon / band / content | top50 mean cos2')
for a in out['aggregate']:
    f = lambda k, n=2: ('-' if a[k]['mean'] is None else f"{a[k]['mean']:.{n}f}")
    print(a['condition'], a['beta'], '|', f('n_alive', 0), '|', f('n_canonical', 1), f"({f('sharing_canonical', 3)})", '|', f('n_band', 1), f"({f('sharing_band', 3)})", '|',
          f('n_content', 0), f"({f('sharing_content', 3)})", '|', f('top50_frac_canonical', 3), '/', f('top50_frac_band', 3), '/', f('top50_frac_content', 3), '|', f('top50_mean_cos2_pc1', 4))
