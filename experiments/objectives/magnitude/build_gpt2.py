"""Assemble the verified GPT-2 B1 sweep from per-run sources. Writes gpt2_b1_sweep.json."""
import json, math, collections
import numpy as np
from pathlib import Path

S = Path(__file__).resolve().parents[3] / "results/additional/objectives/five_seed_beta/inputs"
BR = json.load(open(S / 'BRIDGE_RESULTS.json'))
CFG = json.load(open(S / 'cfg17.json'))
T975_4 = 2.7764451051977987   # Student t, 4 df

def seed_stats(vals):
    a = np.asarray(vals, float)
    sd = float(a.std(ddof=1)) if len(a) > 1 else None
    half = T975_4 * sd / math.sqrt(len(a)) if sd is not None and len(a) == 5 else None
    return dict(n=len(a), mean=float(a.mean()), sd=sd, min=float(a.min()), max=float(a.max()),
                t95=[float(a.mean() - half), float(a.mean() + half)] if half is not None else None)

fid = {(f['condition'], float(f['beta']), f['seed'], f['split']): f for f in BR['fidelity']}
ps = {(r['run'], r['split']): r for r in BR['per_seed']
      if r['baseline'] == 'model' and r['metric'] == 'dce_sum' and r['top_n'] == 50 and r['threshold'] == 0.7}
summ = {(s['condition'], float(s['beta']), s['split']): s for s in BR['summaries']
        if s['baseline'] == 'model' and s['metric'] == 'dce_sum' and s['top_n'] == 50 and s['threshold'] == 0.7}
runs = {r['name']: r for r in CFG['runs']}
seeds = [42, 137, 0, 1, 2]
betas = [0.0, 0.03, 0.1, 0.3]

per_seed = []
for cond in ('standard', 'bypass'):
    for beta in betas:
        for seed in seeds:
            name = f'{cond}_k64_s{seed}_beta{beta:g}'
            e4 = json.load(open(S / 'e4cells' / f'{name}.json'))
            with np.load(S / 'bcal' / f'{name}.npz') as c:
                frozen = int(c['alive'].sum()); fresh = int(c['fresh_alive'].sum()); grp = int(len(c['group']))
                cal_tokens = int(c['n_tokens'])
            row = dict(condition=cond, beta=beta, seed=seed, run=name,
                       checkpoint=f"s3://ssup-research-079415246777/e4real-gpt2/results/e4_real/gpt2/full/training/{name}/sae.pt",
                       checkpoint_sha256=runs[name]['checkpoint_sha256'])
            for split in ('test', 'selection'):
                f = fid[(cond, beta, seed, split)]; p = ps[(name, split)]
                row[split] = dict(n_tokens=f['n_tokens'], top50_pairwise_sharing=p['sharing']['pairwise'],
                                  top50_any_other_seed=p['sharing']['any_other_seed'],
                                  top50_no_other_seed_count=p['sharing']['no_other_seed_count'],
                                  dictwide_pairwise_sharing=p['dictwide']['pairwise'],
                                  l0=f['l0'], fvu_total=f['fvu_total'], fvu_content=f['fvu_perp'], fvu_pc1=f['fvu_pc1'], w1=f['w1'],
                                  splice_dce=f['splice_dce'])
            row['alive_frozen_calibration'] = frozen
            row['alive_fresh_bridge_calibration'] = fresh
            row['d_sae'] = 3072
            row['structural_group_size'] = grp
            row['calibration_tokens'] = cal_tokens
            row['grid_evaluator'] = dict(test_l0=e4['l0'], test_fvu_content=e4['fvu_perp'], test_fvu_total=e4['fvu_total'],
                                         calibration_l0=e4['calibration_l0'], calibration_fvu_content=e4['calibration_fvu_perp'],
                                         test_mean_l1=e4['mean_l1'], n_alive_calibration=e4['n_alive_calibration'])
            per_seed.append(row)

def get(r, path):
    for k in path.split('.'):
        r = r[k]
    return r

aggregate = []
metrics = ['test.top50_pairwise_sharing', 'selection.top50_pairwise_sharing', 'test.dictwide_pairwise_sharing',
           'test.l0', 'selection.l0', 'test.fvu_total', 'test.fvu_content', 'selection.fvu_content', 'test.fvu_pc1', 'test.w1',
           'alive_frozen_calibration', 'alive_fresh_bridge_calibration',
           'grid_evaluator.test_l0', 'grid_evaluator.test_fvu_content', 'grid_evaluator.test_mean_l1']
for cond in ('standard', 'bypass'):
    base = {r['seed']: r for r in per_seed if r['condition'] == cond and r['beta'] == 0.0}
    for beta in betas:
        rs = [r for r in per_seed if r['condition'] == cond and r['beta'] == beta]
        rec = dict(condition=cond, beta=beta, n_seeds=len(rs))
        for m in metrics:
            vals = [get(r, m) for r in rs]
            rec[m] = seed_stats(vals)
            if beta != 0.0:
                rec[m + '.paired_delta_vs_beta0'] = seed_stats([get(r, m) - get(base[r['seed']], m) for r in rs])
        for split in ('test', 'selection'):
            s = summ[(cond, beta, split)]
            rec[f'{split}.bridge_summary_pairwise_sharing'] = s['pairwise_sharing']
            rec[f'{split}.bridge_summary_dictwide_sharing'] = s['dictwide_sharing']
        aggregate.append(rec)

contrasts = [c for c in BR['paired_contrasts'] if c['top_n'] == 50 and c['threshold'] == 0.7 and c['baseline'] == 'model']

out = dict(
    experiment='GPT-2 B1 bridge: TopK k=64 + beta*L1, standard and bypass conditions, 5 seeds, beta in {0, 0.03, 0.1, 0.3}',
    sources=dict(
        bridge_results='results/additional/objectives/five_seed_beta/inputs/BRIDGE_RESULTS.json',
        protocol_id=BR['protocol_id'], complete=BR['complete'], missing_runs=BR['missing_runs'],
        per_run_fidelity='s3://ssup-research-079415246777/{pc1b1-g2-s*,pc1fastb-*}/results/gpt2_b1_bf16/fidelity/<run>/{test,selection}.json',
        per_run_calibration='s3://ssup-research-079415246777/{pc1b1-g2-s*,pc1fastb-*}/results/gpt2_b1_bf16/calibration/<run>.npz',
        grid_training='s3://ssup-research-079415246777/e4real-gpt2/results/e4_real/gpt2/full/training/<run>/ (sae.pt, training.jsonl)',
        grid_evaluation_cells='s3://ssup-research-079415246777/e4real-gpt2/results/e4_real/gpt2/full/evaluation/cells/<run>.json',
        trainer='experiments/objectives/magnitude/train/experiments/train_objective_grid.py',
        evaluator='pc1next (bridgecollect.tgz: pc1next/cache.py calibrate()/fidelity(), analysis.py analyze())'),
    protocol=dict(
        model='openai-community/gpt2 @ 607a30d783dfa663caf39e06633721c8d4cfcd7e',
        site='Output of transformer.h[6] (zero-based block 6; hidden_states[7]).',
        evaluation_dtype='bfloat16 LM forward, attn sdpa',
        training=dict(store='v3 gpt2_wikitext store: first 5M tokens of WikiText-103-v1 train, one document per 128-token sequence, padding excluded',
                      centering='x - train mean (raw scale, no rescaling)', d_sae=3072, k=64, steps=20000, batch=4096, lr=3e-4,
                      optimizer='Adam (torch defaults: betas 0.9/0.999, eps 1e-8)', data_order_seed=20260906,
                      init_seed='torch.manual_seed(seed) and v3 BaseSAE tied init with Generator(seed)', seeds=seeds,
                      loss='mean_batch(sum_d (xhat - x)^2) + beta * mean_batch(sum_j |z_j|)',
                      decoder='rows renormalized to unit norm after every Adam step', grad_clip=None,
                      bypass='x - (x.v1) v1 with the train PC1 before encoding (bypass condition only)'),
        splits=dict(policy='article-level, no overlap (DATA_AUDIT.json)',
                    calibration=dict(tokens=75189, articles=28, source='WikiText-103 validation'),
                    selection=dict(tokens=45120, articles=15, source='WikiText-103 validation (articles a00028+)'),
                    test=dict(tokens=81534, articles=25, source='WikiText-103 test')),
        alive='calibration-alive = fired at least once on the calibration split. The bridge uses the FROZEN alive ids from the grid evaluator (identical sets, checked); fresh_alive is the bridge re-count on its own bfloat16 calibration pass.',
        importance='signed single-feature zero-ablation dCE vs the unmodified model ("model" baseline), token-weighted over the ranking split; every calibration-alive feature evaluated (no screen)',
        top50='top_ids: the 50 calibration-alive features with the largest dCE on the named split; ties broken by feature index',
        sharing='for source seed a and each other seed b in the same (condition, beta): best = max over b\'s calibration-alive decoder rows of the signed cosine with a\'s decoder row. top50_pairwise_sharing(a) = fraction of (other seed, selected feature) pairs with best >= 0.7. Reported sharing = mean over the 5 source seeds. dictwide uses all of a\'s calibration-alive features instead of the top 50.',
        l0='bridge fidelity(): total nonzero codes over the split\'s valid tokens / number of valid tokens (no row cap)',
        fvu='fvu_total = sum||x - xhat||^2 / sum||x - mean_eval||^2; fvu_content (fvu_perp) = same on the complement of the train PC1; w1 = PC1 share of evaluation variance; identity fvu_total = w1*fvu_pc1 + (1-w1)*fvu_content holds to <3e-18',
        uncertainty='sharing: article-resampled paired bootstrap (bridge paired_contrasts, ~1000 draws). L0/FVU/alive: across the 5 seeds (mean, SD, Student-t 95%); beta effects are PAIRED by seed (identical init seed and data order across beta).'),
    per_seed=per_seed, aggregate=aggregate, bridge_paired_contrasts_top50_thr07=contrasts,
)
json.dump(out, open(S / 'gpt2_b1_sweep.json', 'w'), indent=1)

print('cond beta | top50 test (sel) | dictwide | L0 test | L0 sel | FVU tot test | FVU content test (sel) | alive frozen | grid L0 | grid content FVU')
for a in aggregate:
    g = lambda m, n=4: f"{a[m]['mean']:.{n}f}"
    print(a['condition'], a['beta'], '|', g('test.top50_pairwise_sharing', 3), f"({g('selection.top50_pairwise_sharing', 3)})", '|',
          g('test.dictwide_pairwise_sharing', 3), '|', g('test.l0'), '|', g('selection.l0'), '|', g('test.fvu_total', 6), '|',
          g('test.fvu_content', 5), f"({g('selection.fvu_content', 5)})", '|', g('alive_frozen_calibration', 1), '|',
          g('grid_evaluator.test_l0'), '|', g('grid_evaluator.test_fvu_content', 5))
for a in aggregate:
    if a['condition'] == 'standard' and a['beta'] == 0.3:
        for m in ['test.l0', 'test.fvu_content', 'test.fvu_total', 'test.top50_pairwise_sharing', 'test.dictwide_pairwise_sharing', 'alive_frozen_calibration']:
            d = a[m + '.paired_delta_vs_beta0']; print('  paired delta beta0->0.3', m, f"{d['mean']:+.5g} t95 [{d['t95'][0]:+.5g}, {d['t95'][1]:+.5g}] range [{d['min']:+.5g}, {d['max']:+.5g}]")
