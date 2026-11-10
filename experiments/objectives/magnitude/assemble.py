"""Build the magnitude-penalty tables from the saved measurements."""
import csv, json, math, os
import numpy as np
from pathlib import Path

S = Path(__file__).resolve().parents[3] / "results/additional/objectives/five_seed_beta/inputs"
OUT = Path(os.environ.get('SEC5_OUT', str(Path.cwd() / 'generated/magnitude')))
(OUT / 'figure_data').mkdir(parents=True, exist_ok=True)
G = json.load(open(S / 'gpt2_b1_sweep.json'))
GEO = json.load(open(S / 'gpt2_b1_geometry.json'))
GRID = json.load(open(S / 'grid_training_terms.json'))
NAT = json.load(open(S / 'native_gpt2_beta_metrics.json'))
NATT = json.load(open(S / 'native_gpt2_trajectory_summary.json'))
PL = json.load(open(os.environ.get('SEC5_PL', str(S / 'L0_RECOMPUTE.json'))))
ARCH = json.load(open(S / 'archived_pythia-1.4b_phase.json'))
T975_1 = 12.706204736174707
BETAS_G = [0.0, 0.03, 0.1, 0.3]

# ------------------------------------------------------------------ GPT-2 aggregate views
agg = {(a['condition'], a['beta']): a for a in G['aggregate']}
contr = {}
for c in G['bridge_paired_contrasts_top50_thr07']:
    contr[(c['kind'], tuple(c['left']), tuple(c['right']), c['ranking_split'])] = c

def gm(cond, beta, m):
    return agg[(cond, beta)][m]

gpt2_rows = []
for cond in ('standard', 'bypass'):
    for beta in BETAS_G:
        a = agg[(cond, beta)]
        geo = next(x for x in GEO['aggregate'] if x['condition'] == cond and x['beta'] == beta)
        row = dict(experiment='gpt2_b1', condition=cond, beta=beta, n_seeds=a['n_seeds'],
                   top50_sharing_test=a['test.bridge_summary_pairwise_sharing'],
                   top50_sharing_test_seed_min=a['test.top50_pairwise_sharing']['min'],
                   top50_sharing_test_seed_max=a['test.top50_pairwise_sharing']['max'],
                   top50_sharing_selection=a['selection.bridge_summary_pairwise_sharing'],
                   dictwide_sharing=a['test.bridge_summary_dictwide_sharing'],
                   l0_test_mean=a['test.l0']['mean'], l0_test_sd=a['test.l0']['sd'],
                   l0_selection_mean=a['selection.l0']['mean'],
                   fvu_total_test_mean=a['test.fvu_total']['mean'], fvu_total_test_sd=a['test.fvu_total']['sd'],
                   fvu_content_test_mean=a['test.fvu_content']['mean'], fvu_content_test_sd=a['test.fvu_content']['sd'],
                   fvu_content_selection_mean=a['selection.fvu_content']['mean'],
                   fvu_pc1_test_mean=a['test.fvu_pc1']['mean'], w1_test=a['test.w1']['mean'],
                   alive_calibration_mean=a['alive_frozen_calibration']['mean'], alive_calibration_sd=a['alive_frozen_calibration']['sd'],
                   alive_fresh_mean=a['alive_fresh_bridge_calibration']['mean'], d_sae=3072,
                   grid_evaluator_l0_test_mean=a['grid_evaluator.test_l0']['mean'],
                   grid_evaluator_fvu_content_test_mean=a['grid_evaluator.test_fvu_content']['mean'],
                   n_band_atoms_mean=geo['n_band']['mean'], band_sharing_mean=geo['sharing_band']['mean'],
                   n_canonical_atoms_mean=geo['n_canonical']['mean'], canonical_sharing_mean=geo['sharing_canonical']['mean'],
                   content_sharing_mean=geo['sharing_content']['mean'],
                   top50_frac_band_mean=geo['top50_frac_band']['mean'], top50_frac_canonical_mean=geo['top50_frac_canonical']['mean'],
                   top50_frac_content_mean=geo['top50_frac_content']['mean'])
        if beta != 0.0:
            for split in ('test', 'selection'):
                c = contr[('beta_effect', (cond, 0.0), (cond, beta), split)]
                row[f'delta_top50_vs_beta0_{split}'] = c['delta_pairwise_sharing']
                row[f'delta_top50_vs_beta0_{split}_ci95'] = c['interval_95_percentile']
            for m, key in [('test.l0', 'delta_l0_test'), ('test.fvu_content', 'delta_fvu_content_test'), ('test.fvu_total', 'delta_fvu_total_test'),
                           ('test.dictwide_pairwise_sharing', 'delta_dictwide'), ('alive_frozen_calibration', 'delta_alive')]:
                d = a[m + '.paired_delta_vs_beta0']
                row[key] = d['mean']; row[key + '_t95'] = d['t95']; row[key + '_seed_range'] = [d['min'], d['max']]
        gpt2_rows.append(row)
by_cond = {c: {r['beta']: r for r in gpt2_rows if r['condition'] == c} for c in ('standard', 'bypass')}
for cond in ('standard', 'bypass'):
    c = contr[('bypass_effect', ('standard', 0.0), ('bypass', 0.0), 'test')] if cond == 'bypass' else None

# ------------------------------------------------------------------ Pythia views
prow = PL['rows']
arch_topk = {float(r['coef']): r for r in ARCH['rows'] if r['sweep'] == 'topk_l1'}
betas_p = sorted({r['beta'] for r in prow})
pbase = {r['seed']: r for r in prow if r['beta'] == 0.0}
pythia_rows = []
max_repro_err = max(abs(r['reproduced_old_routine'] - r['archived_l0_measured']) for r in prow)
for beta in betas_p:
    rs = sorted([r for r in prow if r['beta'] == beta], key=lambda r: r['seed'])
    ar = arch_topk[beta]
    corr = [r['corrected_l0'] for r in rs]
    d = [r['corrected_l0'] - pbase[r['seed']]['corrected_l0'] for r in rs]
    pythia_rows.append(dict(
        experiment='pythia14b_b2', beta=beta, seeds=[r['seed'] for r in rs],
        l0_archived_per_seed=[r['archived_l0_measured'] for r in rs], l0_archived_mean=ar['l0_measured_mean'],
        l0_old_routine_reproduced_per_seed=[r['reproduced_old_routine'] for r in rs],
        old_routine_rows_encoded=rs[0]['old_routine_rows'], old_routine_denominator=rs[0]['old_routine_denominator'],
        l0_denominator_only_correction_mean=ar['l0_measured_mean'] / 1.06496,
        l0_overrun_sample_per_seed=[r['overrun_sample_l0'] for r in rs],
        l0_corrected_per_seed=corr, l0_corrected_mean=float(np.mean(corr)), l0_corrected_rows=rs[0]['corrected_rows'],
        l0_corrected_delta_vs_beta0_per_seed=d, l0_corrected_delta_vs_beta0_mean=float(np.mean(d)),
        l0_corrected_relative_change_vs_beta0=float(np.mean(corr) / np.mean([pbase[s]['corrected_l0'] for s in pbase]) - 1),
        rows_below_k_per_seed=[r['rows_below_k'] for r in rs], min_row_l0_per_seed=[r['min_row_l0'] for r in rs],
        l0_batch_estimate_seed42_only=ar['l0_batch'], l0_batch_rows='4 random batches x 4096 rows, seed 777, whole store (l1_pressure)',
        l0_train_log_step19000_seed42=ar['l0_train_log_stale'],
        alive_old_routine_per_seed=[r['n_alive_old_routine'] for r in rs], alive_corrected_per_seed=[r['n_alive_corrected'] for r in rs], d_sae=8192,
        fvu_total_first100k_train_rows_per_seed=[r['fvu_total_first100k_train_rows'] for r in rs],
        fvu_content_first100k_train_rows_per_seed=[r['fvu_content_first100k_train_rows'] for r in rs],
        fvu_archived_train_log_step19000_seed42=ar['fvu'],
        top50_sharing_archived_single_pair=ar['top50'], dictwide_sharing_archived_single_pair=ar['dictwide'],
        top50_dense_fraction_archived=ar['top50_dense_frac'], tiling_band_n_archived=ar['tiling_band_n'], tiling_band_sharing_archived=ar['tiling_band_sharing'],
        base_ce_archived=ar['base_ce'],
        checkpoints=[r['checkpoint'] for r in rs], checkpoint_sha256=[r['checkpoint_sha256'] for r in rs]))

# ------------------------------------------------------------------ implementation comparison
ratio_grid = {}
for beta in BETAS_G:
    rs = [v for v in GRID.values() if v['beta'] == beta]
    mse = float(np.mean([r['last10_mean']['mse'] for r in rs])); l1 = float(np.mean([r['last10_mean']['l1'] for r in rs]))
    st0 = [r['first'] for r in rs]
    ratio_grid[beta] = dict(final_mse_per_row=mse, final_l1_per_row=l1, final_penalty_over_recon=beta * l1 / mse,
                            step0_penalty_over_recon=beta * float(np.mean([r['l1'] for r in st0])) / float(np.mean([r['mse'] for r in st0])),
                            final_train_l0=float(np.mean([r['last10_mean']['l0'] for r in rs])))
nat_eval = {}
for arm in ('baseline', 'beta_0.03', 'beta_0.1', 'beta_0.3'):
    rs = [v for f, v in NAT.items() if f.split('__')[1] == arm]
    nat_eval[arm] = dict(n_seeds=len(rs), mean_l0=float(np.mean([r['mean_l0'] for r in rs])), fvu_total=float(np.mean([r['fvu_total'] for r in rs])),
                         fvu_content=float(np.mean([r['fvu_perp'] for r in rs])), alive=float(np.mean([r['alive'] for r in rs])))
per_unit_beta_grid = ratio_grid[0.0]['final_l1_per_row'] / ratio_grid[0.0]['final_mse_per_row']
nb = NATT['summary']['baseline']['final']
per_unit_beta_native = nb['l1'] / nb['mse']
impl = dict(
    identical_in_code=[
        'loss form: mean over batch of (sum over d of squared error) + beta * mean over batch of (sum over latents of |z|)  [grid: train_objective_grid.py loss=mse+beta*z.abs().sum(-1).mean(); native: pc1night/model.py loss_parts (weights=1, loss_axis=1 for the beta arms)]',
        'activation: z = TopK_k(ReLU((x - b_dec) W_enc + b_enc)), k=64, TopK applied after ReLU on pre-activations  [v3 saes.py TopKSAE.encode; pc1night model.py SAE.encode]',
        'tied initialization from a CPU Generator(seed): W_dec rows unit-normalized Gaussian, W_enc = W_dec transpose, zero biases, learned b_dec',
        'Adam lr 3e-4, betas 0.9/0.999, eps 1e-8; 20,000 steps; batch 4,096; width 3,072 (GPT-2); site transformer.h[6]',
        'beta enters only as a constant multiplier of the same L1 term (native beta_0.03/0.1/0.3 arms: constant schedule)'],
    different_in_code=[
        dict(item='input units', grid='x - train mean (raw residual scale)', native='(x - train mean) / s, s = sqrt(mean_train ||x - mean||^2) = 274.83 for GPT-2 (pc1night data.py reference(); reference.json)',
             consequence='per-row MSE scales by 1/s^2 and code magnitudes (hence L1) by 1/s, so at a fixed dictionary the penalty-to-reconstruction ratio at the same numeric beta is ~s times larger in native units'),
        dict(item='gradient clipping', grid='none', native='clip_grad_norm_(params, 1.0) (train.py)', measured='engaged at 2.4% of logged steps (step 1 onward; median pre-clip norm 0.008): not active in steady state'),
        dict(item='decoder constraint', grid='renormalize rows after each Adam step (v3 renorm_decoder)', native='project the W_dec gradient onto the tangent of the unit sphere (tangent_grad) before the step AND renormalize after (constrain)'),
        dict(item='training store and sampling', grid='v3 store: first 5M tokens of WikiText-103-v1 train, one document per 128-token sequence, no BOS; rows drawn uniformly with replacement (default_rng(20260906))',
             native='pc1night harvest (BOS kept in the LM context; 128-token sequences), 5,000,017 train rows; superblocks of 32x4096 indices drawn with replacement (order seed 20260908)'),
        dict(item='evaluation', grid='bridge (pc1next): bfloat16 LM, article splits, frozen calibration-alive masks', native='pc1night evaluation: population-resolved FVU on its own test rows (not directly comparable numbers)')],
    measured=dict(
        grid_gpt2=dict(source='s3://.../e4real-gpt2/results/e4_real/gpt2/full/training/standard_k64_s*_beta*/training.jsonl (last 10 logged steps, 5 seeds)', by_beta=ratio_grid,
                       l1_over_mse_per_unit_beta_at_beta0=per_unit_beta_grid),
        native_gpt2=dict(source=NATT['source'], units=NATT['units'], trajectories=NATT['summary'], evaluation=nat_eval,
                         l1_over_mse_per_unit_beta_at_beta0=per_unit_beta_native),
        strength_ratio_native_over_grid_at_beta0_solution=per_unit_beta_native / per_unit_beta_grid,
        native_top50_sharing_beta03_preliminary=dict(value=0.0025, n_seeds=8, population='all', source='native_prelim/LM_SUMMARY.json (preliminary collect; dictionaries retain ~51 alive atoms, so top-50 sharing is nearly degenerate)')),
    attribution='The only code difference that changes how strong a given numeric beta is, is the native global input rescaling. Measured at the start of training (identical-form objectives, beta=0.3) the penalty/reconstruction ratio is 0.851 native vs 0.0021 grid; at the beta=0 solutions the L1 cost per unit beta relative to MSE is %.0f native vs %.2f grid (x%.0f). Native L0 collapses within the first 500-2,000 steps (0.72 at step 500, 0.04 by step 2,000 at beta=0.3). This establishes that equal numeric beta values are not equal penalties across the two trainers; it does not isolate the rescaling from the other listed differences (decoder tangent projection, store, sampling), which were not ablated.' % (per_unit_beta_native, per_unit_beta_grid, per_unit_beta_native / per_unit_beta_grid),
)

# ------------------------------------------------------------------ write machine-readable table
pythia_protocol = dict(
    experiment='Historical Pythia-1.4B beta phase sweep (exp_b2_l1_phase.py, S3 prefix b2v2/, run 2026-08-20)',
    model='EleutherAI/pythia-1.4b, output of gpt_neox.layers[12] (v3 target_layer 12)', d_sae=8192, k=64, seeds=[42, 137],
    training='v3 get_sae/train_sae: TopKL1SAE, centered raw activations (x - store mean), 20,000 steps, batch 4,096, Adam 3e-4, decoder renormalized after each step; store = first 5M tokens of WikiText-103-v1 train, one document per 128-token sequence, padding excluded, float16',
    betas=betas_p,
    archived_file='results/additional/objectives/five_seed_beta/inputs/archived_pythia-1.4b_phase.json',
    archived_l0_field_origin='l0_measured[seed] = float(rates.sum()) with rates, alive = io.alive(sae, adir) (pc1_v3/v3/common/io_utils.py): for i in range(0, 100000, 8192) encodes acts[i:i+8192] with no clamp (13 slices, 106,496 rows) and divides counts by n = 100,000',
    archived_other_l0_fields=dict(l0_batch='l1_pressure(): mean nonzeros over 4 random batches of 4,096 rows (rng seed 777) drawn from the whole store; seed 42 checkpoint only',
                                  l0_train_log_stale='training log at step 19,000, seed 42 only'),
    archived_fvu_origin='fvu = extra["log"][-1]["fvu"] = training-batch FVU at step 19,000 for seed 42 (checkpoint train log); not an evaluation',
    sharing='single ordered pair: seed 42 reference vs seed 137; importance = zero-ablation dCE for seed 42 over 64 WikiText-103 validation sequences (first 64 non-empty lines, padded to 128, pad fraction 0.44), screened to the top 1,500 alive atoms by E[z^2]; top-50 = highest dCE among seed-42 alive atoms (alive from the unclamped routine, rate > 1e-3); shared if signed max cosine against all seed-137 decoder rows >= 0.7; no held-out split, no uncertainty',
    recompute=dict(job='pc1sec5-pythia-l0 (g6 on-demand), script pl0/recompute_l0.py, output s3://ssup-research-079415246777/pc1_sec5/pythia_l0/L0_RECOMPUTE.json',
                   store='re-harvested with the unmodified b2v2 pc1_v3 package (the original store was never uploaded)', env=PL['env'], store_meta=PL['store'],
                   validation='reproduced_old_routine vs archived l0_measured: max abs difference %.6g over %d checkpoints' % (max_repro_err, len(prow)),
                   corrected_definition=PL['definition'], rows=PL['n']))
table = dict(generated_by='assemble.py (Section 5 verification, 2026-09-13)',
             gpt2_b1=dict(protocol=G['protocol'], sources=G['sources'], by_condition_beta=gpt2_rows,
                          per_seed=G['per_seed'], geometry=GEO, bridge_paired_contrasts=G['bridge_paired_contrasts_top50_thr07']),
             pythia14b_b2=dict(protocol=pythia_protocol, by_beta=pythia_rows, per_checkpoint=prow),
             implementation_comparison=impl)
json.dump(table, open(OUT / 'sec5_sweeps.json', 'w'), indent=1)

# ------------------------------------------------------------------ figure data
def wcsv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})

main = []
for r in gpt2_rows:
    b0 = by_cond[r['condition']][0.0]
    main.append(dict(condition=r['condition'], beta=r['beta'], n_seeds=r['n_seeds'],
                     top50_sharing=r['top50_sharing_test'], top50_seed_min=r['top50_sharing_test_seed_min'], top50_seed_max=r['top50_sharing_test_seed_max'],
                     delta_top50_ci95_low=(r.get('delta_top50_vs_beta0_test_ci95') or [None, None])[0],
                     delta_top50_ci95_high=(r.get('delta_top50_vs_beta0_test_ci95') or [None, None])[1],
                     dictwide_sharing=r['dictwide_sharing'],
                     l0=r['l0_test_mean'], l0_sd=r['l0_test_sd'], l0_rel=r['l0_test_mean'] / b0['l0_test_mean'],
                     fvu_content=r['fvu_content_test_mean'], fvu_content_sd=r['fvu_content_test_sd'], fvu_content_rel=r['fvu_content_test_mean'] / b0['fvu_content_test_mean'],
                     fvu_total=r['fvu_total_test_mean'], fvu_total_rel=r['fvu_total_test_mean'] / b0['fvu_total_test_mean'],
                     alive=r['alive_calibration_mean']))
wcsv(OUT / 'figure_data' / 'fig5_gpt2_main.csv', main)
seedrows = []
for p in G['per_seed']:
    seedrows.append(dict(condition=p['condition'], beta=p['beta'], seed=p['seed'], top50_sharing_test=p['test']['top50_pairwise_sharing'],
                         top50_sharing_selection=p['selection']['top50_pairwise_sharing'], dictwide=p['test']['dictwide_pairwise_sharing'],
                         l0_test=p['test']['l0'], fvu_total_test=p['test']['fvu_total'], fvu_content_test=p['test']['fvu_content'],
                         alive_calibration=p['alive_frozen_calibration']))
wcsv(OUT / 'figure_data' / 'fig5_gpt2_per_seed.csv', seedrows)
wcsv(OUT / 'figure_data' / 'fig5_gpt2_geometry.csv', GEO['rows'])
py = []
for r in pythia_rows:
    for i, s in enumerate(r['seeds']):
        py.append(dict(beta=r['beta'], seed=s, l0_corrected=r['l0_corrected_per_seed'][i], l0_archived_inflated=r['l0_archived_per_seed'][i],
                       l0_old_routine_reproduced=r['l0_old_routine_reproduced_per_seed'][i], alive_corrected=r['alive_corrected_per_seed'][i],
                       fvu_total_first100k_train=r['fvu_total_first100k_train_rows_per_seed'][i], fvu_content_first100k_train=r['fvu_content_first100k_train_rows_per_seed'][i],
                       top50_sharing_single_pair=r['top50_sharing_archived_single_pair'], dictwide_single_pair=r['dictwide_sharing_archived_single_pair']))
wcsv(OUT / 'figure_data' / 'fig5_pythia_supporting.csv', py)
nat = [dict(arm=k, **v) for k, v in nat_eval.items()]
wcsv(OUT / 'figure_data' / 'native_gpt2_beta.csv', nat)

# long csv of everything tabular
longrows = []
for r in gpt2_rows:
    for k, v in r.items():
        if k in ('experiment', 'condition', 'beta'):
            continue
        longrows.append(dict(experiment='gpt2_b1', condition=r['condition'], beta=r['beta'], seed='aggregate', metric=k, value=json.dumps(v) if isinstance(v, list) else v))
for p in G['per_seed']:
    for split in ('test', 'selection'):
        for k, v in p[split].items():
            longrows.append(dict(experiment='gpt2_b1', condition=p['condition'], beta=p['beta'], seed=p['seed'], metric=f'{split}.{k}', value=v))
    for k in ('alive_frozen_calibration', 'alive_fresh_bridge_calibration', 'structural_group_size'):
        longrows.append(dict(experiment='gpt2_b1', condition=p['condition'], beta=p['beta'], seed=p['seed'], metric=k, value=p[k]))
for r in prow:
    for k in ('archived_l0_measured', 'reproduced_old_routine', 'old_routine_rows', 'overrun_sample_l0', 'corrected_l0', 'corrected_rows',
              'rows_below_k', 'n_alive_old_routine', 'n_alive_corrected', 'fvu_total_first100k_train_rows', 'fvu_content_first100k_train_rows'):
        longrows.append(dict(experiment='pythia14b_b2', condition='standard', beta=r['beta'], seed=r['seed'], metric=k, value=r[k]))
wcsv(OUT / 'sec5_sweeps_long.csv', longrows)
json.dump(dict(gpt2=gpt2_rows, pythia=pythia_rows, impl=impl, max_repro_err=max_repro_err), open(OUT / 'assembled_views.json', 'w'), indent=1)
print('wrote', OUT)
print('pythia max |reproduced old - archived| =', max_repro_err)
for r in pythia_rows:
    print(f"beta={r['beta']:g} archived {r['l0_archived_mean']:.4f} corrected {r['l0_corrected_mean']:.4f} per-seed {['%.4f' % x for x in r['l0_corrected_per_seed']]} delta {r['l0_corrected_delta_vs_beta0_mean']:+.4f} alive old/corr {r['alive_old_routine_per_seed']}/{r['alive_corrected_per_seed']} fvu_tot {['%.5f' % x for x in r['fvu_total_first100k_train_rows_per_seed']]} fvu_content {['%.4f' % x for x in r['fvu_content_first100k_train_rows_per_seed']]}")
print('strength ratio native/grid at beta=0 solution:', per_unit_beta_native / per_unit_beta_grid, per_unit_beta_native, per_unit_beta_grid)
