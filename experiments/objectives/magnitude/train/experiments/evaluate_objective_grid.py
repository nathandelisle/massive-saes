"""Evaluate every completed E4 cell on independent calibration/test activations.
Optional planted-truth scoring uses one-to-one assignments and calibration-only
affine coefficient fits. No LM loading and no selection using test coefficients.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import slices, unit_rows, sha256, write_json, start_run, finish_run
from runtime import load_sae, sae_output


def assign(scores, alive):
    """One learned latent per truth feature; missing assignments remain explicit."""
    ids = np.full(len(scores), -1, int)
    candidates = np.flatnonzero(alive)
    if len(candidates):
        rows, cols = linear_sum_assignment(-scores[:, candidates])
        ids[rows] = candidates[cols]
    return ids


def affine_map(cross, zsum, zsq, ysum, n, ids):
    valid = ids >= 0
    slopes = np.zeros(len(ids)); intercept = ysum / n
    for j in np.flatnonzero(valid):
        i = ids[j]; var = max(0., zsq[i] / n - (zsum[i] / n)**2)
        if var > 1e-12:
            slopes[j] = max(0., (cross[j, i] / n - ysum[j] * zsum[i] / n**2) / var)
        intercept[j] -= slopes[j] * zsum[i] / n
    return slopes, intercept


def truth_codes(path, n_features):
    with np.load(path, allow_pickle=False) as b:
        y = np.zeros((len(b['content_indices']), n_features))
        np.put_along_axis(y, b['content_indices'], b['content_magnitudes'], axis=1)
    return y


def evaluate_cell(cfg, run, calibration, test, truth=None, batch=512):
    import torch
    sae = load_sae(cfg, run); dev = cfg.get('device', 'cpu')
    d = sae.W_dec.detach().cpu().numpy()
    v = np.load(cfg['pc1_path']).astype(float); v /= np.linalg.norm(v)
    mu = np.load(cfg['mean_path']).astype(float)
    ncal = len(calibration); counts = np.zeros(len(d)); zsum = counts.copy(); zsq = counts.copy()
    cal_center=np.zeros(calibration.shape[1])
    for sl in slices(ncal,batch):cal_center+=np.asarray(calibration[sl],float).sum(0)
    cal_center/=ncal;cal_error=0.;cal_variance=0.
    yc, yt, target = truth if truth is not None else (None, None, None)
    cross = np.zeros((len(target), len(d))) if truth is not None else None
    for sl in slices(ncal, batch):
        with torch.no_grad():
            ch, z, _ = sae_output(sae, torch.tensor(np.asarray(calibration[sl]), device=dev), cfg, run)
        cx=np.asarray(calibration[sl],float);ce=cx-ch.cpu().numpy();ca=cx-cal_center
        cal_error+=((ce-np.outer(ce@v,v))**2).sum();cal_variance+=((ca-np.outer(ca@v,v))**2).sum()
        z = z.cpu().numpy().astype(float)
        counts += np.count_nonzero(z, axis=0); zsum += z.sum(0); zsq += (z*z).sum(0)
        if cross is not None: cross += yc[sl].T @ z
    alive = counts > 0
    dn, _ = unit_rows(d); align = (dn @ v)**2
    group = np.flatnonzero(alive & (align > 5*np.median(align[alive]))) if alive.any() else np.array([], int)
    maps = {}; truth_report = None
    if truth is not None:
        tn, _ = unit_rows(target); scores = tn @ dn.T
        ysum = yc.sum(0); ysq = (yc*yc).sum(0)
        vz = np.maximum(zsq-zsum*zsum/ncal, 0)
        vy = np.maximum(ysq-ysum*ysum/ncal, 0)
        corr = (cross - np.outer(ysum, zsum)/ncal) / np.maximum(np.sqrt(np.outer(vy, vz)), 1e-12)
        for label, criterion in [('decoder', scores), ('activation', corr)]:
            ids = assign(criterion, alive)
            a, b = affine_map(cross, zsum, zsq, ysum, ncal, ids)
            maps[label] = dict(ids=ids, slope=a, intercept=b, sse=np.zeros(len(target)))
        ids = maps['decoder']['ids']; good = ids >= 0
        matched_cos = np.full(len(target), -1.)
        matched_cos[good] = scores[np.flatnonzero(good), ids[good]]
        truth_report = dict(one_to_one_recall_cos07=float((matched_cos > .7).mean()),
            mean_assigned_cos=float(matched_cos.mean()), n_truth=len(target), n_assigned=int(good.sum()),
            alive_latent_precision_cos07=float((scores[:, alive].max(0) > .7).mean()) if alive.any() else None)
    # Use one evaluation mean for both conditions. Float64 projections avoid
    # subtracting two large totals to estimate the complementary error.
    center = np.zeros(test.shape[1])
    for sl in slices(len(test), batch): center += np.asarray(test[sl], float).sum(0)
    center /= len(test)
    error = np.zeros(2); variance = np.zeros(2); l0 = 0; l1 = 0.
    individual = 0.; combined = 0.; group_pc = 0.; group_total = 0.
    for sl in slices(len(test), batch):
        x = np.asarray(test[sl], float)
        with torch.no_grad():
            hat, z, _ = sae_output(sae, torch.tensor(x, dtype=torch.float32, device=dev), cfg, run)
        hat = hat.cpu().numpy().astype(float); z = z.cpu().numpy().astype(float)
        e = x-hat; a = x-center; e1 = e@v; a1 = a@v
        error += [(e1*e1).sum(), ((e-e1[:, None]*v)**2).sum()]
        variance += [(a1*a1).sum(), ((a-a1[:, None]*v)**2).sum()]
        l0 += np.count_nonzero(z); l1 += np.abs(z).sum()
        if len(group):
            zg = z[:, group]; dg = d[group]; dp = dg-np.outer(dg@v, v)
            y = zg@dg; yp = y-np.outer(y@v, v)
            individual += ((zg*zg).sum(0)*(dp*dp).sum(1)).sum()
            combined += (yp*yp).sum(); group_pc += ((y@v)**2).sum(); group_total += (y*y).sum()
        for mapping in maps.values():
            ids = mapping['ids']; selected = np.zeros((len(z), len(ids))); good = ids >= 0
            selected[:, good] = z[:, ids[good]]
            pred = selected*mapping['slope']+mapping['intercept']
            mapping['sse'] += ((yt[sl]-pred)**2).sum(0)
    if (variance <= 0).any(): raise ValueError('Degenerate test component variance')
    if truth_report is not None:
        sst = ((yt-yt.mean(0))**2).sum(0); valid = sst > 1e-12
        truth_report['coefficient_fits'] = {}
        for label, mapping in maps.items():
            r2 = np.full(len(sst), np.nan); r2[valid] = 1-mapping['sse'][valid]/sst[valid]
            truth_report['coefficient_fits'][label] = dict(mean_r2=float(np.nanmean(r2)),
                median_r2=float(np.nanmedian(r2)), fraction_r2_above_half=float((r2[valid] > .5).mean()),
                n_nonconstant_test_features=int(valid.sum()), n_constant_test_features=int((~valid).sum()),
                r2=r2, matched_latents=mapping['ids'])
    record = dict(name=run['name'], condition=run['condition'], seed=run['seed'],
        n_test=len(test), n_calibration=ncal, l0=l0/len(test), mean_l1=l1/len(test),
        calibration_l0=float(counts.sum()/ncal),calibration_fvu_perp=float(cal_error/cal_variance) if cal_variance>0 else None,
        n_alive_calibration=int(alive.sum()), fvu_total=float(error.sum()/variance.sum()),
        calibration_alive_ids=np.flatnonzero(alive).tolist(), structural_group_ids=group.tolist(),
        fvu_pc1=float(error[0]/variance[0]), fvu_perp=float(error[1]/variance[1]),
        w1=float(variance[0]/variance.sum()), test_variance_sum=float(variance.sum()),
        group=dict(n_atoms=len(group), offaxis_individual_energy=individual,
            offaxis_group_energy=combined, offaxis_cross_term=combined-individual,
            cancellation_ratio=individual/combined if combined > 1e-25 else None,
            pc1_energy_fraction=group_pc/group_total if group_total else None), truth=truth_report)
    return record, d, alive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid', required=True); p.add_argument('--calibration', required=True)
    p.add_argument('--test', required=True); p.add_argument('--out', required=True)
    p.add_argument('--truth-dir'); p.add_argument('--heldout-statement', required=True)
    p.add_argument('--resume', action='store_true'); p.add_argument('--batch', type=int, default=512)
    args = p.parse_args(); grid = Path(args.grid)
    training = json.loads((grid/'manifest.json').read_text()); cfg = training['config']
    if not args.heldout_statement.strip() or 'REPLACE' in args.heldout_statement: raise ValueError('Supply exclusion provenance')
    spec = {k:v for k,v in vars(args).items() if k != 'resume'}
    inputs = [args.calibration, args.test, cfg['mean_path'], cfg['pc1_path'],
              Path(cfg['v3_path'])/'v3/common/saes.py']
    if args.truth_dir:
        t = Path(args.truth_dir)
        inputs += [t/'ground_truth_decoder.npy', t/'calibration_truth.npz', t/'test_truth.npz']
    if args.resume:
        out = Path(args.out); meta = json.loads((out/'manifest.json').read_text())
        if meta['config'] != spec: raise ValueError('Evaluation configuration changed')
        for dependency in ['evaluate_objective_grid.py','runtime.py','common.py']:
            if meta['code_sha256'].get(dependency)!=sha256(Path(__file__).parent/dependency):raise ValueError('Evaluation code changed; use a new output version')
        if any(sha256(r['path']) != r['sha256'] for r in meta['inputs']): raise ValueError('Evaluation input changed')
    else: out, meta = start_run(args.out, spec, inputs)
    if sha256(args.calibration) == sha256(args.test): raise ValueError('Identical calibration/test arrays')
    train_sha = next((r['sha256'] for r in training['inputs'] if Path(r['path']) == Path(cfg['activations']).resolve()), None)
    if train_sha in [sha256(args.calibration), sha256(args.test)]: raise ValueError('Evaluation duplicates training store')
    cal = np.load(args.calibration, mmap_mode='r'); test = np.load(args.test, mmap_mode='r')
    if cal.ndim != 2 or test.ndim != 2 or cal.shape[1] != test.shape[1]: raise ValueError('Activation shapes differ')
    truth = None
    if args.truth_dir:
        target = np.load(t/'ground_truth_decoder.npy')
        yc = truth_codes(t/'calibration_truth.npz', len(target)); yt = truth_codes(t/'test_truth.npz', len(target))
        if len(yc) != len(cal) or len(yt) != len(test): raise ValueError('Truth rows do not match activations')
        truth = yc, yt, target
    records = []; completed = 0
    for cond in cfg.get('conditions', ['standard', 'bypass']):
      for k in cfg.get('k_values', [64]):
       for seed in cfg.get('seeds', [42, 137, 0, 1, 2]):
        for beta in cfg.get('betas', [0, .03, .1, .3]):
            name = f'{cond}_k{k}_s{seed}_beta{beta:g}'; rd = grid/name
            if not (rd/'complete.json').exists(): continue
            checkpoint = rd/'sae.pt'; digest = sha256(checkpoint)
            if json.loads((rd/'complete.json').read_text())['checkpoint_sha256'] != digest: raise ValueError('Checkpoint checksum mismatch')
            dest = out/'cells'/(name+'.json')
            if dest.exists():
                rec = json.loads(dest.read_text())
                if rec['checkpoint_sha256'] != digest: raise ValueError('Evaluated checkpoint changed')
            else:
                run = dict(name=name, condition=cond, seed=seed, checkpoint=str(checkpoint))
                rec, _, _ = evaluate_cell(cfg, run, cal, test, truth, args.batch)
                rec.update(k=k, beta=beta, checkpoint_sha256=digest,
                    nominal_sparse_plus_bypass_budget=k+int(cond=='bypass'),
                    bypass_scalar_channels=int(cond=='bypass'),
                    heldout_statement=args.heldout_statement, data_order_seed=cfg.get('data_order_seed', 20260906))
                write_json(dest, rec)
            records.append(rec); completed += 1
            print(f'[pc1-E4-eval] {name} FVU_perp={rec["fvu_perp"]:.5g} L0={rec["l0"]:.3f}', flush=True)
    expected = len(cfg.get('conditions', ['standard','bypass']))*len(cfg.get('k_values',[64]))*len(cfg.get('seeds',[42,137,0,1,2]))*len(cfg.get('betas',[0,.03,.1,.3]))
    write_json(out/'summary.json', dict(runs=records, completed_cells=completed, expected_cells=expected,
        complete=completed == expected, warning='File hashes cannot prove historical document exclusion; retain the supplied provenance.'))
    if completed == expected: finish_run(out, meta)
    else: print(f'[pc1-E4-eval] INCOMPLETE {completed}/{expected}', flush=True)


if __name__ == '__main__': main()
