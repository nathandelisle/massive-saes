"""B5: independent planted-truth audit of existing synthetic fits, without retraining.

Manifest cases provide cal_codes/test_codes and cal_truth/test_truth .npy matrices.
One-to-one signed-correlation matching, scale and intercept use calibration only.
Report held-out R2 for every planted factor, including negative R2; a dense ridge
decoder is a separate information-retention ceiling, never called atom recovery.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .io import atomic_json,digest,key,load_json,ratio,canonical,rng_for


def score(zc,zt,sc,st,ridge=1e-3):
    if zc.ndim!=2 or zt.shape[1]!=zc.shape[1] or sc.ndim!=2 or st.shape[1]!=sc.shape[1] or len(zc)!=len(sc) or len(zt)!=len(st):raise ValueError('Truth/code axis mismatch')
    if not all(np.isfinite(x).all() for x in (zc,zt,sc,st)):raise ValueError('Nonfinite codes/truth')
    zm=zc.mean(0);sm=sc.mean(0);x=zc-zm;y=sc-sm
    zx=np.sqrt(np.square(x).sum(0));sy=np.sqrt(np.square(y).sum(0))
    corr=np.divide(y.T@x,sy[:,None]*zx[None,:],out=np.zeros((sc.shape[1],zc.shape[1])),where=sy[:,None]*zx[None,:]>1e-20)
    factor,atom=linear_sum_assignment(-corr) # signed: a negated concept is not the same nonnegative atom
    match={int(f):int(a) for f,a in zip(factor,atom)};pred=np.broadcast_to(sm,st.shape).copy();rows=[]
    for f in range(sc.shape[1]):
        a=match.get(f)
        if a is not None:
            den=x[:,a]@x[:,a];scale=max(0.,float(x[:,a]@y[:,f]/den)) if den>1e-20 else 0.
            pred[:,f]=sm[f]+(zt[:,a]-zm[a])*scale
        else:scale=0.
        testvar=np.square(st[:,f]-st[:,f].mean()).sum();sse=np.square(st[:,f]-pred[:,f]).sum()
        rows.append(dict(factor=f,atom=a,cal_correlation=float(corr[f,a]) if a is not None else None,scale=scale,
            heldout_r2=1-sse/testvar if testvar>1e-20 else None,heldout_variance=float(testvar)))
    # Standardize each code on calibration; ridge is fixed in the register.
    std=np.sqrt(np.mean(x*x,axis=0));std=np.maximum(std,1e-8);xs=x/std;xt=(zt-zm)/std
    if xs.shape[1]<=xs.shape[0]:coef=np.linalg.solve(xs.T@xs+ridge*len(xs)*np.eye(xs.shape[1]),xs.T@y)
    else:coef=xs.T@np.linalg.solve(xs@xs.T+ridge*len(xs)*np.eye(len(xs)),y)
    dense=sm+xt@coef;var=np.square(st-st.mean(0)).sum(0);sse=np.square(st-dense).sum(0)
    dense_r2=np.where(var>1e-20,1-sse/np.maximum(var,1e-20),np.nan)
    valid=[r['heldout_r2'] for r in rows if r['heldout_r2'] is not None]
    return dict(n_factors=len(rows),n_factors_without_test_variance=len(rows)-len(valid),one_to_one=rows,
        mean_one_to_one_r2=float(np.mean(valid)) if valid else None,
        fraction_factors_r2_above_0_8=float(np.mean(np.asarray(valid)>=.8)) if valid else None,
        dense_ridge_r2=dense_r2,mean_dense_ridge_r2=float(np.nanmean(dense_r2)),ridge=ridge)


def summarize_cases(manifest,completed,missing):
    expected={};observed={r['case']['name']:r for r in completed};contrasts=[]
    for case in manifest['cases']:
        group=(case['condition'],canonical(case.get('stratum',{})))
        index=(case['corpus_id'],case['seed'],float(case['beta']))
        cells=expected.setdefault(group,{})
        if index in cells:raise ValueError('Duplicate corpus/seed/beta: use stratum for different architectures, k or widths')
        cells[index]=case
    for group,cells in expected.items():
        betas=sorted({b for c,s,b in cells});corpora=sorted({c for c,s,b in cells});seeds=sorted({s for c,s,b in cells})
        for beta in betas[1:]:
            absent=[];mat=np.full((len(corpora),len(seeds)),np.nan)
            for ci,corpus in enumerate(corpora):
                for si,seed in enumerate(seeds):
                    a=cells.get((corpus,seed,betas[0]));b=cells.get((corpus,seed,beta))
                    if not a or not b or a['name'] not in observed or b['name'] not in observed:
                        absent.append([corpus,seed]);continue
                    aa=observed[a['name']]['metrics']['mean_one_to_one_r2'];bb=observed[b['name']]['metrics']['mean_one_to_one_r2']
                    if aa is None or bb is None:absent.append([corpus,seed,'undefined_truth_variance']);continue
                    mat[ci,si]=bb-aa
            record=dict(condition=group[0],stratum=json.loads(group[1]),beta0=betas[0],beta1=beta,
                n_registered_corpora=len(corpora),n_training_seeds=len(seeds),missing_pairs=absent,
                complete=not absent)
            if not absent:
                by_corpus=mat.mean(1);rng=rng_for('truth-bootstrap',group,beta);draws=[]
                for _ in range(manifest.get('bootstrap_draws',2000)):
                    cw=rng.multinomial(len(corpora),np.full(len(corpora),1/len(corpora)))
                    sw=rng.multinomial(len(seeds),np.full(len(seeds),1/len(seeds)))
                    draws.append(float(cw@mat@sw/(cw.sum()*sw.sum())))
                record.update(mean_delta=float(mat.mean()),by_corpus=dict(zip(corpora,by_corpus.tolist())),
                    interval_95_percentile=np.quantile(draws,[.025,.975]).tolist(),
                    fraction_corpora_positive=float((by_corpus>0).mean()),
                    weighting='Equal independent-corpus weight, then paired training seeds. Bootstrap resamples corpus and seed axes with common beta weights.')
            contrasts.append(record)
    return dict(complete=not missing and all(r['complete'] for r in contrasts),
        expected_cases=len(manifest['cases']),observed_cases=len(completed),contrasts=contrasts)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();manifest=load_json(args.manifest);base=Path(args.manifest).resolve().parent;out=Path(args.out)
    if manifest.get('expected_cases')!=len(manifest['cases']):raise ValueError('List every registered case, including missing files; expected_cases must agree')
    if set(manifest.get('expected_corpus_ids',[]))!={c['corpus_id'] for c in manifest['cases']}:raise ValueError('Registered corpus coverage differs from manifest cases')
    if len({c['name'] for c in manifest['cases']})!=len(manifest['cases']):raise ValueError('Duplicate synthetic case IDs')
    completed=[];missing=[]
    for case in manifest['cases']:
        if any(s in case['name'] for s in ('/','\\','..')):raise ValueError('Unsafe case name')
        paths=[(base/case[k]).resolve() for k in ('cal_codes','test_codes','cal_truth','test_truth')]
        if not all(q.is_file() for q in paths):missing.append(dict(case=case['name'],paths=[str(q) for q in paths if not q.is_file()]));continue
        # Require explicit disjoint generation seeds or source IDs, not a claimed train/test label.
        if not case.get('cal_data_id') or not case.get('test_data_id') or case['cal_data_id']==case['test_data_id']:raise ValueError('Distinct calibration/test data IDs required')
        identity=key(dict(case=case,hashes=[digest(q) for q in paths],code=digest(__file__),ridge=manifest.get('ridge',1e-3)))
        target=out/(case['name']+'.json')
        if target.exists():
            r=load_json(target)
            if r['input_id']!=identity:raise ValueError('Inputs changed for '+case['name'])
        else:
            arrays=[np.load(q,allow_pickle=False).astype(np.float64) for q in paths]
            r=dict(case=case,input_id=identity,metrics=score(*arrays,ridge=manifest.get('ridge',1e-3)))
            atomic_json(target,r)
        completed.append(r);print('[pc1-truth]',case['name'],r['metrics']['mean_one_to_one_r2'],flush=True)
    atomic_json(out/'coverage.json',dict(complete=not missing,expected=len(manifest['cases']),present=len(completed),missing=missing,
        aggregation='Group by corpus_id first, then paired training seed and beta. Fits sharing a corpus are not independent corpora.'))
    summary=summarize_cases(manifest,completed,missing)
    atomic_json(out/'TRUTH_RESULTS.json',summary)
    if not summary['complete']:raise SystemExit(2)


if __name__=='__main__':main()
