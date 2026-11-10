"""Paper-facing mechanism contrasts, with paired seed and article resampling."""
import argparse
from pathlib import Path
import numpy as np
from .io import atomic_json,load_json,rng_for,ratio


def paired_effect(pairs,metric,draws,article_resample,tag):
    """pairs: (left row,right row,recipient seed,optional donor seed)."""
    if not pairs:return None
    seeds=sorted({s for _,_,a,b in pairs for s in (a,b) if s is not None})
    seed_index={s:i for i,s in enumerate(seeds)}
    arts=sorted(set.intersection(*[set(r['articles']) for l,r,_,_ in pairs]))
    if not arts:return dict(unavailable='No shared articles')
    for l,r,_,_ in pairs:
        if set(l['articles'])!=set(arts) or set(r['articles'])!=set(arts):raise ValueError('Unequal article coverage; no complete-case trimming')
    arrays=[]
    field={'kl':'kl_sum','dce':'dce_sum','msd':'energy_sum'}[metric]
    for l,r,a,b in pairs:
        arrays.append((np.array([[l['articles'][x][field],l['articles'][x]['count'],r['articles'][x][field],r['articles'][x]['count']] for x in arts]),a,b))
    def values(sf,aw):
        left=right=den=0.
        for mat,a,b in arrays:
            weight=sf[seed_index[a]]*(sf[seed_index[b]] if b is not None else 1)
            if not weight:continue
            vv=(mat*aw[:,None]).sum(0)
            if min(vv[1],vv[3])<=0:continue
            left+=weight*vv[0]/vv[1];right+=weight*vv[2]/vv[3];den+=weight
        return (left/den,right/den) if den else (np.nan,np.nan)
    ones=np.ones(len(seeds));aw=np.ones(len(arts));l,r=values(ones,aw)
    rng=rng_for('mechanism-bootstrap',tag,metric);boots=[]
    for _ in range(draws):
        sf=rng.multinomial(len(seeds),np.full(len(seeds),1/len(seeds)))
        weights=rng.multinomial(len(arts),np.full(len(arts),1/len(arts))) if article_resample else aw
        ll,rr=values(sf,weights)
        if np.isfinite(ll+rr):boots.append(rr-ll)
    loso=[]
    for i in range(len(seeds)):
        sf=ones.copy();sf[i]=0;ll,rr=values(sf,aw)
        if np.isfinite(ll+rr):loso.append(rr-ll)
    return dict(left_mean=l,right_mean=r,delta=r-l,ratio_of_means=ratio(r,l),
        delta_interval_95=np.quantile(boots,[.025,.975]).tolist() if boots else None,
        n_training_seeds=len(seeds),n_articles=len(arts),article_resampled=article_resample,
        leave_one_seed_out_delta=loso,n_rows=len(pairs),bootstrap_valid_draws=len(boots),
        weighting='Average token-weighted effect across training seeds or distinct ordered seed pairs. Bootstrap reuses the same seed/article draws in both arms.')


def run(cfg,out):
    out=Path(out);records=[];missing=[];sources={}
    expected=[r for r in cfg['runs'] if float(r['beta']) in cfg.get('mechanism_betas',[0.,.3])]
    for stage in ('interventions','interchange','surgery'):
        sources[stage]=[]
        for r in expected:
            p=out/'interventions'/(r['name']+'.json') if stage=='interventions' else out/stage/r['name']/'summary.json'
            if not p.exists():missing.append(f'{stage}/{r["name"]}');continue
            obj=load_json(p);sources[stage].append((r,obj))
            if not obj.get('complete'):missing.append(f'{stage}/{r["name"]}:incomplete')
    grouped={}
    for r,obj in sources['interventions']:
        for row in obj['rows']:
            converted=dict(row);converted['articles']=converted.pop('per_article')
            # B3 units store energy_sum; include it under the common schema unchanged.
            group=(r['condition'],float(r['beta']),row['population'],row['dose'],row['baseline'])
            grouped.setdefault(group,{})[(r['seed'],row['mode'])]=converted
    comparisons=[('positive_resample','zero'),('positive_resample','zero_direction_energy_matched'),('positive_resample','random_perp_energy_matched')]
    for group,items in grouped.items():
        seeds=sorted({s for s,m in items})
        for left,right in comparisons:
            pairs=[(items[(s,left)],items[(s,right)],s,None) for s in seeds if (s,left) in items and (s,right) in items]
            for metric in ('kl','dce','msd'):
                records.append(dict(stage='interventions',group=group,left=left,right=right,metric=metric,
                    estimate=paired_effect(pairs,metric,cfg.get('bootstrap_draws',1000),cfg.get('split_policy','article')=='article',[group,left,right])))
    grouped={}
    for r,obj in sources['interchange']:
        for row in obj['rows']:
            group=(r['condition'],float(r['beta']),row['dose'],row['baseline'])
            grouped.setdefault(group,{})[(row['recipient_seed'],row['donor_seed'],row['mode'])]=row
    for group,items in grouped.items():
        pairs_of_seeds=sorted({(a,b) for a,b,m in items})
        for right in ('shift_perp','calibration_mean_perp','shift_perp_energy_matched'):
            pairs=[(items[(a,b,'same_input_perp')],items[(a,b,right)],a,b) for a,b in pairs_of_seeds if (a,b,'same_input_perp') in items and (a,b,right) in items]
            for metric in ('kl','dce','msd'):
                records.append(dict(stage='interchange',group=group,left='same_input_perp',right=right,metric=metric,
                    estimate=paired_effect(pairs,metric,cfg.get('bootstrap_draws',1000),cfg.get('split_policy','article')=='article',[group,right])))
    surgery=[]
    for run,obj in sources['surgery']:
        rows=[r for r in obj.get('records',[]) if r['fraction']>0];valid=[r for r in rows if r['admissible_on_heldout']]
        surgery.append(dict(run=run['name'],condition=run['condition'],beta=run['beta'],seed=run['seed'],
            available_pairs=obj.get('available_ordered_pairs',0),nonzero_trials=len(rows),admissible_trials=len(valid),
            max_roundtrip_kl=max([r['roundtrip_kl'] for r in rows],default=None),
            max_relative_reconstruction_squared_error=max([r['reconstruction_relative_squared_error'] for r in rows],default=None),
            admissible_mean_l1_ratio=float(np.mean([r['unit_decoder_l1_ratio'] for r in valid])) if valid else None,
            admissible_mean_first_atom_kl_change=float(np.mean([r['effects']['kl'][2]-r['effects']['kl'][0] for r in valid])) if valid else None))
    report=dict(complete=not missing,missing=missing,contrasts=records,surgery=surgery,
        warning='Do not turn exploratory dose/population contrasts into independent confirmatory tests; the experiment register names primary contrasts.')
    atomic_json(out/'MECHANISM_RESULTS.json',report);print(f'[pc1-summary] complete={not missing} contrasts={len(records)}',flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=run(load_json(a.config),a.out)
    if not result['complete']:raise SystemExit(2)
