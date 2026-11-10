"""B2: exact projected agreement, shared-offset controls, and joint subset nulls.

Null draw r selects a subset in EVERY seed and then computes the same mean over
seed pairs as the real group. Pair/draw combinations are not independent draws.
Subset matching uses calibration firing rate and individual contribution energy.
Actual aggregate-energy and PC1-fraction mismatches are reported, never renamed
as exact matching. Component calibration is a separately labelled sensitivity.
"""
import itertools
from pathlib import Path
import numpy as np
from .cache import blocks,cache_path
from .io import atomic_json,atomic_npz,load_json,rng_for,ratio
from .projection import compare_outputs


def token_arrays(cfg,out,run,split,data):
    zs=[];articles=[];seq=[]
    for start,sl in blocks(data,cfg.get('sequence_batch',2)):
        with np.load(cache_path(out,run,split,start)) as n:zs.append(n['z'][data['mask'][sl]])
        for j in range(sl.start,sl.stop):
            positions=np.flatnonzero(data['mask'][j])
            articles.extend([data['articles'][j]]*len(positions));seq.extend([str(data['seq'][j])+':'+str(t) for t in positions])
    return np.concatenate(zs),np.asarray(articles),np.asarray(seq)


def aggregate(z,d,ids,chunk=2048):
    out=np.empty((len(z),d.shape[1]),dtype=np.float64)
    for s in range(0,len(z),chunk):out[s:s+chunk]=z[s:s+chunk,ids].astype(np.float64)@d[ids]
    return out


def properties(y,z,d,ids,v):
    yp=y-(y@v)[:,None]*v
    dq=d[ids]-(d[ids]@v)[:,None]*v
    individual=np.square(z[:,ids].astype(np.float64)).sum(0)@np.square(dq).sum(1)
    total=np.square(y).sum();perp=np.square(yp).sum()
    return dict(n_atoms=len(ids),total_energy=float(total),perp_energy=float(perp),
        pc1_fraction=ratio(total-perp,total),cancellation=ratio(individual,perp),individual_perp_energy=float(individual))


def matched_subset(cal,real,rng,disjoint=False):
    pool=np.flatnonzero(cal['alive'])
    if disjoint:pool=np.setdiff1d(pool,real)
    if len(pool)<len(real):return None
    # Deterministic randomized greedy assignment without replacement. This is not
    # an exact exchangeability test: diagnostics determine where matching worked.
    cov=np.column_stack([np.log(np.maximum(cal['firing_rate'],1e-12)),
                         np.log(np.maximum(cal['individual_energy'],1e-20))])
    scale=np.maximum(np.std(cov[pool],axis=0),.1);cov=cov/scale
    used=np.zeros(len(pool),bool);chosen=[]
    for target in rng.permutation(real):
        distance=np.square(cov[pool]-cov[target]).sum(1);distance[used]=np.inf
        # Sample among 8 nearest, so draws do not just repeat a single greedy map.
        count=min(8,int((~used).sum()));near=np.argpartition(distance,count-1)[:count]
        weights=np.exp(-(distance[near]-distance[near].min()));weights/=weights.sum()
        j=rng.choice(near,p=weights);used[j]=True;chosen.append(pool[j])
    return np.sort(chosen)


def shift_within_articles(y,articles,rng):
    shifted=y.copy();moved=0
    for a in np.unique(articles):
        ix=np.flatnonzero(articles==a)
        if len(ix)<2:continue
        # Large circular offsets reduce nearly identical adjacent-token controls.
        low=max(1,len(ix)//4);high=max(low+1,3*len(ix)//4+1)
        shift=int(rng.integers(low,high))
        shifted[ix]=np.roll(y[ix],shift,axis=0);moved+=len(ix)
    return shifted,moved/len(y)


def compare_set(ys,v,articles):
    rows=[compare_outputs(ys[i],ys[j],v,articles) for i,j in itertools.combinations(range(len(ys)),2)]
    def avg(path):
        vals=[]
        for r in rows:
            for part in path:r=r.get(part,{}) if isinstance(r,dict) else None
            if isinstance(r,(int,float)) and np.isfinite(r):vals.append(r)
        return float(np.mean(vals)) if vals else None
    return dict(n_seed_pairs=len(rows),total_mse=avg(['total_normalized_mse']),
        exact_offaxis=avg(['exact_offaxis_disagreement']),
        globally_centered_offaxis=avg(['evaluation_centered','centered_disagreement']),
        within_article_centered_offaxis=avg(['within_article_centered','disagreement']),
        shared_mean_fraction=avg(['mean_vector_fraction_of_offaxis_energy']),pairs=rows)


def execute(cfg,out,data,pid):
    cases={}
    for run in cfg['runs']:
        if float(run['beta']) not in cfg.get('mechanism_betas',[0.0,0.3]):continue
        cases.setdefault((run['condition'],float(run['beta'])),[]).append(run)
    results=[];missing=[]
    for (condition,beta),runs in cases.items():
        root=Path(out)/'geometry'/f'{condition}_beta{beta:g}';root.mkdir(parents=True,exist_ok=True)
        if len(runs)<2:continue
        loaded=[]
        for run in runs:
            cp=Path(out)/'calibration'/(run['name']+'.npz')
            if not cp.exists():missing.append(run['name']);continue
            try:
                with np.load(cp) as n:cal={k:n[k] for k in n.files}
                zc,_,_=token_arrays(cfg,out,run,'calibration',data['calibration'])
                zt,articles,seq=token_arrays(cfg,out,run,'test',data['test'])
            except FileNotFoundError:missing.append(run['name']);continue
            if not len(cal['group']):missing.append(run['name']+':empty_group');continue
            d=cal['decoder'];v=cal['pc1'];g=cal['group']
            yc=aggregate(zc,d,g);yt=aggregate(zt,d,g)
            props=properties(yt,zt,d,g,v)
            loaded.append(dict(run=run,cal=cal,zc=zc,zt=zt,yc=yc,yt=yt,properties=props))
            atomic_npz(Path(out)/'aggregates'/(run['name']+'.npz'),protocol_id=pid,
                y=yt.astype(np.float32),calibration_mean=yc.mean(0),article_ids=articles,token_ids=seq,feature_ids=g)
        if len(loaded)!=len(runs):continue
        real=compare_set([r['yt'] for r in loaded],v,articles)
        real['per_seed']=[dict(run=r['run']['name'],**r['properties']) for r in loaded]
        draws=[]
        for draw in range(cfg.get('null_draws',100)):
            path=root/f'draw{draw:04d}.json'
            if path.exists():
                old=load_json(path)
                if old['protocol_id']!=pid:raise ValueError('Stale geometry draw')
                draws.append(old);continue
            row=dict(protocol_id=pid,draw=draw,nulls={})
            for mode in ('inclusive','disjoint'):
                ys=[];scaled=[];diagnostics=[]
                for r in loaded:
                    cal=r['cal'];d=cal['decoder'];g=cal['group']
                    ids=matched_subset(cal,g,rng_for('subset',r['run']['name'],mode,draw),mode=='disjoint')
                    if ids is None:break
                    yc=aggregate(r['zc'],d,ids);yt=aggregate(r['zt'],d,ids)
                    props=properties(yt,r['zt'],d,ids,v)
                    calreal=properties(r['yc'],r['zc'],d,g,v);calnull=properties(yc,r['zc'],d,ids,v)
                    # Independently scale parallel and perpendicular components to
                    # match real CALIBRATION component energies, never test energy.
                    rp=calreal['total_energy']-calreal['perp_energy'];np_=calnull['total_energy']-calnull['perp_energy']
                    if min(np_,calnull['perp_energy'])<=1e-20:break
                    ap=np.sqrt(max(rp,0)/np_);aq=np.sqrt(calreal['perp_energy']/calnull['perp_energy'])
                    par=(yt@v)[:,None]*v;scaled.append(ap*par+aq*(yt-par));ys.append(yt)
                    diagnostics.append(dict(run=r['run']['name'],features=ids.tolist(),
                        overlap_fraction=len(set(ids)&set(g))/len(g),
                        calibration_total_energy_ratio=ratio(calnull['total_energy'],calreal['total_energy']),
                        test_total_energy_ratio=ratio(props['total_energy'],r['properties']['total_energy']),
                        calibration_pc1_fraction_real=calreal['pc1_fraction'],calibration_pc1_fraction_null=calnull['pc1_fraction'],
                        test_pc1_fraction_real=r['properties']['pc1_fraction'],test_pc1_fraction_null=props['pc1_fraction'],
                        test_perp_energy_ratio=ratio(props['perp_energy'],r['properties']['perp_energy']),
                        log_rate_mean_difference=float(np.log(np.maximum(cal['firing_rate'][ids],1e-12)).mean()-np.log(np.maximum(cal['firing_rate'][g],1e-12)).mean()),
                        cancellation_null=props['cancellation'],cancellation_real=r['properties']['cancellation'],
                        parallel_scale=float(ap),perpendicular_scale=float(aq)))
                if len(ys)==len(loaded):
                    row['nulls'][mode]=dict(raw=compare_set(ys,v,articles),component_calibrated=compare_set(scaled,v,articles),diagnostics=diagnostics)
                else:row['nulls'][mode]=dict(infeasible=True,reason='Insufficient disjoint pool or zero calibration component energy')
            shifted=[loaded[0]['yt']];moved=[]
            for r in loaded[1:]:
                yy,frac=shift_within_articles(r['yt'],articles,rng_for('shift',r['run']['name'],draw));shifted.append(yy);moved.append(frac)
            row['nulls']['within_article_shift']=dict(raw=compare_set(shifted,v,articles),moved_token_fraction=moved)
            atomic_json(path,row);draws.append(row)
            print(f'[pc1-geometry] {condition} beta={beta:g} joint_null_draw={draw}',flush=True)
        comparison=[]
        for mode in ('inclusive','disjoint','within_article_shift'):
            for version in ('raw','component_calibrated'):
                for metric in ('total_mse','exact_offaxis','globally_centered_offaxis','within_article_centered_offaxis'):
                    null=[r['nulls'].get(mode,{}).get(version,{}).get(metric) for r in draws]
                    null=[x for x in null if x is not None];rv=real[metric]
                    if not null or rv is None:continue
                    comparison.append(dict(null=mode,version=version,metric=metric,real=rv,null_mean=float(np.mean(null)),
                        null_over_real=ratio(np.mean(null),rv),n_joint_draws=len(null),
                        monte_carlo_lower_tail=(1+sum(x<=rv for x in null))/(1+len(null)),
                        inference='Monte Carlo rank under this constructed null; not an independent-sample or exchangeability p-value'))
        result=dict(condition=condition,beta=beta,real=real,comparisons=comparison,n_joint_draws=len(draws))
        atomic_json(root/'summary.json',result);results.append(result)
    report=dict(protocol_id=pid,complete=not missing,missing=missing,cases=results)
    atomic_json(Path(out)/'GEOMETRY_RESULTS.json',report);return report
