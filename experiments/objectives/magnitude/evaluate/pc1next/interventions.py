"""B3: dose, displacement, direction, baseline, and equal-size populations.

An energy-matched partial zero is a direction test, not support removal. Every
unit reports the fraction of active coordinates extinguished or made negative.
No model family is deemed special merely because unscaled zeroing travels farther.
"""
from pathlib import Path
import numpy as np
import torch
from .backend import scores,valid_predictors
from .analysis import top_ids,estimate
from .cache import blocks,cache_path
from .geometry import token_arrays,matched_subset
from .io import atomic_json,atomic_npz,completed,rng_for,ratio,load_json


def populations(cfg,out,run,cal):
    path=Path(out)/'rank_tables'/run['name']/'selection.npz'
    if not path.exists():raise FileNotFoundError('B1 selection ranks required for '+run['name'])
    with np.load(path) as n:tab={k:n[k] for k in n.files}
    score=estimate(tab,'dce_sum',cfg.get('primary_baseline','model'))
    sets={};missing=[]
    for budget in cfg.get('intervention_budgets',[12,50]):
        if cal['alive'].sum()<budget:missing.append(f'importance_{budget}:insufficient_alive');continue
        imp=top_ids(score,cal['alive'],budget);sets[f'importance_{budget}']=imp
        group=cal['group']
        if len(group)<budget:missing.append(f'structural_{budget}:insufficient_group');continue
        # Calibration-only random sample: never pick this set using test losses.
        g=np.sort(rng_for('structural-budget',run['name'],budget).choice(group,budget,replace=False))
        sets[f'structural_{budget}']=g
        control_cal={k:v for k,v in cal.items()};control_cal['alive']=cal['alive'].copy();control_cal['alive'][group]=False
        ctrl=matched_subset(control_cal,g,rng_for('structural-control',run['name'],budget),disjoint=True)
        if ctrl is None:missing.append(f'control_{budget}:infeasible')
        else:sets[f'control_{budget}']=ctrl
    if cfg.get('full_group_interventions',False) and len(cal['group']):sets['full_structural_group']=cal['group']
    return sets,missing


def positive_donor(z,bank,ids,sequence_ids,positions,run_name,pop,draw):
    """Independent magnitude donors, stratified position zero / remaining positions."""
    donor=np.zeros_like(z);fallback=np.zeros(z.shape[:2],dtype=float)
    for b,sequence in enumerate(sequence_ids):
        for j,f in enumerate(ids):
            for stratum in (0,1):
                pos=np.flatnonzero((positions==0) if stratum==0 else (positions>0))
                choices=bank.get((int(f),stratum),np.array([]))
                if not len(choices):
                    choices=bank.get((int(f),2),np.array([]));fallback[b,pos]+=1/len(ids)
                if not len(choices):continue
                rng=rng_for('positive-donor',run_name,pop,int(f),str(sequence),draw,stratum)
                donor[b,pos,j]=rng.choice(choices,size=len(pos))
    return donor*(z>0),fallback


def norm_match(direction,target):
    """Exact per-token Euclidean match wherever the direction has nonzero norm."""
    a=torch.linalg.vector_norm(direction.double(),dim=-1)
    b=torch.linalg.vector_norm(target.double(),dim=-1)
    scale=torch.where(a>1e-20,b/a,torch.zeros_like(a)).float()
    return direction*scale[...,None],scale,(a<=1e-20)&(b>1e-20)


def execute(be,sae,cfg,run,data,cal,out,pid):
    sets,missing=populations(cfg,out,run,cal)
    zc,_,_=token_arrays(cfg,out,run,'calibration',data['calibration'])
    positions=np.concatenate([np.flatnonzero(m) for m in data['calibration']['mask']])
    all_features=sorted({int(j) for ids in sets.values() for j in ids});bank={}
    for j in all_features:
        for s in (0,1,2):
            ix=(zc[:,j]>0)&((positions==0) if s==0 else ((positions>0) if s==1 else True))
            bank[(j,s)]=zc[ix,j]
    d=torch.as_tensor(cal['decoder'],dtype=torch.float32,device=be.device);v=be.v
    modes=['zero','positive_resample','zero_direction_energy_matched','random_perp_energy_matched','zero_pc1','zero_perp']
    doses=cfg.get('doses',[0.01,0.03,0.1,0.3,1.0]);baselines=cfg.get('baselines',['model','splice'])
    unit_paths=[]
    for start,sl in blocks(data['test'],cfg.get('sequence_batch',2)):
        ids,mask=be.tensors(data['test'],sl);original,x=be.forward(ids,mask);z,hat=be.encode(sae,x,run)
        spliced,_=be.forward(ids,mask,hat);refs={'model':original,'splice':spliced};states={'model':x.float(),'splice':hat}
        pred=valid_predictors(mask);nb,nt=ids.shape
        for pop,features in sets.items():
            f=torch.as_tensor(features,device=be.device);zg=z[...,f];dg=d[f]
            for draw in range(cfg.get('resample_draws',3)):
                path=Path(out)/'intervention_units'/run['name']/pop/f'b{start:07d}_r{draw:03d}.npz';unit_paths.append(path)
                if completed(path,pid):continue
                donor,fallback=positive_donor(zg.cpu().numpy(),bank,features,data['test']['seq'][sl],np.arange(nt),run['name'],pop,draw)
                donor=torch.as_tensor(donor,device=be.device)
                delta0=-(zg@dg)*mask[...,None];deltar=((donor-zg)@dg)*mask[...,None]
                matched,scale,infeasible=norm_match(delta0,deltar)
                random=np.stack([rng_for('random-perp',run['name'],pop,str(s),draw).normal(size=(nt,len(v))) for s in data['test']['seq'][sl]])
                random=torch.as_tensor(random,dtype=torch.float32,device=be.device);random-= (random@v)[...,None]*v
                random,_,rand_infeasible=norm_match(random,deltar)
                par=(delta0@v)[...,None]*v
                deltas=[delta0,deltar,matched,random,par,delta0-par]
                rows=[];labels=[]
                for mode,delta in zip(modes,deltas):
                    for dose in doses:
                        perturb=delta*dose
                        for baseline in baselines:
                            logits,_=be.forward(ids,mask,states[baseline]+perturb)
                            ce,kl,count=scores(refs[baseline],logits,ids,mask)
                            energy=(perturb.double().square().sum(-1)*pred).sum(-1)
                            par_energy=((perturb.double()@v.double()).square()*pred).sum(-1)
                            # Only the three coefficient-space operations have a
                            # meaningful active-support count in this dictionary.
                            zz=None
                            if mode=='zero':zz=(1-dose)*zg
                            elif mode=='positive_resample':zz=zg+dose*(donor-zg)
                            elif mode=='zero_direction_energy_matched':zz=zg*(1-dose*scale[...,None])
                            active=(zg>0)&pred[...,None]
                            extinguished=((zz==0)&active).sum((1,2)) if zz is not None else torch.full((nb,),float('nan'),device=be.device)
                            negative=((zz<0)&active).sum((1,2)) if zz is not None else torch.full((nb,),float('nan'),device=be.device)
                            rows.append(np.stack([ce.cpu().numpy(),kl.cpu().numpy(),count.cpu().numpy(),energy.cpu().numpy(),par_energy.cpu().numpy(),
                                active.sum((1,2)).cpu().numpy(),extinguished.cpu().numpy(),negative.cpu().numpy()]))
                            labels.append([mode,str(dose),baseline])
                atomic_npz(path,protocol_id=pid,features=features,labels=np.asarray(labels),
                    fields=np.asarray(['dce_sum','kl_sum','count','energy_sum','parallel_energy_sum','active_count','extinguished_count','negative_count']),
                    values=np.asarray(rows),sequence_indices=np.arange(sl.start,sl.stop),
                    energy_match_infeasible=infeasible.cpu().numpy(),random_match_infeasible=rand_infeasible.cpu().numpy(),
                    alpha=scale.cpu().numpy(),positive_bank_fallback=fallback)
                print(f'[pc1-intervention] {run["name"]} {pop} b={start} draw={draw}',flush=True)
    report=summarize(cfg,run,data['test'],out,pid,sets,unit_paths,missing)
    return report


def summarize(cfg,run,test,out,pid,sets,paths,missing):
    sums={};article_sums={};fallback=0.;unmatch=0.;nmatch=0
    for path in paths:
        if not completed(path,pid):missing.append(str(path));continue
        pop=path.parent.name
        with np.load(path) as n:
            ix=n['sequence_indices'];fields=n['fields'].tolist()
            fallback+=float(n['positive_bank_fallback'].sum());unmatch+=float(n['energy_match_infeasible'].sum());nmatch+=n['alpha'].size
            for label,values in zip(n['labels'],n['values']):
                identity=(pop,*label);acc=sums.setdefault(identity,np.zeros(len(fields)))
                acc+=np.nansum(values,axis=1)
                for j,s in enumerate(ix):
                    art=str(test['articles'][s]);article_sums.setdefault(identity,{}).setdefault(art,np.zeros(len(fields)))
                    article_sums[identity][art]+=np.nan_to_num(values[:,j])
    rows=[]
    for identity,s in sums.items():
        pop,mode,dose,baseline=identity
        row=dict(population=pop,mode=mode,dose=float(dose),baseline=baseline,n_features=len(sets[pop]),
            dce=ratio(s[0],s[2]),kl=ratio(s[1],s[2]),msd=ratio(s[3],s[2]),parallel_msd=ratio(s[4],s[2]),
            kl_per_squared_displacement=ratio(s[1],s[3]),
            extinguished_fraction=ratio(s[6],s[5]) if mode in ('zero','positive_resample','zero_direction_energy_matched') else None,
            negative_fraction=ratio(s[7],s[5]) if mode in ('zero','positive_resample','zero_direction_energy_matched') else None,
            per_article={a:dict(zip(fields,ss.tolist())) for a,ss in article_sums[identity].items()})
        rows.append(row)
    index={(r['population'],r['mode'],r['dose'],r['baseline']):r for r in rows};interactions=[]
    for r in rows:
        if r['mode']!='zero':continue
        pref=(r['population'],);suf=(r['dose'],r['baseline'])
        a=index.get((*pref,'zero_pc1',*suf));b=index.get((*pref,'zero_perp',*suf))
        if a and b:interactions.append(dict(population=r['population'],dose=r['dose'],baseline=r['baseline'],
            kl_nonadditivity=r['kl']-a['kl']-b['kl'],dce_nonadditivity=r['dce']-a['dce']-b['dce']))
    report=dict(protocol_id=pid,run=run['name'],complete=not missing,missing=missing,rows=rows,
        decomposition_interactions=interactions,energy_match_infeasible_fraction=ratio(unmatch,nmatch),positive_bank_fallback_fraction=ratio(fallback,nmatch),
        support_warning='Only dose-one zero removes support. Matched zero direction is generally partial or extrapolated; inspect extinguished/negative fractions.',
        inference='Article sufficient statistics retained; repeated donor draws are averaged within article, not counted as new articles.')
    atomic_json(Path(out)/'interventions'/(run['name']+'.json'),report);return report
