"""Frozen top-importance and tile groups: role localization and calibrated displacement controls."""
from pathlib import Path
import numpy as np
import torch
from .io import loadnp,npz,read,write,rng,sha,ratio
from .model import make
from .lm import sae_forward,select_sequences,population_mask,scores

@torch.no_grad()
def execute(job,site,ref,settings,out,device,lm):
    out=Path(out);root=out/'causal'/job['id'];rank=out/'lm'/job['id'];fit=out/'fits'/job['id']
    if not (rank/'complete.json').exists():return dict(status='blocked_missing_actual_ranking')
    cp=fit/'final.pt';h=sha(cp)
    if read(rank/'complete.json')['checkpoint_sha']!=h:raise ValueError('Rank/SAE checkpoint mismatch')
    if (root/'complete.json').exists():
        done=read(root/'complete.json')
        if done['checkpoint_sha']!=h:raise ValueError('Stale causal result')
        return done
    blob=torch.load(cp,weights_only=True,map_location=device);model=make(blob['cfg'],ref['pc1']).to(device);model.load_state_dict(blob['state']);model.eval()
    table=loadnp(rank/'all_selection_table.npz');score=table['ce'][0].sum(1)/table['count'].sum()
    top=table['features'][np.lexsort((table['features'],-score))[:12]]
    arr=loadnp(fit/'arrays.npz');tiles=np.flatnonzero(arr['mask_tiles'])
    # Both a fixed size angular population and the actual top-importance population.
    tile_order=np.argsort(arr['cal_energy'][tiles])[::-1][:12];tile12=tiles[tile_order]
    pool=np.flatnonzero(arr['alive']&~arr['mask_structural']);pool=np.setdiff1d(pool,top)
    controls=[];target=arr['cal_energy'][top];rates=arr['cal_firing_rate']
    for f,e in zip(top,target):
        choices=np.setdiff1d(pool,controls)
        if not len(choices):break
        cost=abs(np.log((arr['cal_energy'][choices]+1e-20)/(e+1e-20)))+abs(rates[choices]-rates[f])*10
        controls.append(int(choices[np.argmin(cost)]))
    groups={'top12':top,'tile12':tile12,'matched_controls':np.array(controls,int)}
    cal=np.load(site['splits']['calibration']['x'],mmap_mode='r');calix=rng('causal_calibration',site['name']).choice(len(cal),min(8192,len(cal)),False)
    _,cz=sae_forward(model,torch.tensor(np.array(cal[calix],np.float32),device=device),ref,job['arm'])
    D=model.W_dec*float(ref['scale']);v=torch.tensor(ref['pc1'],dtype=torch.float32,device=device)
    pools={int(f):cz[:,f][cz[:,f]>0] for f in np.unique(np.concatenate(list(groups.values())))}
    def replacement(zg,features,seed):
        r=rng('causal_donors',seed);replacement=torch.zeros_like(zg)
        for j,f in enumerate(features):
            p=pools[int(f)]
            if not len(p):raise ValueError('Selected candidate lacks positive calibration support')
            ix=r.integers(len(p),size=zg[...,j].shape);val=p[torch.tensor(ix,device=device)]
            replacement[...,j]=torch.where(zg[...,j]>0,val,torch.zeros_like(val))
        return replacement
    alpha={}
    for name,features in groups.items():
        if not len(features):continue
        gz=cz[:,features];zd=-gz@D[features];rd=(replacement(gz,features,name)-gz)@D[features]
        alpha[name]=float((rd.square().sum()/zd.square().sum().clamp_min(1e-30)).sqrt())
    data=loadnp(site['splits']['test']['tokens']);seq=select_sequences(data,settings.get('causal_sequences',128));rows=[]
    for start in range(0,len(seq),settings.get('sequence_batch',1)):
        ix=seq[start:start+settings.get('sequence_batch',1)];path=root/f'block_{start:05d}.json'
        if path.exists():rows.extend(read(path));continue
        ids=torch.tensor(data['input_ids'][ix],device=device);mask=torch.tensor(data['attention_mask'][ix],dtype=torch.bool,device=device)
        original,x=lm.forward(ids,mask);hat,z=sae_forward(model,x,ref,job['arm']);spliced,_=lm.forward(ids,mask,hat)
        a=((x.float()-torch.tensor(ref['mean'],dtype=torch.float32,device=device))@v)
        first=mask&~population_mask(ids,mask,'after_first',site['lm'])
        pops={'all':mask,'first':first,'ordinary':population_mask(ids,mask,'native',site['lm'])&~first,
            'tail':mask&(abs(a)>float(ref['clip_q']))}
        block=[]
        for group,features in groups.items():
            if not len(features):continue
            gz=z[...,features];zd=-gz@D[features];rd=(replacement(gz,features,(group,start))-gz)@D[features]
            gen=torch.Generator(device=device).manual_seed(int(start+18273));noise=torch.randn(zd.shape,generator=gen,device=device)
            noise=noise/noise.norm(dim=-1,keepdim=True).clamp_min(1e-30)*rd.norm(dim=-1,keepdim=True)
            for pop,pm in pops.items():
                for mode,base_delta in [('zero',zd),('positive_resample',rd),('zero_matched',zd*alpha[group]),('random_direction',noise)]:
                    for dose in (.1,1.):
                        delta=base_delta*pm[...,None]*dose
                        for baseline,base,state in [('model',original,x),('splice',spliced,hat)]:
                            logits,_=lm.forward(ids,mask,state+delta);ce,kl,count=scores(base,logits,ids,mask)
                            for b,art in enumerate(data['article_ids'][ix]):
                                block.append(dict(group=group,population=pop,mode=mode,dose=dose,baseline=baseline,article=str(art),
                                    n_features=len(features),dce_sum=float(ce[b]),kl_sum=float(kl[b]),count=int(count[b]),
                                    displacement=float(delta[b].double().square().sum()),edited_positions=int(pm[b].sum())))
        write(path,block);rows.extend(block);print(f'[pc1night-causal] {job["id"]} seq={start+len(ix)}/{len(seq)}',flush=True)
    grouped={}
    for r in rows:
        key=tuple(r[k] for k in ('group','population','mode','dose','baseline'));grouped.setdefault(key,[]).append(r)
    report=[]
    for k,rr in grouped.items():
        n=sum(r['count'] for r in rr);report.append(dict(zip(('group','population','mode','dose','baseline'),k),
            dce=sum(r['dce_sum'] for r in rr)/n,kl=sum(r['kl_sum'] for r in rr)/n,
            displacement_per_prediction=sum(r['displacement'] for r in rr)/n,n_predictions=n))
    result=dict(status='complete',checkpoint_sha=h,summary=report,groups=groups,calibration_alpha=alpha,
        control_balance={name:dict(n=len(fs),mean_energy=float(arr['cal_energy'][fs].mean()) if len(fs) else None,
            mean_rate=float(arr['cal_firing_rate'][fs].mean()) if len(fs) else None) for name,fs in groups.items()},
        note='Dose calibrated on calibration activations; report actual test displacement mismatch. Signed-code arm is excluded by default. All causal rows carry actual article IDs.')
    write(root/'complete.json',result);return result

def summarize(plan,out):
    out=Path(out);arms=plan['config'].get('rank_arms',['baseline','beta_0.3','no_first','canonical_init','clip_restore'])
    expected=[j for j in plan['jobs'] if j['kind']=='real' and j['arm']['name'] in arms];missing=[];reports=[]
    for job in expected:
        p=out/'causal'/job['id']/'complete.json'
        if not p.exists():missing.append(job['id']);continue
        rec=read(p);cells={tuple(r[k] for k in ('group','population','mode','dose','baseline')):r for r in rec['summary']}
        comparisons=[]
        for group in rec['groups']:
            for pop in ('all','first','ordinary','tail'):
                for dose in (.1,1.):
                    for baseline in ('model','splice'):
                        z=cells.get((group,pop,'zero_matched',dose,baseline));r=cells.get((group,pop,'positive_resample',dose,baseline))
                        if z and r:comparisons.append(dict(group=group,population=pop,dose=dose,baseline=baseline,
                            kl_ratio=ratio(z['kl'],r['kl']),dce_difference=z['dce']-r['dce'],
                            actual_test_displacement_ratio=ratio(z['displacement_per_prediction'],r['displacement_per_prediction'])))
        reports.append(dict(job=job,comparisons=comparisons,calibration_alpha=rec['calibration_alpha'],control_balance=rec['control_balance']))
    result=dict(expected=len(expected),finished=len(reports),missing=missing,complete=not missing,per_run=reports,
        uncertainty='Per-article sufficient statistics retained in block JSONs. Ratios alone do not establish a cross-model law; no such law is automatically declared.')
    write(out/'CAUSAL_SUMMARY.json',result);return result
