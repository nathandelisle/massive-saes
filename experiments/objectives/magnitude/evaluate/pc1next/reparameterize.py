"""B4: an exact reconstruction-preserving coordinate intervention on trained SAEs.

For an ordered coactive pair, replace d1 by d1+t*d2 and z2 by z2-t*z1.
Renormalize decoder rows and compensate coefficients. The aggregate is unchanged
algebraically. t is chosen using calibration only; held-out nonnegativity/support
violations are reported and invalidate the admissible-coordinate interpretation.

This is a POST-ENCODER reparameterization. It does not establish realizability by
another affine TopK encoder or a global continuum of trained optima.
"""
from pathlib import Path
import numpy as np
import torch
from .geometry import token_arrays
from .cache import blocks
from .backend import scores,valid_predictors
from .io import atomic_json,atomic_npz,completed,ratio


def choose_pairs(z,cal,max_candidates=64,max_pairs=12):
    g=cal['group'];g=g[np.argsort(-cal['firing_rate'][g],kind='stable')][:max_candidates]
    candidates=[]
    for i in g:
        on=z[:,i]>0
        if not on.any():continue
        for j in g:
            if i==j or np.any(z[on,j]<=0):continue
            bound=float(np.min(z[on,j].astype(np.float64)/z[on,i]))
            if np.isfinite(bound) and bound>1e-8:candidates.append((int(i),int(j),bound))
    candidates.sort(key=lambda x:(-min(cal['firing_rate'][x[0]],cal['firing_rate'][x[1]]),-x[2],x[0],x[1]))
    chosen=[];used=set()
    for c in candidates:
        if c[0] in used or c[1] in used:continue
        chosen.append(c);used.update(c[:2])
        if len(chosen)>=max_pairs:break
    return chosen,len(candidates)


def transform_pair(z,d,i,j,t):
    first=d[i]+t*d[j];n1=torch.linalg.vector_norm(first);n2=torch.linalg.vector_norm(d[j])
    if min(n1,n2)<=1e-12:raise ValueError('Degenerate transformed decoder')
    nd=torch.stack((first/n1,d[j]/n2))
    nz=torch.stack((z[...,i]*n1,(z[...,j]-t*z[...,i])*n2),dim=-1)
    return nz,nd


def execute(be,sae,cfg,run,data,cal,out,pid):
    zc,_,_=token_arrays(cfg,out,run,'calibration',data['calibration'])
    pairs,available=choose_pairs(zc,cal,cfg.get('surgery_candidates',64),cfg.get('surgery_pairs',12))
    root=Path(out)/'surgery'/run['name'];root.mkdir(parents=True,exist_ok=True)
    if not pairs:
        result=dict(protocol_id=pid,run=run['name'],complete=True,admissible_calibration_pairs=0,
            verdict='Infeasible under the registered exact support constraint; do not relax using test data.')
        atomic_json(root/'summary.json',result);return result
    d=torch.as_tensor(cal['decoder'],dtype=torch.float32,device=be.device)
    fractions=cfg.get('surgery_fractions',[0.,0.25,0.5,0.9]);records=[]
    for pair_index,(i,j,bound) in enumerate(pairs):
        for fraction in fractions:
            t=fraction*bound
            accum=dict(count=0.,negative=0.,support_changed=0.,coeff_count=0.,before_l1=0.,after_l1=0.,
                aggregate_squared_error=0.,aggregate_energy=0.,roundtrip_kl=0.,roundtrip_dce=0.)
            ce=np.zeros(4);kl=np.zeros(4);energy=np.zeros(4);articles={}
            for start,sl in blocks(data['test'],cfg.get('sequence_batch',2)):
                path=root/f'p{pair_index:03d}_f{fraction:g}_b{start:07d}.npz'
                if not completed(path,pid):
                    ids,mask=be.tensors(data['test'],sl);_,x=be.forward(ids,mask);z,hat=be.encode(sae,x,run)
                    base,_=be.forward(ids,mask,hat);pred=valid_predictors(mask)
                    nz,nd=transform_pair(z,d,i,j,t)
                    old=z[...,i,None]*d[i]+z[...,j,None]*d[j];new=nz@nd
                    reconstruction_delta=(new-old)*mask[...,None]
                    roundtrip,_=be.forward(ids,mask,hat+reconstruction_delta)
                    rtce,rtkl,count=scores(base,roundtrip,ids,mask)
                    deltas=[-z[...,i,None]*d[i],-z[...,j,None]*d[j],-nz[...,0,None]*nd[0],-nz[...,1,None]*nd[1]]
                    cez=[];klz=[];ez=[]
                    for delta in deltas:
                        perturb=delta*mask[...,None];logits,_=be.forward(ids,mask,hat+perturb)
                        a,b,_=scores(base,logits,ids,mask);cez.append(a.cpu().numpy());klz.append(b.cpu().numpy())
                        ez.append((perturb.double().square().sum(-1)*pred).sum(-1).cpu().numpy())
                    zz=z[...,[i,j]];on=(zz>0);valid=pred[...,None]
                    before_l1=(zz*torch.linalg.vector_norm(d[[i,j]],dim=1)).sum(-1)
                    after_l1=nz.abs().sum(-1)
                    stats=np.stack([count.cpu().numpy(),((nz<0)&valid).sum((1,2)).cpu().numpy(),
                        (((nz>0)!=on)&valid).sum((1,2)).cpu().numpy(),(pred.sum(1)*2).cpu().numpy(),
                        (before_l1*pred).sum(1).cpu().numpy(),(after_l1*pred).sum(1).cpu().numpy(),
                        (reconstruction_delta.double().square().sum(-1)*pred).sum(1).cpu().numpy(),
                        (old.double().square().sum(-1)*pred).sum(1).cpu().numpy(),rtkl.cpu().numpy(),rtce.cpu().numpy()])
                    atomic_npz(path,protocol_id=pid,pair=np.asarray([i,j]),t=t,stats=stats,ce=np.asarray(cez),kl=np.asarray(klz),
                        energy=np.asarray(ez),sequence_indices=np.arange(sl.start,sl.stop))
                with np.load(path) as n:
                    for k,s in zip(accum,n['stats']):accum[k]+=float(s.sum())
                    ce+=n['ce'].sum(1);kl+=n['kl'].sum(1);energy+=n['energy'].sum(1)
                    for q,s in enumerate(n['sequence_indices']):
                        a=str(data['test']['articles'][s]);r=articles.setdefault(a,dict(count=0.,ce=np.zeros(4),kl=np.zeros(4)))
                        r['count']+=float(n['stats'][0,q]);r['ce']+=n['ce'][:,q];r['kl']+=n['kl'][:,q]
            records.append(dict(pair=[i,j],fraction=fraction,t=t,calibration_bound=bound,
                admissible_on_heldout=accum['negative']==0 and accum['support_changed']==0,
                negative_fraction=ratio(accum['negative'],accum['coeff_count']),support_changed_fraction=ratio(accum['support_changed'],accum['coeff_count']),
                reconstruction_relative_squared_error=ratio(accum['aggregate_squared_error'],accum['aggregate_energy']),
                roundtrip_kl=ratio(accum['roundtrip_kl'],accum['count']),roundtrip_dce=ratio(accum['roundtrip_dce'],accum['count']),
                unit_decoder_l1_ratio=ratio(accum['after_l1'],accum['before_l1']),
                effects=dict(labels=['old_first','old_second','new_first','new_second'],ce=ce/accum['count'],kl=kl/accum['count'],msd=energy/accum['count']),
                articles=articles))
            print(f'[pc1-surgery] {run["name"]} pair={pair_index} fraction={fraction} support_violations={accum["support_changed"]}',flush=True)
    result=dict(protocol_id=pid,run=run['name'],complete=True,available_ordered_pairs=available,selected_disjoint_pairs=len(pairs),records=records,
        scope='Exact linear reconstruction symmetry after the trained encoder. Do not claim a new affine-TopK encoder or universal non-identifiability.')
    atomic_json(root/'summary.json',result);return result
