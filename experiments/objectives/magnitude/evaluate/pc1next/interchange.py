"""B6: does cross-seed aggregate agreement preserve model behavior outside PC1?

Replace a recipient SAE group's contribution with another seed's contribution on
the SAME input. Hold PC1 exactly fixed in the primary arms. Compare with shifted
donor outputs, a calibration-only mean, and a shift at matched per-token norm.
This is group interchange, not paired clean/corrupt task activation patching.
"""
from pathlib import Path
import numpy as np
import torch
from .backend import scores,valid_predictors
from .cache import blocks,cache_path
from .geometry import shift_within_articles
from .interventions import norm_match
from .io import atomic_json,atomic_npz,completed,rng_for,ratio


def read_aggregate(out,run):
    with np.load(Path(out)/'aggregates'/(run['name']+'.npz')) as n:return {k:n[k] for k in n.files}


def execute(be,cfg,run,data,out,pid):
    root=Path(out)/'interchange'/run['name'];root.mkdir(parents=True,exist_ok=True)
    recipients=read_aggregate(out,run)
    expected=np.asarray([str(data['test']['seq'][i])+':'+str(t) for i,m in enumerate(data['test']['mask']) for t in np.flatnonzero(m)])
    if not np.array_equal(recipients['token_ids'],expected):raise ValueError('Recipient aggregate token alignment differs')
    candidates=[r for r in cfg['runs'] if r['condition']==run['condition'] and float(r['beta'])==float(run['beta']) and r['seed']!=run['seed']]
    offsets=np.concatenate([[0],np.cumsum(data['test']['mask'].sum(1))]);rows=[];missing=[]
    for donor_run in candidates:
        try:donor=read_aggregate(out,donor_run)
        except FileNotFoundError:missing.append(donor_run['name']);continue
        if not np.array_equal(recipients['token_ids'],donor['token_ids']):raise ValueError('Cross-seed aggregate token alignment differs')
        shifted,moved=shift_within_articles(donor['y'],donor['article_ids'],rng_for('interchange-shift',run['name'],donor_run['name']))
        sums={};art_stats={}
        for start,sl in blocks(data['test'],cfg.get('sequence_batch',2)):
            path=root/donor_run['name']/f'b{start:07d}.npz'
            if not completed(path,pid):
                ids,mask=be.tensors(data['test'],sl);original,x=be.forward(ids,mask)
                with np.load(cache_path(out,run,'test',start)) as n:hat=torch.as_tensor(n['hat'],device=be.device)
                spliced,_=be.forward(ids,mask,hat);refs={'model':original,'splice':spliced};states={'model':x.float(),'splice':hat}
                valid=data['test']['mask'][sl];shape=(*valid.shape,len(be.v));v=be.v
                def restore(y):
                    arr=np.zeros(shape,dtype=np.float32);arr[valid]=y[offsets[sl.start]:offsets[sl.stop]]
                    return torch.as_tensor(arr,device=be.device)
                a=restore(recipients['y']);b=restore(donor['y']);s=restore(shifted)
                full=b-a;perp=full-(full@v)[...,None]*v
                shuffle=s-a;shuffle=shuffle-(shuffle@v)[...,None]*v
                mean=torch.as_tensor(donor['calibration_mean'],dtype=torch.float32,device=be.device)-a
                mean=mean-(mean@v)[...,None]*v;mean=mean*mask[...,None]
                matched,alpha,infeasible=norm_match(shuffle,perp)
                deltas={'same_input_full':full,'same_input_perp':perp,'shift_perp':shuffle,
                    'calibration_mean_perp':mean,'shift_perp_energy_matched':matched}
                labels=[];values=[];pred=valid_predictors(mask)
                for mode,delta in deltas.items():
                    for dose in cfg.get('interchange_doses',[.1,1.]):
                        for baseline in cfg.get('baselines',['model','splice']):
                            perturb=delta*dose;logits,_=be.forward(ids,mask,states[baseline]+perturb)
                            ce,kl,count=scores(refs[baseline],logits,ids,mask)
                            energy=(perturb.double().square().sum(-1)*pred).sum(-1)
                            pcenergy=((perturb.double()@v.double()).square()*pred).sum(-1)
                            values.append(np.stack([ce.cpu().numpy(),kl.cpu().numpy(),count.cpu().numpy(),energy.cpu().numpy(),pcenergy.cpu().numpy()]))
                            labels.append([mode,str(dose),baseline])
                atomic_npz(path,protocol_id=pid,labels=np.asarray(labels),values=np.asarray(values),
                    sequence_indices=np.arange(sl.start,sl.stop),alpha=alpha.cpu().numpy(),infeasible=infeasible.cpu().numpy())
                print(f'[pc1-interchange] {run["name"]} <- {donor_run["name"]} b={start}',flush=True)
            with np.load(path) as n:
                for label,values in zip(n['labels'],n['values']):
                    identity=tuple(label);sums.setdefault(identity,np.zeros(5));sums[identity]+=values.sum(1)
                    for j,i in enumerate(n['sequence_indices']):
                        a=str(data['test']['articles'][i]);art_stats.setdefault(identity,{}).setdefault(a,np.zeros(5));art_stats[identity][a]+=values[:,j]
        for (mode,dose,baseline),ss in sums.items():
            rows.append(dict(recipient=run['name'],donor=donor_run['name'],recipient_seed=run['seed'],donor_seed=donor_run['seed'],
                mode=mode,dose=float(dose),baseline=baseline,kl=ss[1]/ss[2],dce=ss[0]/ss[2],msd=ss[3]/ss[2],
                pc1_energy_fraction=ratio(ss[4],ss[3]),shift_moved_fraction=moved,
                articles={a:dict(zip(['dce_sum','kl_sum','count','energy_sum','pc1_energy_sum'],vals.tolist())) for a,vals in art_stats[(mode,dose,baseline)].items()}))
    report=dict(protocol_id=pid,run=run['name'],complete=not missing,missing=missing,rows=rows,
        interpretation='Same-input interchange tests cross-seed functional agreement. Primary perpendicular arms retain the original PC1 coefficient. Mean and shift controls distinguish correspondence from shared offsets.')
    atomic_json(root/'summary.json',report);return report
