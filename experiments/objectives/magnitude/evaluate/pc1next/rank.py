"""B1: exhaustive SINGLE-feature zero-ablation, using actual next-token CE and KL.

Every calibration-alive candidate is evaluated. E[z^2] is recorded only as a proxy
audit. The original-state and SAE-splice baselines are both measured and named.
Units contain all baselines for a sequence block and feature tile. No completed
flag is emitted until the collector verifies the entire Cartesian design.
"""
from pathlib import Path
import time
import numpy as np
import torch
from .backend import scores, valid_predictors
from .cache import blocks, cache_path, make_cache, fidelity
from .io import atomic_json, atomic_npz, completed, assigned


def unit_path(out, run, split, start, tile):
    return Path(out)/"rank_units"/run['name']/split/f"b{start:07d}_f{tile:06d}.npz"


def rank_units(cfg, run, data, cal):
    alive=np.flatnonzero(cal['alive']); tile=cfg.get('feature_batch',8)
    for split in ('selection','test'):
        for start,sl in blocks(data[split],cfg.get('sequence_batch',2)):
            for f in range(0,len(alive),tile):
                yield split,start,sl,f,alive[f:f+tile]


def execute(be,sae,cfg,run,data,cal,out,pid,shard=0,nshards=1,limit=None):
    for split in ('selection','test'):
        make_cache(be,sae,run,split,data[split],out,pid,cfg.get('sequence_batch',2))
        fidelity(out,run,split,data[split],pid,cfg.get('sequence_batch',2))
    done=0; started=time.monotonic(); baselines=cfg.get('baselines',['model','splice'])
    d=torch.as_tensor(cal['decoder'],dtype=torch.float32,device=be.device)
    current=None
    for split,start,sl,f,features in rank_units(cfg,run,data,cal):
        identity=['rank',run['name'],split,start,f]
        if not assigned(identity,shard,nshards):continue
        path=unit_path(out,run,split,start,f)
        if completed(path,pid):continue
        if current != (split,start):
            ids,mask=be.tensors(data[split],sl)
            original,x=be.forward(ids,mask)
            z,hat=be.encode(sae,x,run)
            spliced,_=be.forward(ids,mask,hat)
            references={'model':original,'splice':spliced}; states={'model':x.float(),'splice':hat}
            current=(split,start)
        ix=torch.as_tensor(features,device=be.device)
        # F x B x T x D: erase exactly one coefficient in each copy, keep decoder bias.
        delta=-z[...,ix].permute(2,0,1)[...,None]*d[ix,None,None,:]
        delta=delta*mask[None,...,None]
        nf,nb,nt,dm=delta.shape
        fid=ids.unsqueeze(0).expand(nf,-1,-1).reshape(nf*nb,nt)
        fmask=mask.unsqueeze(0).expand(nf,-1,-1).reshape(nf*nb,nt)
        ce_rows=[];kl_rows=[]
        for baseline in baselines:
            edited=(states[baseline][None]+delta).reshape(nf*nb,nt,dm)
            logits,_=be.forward(fid,fmask,edited)
            logits=logits.reshape(nf,nb,nt,-1)
            ce,kl,count=scores(references[baseline],logits,ids,mask)
            ce_rows.append(ce.cpu().numpy());kl_rows.append(kl.cpu().numpy())
        pred=valid_predictors(mask)
        energy=((delta.double()**2).sum(-1)*pred[None]).sum(-1)
        z2=((z[...,ix].double()**2)*pred[...,None]).sum(1).T
        atomic_npz(path,protocol_id=pid,features=features,sequence_indices=np.arange(sl.start,sl.stop),
            baselines=np.asarray(baselines),dce_sum=np.asarray(ce_rows),kl_sum=np.asarray(kl_rows),
            count=count.cpu().numpy(),displacement_sum=energy.cpu().numpy(),z2_sum=z2.cpu().numpy())
        done+=1
        print(f"[pc1-rank] {run['name']} {split} b={start} f={f} n={nf} complete=1",flush=True)
        if limit is not None and done>=limit:break
    return dict(new_units=done,seconds=time.monotonic()-started,limited=limit is not None)


def collect(cfg,run,data,cal,out,pid):
    """Per-article sufficient statistics retain rank reselection and paired resampling."""
    baselines=cfg.get('baselines',['model','splice']); d=len(cal['alive'])
    summaries={}; missing=[]
    for split in ('selection','test'):
        arts=np.unique(data[split]['articles']); lookup={a:i for i,a in enumerate(arts)}
        shape=(len(baselines),len(arts),d)
        ce=np.zeros(shape);kl=np.zeros(shape);energy=np.zeros((len(arts),d));z2=energy.copy()
        count=np.zeros((len(arts),d));seen=np.zeros((len(data[split]['ids']),d),dtype=bool)
        for sp,start,sl,f,feat in rank_units(cfg,run,data,cal):
            if sp!=split:continue
            path=unit_path(out,run,split,start,f)
            if not completed(path,pid):missing.append(str(path.relative_to(out)));continue
            with np.load(path) as n:
                if not np.array_equal(n['features'],feat) or not np.array_equal(n['baselines'],baselines):raise ValueError('Unit axes disagree')
                seq=n['sequence_indices']
                if not np.array_equal(seq,np.arange(sl.start,sl.stop)):raise ValueError('Unit row mismatch')
                if seen[np.ix_(seq,feat)].any():raise ValueError('Duplicate rank observations')
                seen[np.ix_(seq,feat)]=True
                for j,s in enumerate(seq):
                    a=lookup[data[split]['articles'][s]]
                    for bi in range(len(baselines)):
                        ce[bi,a,feat]+=n['dce_sum'][bi,:,j];kl[bi,a,feat]+=n['kl_sum'][bi,:,j]
                    energy[a,feat]+=n['displacement_sum'][:,j];z2[a,feat]+=n['z2_sum'][:,j]
                    count[a,feat]+=n['count'][j]
        complete=seen[:,cal['alive']].all()
        target=Path(out)/'rank_tables'/run['name']/(split+'.npz')
        if complete:
            atomic_npz(target,protocol_id=pid,baselines=np.asarray(baselines),article_ids=arts,
                dce_sum=ce,kl_sum=kl,displacement_sum=energy,z2_sum=z2,count=count,alive=cal['alive'])
        elif target.exists():raise ValueError('Previously complete table now has missing source units')
        summaries[split]=dict(complete=bool(complete),observed=int(seen[:,cal['alive']].sum()),
            expected=int(len(seen)*cal['alive'].sum()),articles=len(arts),article_scope=data[split]['scope'])
    report=dict(run=run['name'],complete=not missing,splits=summaries,missing=missing)
    atomic_json(Path(out)/'coverage'/(run['name']+'.json'),report)
    return report
