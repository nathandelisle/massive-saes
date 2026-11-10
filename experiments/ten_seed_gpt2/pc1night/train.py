"""Paired initialization, exact stream resume, optimizer checkpoints, held-out diagnostics."""
import os,time
from pathlib import Path
import numpy as np
import torch
from .io import read,write,save_torch,npz,key,sha,loadnp,rng
from .model import make,loss_parts
from .data import Store,TrainingView,evaluate_input
from .diagnostics import diagnose,sample_rows,gradients,refine_codes,infer

def current_beta(arm,step,steps):
    b=arm.get('beta',0.);schedule=arm.get('beta_schedule')
    if schedule=='late':return b if step>=steps//2 else 0.
    if schedule=='removed':return b if step<steps//2 else 0.
    return b

def execute(job,site,ref,settings,root,protocol,device='cuda',stop_after=None):
    root=Path(root)/'fits'/job['id'];root.mkdir(parents=True,exist_ok=True)
    complete=root/'complete.json'
    if complete.exists():
        c=read(complete)
        if c['protocol']!=protocol:raise ValueError('Stale complete marker')
        for f,h in c['hashes'].items():
            if sha(root/f)!=h:raise ValueError('Corrupt completed artifact '+f)
        return dict(status='skipped',**c)
    st=Store(site['splits']['train']);cal=Store(site['splits']['calibration']);test=Store(site['splits']['test'])
    arm=job['arm'];d=st.x.shape[1];syn=job['kind']=='synthetic'
    base=site.get('training',{});steps=int(base.get('steps',settings.get('synthetic_steps',10000) if syn else settings.get('steps',20000)))
    batch=int(base.get('batch',settings.get('synthetic_batch',512) if syn else settings.get('batch',4096)))
    width=int(round(base.get('width',256 if syn else 4*d)*arm.get('width_factor',1.)))
    k=int(round(base.get('k',32 if syn else 64)*arm.get('k_factor',1.)))
    mc=dict(d=d,m=width,k=min(k,width),seed=job['seed'],**{t:arm[t] for t in ('mode','init','bias','encoder') if t in arm})
    torch.set_grad_enabled(True);model=make(mc,ref['pc1']).to(device);model.train()
    lr=float(base.get('lr',settings.get('lr',.0003)));optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8)
    view=TrainingView(st,ref,arm,settings.get('order_seed',20260908));start=0;history=[];gradlog=[];elapsed=0.
    if (root/'latest.pt').exists():
        ck=torch.load(root/'latest.pt',weights_only=True,map_location=device)
        if ck['protocol']!=protocol or ck['job_id']!=job['id']:raise ValueError('Wrong checkpoint provenance')
        model.load_state_dict(ck['state']);optimizer.load_state_dict(ck['optimizer']);start=ck['step'];history=ck['history'];gradlog=ck['gradients'];elapsed=ck['elapsed']
    started=time.monotonic();checkpoint_every=settings.get('checkpoint_every',500)
    probe=sample_rows(cal,min(settings.get('probe_rows',512),len(cal.x)));rawq,co=evaluate_input(np.asarray(cal.x[probe],float),ref,arm)
    # Optimization diagnostics are on train-distribution rows with registered transforms.
    gp=rng('gradient_probe',site['name']).choice(view.eligible,min(1024,len(view.eligible)),False)
    gx,gw=view.batch(gp)
    if start==0:
        gradlog.append(dict(step=0,**gradients(model,torch.as_tensor(gx,device=device),torch.as_tensor(gw,device=device),
            current_beta(arm,0,steps),ref['coef'][gp],st.first[gp],arm.get('loss_axis',1.))))
    def checkpoint(step):
        ck=dict(protocol=protocol,job_id=job['id'],step=step,cfg=mc,state=model.state_dict(),optimizer=optimizer.state_dict(),
            history=history,gradients=gradlog,elapsed=elapsed+time.monotonic()-started,arm=arm,
            reference=dict(mean=torch.tensor(ref['mean']),pc1=torch.tensor(ref['pc1']),scale=float(ref['scale'])))
        save_torch(root/'latest.pt',ck)
    for step in range(start,steps):
        ix=view.indices(step,batch,steps);x,w=view.batch(ix)
        x=torch.as_tensor(x,device=device);w=torch.as_tensor(w,device=device);beta=current_beta(arm,step,steps)
        optimizer.zero_grad(set_to_none=True);loss,hat,z,mse,l1=loss_parts(model,x,w,beta,arm.get('loss_axis',1.))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss; no automatic dtype/protocol fallback')
        loss.backward();model.tangent_grad();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),settings.get('grad_clip',1.))
        optimizer.step();model.constrain()
        if step==0 or (step+1)%settings.get('log_every',500)==0 or step==steps-1:
            with torch.no_grad():
                q=torch.as_tensor(rawq,device=device);hh,zz=model(q);err=(hh-q).square().sum(-1)
                al=(model.W_dec@model.v).square();alive=zz.ne(0).any(0)
            rec=dict(step=step+1,loss=float(loss.detach()),mse=float(mse.detach().mean()),weighted_mse=float((w*mse.detach()).mean()),
                l1=float(l1.detach().mean()),l0=float(z.ne(0).sum(-1).float().mean()),grad_norm_preclip=float(gn),
                clipping_engaged=bool(float(gn)>settings.get('grad_clip',1.)),beta=beta,
                probe_mse=float(err.mean()),canonical=int((alive&(al>=.6)).sum()),
                canonical_ids=torch.nonzero(alive&(al>=.6)).flatten().cpu().tolist(),
                probe_alive_ids=torch.nonzero(alive).flatten().cpu().tolist(),
                first_batch_fraction=float(st.first[ix].mean()),batch_pc1_energy=float(np.square(ref['coef'][ix]).mean()),
                mean_importance_weight=float(w.mean()),batch_indices_hash=key(ix.tolist()))
            history.append(rec);print(f"[pc1night-train] {job['id']} step={step+1}/{steps} loss={rec['loss']:.6g} canon={rec['canonical']}",flush=True)
        if step in sorted(set([0,steps//20,steps//4,steps//2,steps-1])):
            gradlog.append(dict(step=step+1,**gradients(model,torch.as_tensor(gx,device=device),torch.as_tensor(gw,device=device),beta,
                ref['coef'][gp],st.first[gp],arm.get('loss_axis',1.))))
        if (step+1)%checkpoint_every==0 or step==steps-1 or (stop_after and step+1>=stop_after):
            checkpoint(step+1)
            if stop_after and step+1>=stop_after and step+1<steps:return dict(status='interrupted_for_test',step=step+1)
    model.eval();result,arrays=diagnose(model,cal,test,ref,arm,settings.get('audit_rows',8192))
    from .manipulation import audit
    result['manipulation']=audit(site,ref,arm)
    qi,_=evaluate_input(np.asarray(test.x[sample_rows(test,settings.get('oracle_rows',128))],float),ref,arm)
    result['code_refinement']=refine_codes(model,torch.as_tensor(qi,device=device),settings.get('oracle_steps',100))
    if syn:
        # Linear coefficient prediction is fitted on calibration, tested independently.
        from .truth import evaluate_truth
        result['planted_truth']=evaluate_truth(model,cal,test,ref,arm,site,settings.get('truth_rows',2048))
    result.update(job=job,cfg=mc,protocol=protocol,steps=steps,batch=batch,lr=lr,seconds=elapsed+time.monotonic()-started,
        evaluation_domain='Original, unmodified test activations for ALL train-distribution arms',
        parameterization='New explicit trainer; not silently interchangeable with historical v3 beta units')
    npz(root/'arrays.npz',**arrays);write(root/'metrics.json',result);write(root/'trajectory.json',dict(training=history,gradients=gradlog))
    save_torch(root/'final.pt',dict(protocol=protocol,job_id=job['id'],cfg=mc,arm=arm,state=model.state_dict(),
        reference=dict(mean=torch.tensor(ref['mean']),pc1=torch.tensor(ref['pc1']),scale=float(ref['scale']))))
    hashes={f:sha(root/f) for f in ('final.pt','metrics.json','arrays.npz','trajectory.json')}
    write(complete,dict(protocol=protocol,job_id=job['id'],status='complete',hashes=hashes,seconds=result['seconds']))
    return read(complete)
