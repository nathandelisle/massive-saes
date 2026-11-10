"""Allocation, concentration, competition, code optimization, and gradient diagnostics."""
import numpy as np
import torch
from .io import ratio,rng
from .data import evaluate_input

def sample_rows(st,n=8192):
    # Round-robin across real articles; do not give a long article every row.
    arts=st.meta['article'].astype(str);groups=[]
    for a in sorted(set(arts)):
        ix=np.flatnonzero(arts==a);groups.append(rng('audit_rows',a).permutation(ix))
    rows=[]
    for j in range(max(map(len,groups))):
        for g in groups:
            if j<len(g):rows.append(g[j])
            if len(rows)>=n:return np.array(rows,dtype=int)
    return np.array(rows,dtype=int)

def concentration(a):
    e=np.square(np.asarray(a,dtype=float));s=e.sum();desc=np.sort(e)[::-1]
    return dict(n=len(a),second_moment=e.mean(),mean=float(np.mean(a)),sd=float(np.std(a)),
        negative_frequency=float(np.mean(a<0)),negative_energy_fraction=ratio(e[a<0].sum(),s),
        positive_ray_error=ratio(e[a<0].sum(),s),negative_ray_error=ratio(e[a>0].sum(),s),
        energy_effective_n=ratio(s*s,np.square(e).sum()),
        top_energy={str(q):ratio(desc[:max(1,int(np.ceil(len(e)*q)))].sum(),s) for q in (.001,.005,.01,.02,.05,.1)},
        abs_quantiles=np.quantile(abs(a),[0,.5,.9,.95,.99,.999,1]))

def spectral(a):
    a=np.asarray(a,float)
    if not a.size:return dict(rank=0,effective_rank=0.)
    val=np.linalg.eigvalsh((a+a.T)/2);val=np.maximum(val,0);s=val.sum()
    p=val[val>max(s,1)*1e-14]/max(s,1e-30)
    return dict(rank=int((val>max(s,1)*1e-8).sum()),effective_rank=float(np.exp(-(p*np.log(p)).sum())),
                participation_rank=ratio(s*s,val@val),top_eigenvalues=val[-10:][::-1])

@torch.no_grad()
def infer(model,q,batch=256):
    device=next(model.parameters()).device;zs=[];hs=[];margin=[]
    for start in range(0,len(q),batch):
        x=torch.as_tensor(q[start:start+batch],device=device);h,z=model(x)
        p=model.pre(x);p=p.abs() if model.mode=='signed' else torch.relu(p)
        if model.k<p.shape[-1]:
            top=p.topk(model.k+1,-1).values;margin.extend((top[:,-2]-top[:,-1]).cpu().tolist())
        zs.append(z.cpu().numpy());hs.append(h.cpu().numpy())
    return np.concatenate(hs),np.concatenate(zs),np.array(margin)

def diagnose(model,cal,test,ref,arm,max_rows=8192):
    ci=sample_rows(cal,max_rows);ti=sample_rows(test,max_rows)
    cq,ca=evaluate_input(np.asarray(cal.x[ci],float),ref,arm)
    tq,ta=evaluate_input(np.asarray(test.x[ti],float),ref,arm)
    ch,cz,_=infer(model,cq);hat,z,margin=infer(model,tq)
    D=model.W_dec.detach().cpu().numpy();v=ref['pc1'];dp=D@v;cos2=dp*dp/np.square(D).sum(1)
    alive=np.any(cz!=0,axis=0);med=float(np.median(cos2[alive])) if alive.any() else 0.
    masks=dict(structural=alive&(cos2>5*med),canonical=alive&(cos2>=.6),
        # Explicit fixed angular band; not an outcome-selected sharing cliff.
        tiles=alive&(cos2>=.03)&(cos2<.6))
    masks['other']=alive&~(masks['canonical']|masks['tiles'])
    original=(np.asarray(test.x[ti],float)-ref['mean'])/ref['scale']
    if arm.get('transform')=='bypass':hat=hat+ta[:,None]*v
    err=hat-original;ec=err@v;ev=original-original.mean(0);evc=ev@v
    et=np.square(err).sum();ep=np.square(ec).sum();vt=np.square(ev).sum();vp=np.square(evc).sum()
    fvu=dict(total=ratio(et,vt),parallel=ratio(ep,vp),perp=ratio(max(0,et-ep),max(0,vt-vp)),w1=ratio(vp,vt))
    fvu['identity_residual']=None if None in fvu.values() else fvu['total']-fvu['w1']*fvu['parallel']-(1-fvu['w1'])*fvu['perp']
    b=model.b_dec.detach().cpu().numpy();a=original@v-b@v
    row_masks={k:m[ti] for k,m in test.masks().items()}
    row_masks.update(target_positive=a>0,target_negative=a<0,
        train_threshold_tail=abs(a*ref['scale'])>ref['clip_q'])
    fvu_by_population={}
    for pop,rm in row_masks.items():
        if not rm.any():continue
        xx=original[rm];xx=xx-xx.mean(0);ee=err[rm];tt=np.square(xx).sum();pp=np.square(xx@v).sum()
        en=np.square(ee).sum();pn=np.square(ee@v).sum()
        fvu_by_population[pop]=dict(n=int(rm.sum()),total=ratio(en,tt),parallel=ratio(pn,pp),
            perp=ratio(max(en-pn,0),max(tt-pp,0)),fixed_axis_variance_share=ratio(pp,tt))
    allocations={};group_vectors={};norms=np.square(D).sum(1)
    for name,gm in masks.items():
        gz=z[:,gm];gd=D[gm];g= gz@gd
        gp=g@v;off=g-gp[:,None]*v;individual=np.square(gz)*norms[gm]
        signed=gz*dp[gm];positive=np.maximum(signed,0).sum(1);negative=np.maximum(-signed,0).sum(1)
        group_vectors[name]=g.astype(np.float32)
        cells={}
        for pop,rm in row_masks.items():
            if not rm.any():continue
            # Own-signed contribution mass vs target-sign conditioning are distinct.
            tg=(z[rm]*dp);denp=np.maximum(tg,0).sum();denn=np.maximum(-tg,0).sum()
            cells[pop]=dict(n=int(rm.sum()),sum_individual_energy=float(individual[rm].sum()),
                joint_energy=float(np.square(g[rm]).sum()),pc1_joint_energy=float(np.square(gp[rm]).sum()),
                offaxis_joint_energy=float(np.square(off[rm]).sum()),
                cancellation=ratio(individual[rm].sum(),np.square(g[rm]).sum()),
                positive_mass_share=ratio(positive[rm].sum(),denp),negative_mass_share=ratio(negative[rm].sum(),denn),
                target_projection_mse=float(np.mean((a[rm]-gp[rm])**2)),
                mean_support=float(np.mean(np.count_nonzero(gz[rm],axis=1))))
        # Limit spectrum deterministically by calibration energy; record limit.
        members=np.flatnonzero(gm);order=np.argsort(np.square(cz[:,gm]).sum(0))[::-1][:256]
        zz=gz[:,order].astype(float);gram=zz.T@zz/max(len(zz),1)
        allocations[name]=dict(n_atoms=int(gm.sum()),members=members,cells=cells,
            code_gram=spectral(gram),spectral_members=members[order],spectrum_cap=256)
    # Structure's union size and per-token total competition are measured separately.
    active=(z!=0);support=active.sum(1);gs=active[:,masks['structural']].sum(1)
    articles={}
    for art in sorted(set(test.meta['article'][ti].astype(str))):
        m=test.meta['article'][ti].astype(str)==art
        articles[art]=dict(n=int(m.sum()),error=float(np.square(err[m]).sum()),
            perpendicular_error=float(np.square(err[m]-ec[m,None]*v).sum()),coef_energy=float(np.square(a[m]).sum()))
    # Encoder routing counterfactual holds every residual coordinate fixed.
    device=next(model.parameters()).device
    with torch.no_grad():
        rp=torch.as_tensor(tq[:min(512,len(tq))],device=device)
        bb=model.b_dec;aa=(rp-bb)@model.v;rr=rp-aa[:,None]*model.v
        active_full=model.encode(rp).ne(0);active_no_pc1=model.encode(rr).ne(0)
        intersection=(active_full&active_no_pc1).sum(1).float();union=(active_full|active_no_pc1).sum(1).float()
        routing=dict(n=len(rp),support_jaccard_remove_coefficient=float((intersection/union.clamp_min(1)).mean()),
            coefficient_origin='learned decoder bias in fixed train-centered coordinates',
            caveat='Counterfactual encoder input, not an in-distribution model intervention')
    # Sign/role histograms and weighted effective sample sizes are fully joint.
    energy={name:concentration(a[m]) for name,m in row_masks.items() if m.any()}
    from .replay import replay
    gm=masks['structural'];decoder_replay=replay(cz[:,gm],z[:,gm],D[gm],np.square(cz[:,gm]).mean(0),residual_test=err,reference_axis=v)
    result=dict(n_cal=len(ci),n_test=len(ti),alive=int(alive.sum()),fvu=fvu,fvu_by_population=fvu_by_population,allocations=allocations,routing=routing,decoder_replay=decoder_replay,
        energy=energy,articles=articles,mean_l0=float(support.mean()),mean_l1=float(abs(z).sum(1).mean()),
        total_support_quantiles=np.quantile(support,[.5,.9,.99,1]),group_support_quantiles=np.quantile(gs,[.5,.9,.99,1]),
        at_total_k_fraction=float(np.mean(support==model.k)),group_at_k_fraction=float(np.mean(gs==model.k)),
        topk_margin_quantiles=np.quantile(margin,[0,.01,.1,.5,.9,1]) if len(margin) else [],
        bias_projection=float(b@v),canonical_threshold_sensitivity={str(t):int((alive&(cos2>=t)).sum()) for t in (.4,.5,.6,.7,.8,.9,.95)},
        unit_decoder_max_residual=float(np.max(abs(np.linalg.norm(D,axis=1)-1))),
        selection='Groups and alive mask defined only on calibration; identical deterministic test sample for all arms')
    vr=min(1024,len(ti))
    result['stored_vector_rows']=vr
    arrays=dict(decoder=D,alive=alive,alignment=cos2,cal_firing_rate=(cz!=0).mean(0),
        cal_energy=np.square(cz).mean(0)*norms,test_energy=np.square(z).mean(0)*norms,
        test_rows=ti[:vr],projection=a[:vr],test_hat=hat[:vr].astype(np.float32),
        **{'group_'+k:val[:vr] for k,val in group_vectors.items()},**{'mask_'+k:val for k,val in masks.items()})
    return result,arrays

def gradients(model,x,weights,beta,refcoef,first,axis_weight=1.):
    """Exact gradients of sampled stratum contributions; energy is not a gradient proxy."""
    from .model import loss_parts
    ps=[p for p in model.parameters() if p.requires_grad];n=len(x)
    loss,_,_,_,_=loss_parts(model,x,weights,beta,axis_weight)
    total=torch.autograd.grad(loss,ps);tn=sum(g.square().sum() for g in total)
    a=np.asarray(refcoef);groups={'first':first,'ordinary':~first,
        'tail':abs(a)>=np.quantile(abs(a),.99),'positive':a>0,'negative':a<0}
    out={}
    for name,mask in groups.items():
        if not mask.any():out[name]=dict(n=0);continue
        ix=torch.as_tensor(mask,device=x.device);l,_,_,_,_=loss_parts(model,x[ix],weights[ix],beta,axis_weight)
        # Fraction-of-sample weighting yields additive stratum contributions.
        g=torch.autograd.grad(l*float(mask.mean()),ps);gn=sum(t.square().sum() for t in g);dot=sum((u*t).sum() for u,t in zip(total,g))
        out[name]=dict(n=int(mask.sum()),gradient_norm=float(gn.sqrt()),
            cosine_with_full=ratio(float(dot),float((gn*tn).sqrt())),
            projection_on_full=ratio(float(dot),float(tn)))
    return dict(n=n,full_gradient_norm=float(tn.sqrt()),groups=out,
        caveat='Exact gradients on a fixed diagnostic sample. Adam updates additionally depend on optimizer history.')

@torch.no_grad()
def refine_codes(model,x,iterations=100):
    """Monotone projected-gradient heuristic, NOT a certified global sparse optimum."""
    D=model.W_dec;target=x-model.b_dec;z=model.encode(x);initial=(target-z@D).square().sum(1)
    best=z.clone();besterr=initial.clone()
    # Fast spectral estimate. Every row still uses an explicit monotone acceptance
    # test, so an imperfect Lipschitz estimate cannot silently worsen the result.
    q=torch.ones(D.shape[1],device=D.device);q=q/q.norm()
    for _ in range(20):q=D.T@(D@q);q=q/q.norm().clamp_min(1e-30)
    spectral_est=float((D@q).square().sum());lr=1./max(1.1*spectral_est,1e-8);traces=[]
    for _ in range(iterations):
        cand=z-lr*((z@D-target)@D.T)
        if model.mode!='signed':cand=torch.relu(cand)
        _,ix=cand.abs().topk(model.k,-1);cand=torch.zeros_like(cand).scatter(-1,ix,cand.gather(-1,ix))
        e=(target-cand@D).square().sum(1);ok=e<=besterr
        best[ok]=cand[ok];besterr[ok]=e[ok];z=best.clone()
        traces.append(float(besterr.mean()))
    return dict(initial_mse=float(initial.mean()),refined_mse=float(besterr.mean()),
                relative_reduction=ratio(float((initial-besterr).mean()),float(initial.mean())),
                iterations=iterations,trace=traces,max_support=int((best!=0).sum(1).max()),spectral_estimate=spectral_est,step_size=lr)
