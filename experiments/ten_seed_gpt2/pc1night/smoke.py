"""CPU integration test with falsifying fixtures; not research evidence."""
from pathlib import Path
import time
import numpy as np
import torch
from .io import read,write,loadnp,npz,sha

def run(out):
    from .data import resolve_site,prepare,Store,TrainingView,verify_site
    from .synthetic import generate
    from .design import make_plan
    from .train import execute
    from .diagnostics import concentration,refine_codes
    from .model import make
    from .summary import collect
    started=time.monotonic();torch.set_num_threads(1);out=Path(out);out.mkdir(parents=True,exist_ok=True);checks=[]
    v=np.array([1.,0.]);amp=np.logspace(-6,6,100)
    assert np.allclose(amp[:,None]*v,np.maximum(amp,0)[:,None]*v);checks.append('unbounded positive amplitudes need one ray')
    # Fixed two-atom support exact continuum; L1 increases with tilt.
    for angle in (.1,.5,1.):
        D=np.array([[np.cos(angle),np.sin(angle)],[np.cos(angle),-np.sin(angle)]])
        z=np.repeat((amp/(2*np.cos(angle)))[:,None],2,axis=1)
        assert np.allclose(z@D,amp[:,None]*v,atol=1e-8)
        assert np.all(z.sum(1)>amp)
    checks.append('fixed-support continuum and L1 tilt cost')
    from .replay import replay
    r=np.random.default_rng(19);co=r.uniform(.1,1,8);dc=r.normal(size=(8,16));dc/=np.linalg.norm(dc,axis=1,keepdims=True)
    zz=np.linspace(.1,2,100)[:,None]*co;zt=np.linspace(.01,10,200)[:,None]*co
    rep=replay(zz,zt,dc,np.square(zz).mean(0))
    assert all(c['retained_rank']==1 and c['heldout_relative_aggregate_error']<1e-12 for c in rep['curves'])
    assert any(c['mean_best_signed_cosine_to_original_members']<.8 for c in rep['curves'])
    checks.append('rank-one code replay changes decoder directions while preserving held-out aggregate')
    oracle=make(dict(d=4,m=4,k=2,seed=0),np.array([1.,0.,0.,0.]))
    with torch.no_grad():oracle.W_dec.copy_(torch.eye(4));oracle.W_enc.zero_();oracle.b_enc.zero_()
    orc=refine_codes(oracle,torch.tensor([[2.,1.,0.,0.],[0.,0.,3.,1.]]),20)
    assert orc['refined_mse']<1e-8 and orc['initial_mse']>1
    checks.append('code optimization detects a planted encoder failure with a perfect decoder')
    a=np.r_[-99.,np.ones(99)];c=concentration(a)
    assert c['negative_frequency']==.01 and c['negative_energy_fraction']>.98
    checks.append('frequency-energy reversal fixture')
    path=generate(out/'data','fixture',dict(w1=.99,tail_p=.01,dependence=.8,content_support=4),1024,16)
    # d=16 allows 15 planted directions; fix support <= this bound.
    site=resolve_site(path);site['training']=dict(width=32,k=4,steps=8,batch=32)
    _,hashes=verify_site(site);ref=prepare(site,out/'references'/'fixture')
    st=Store(site['splits']['train']);base=TrainingView(st,ref,{});clip=TrainingView(st,ref,dict(transform='clip_restore'));shuffle=TrainingView(st,ref,dict(transform='shuffle'))
    xx=np.asarray(st.x,float);xx-=xx.mean(0);cov=xx.T@xx/len(xx);vref=ref['pc1'];eig=float(vref@cov@vref)
    assert np.linalg.norm(cov@vref-eig*vref)/eig<1e-6 and np.isclose(eig/np.trace(cov),.99,atol=1e-6)
    checks.append('planted nuisance axis is actual empirical PC1 at the specified variance share')
    ix=np.arange(len(st.x));bx,_=base.batch(ix);cx,_=clip.batch(ix);sx,_=shuffle.batch(ix);vv=ref['pc1']
    assert np.allclose(np.var(bx@vv),np.var(cx@vv),rtol=1e-5)
    assert np.allclose(np.sort(bx@vv),np.sort(sx@vv),rtol=1e-4,atol=1e-5)
    assert np.allclose(bx-(bx@vv)[:,None]*vv,sx-(sx@vv)[:,None]*vv,atol=1e-6)
    checks+=['clipping preserves train coefficient variance','shuffle preserves marginal and perpendicular residual']
    imp=TrainingView(st,ref,dict(sampling='importance'));_,ww=imp.batch(ix);pp=ref['sample_p']/ref['sample_p'].sum()
    assert np.allclose(pp*ww,np.full(len(ix),1/len(ix)),rtol=1e-6)
    checks.append('importance weights exactly recover the uniform expected objective')
    tf=TrainingView(st,ref,dict(order='tail_first'));tl=TrainingView(st,ref,dict(order='tail_last'))
    ia=np.concatenate([base.indices(s,32,8) for s in range(8)]);ib=np.concatenate([tf.indices(s,32,8) for s in range(8)]);ic=np.concatenate([tl.indices(s,32,8) for s in range(8)])
    assert np.array_equal(np.sort(ia),np.sort(ib)) and np.array_equal(np.sort(ia),np.sort(ic));checks.append('curricula use exactly identical sampled row multisets')
    cfg=dict(sites=[dict(name='fixture',manifest=str(path))],synthetic=False,seeds=[0,1],
        arms=['baseline','beta_0.3','clip_restore','canonical_init'],steps=8,batch=32,audit_rows=64,probe_rows=32,oracle_rows=8,oracle_steps=4,
        truth_rows=64,checkpoint_every=3,log_every=4,rank_arms=['baseline'],rank_populations=['all'],selection_sequences=2,test_sequences=2,sequence_batch=1,feature_batch=4)
    plan=make_plan(cfg);pid='SMOKE_ONLY';write(out/'plan.json',plan)
    for job in plan['jobs']:execute(job,site,ref,cfg,out,pid,'cpu')
    checks.append('eight complete fits with held-out allocation and optimization diagnostics')
    # Exact optimizer/stream resume compared to a complete run of the identical cell.
    job=plan['jobs'][0];resumedir=out/'resume'
    execute(job,site,ref,cfg,resumedir,pid,'cpu',stop_after=3);execute(job,site,ref,cfg,resumedir,pid,'cpu')
    c1=torch.load(out/'fits'/job['id']/'final.pt',weights_only=True);c2=torch.load(resumedir/'fits'/job['id']/'final.pt',weights_only=True)
    assert all(torch.equal(c1['state'][k],c2['state'][k]) for k in c1['state']);checks.append('eviction/resume is bit-identical to uninterrupted CPU training')
    for j in plan['jobs']:
        m=read(out/'fits'/j['id']/'metrics.json');assert abs(m['fvu']['identity_residual'])<1e-8
        assert m['code_refinement']['refined_mse']<=m['code_refinement']['initial_mse']+1e-7
    checks+=['exact FVU decomposition on all fitted cells','code refinement never worsens reconstruction']
    summary=collect(plan,out,20);assert summary['complete'];checks.append('complete Cartesian coverage and seed-vertex analysis')
    # This fixture checks downstream hooking and real CE, using a small causal map.
    class ToyLM:
        def forward(self,ids,mask,replace=None):
            x=torch.nn.functional.one_hot(ids%16,16).float();x=x+.05*x.cumsum(1)
            h=x if replace is None else replace(x) if callable(replace) else replace
            logits=h+.03*h.cumsum(1)
            return logits,x
    for split in ('selection','test'):
        t=out/(split+'_tokens.npz');npz(t,input_ids=np.array([[1,2,3,4],[4,3,2,1]]),attention_mask=np.ones((2,4),bool),article_ids=np.array([split+'a',split+'b']))
        site['splits'][split]['tokens']=str(t)
    site['lm']=dict(bos_token_id=None,eos_token_id=None)
    from .lm import execute as lmexecute,summarize
    for j in plan['jobs']:
        if j['arm']['name']=='baseline':lmexecute(j,site,ref,cfg,out,'cpu',ToyLM())
    lm=summarize(plan,out);assert lm['complete'] and len(lm['results'])>0
    checks.append('exhaustive actual CE/KL selection and independent frozen-set evaluation')
    result=dict(passed=len(checks),checks=checks,seconds=time.monotonic()-started,scope='CPU synthetic integration, not pretrained-model scientific validation')
    write(out/'SMOKE_RESULTS.json',result);print(f"[pc1night-smoke] PASS {len(checks)} checks {result['seconds']:.1f}s",flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
