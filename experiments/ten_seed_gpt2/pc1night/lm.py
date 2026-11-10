"""Real single-feature ablation. Exhaustive selection, independent selected-feature test.

Units are sequence blocks containing ALL candidate features, not millions of
individual feature files. A partial block resumes locally from a single checkpoint.
"""
from pathlib import Path
import numpy as np
import torch
from .io import loadnp,read,npz,write,key,sha,rng,ratio
from .model import make

class LM:
    def __init__(self,site,device):
        from transformers import AutoModelForCausalLM
        cfg=site['lm'];rev=cfg['revision']
        if len(rev)!=40:raise ValueError('Pin model revision to a 40-character HF commit')
        self.model=AutoModelForCausalLM.from_pretrained(cfg['model_path'],revision=rev,
            torch_dtype=getattr(torch,cfg.get('dtype','bfloat16')),attn_implementation=cfg.get('attn_implementation','eager')).to(device).eval()
        self.device=device;self.block=self.model
        for p in cfg['block_path'].split('.'):self.block=getattr(self.block,p)
        self.block=self.block[cfg['block_index']]
    @torch.no_grad()
    def forward(self,ids,mask,replace=None):
        saved={}
        def hook(_m,_a,o):
            x=o[0] if isinstance(o,tuple) else o;saved['x']=x
            if replace is None:return o
            y=replace(x) if callable(replace) else replace
            if not torch.isfinite(y).all():raise FloatingPointError('Nonfinite edit BEFORE cast')
            y=y.to(x.dtype)
            if not torch.isfinite(y).all():raise FloatingPointError('Edit overflows model dtype')
            return (y,)+o[1:] if isinstance(o,tuple) else y
        h=self.block.register_forward_hook(hook)
        try:logits=self.model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float()
        finally:h.remove()
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite downstream logits')
        return logits,saved['x']

def scores(base,logits,ids,mask):
    lp=base[:,:-1].log_softmax(-1);lq=logits[...,:-1,:].log_softmax(-1);valid=mask[:,:-1]&mask[:,1:]
    target=ids[:,1:,None]
    if lq.ndim==4:target=target[None].expand(lq.shape[:-1]+(1,))
    ce=lp.gather(-1,ids[:,1:,None]).squeeze(-1)-lq.gather(-1,target).squeeze(-1)
    kl=(lp.exp()*(lp-lq)).sum(-1)
    return (ce*valid).sum(-1),(kl*valid).sum(-1),valid.sum(-1)

def sae_forward(model,x,ref,arm):
    v=torch.as_tensor(ref['pc1'],dtype=torch.float32,device=x.device);mu=torch.as_tensor(ref['mean'],dtype=torch.float32,device=x.device)
    q=(x.float()-mu)/float(ref['scale']);a=q@v
    if arm.get('transform')=='bypass':q=q-a[...,None]*v
    hat,z=model(q)
    if arm.get('transform')=='bypass':hat=hat+a[...,None]*v
    return hat*float(ref['scale'])+mu,z

def select_sequences(data,count):
    arts=data['article_ids'].astype(str);groups=[]
    for a in sorted(set(arts)):groups.append(rng('lm-sequence',a).permutation(np.flatnonzero(arts==a)))
    idx=[]
    for i in range(max(map(len,groups))):
        for g in groups:
            if i<len(g):idx.append(g[i])
            if len(idx)>=count:return np.array(idx)
    return np.array(idx)

def population_mask(ids,mask,pop,lmcfg):
    m=mask.clone()
    if pop=='after_first':
        first=mask.long().argmax(1);m[torch.arange(len(m),device=m.device),first]=False
    elif pop=='native':
        for k in ('bos_token_id','eos_token_id'):
            if lmcfg.get(k) is not None:m&=ids!=lmcfg[k]
    elif pop!='all':raise ValueError(pop)
    return m

@torch.no_grad()
def execute(job,site,ref,settings,out,device='cuda',lm=None):
    root=Path(out)/'lm'/job['id'];fit=Path(out)/'fits'/job['id'];ck=fit/'final.pt'
    if not ck.exists():return dict(status='blocked_missing_fit')
    if not (fit/'complete.json').exists():return dict(status='blocked_fit_not_committed')
    if not site.get('lm'):return dict(status='blocked_missing_lm_metadata')
    h=sha(ck);done=root/'complete.json'
    if read(fit/'complete.json')['hashes']['final.pt']!=h:raise ValueError('Fitted checkpoint hash mismatch')
    if done.exists():
        d=read(done)
        if d['checkpoint_sha']!=h:raise ValueError('LM result belongs to another SAE')
        return d
    blob=torch.load(ck,map_location=device,weights_only=True);model=make(blob['cfg'],ref['pc1']).to(device);model.load_state_dict(blob['state']);model.eval()
    lm=lm or LM(site,device);arm=job['arm'];width=blob['cfg']['m'];baselines=('model','splice')
    # Full calibration store determines alive candidates; no importance proxy screen.
    cal=np.load(site['splits']['calibration']['x'],mmap_mode='r');alive=np.zeros(width,bool)
    with torch.no_grad():
        for start in range(0,len(cal),256):
            x=torch.as_tensor(np.array(cal[start:start+256],dtype=np.float32,copy=True),device=device)
            _,z=sae_forward(model,x,ref,arm);alive|=z.ne(0).any(0).cpu().numpy()
    npz(root/'calibration.npz',alive=alive,decoder=model.W_dec.detach().cpu().numpy(),checkpoint_sha=h)
    candidates=np.flatnonzero(alive);selection={};all_records=[];tile=settings.get('feature_batch',4);bs=settings.get('sequence_batch',1)
    if not len(candidates):raise ValueError('No calibration-alive candidates')
    for pop in settings.get('rank_populations',['all','after_first','native']):
        for split in ('selection','test'):
            spec=site['splits'][split]
            if not spec.get('tokens'):return dict(status='blocked_missing_token_cache',split=split)
            data=loadnp(spec['tokens']);seq=select_sequences(data,settings.get('selection_sequences',64) if split=='selection' else settings.get('test_sequences',256))
            features=candidates if split=='selection' else selection[pop]
            for block,start in enumerate(range(0,len(seq),bs)):
                seqix=seq[start:start+bs];target=root/f'{pop}_{split}_{block:05d}.npz';partial=target.with_suffix('.partial.npz')
                if target.exists():
                    old=loadnp(target)
                    if str(old['checkpoint_sha'])!=h or not np.array_equal(old['features'],features) or not np.array_equal(old['sequence_indices'],seqix):raise ValueError('Stale completed rank block')
                    continue
                ids=torch.as_tensor(data['input_ids'][seqix],device=device);mask=torch.as_tensor(data['attention_mask'][seqix],dtype=torch.bool,device=device)
                with torch.no_grad():
                    original,x=lm.forward(ids,mask);hat,z=sae_forward(model,x,ref,arm);spliced,_=lm.forward(ids,mask,hat)
                if block==0:
                    roundtrip,_=lm.forward(ids,mask,x)
                    if not torch.equal(original,roundtrip):raise AssertionError('Hook identity fails')
                em=population_mask(ids,mask,pop,site['lm']);shape=(2,len(features),len(seqix))
                ce=np.zeros(shape);kl=np.zeros(shape);displacement=np.zeros((len(features),len(seqix)));cursor=0
                if partial.exists():
                    old=loadnp(partial)
                    if str(old['checkpoint_sha'])!=h or not np.array_equal(old['features'],features):raise ValueError('Stale partial rank block')
                    ce=old['ce'];kl=old['kl'];displacement=old['displacement'];cursor=int(old['cursor'])
                for f in range(cursor,len(features),tile):
                    fids=features[f:f+tile];ix=torch.as_tensor(fids,device=device);nf=len(ix);nb=len(ids)
                    delta=-z[...,ix].permute(2,0,1)[...,None]*model.W_dec[ix,None,None,:]*float(ref['scale'])
                    delta*=em[None,...,None]
                    fi=ids[None].expand(nf,-1,-1).reshape(nf*nb,-1);fm=mask[None].expand(nf,-1,-1).reshape(nf*nb,-1)
                    for bi,(base,state) in enumerate(((original,x),(spliced,hat))):
                        logits,_=lm.forward(fi,fm,(state[None]+delta).reshape(nf*nb,ids.shape[1],-1))
                        c,k,count=scores(base,logits.reshape(nf,nb,ids.shape[1],-1),ids,mask)
                        ce[bi,f:f+nf]=c.cpu().numpy();kl[bi,f:f+nf]=k.cpu().numpy()
                    displacement[f:f+nf]=delta.double().square().sum((-1,-2)).cpu().numpy()
                    if (f//tile)%settings.get('rank_checkpoint_tiles',32)==0:
                        npz(partial,checkpoint_sha=h,features=features,ce=ce,kl=kl,displacement=displacement,cursor=f+nf)
                npz(target,checkpoint_sha=h,features=features,ce=ce,kl=kl,displacement=displacement,
                    count=count.cpu().numpy(),sequence_indices=seqix,article_ids=data['article_ids'][seqix],
                    baselines=np.array(baselines),edit_population=pop,score_population='all_valid_next_token_predictions')
                if partial.exists():partial.unlink()
                print(f'[pc1night-rank] {job["id"]} {pop}/{split} block={block+1} candidates={len(features)}',flush=True)
            # Preserve article-by-feature statistics; independent test only evaluates the frozen selection union.
            paths=[root/f'{pop}_{split}_{b:05d}.npz' for b in range((len(seq)+bs-1)//bs)];loaded=[loadnp(p) for p in paths]
            ce=np.concatenate([l['ce'] for l in loaded],axis=2);kl=np.concatenate([l['kl'] for l in loaded],axis=2)
            count=np.concatenate([l['count'] for l in loaded]);arts=np.concatenate([l['article_ids'] for l in loaded]).astype(str)
            unique=sorted(set(arts));ac=np.stack([ce[:,:,arts==a].sum(2) for a in unique],axis=2)
            ak=np.stack([kl[:,:,arts==a].sum(2) for a in unique],axis=2);an=np.array([count[arts==a].sum() for a in unique])
            npz(root/f'{pop}_{split}_table.npz',features=features,ce=ac,kl=ak,count=an,article_ids=np.array(unique),baselines=np.array(baselines))
            if split=='selection':
                chosen=set()
                for metric in (ac,ak):
                    for bi in range(2):
                        score=metric[bi].sum(1)/an.sum();order=np.lexsort((features,-score));chosen.update(features[order[:100]].tolist())
                rest=np.setdiff1d(candidates,list(chosen));chosen.update(rng('random_test_features',job['seed'],pop).choice(rest,min(100,len(rest)),False).tolist())
                selection[pop]=np.array(sorted(chosen))
            all_records.append(dict(population=pop,split=split,n_candidates=len(features),articles=len(unique),sequences=len(seq),
                mean_single_dce=ac.sum((1,2))/an.sum()/len(features),mean_single_kl=ak.sum((1,2))/an.sum()/len(features)))
    result=dict(status='complete',checkpoint_sha=h,records=all_records,ranking='Exhaustive calibration-alive candidates on selection articles',
        test='Independent articles; frozen top100 CE/KL union across baselines plus 100 random controls. NOT exhaustive test rankings.')
    write(done,result);return result

def summarize(plan,out):
    """Top-importance matching under each intervention population, both named baselines."""
    from .summary import matching,seed_bootstrap_matrix
    out=Path(out);cfg=plan['config'];expected=[j for j in plan['jobs'] if j['kind']=='real' and j['arm']['name'] in cfg.get('rank_arms',['baseline','beta_0.3','no_first','canonical_init','clip_restore'])]
    groups={};missing=[];result=[]
    for j in expected:
        root=out/'lm'/j['id']
        if not (root/'complete.json').exists():missing.append(j['id']);continue
        groups.setdefault((j['site'],j['arm']['name']),[]).append(j)
    for (site,arm),jobs in groups.items():
        arrays=[loadnp(out/'lm'/j['id']/'calibration.npz') for j in jobs];n=len(jobs)
        for pop in cfg.get('rank_populations',['all','after_first','native']):
            tabs=[loadnp(out/'lm'/j['id']/f'{pop}_selection_table.npz') for j in jobs]
            tests=[loadnp(out/'lm'/j['id']/f'{pop}_test_table.npz') for j in jobs]
            for baseline in (0,1):
                for metric in ('ce','kl'):
                    scores_=[t[metric][baseline].sum(1)/t['count'].sum() for t in tabs]
                    for topn in (10,50,100):
                        selected=[t['features'][np.lexsort((t['features'],-s))[:topn]] for t,s in zip(tabs,scores_)]
                        M=np.full((n,n),np.nan);D=M.copy();held=[];one_to_one=[]
                        from scipy.optimize import linear_sum_assignment
                        for i,a in enumerate(arrays):
                            loc={f:k for k,f in enumerate(tests[i]['features'])};ix=np.array([loc[f] for f in selected[i]])
                            held.append(float(tests[i][metric][baseline,ix].sum()/tests[i]['count'].sum()/len(ix)))
                            for j,b in enumerate(arrays):
                                if i==j:continue
                                c=matching(a,b);M[i,j]=np.mean(c[selected[i]]>=.7);D[i,j]=np.mean(c[a['alive']]>=.7)
                                aa=a['decoder'][selected[i]];bb=b['decoder'][b['alive']]
                                cos=aa@bb.T/np.maximum(np.linalg.norm(aa,axis=1)[:,None]*np.linalg.norm(bb,axis=1)[None],1e-30)
                                ii,jj=linear_sum_assignment(-((cos>=.7).astype(float)+cos/(4*max(len(aa),1))))
                                one_to_one.append(float((cos[ii,jj]>=.7).sum()/len(aa)))
                        result.append(dict(site=site,arm=arm,population=pop,baseline=('model','splice')[baseline],metric=metric,top_n=topn,
                            actual_ns=list(map(len,selected)),n_seeds=n,seeds=[j['seed'] for j in jobs],
                            pairwise_sharing=float(np.nanmean(M)) if n>1 else None,dictwide=float(np.nanmean(D)) if n>1 else None,
                            sharing_interval95=seed_bootstrap_matrix(M),one_to_one=float(np.mean(one_to_one)) if one_to_one else None,
                            heldout_mean_feature_effect=held,selected_ids=[s.tolist() for s in selected],sharing_matrix=M))
    report=dict(complete=not missing,expected=len(expected),finished=len(expected)-len(missing),missing=missing,results=result,
        caveat='SEED uncertainty only for displayed sharing interval. Selection article sensitivity is separate; no corpus-generalization CI claimed here.')
    write(out/'LM_SUMMARY.json',report);return report
