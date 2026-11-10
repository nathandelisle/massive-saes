"""Evaluate frozen released checkpoints against cached activations; resume per job."""
import argparse,collections,gc,hashlib,json,pathlib,time,traceback
import numpy as np
from metrics import sufficient_stats,merge_stats,summarize


def atomic_json(path,data):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False,default=str));tmp.replace(path)


def populations(folder,extra_position_filter=0):
    pops={'all':np.ones(len(np.load(folder/'doc.npy',mmap_mode='r')),dtype=bool),
          'no_special':~np.load(folder/'special.npy'),
          'after_first':np.load(folder/'pos.npy')>0}
    if extra_position_filter:
        # For releases that document a positional training filter wider than position 0 (e.g.
        # andyrdt/saes-qwen2.5-7b-instruct filters the first eight positions AND a batch-relative norm
        # cutoff). This reproduces only the POSITION half of such a filter; the norm cutoff is not applied.
        pops[f'after_{extra_position_filter}']=np.load(folder/'pos.npy')>=extra_position_filter
    return pops


def main():
    import torch
    from adapters import Adapter,fit_axis_gpu
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='manifest.json');p.add_argument('--group',required=True)
    p.add_argument('--cache',default='cache');p.add_argument('--out',default='results');p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--device',default='cuda:0');p.add_argument('--context',type=int);p.add_argument('--job-id')
    p.add_argument('--limit-jobs',type=int);p.add_argument('--deadline-unix',type=float)
    p.add_argument('--position-filter',type=int,default=0,help='add an after_<n> population for releases documenting one')
    a=p.parse_args();m=json.loads(pathlib.Path(a.manifest).read_text());ctx=a.context or m['context_length']
    jobs=[j for j in m['jobs'] if j['group']==a.group and (not a.job_id or j['job_id']==a.job_id)]
    if a.limit_jobs:jobs=jobs[:a.limit_jobs]
    root=pathlib.Path(a.cache)/a.group/f'ctx{ctx}';meta=json.loads((root/'metadata.json').read_text())
    if not meta['complete']:raise RuntimeError('Activation cache incomplete')
    for key in ['protocol','calibration_rows','evaluation_rows']:
        if meta[key]!=m[key]:raise RuntimeError('Activation cache uses a different '+key)
    for job in jobs:
        if (meta['model']!=job['model'] or meta['model_revision']!=job['model_revision'] or
            meta['prepend_bos']!=job['prepend_bos'] or meta['context_length']!=ctx):
            raise RuntimeError('Activation cache/model conventions do not match')
    out=pathlib.Path(a.out)/a.group/f'ctx{ctx}';out.mkdir(parents=True,exist_ok=True)
    doc=np.load(root/'test/doc.npy')
    train_masks=populations(root/'train',a.position_filter);test_masks=populations(root/'test',a.position_filter)
    axis_cache={};torch.backends.cuda.matmul.allow_tf32=False
    code_sha=hashlib.sha256(b''.join((pathlib.Path(__file__).parent/f).read_bytes() for f in ['audit.py','adapters.py','metrics.py'])).hexdigest()
    for job in jobs:
        if a.deadline_unix and time.time()>a.deadline_unix:break
        path=out/(job['job_id']+'.json')
        inputs={k:v for k,v in job.items() if k not in ['primary_space','primary_population','conventions_reviewed','convention_source','evaluation_interpretation']}
        fingerprint=hashlib.sha256(json.dumps({'job':inputs,'cache':meta['fingerprint'],'code':code_sha},sort_keys=True).encode()).hexdigest()
        if path.exists():
            old=json.loads(path.read_text())
            if old.get('status')=='ok' and old.get('fingerprint')==fingerprint:
                print('Already complete',job['job_id'],flush=True);continue
            if old.get('status')=='ok':raise RuntimeError('Successful result has different inputs; use another result directory')
        started=time.time();adapter=None
        try:
            adapter=Adapter(job,a.device)
            xtr=np.load(root/'train'/f"layer_{job['layer']}.npy",mmap_mode='r')
            xte=np.load(root/'test'/f"layer_{job['layer']}.npy",mmap_mode='r')
            if xtr.shape[1]!=adapter.sae.W_dec.shape[1]:raise RuntimeError('Hook dimension mismatch')
            smoke=adapter.smoke(np.asarray(xte[:128]))
            spaces=['raw']+(['sae_input'] if adapter.norm!='none' else [])
            axes={};groups={};l0s={pop:[0.,0] for pop in test_masks}
            alive=np.zeros(adapter.n_features,dtype=bool)
            for space in spaces:
                for pop,mask in train_masks.items():
                    key=(job['layer'],adapter.norm if space=='sae_input' else 'raw',pop)
                    if key not in axis_cache:
                        idx=np.flatnonzero(mask);x=np.asarray(xtr[idx])
                        if space=='sae_input':
                            x=np.concatenate([adapter.transform_input(x[s:s+a.batch_size]) for s in range(0,len(x),a.batch_size)])
                        origin,v,info=fit_axis_gpu(x,a.device)
                        if not info['converged']:raise RuntimeError(f'PC1 convergence requires review: {info}')
                        axis_cache[key]=(origin,v,info)
                    axes[(space,pop)]=axis_cache[key];groups[(space,pop)]={}
            for start in range(0,len(xte),a.batch_size):
                if a.deadline_unix and time.time()>a.deadline_unix:raise TimeoutError('Campaign deadline reached')
                stop=min(start+a.batch_size,len(xte));rec=adapter.reconstruct(xte[start:stop]);alive|=rec['alive']
                for pop,mask in test_masks.items():
                    keep=mask[start:stop];l0s[pop][0]+=float(rec['l0'][keep].sum());l0s[pop][1]+=int(keep.sum())
                for (space,pop),(origin,v,info) in axes.items():
                    keep=test_masks[pop][start:stop];x,h=rec[space];labs=doc[start:stop]
                    for d in np.unique(labs[keep]):
                        ix=keep&(labs==d)
                        st=sufficient_stats(x[ix],h[ix],origin,v)
                        old=groups[(space,pop)].get(int(d))
                        groups[(space,pop)][int(d)]=st if old is None else merge_stats([old,st],v)
            rows=[]
            for (space,pop),(origin,v,info) in axes.items():
                ids=sorted(groups[(space,pop)]);g=np.stack([groups[(space,pop)][d] for d in ids])
                st=merge_stats(g,v);values=summarize(st,v)
                if values['identity_error'] is not None and values['identity_error']>1e-6:raise RuntimeError('Energy partition identity failed')
                stats_file=f"{job['job_id']}__{space}__{pop}.npz"
                # Save before JSON; incomplete attempts never count as complete rows.
                np.savez_compressed(out/stats_file,group_stats=g,axis=v,origin=origin,document_indices=ids)
                rows.append({**values,'space':space,'population':pop,'pc1_fit':info,
                             'n_documents':len(ids),'l0':l0s[pop][0]/l0s[pop][1],'statistics_file':stats_file})
            result={'status':'ok','protocol':m['protocol'],'job':job,'cache_metadata':meta,
                    'fingerprint':fingerprint,'code_sha256':code_sha,'rows':rows,'adapter_smoke':smoke,
                    'loaded_config':adapter.cfg,'download_records':adapter.downloads,
                    'n_alive_any_eval_token':int(alive.sum()),'n_features':adapter.n_features,
                    'seconds':time.time()-started,'context_length':ctx}
            atomic_json(path,result);print(json.dumps({'job':job['job_id'],'status':'ok','seconds':round(time.time()-started)}),flush=True)
        except Exception as e:
            atomic_json(path.with_name(path.stem+f'.{time.time_ns()}.error.json'),{'status':'error','job':job,'fingerprint':fingerprint,
                'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'seconds':time.time()-started})
            print(json.dumps({'job':job['job_id'],'status':'error','error':str(e)}),flush=True)
        finally:
            if adapter is not None:del adapter
            gc.collect()
            torch.cuda.empty_cache()


if __name__=='__main__':main()
