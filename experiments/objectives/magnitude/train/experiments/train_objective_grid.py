"""E4: additive TopK/beta/bypass grid on an explicit activation .npy store.
Reuses source SAE classes; implements an independent, checkpointed trainer.
Uniform samples, initialization seed and data-order seed are separated.
Never modifies old caches. Exact resume includes optimizer and RNG position.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from common import sha256,write_json,start_run,finish_run


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    sys.path.insert(0,str(Path(cfg['v3_path']).resolve()))
    from v3.common.saes import TopKL1SAE
    acts=np.load(cfg['activations'],mmap_mode='r');mu=np.load(cfg['mean_path']);v=np.load(cfg['pc1_path']);v=v/np.linalg.norm(v)
    if acts.ndim!=2 or acts.shape[1]!=len(mu):raise ValueError('Bad activations')
    if cfg.get('training_split')!='train':raise ValueError('Training split must be explicit')
    source=Path(cfg['v3_path'])/'v3/common/saes.py'
    inputs=[a.config,cfg['activations'],cfg['mean_path'],cfg['pc1_path'],source]
    if not a.resume:out,meta=start_run(a.out,cfg,inputs)
    else:
        out=Path(a.out);meta=json.loads((out/'manifest.json').read_text())
        if meta['config']!=cfg:raise ValueError('Resume configuration changed')
        if any(sha256(r['path'])!=r['sha256'] for r in meta['inputs']):raise ValueError('Resume input/source changed')
        if meta['code_sha256'].get(Path(__file__).name)!=sha256(__file__):raise ValueError('Trainer code changed; use a new output version')
    remote=None
    if cfg.get('checkpoint_s3_uri'):
        from checkpoint_store import S3CheckpointStore,protocol_fingerprint
        remote=S3CheckpointStore(cfg['checkpoint_s3_uri'],protocol_fingerprint(cfg,
            dict(inputs=meta['inputs'],trainer_sha256=meta['code_sha256'][Path(__file__).name],torch_version=str(torch.__version__))))
    dev=cfg.get('device','cuda');torch.use_deterministic_algorithms(cfg.get('deterministic',True))
    for cond in cfg.get('conditions',['standard','bypass']):
      for k in cfg.get('k_values',[64]):
       for seed in cfg.get('seeds',[42,137,0,1,2]):
        for beta in cfg.get('betas',[0,.03,.1,.3]):
            name=f'{cond}_k{k}_s{seed}_beta{beta:g}';rd=out/name;rd.mkdir(exist_ok=True)
            if (rd/'complete.json').exists():
                if not (rd/'sae.pt').exists() and remote:remote.restore(name,'final',rd/'sae.pt')
                if sha256(rd/'sae.pt')!=json.loads((rd/'complete.json').read_text())['checkpoint_sha256']:
                    raise ValueError('Completed checkpoint changed')
                continue
            if remote and remote.restore(name,'final',rd/'sae.pt'):
                write_json(rd/'complete.json',dict(checkpoint_sha256=sha256(rd/'sae.pt')))
                print('[pc1-E4] restored complete',name,flush=True)
                continue
            torch.manual_seed(seed)
            sae=TopKL1SAE(acts.shape[1],cfg['d_sae'],seed=seed,k=k,beta=beta).to(dev)
            opt=torch.optim.Adam(sae.parameters(),lr=cfg.get('lr',3e-4))
            rng=np.random.default_rng(cfg.get('data_order_seed',20260906));step0=0
            ck=rd/'resume.pt'
            if not ck.exists() and remote:remote.restore(name,'resume',ck)
            if ck.exists():
                state=torch.load(ck,map_location=dev,weights_only=True)
                sae.load_state_dict(state['state']);opt.load_state_dict(state['optimizer'])
                rng.bit_generator.state=state['numpy_rng'];step0=state['next_step']
                torch.set_rng_state(state['torch_rng'].cpu())
                if dev.startswith('cuda'):torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
            start=time.time();sae.train()
            mt=torch.tensor(mu,dtype=torch.float32,device=dev);vt=torch.tensor(v,dtype=torch.float32,device=dev)
            for step in range(step0,cfg.get('steps',20000)):
                ids=rng.integers(acts.shape[0],size=cfg.get('batch_size',4096))
                x=torch.tensor(np.asarray(acts[ids]),dtype=torch.float32,device=dev)-mt
                if cond=='bypass':x=x-(x@vt)[:,None]*vt
                elif cond!='standard':raise ValueError(cond)
                hat,z=sae(x);mse=(hat-x).square().sum(-1).mean();loss=mse+beta*z.abs().sum(-1).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                opt.zero_grad(set_to_none=True);loss.backward();opt.step();sae.renorm_decoder()
                if step%100==0:
                    rec=dict(step=step,objective=float(loss.item()),mse=float(mse.item()),
                              l0=float((z!=0).sum(-1).float().mean().item()),
                              l1=float(z.abs().sum(-1).mean().item()),elapsed_seconds=time.time()-start)
                    if rec['l0']>k+1e-5:raise RuntimeError('TopK support exceeded')
                    with open(rd/'training.jsonl','a') as f:f.write(json.dumps(rec)+'\n')
                    print(name,rec,flush=True)
                if (step+1)%cfg.get('checkpoint_every',500)==0 or step+1==cfg.get('steps',20000):
                    state=dict(state=sae.state_dict(),optimizer=opt.state_dict(),next_step=step+1,
                         numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if dev.startswith('cuda') else [])
                    torch.save(state,rd/'resume.tmp');(rd/'resume.tmp').replace(ck)
                    if remote:remote.save(name,'resume',ck)
            final=dict(state=sae.state_dict(),cfg=dict(arch='topk_l1',d_model=acts.shape[1],d_sae=cfg['d_sae'],k=k,beta=beta),
                       extra=dict(condition=cond,seed=seed,data_order_seed=cfg.get('data_order_seed',20260906)))
            torch.save(final,rd/'sae.tmp');(rd/'sae.tmp').replace(rd/'sae.pt')
            if remote:remote.save(name,'final',rd/'sae.pt')
            write_json(rd/'complete.json',dict(checkpoint_sha256=sha256(rd/'sae.pt')))
    finish_run(out,meta)

if __name__=='__main__':main()
