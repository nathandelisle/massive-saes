"""Tiny local integration fixture. NEVER a replacement for a supplied v3 SAE."""
import json
from pathlib import Path
import numpy as np
import torch
from .io import atomic_json,atomic_npz


class ToySAE(torch.nn.Module):
    def __init__(self,d_model,d_sae,k=4):
        super().__init__();self.d_sae=d_sae;self.k=k
        self.W_enc=torch.nn.Parameter(torch.randn(d_model,d_sae)*.05)
        self.W_dec=torch.nn.Parameter(torch.nn.functional.normalize(torch.randn(d_sae,d_model),dim=1))
        self.b_enc=torch.nn.Parameter(torch.ones(d_sae)*.2)
        self.b_dec=torch.nn.Parameter(torch.zeros(d_model))
    def encode(self,x):
        y=torch.relu(x@self.W_enc+self.b_enc)
        val,ix=y.topk(self.k,dim=-1);return torch.zeros_like(y).scatter(-1,ix,val)
    def decode(self,z):return z@self.W_dec+self.b_dec


def fixture(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);dm=8;ds=8
    np.save(root/'mean.npy',np.zeros(dm,dtype=np.float32));v=np.zeros(dm,dtype=np.float32);v[0]=1;np.save(root/'pc1.npy',v)
    tokens={}
    for split in ('calibration','selection','test'):
        seed={'calibration':1,'selection':2,'test':3}[split];rng=np.random.default_rng(seed)
        ids=rng.integers(1,31,size=(4,6));mask=np.ones_like(ids,dtype=bool);mask[-1,-2:]=False;ids[~mask]=0
        atomic_npz(root/(split+'.npz'),input_ids=ids,attention_mask=mask,
            article_ids=np.array([f'{split}_a{i//2}' for i in range(4)]),sequence_ids=np.array([f'{split}_r{i}' for i in range(4)]))
        tokens[split]=str((root/(split+'.npz')).resolve())
    runs=[]
    for cond in ('standard','bypass'):
        for seed in (0,1,2):
            for beta in (0.,.3):
                torch.manual_seed(seed+10);sae=ToySAE(dm,ds,k=8)
                # Dense, positive pair with nested support exercises admissible surgery.
                with torch.no_grad():
                    sae.W_enc.mul_(.1);sae.b_enc.copy_(torch.linspace(.2,.8,ds))
                    sae.W_dec[:4,0]+=1.5+beta
                    sae.W_dec.copy_(torch.nn.functional.normalize(sae.W_dec,dim=1))
                name=f'{cond}_s{seed}_beta{beta:g}';cp=root/(name+'.pt')
                torch.save(dict(cfg=dict(arch='toy',d_model=dm,d_sae=ds,k=8),state=sae.state_dict()),cp)
                runs.append(dict(name=name,condition=cond,seed=seed,beta=beta,checkpoint=str(cp.resolve())))
    cfg=dict(model_path='LOCAL_RANDOM_GPT2_NOT_RESEARCH',toy_model=dict(vocab_size=32,n_positions=16,n_ctx=16,n_embd=dm,n_layer=2,n_head=2,
        resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,use_cache=False),model_seed=991,
        block_path='transformer.h',block_index=0,device='cpu',dtype='float32',mean_path=str((root/'mean.npy').resolve()),
        pc1_path=str((root/'pc1.npy').resolve()),tokens=tokens,runs=runs,sequence_batch=2,feature_batch=3,
        baselines=['model','splice'],primary_baseline='model',split_policy='article',group_multiplier=1.1,
        top_ns=[2,4],primary_top_n=2,cosine_thresholds=[.7],bootstrap_draws=24,null_draws=2,
        mechanism_betas=[0.,.3],intervention_budgets=[2],resample_draws=1,doses=[.1,1.],surgery_pairs=1,surgery_fractions=[0.,.5])
    atomic_json(root/'config.json',cfg);return root/'config.json'


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();print(fixture(a.out))
