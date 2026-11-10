"""Optional GPU adapter for the supplied v3 SAE implementation and Hugging Face.
No cloud SDKs, cache guessing, train-if-missing calls, or implicit data harvests.
"""
import json, sys
from pathlib import Path
import numpy as np


def load_config(path):
    cfg=json.loads(Path(path).read_text())
    for key in ['v3_path','model_path','block_path','block_index','mean_path','pc1_path']:
        if key not in cfg:raise ValueError('Missing configuration field: '+key)
        if isinstance(cfg[key],str) and 'REPLACE' in cfg[key]:raise ValueError('Unresolved configuration field: '+key)
    if not Path(cfg['v3_path']).joinpath('v3/common/saes.py').is_file():
        raise ValueError('v3_path must contain v3/common/saes.py')
    return cfg


def load_lm(cfg):
    import torch
    from transformers import AutoModelForCausalLM
    if cfg.get('device_map'):
        raise ValueError('This first-pass runner supports one GPU. Defer sharded 70B runs.')
    dtype=getattr(torch,cfg.get('dtype','bfloat16'))
    model=AutoModelForCausalLM.from_pretrained(cfg['model_path'],
            revision=cfg.get('revision'),torch_dtype=dtype,
            local_files_only=cfg.get('local_files_only',True),attn_implementation='eager')
    model.to(cfg.get('device','cuda'));model.eval()
    module=model
    for attr in cfg['block_path'].split('.'):module=getattr(module,attr)
    return model,module[int(cfg['block_index'])]


def load_sae(cfg,run):
    import torch
    sys.path.insert(0,str(Path(cfg['v3_path']).resolve()))
    from v3.common.saes import ARCHS
    # Existing checkpoints contain state/cfg/extra dictionaries of tensors and
    # primitive values. Use the restricted loader; do not unpickle unknown objects.
    blob=torch.load(run['checkpoint'],map_location='cpu',weights_only=True)
    c=dict(blob['cfg']);arch=c.pop('arch');dm=c.pop('d_model');ds=c.pop('d_sae')
    c.pop('seed',None)
    sae=ARCHS[arch](dm,ds,**c).to(cfg.get('device','cuda'));sae.load_state_dict(blob['state']);sae.eval()
    return sae


def sae_output(sae,x,cfg,run):
    import torch
    mean=torch.as_tensor(np.load(cfg['mean_path']),dtype=torch.float32,device=x.device)
    v=torch.as_tensor(np.load(cfg['pc1_path']),dtype=torch.float32,device=x.device)
    v=v/torch.linalg.vector_norm(v)
    xc=x.float()-mean;coef=xc@v
    if run['condition']=='bypass':xc=xc-coef[...,None]*v
    z=sae.encode(xc);hat=sae.decode(z)
    if run['condition']=='bypass':hat=hat+coef[...,None]*v
    elif run['condition']!='standard':raise ValueError('Unknown condition')
    return hat+mean,z,v


def read_tokens(path):
    with np.load(path,allow_pickle=False) as n:
        ids=n['input_ids'].astype(np.int64);mask=n['attention_mask'].astype(bool)
        hashes=n['document_hashes'].astype(str)
    if ids.ndim!=2 or ids.shape!=mask.shape or len(hashes)!=len(ids):raise ValueError('Bad token cache')
    if not (mask.sum(1)>1).all():raise ValueError('Sequences need >=2 valid tokens')
    return ids,mask,hashes


def disjoint_tokens(*paths):
    records=[read_tokens(p) for p in paths]
    for i in range(len(records)):
        for j in range(i):
            if set(records[i][2])&set(records[j][2]):raise ValueError('Document overlap between splits')
    return records


def forward(model,module,ids,mask,edit=None):
    import torch
    captured={}
    def hook(_module,_args,output):
        x=output[0] if isinstance(output,tuple) else output
        captured['x']=x.detach()
        if edit is None:return output
        y=edit(x).to(x.dtype)
        if y.shape!=x.shape or not torch.isfinite(y).all():raise RuntimeError('Invalid intervention')
        return (y,)+output[1:] if isinstance(output,tuple) else y
    handle=module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits.float()
    finally:handle.remove()
    if 'x' not in captured:raise RuntimeError('Target block hook did not fire')
    if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite model logits')
    return logits,captured['x']


def sequence_scores(ref,other,ids,mask):
    """Per-sequence KL and signed CE change, teacher forced, valid next tokens."""
    import torch
    p=torch.log_softmax(ref[:,:-1],-1);q=torch.log_softmax(other[:,:-1],-1)
    valid=(mask[:,:-1].bool()&mask[:,1:].bool()).float()
    count=valid.sum(-1)
    kl=(p.exp()*(p-q)).sum(-1)
    ix=ids[:,1:,None]
    ce_delta=p.gather(-1,ix).squeeze(-1)-q.gather(-1,ix).squeeze(-1)
    # Preserve sums/counts so token-weighted aggregation and document bootstrap
    # both remain possible; the two estimands are not silently conflated.
    return dict(kl_sum=(kl*valid).sum(-1).cpu().numpy(),
                dce_sum=(ce_delta*valid).sum(-1).cpu().numpy(),
                token_count=count.cpu().numpy())
