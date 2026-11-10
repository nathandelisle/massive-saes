"""One HF model pass caches all chosen residual-post sites, with token/document IDs."""
import argparse,hashlib,json,pathlib,time
import numpy as np


def token_batches(tokenizer,path,context,prepend_bos,batch_size):
    docs=[json.loads(s) for s in pathlib.Path(path).read_text().splitlines() if s]
    queues=[]
    bos=[tokenizer.bos_token_id] if prepend_bos else []
    if bos==[None]:raise ValueError('Requested BOS but tokenizer has no BOS id')
    budget=context-len(bos)
    for i,d in enumerate(docs):
        ids=tokenizer.encode(d['text'],add_special_tokens=False)
        chunks=[bos+ids[j:j+budget] for j in range(0,len(ids),budget) if len(ids[j:j+budget])>=8]
        queues.append((i,chunks))
    # Round-robin documents prevents the first long article owning the sample.
    seqs=[];labels=[]
    for depth in range(max((len(q) for _,q in queues),default=0)):
        for doc,chunks in queues:
            if depth>=len(chunks):continue
            seqs.append(chunks[depth]);labels.append(doc)
            if len(seqs)==batch_size:
                yield seqs,labels,docs;seqs=[];labels=[]
    if seqs:yield seqs,labels,docs


def get_blocks(model):
    for name in ['layers','h','gpt_neox.layers','model.layers','transformer.h','language_model.layers','language_model.model.layers']:
        obj=model
        try:
            for part in name.split('.'):obj=getattr(obj,part)
            if len(obj)>0:return obj
        except (AttributeError,TypeError):pass
    raise RuntimeError('Unrecognized block layout; inspect native HF model, do not substitute hidden_states[-1]')


def main():
    import torch
    from transformers import AutoModel,AutoTokenizer
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='manifest.json');p.add_argument('--group',required=True)
    p.add_argument('--corpus',default='corpus');p.add_argument('--cache',default='cache');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='cuda:0');p.add_argument('--context',type=int)
    p.add_argument('--attn',default='',help='override attn_implementation; default eager for gemma, else sdpa')
    a=p.parse_args();m=json.loads(pathlib.Path(a.manifest).read_text());jobs=[j for j in m['jobs'] if j['group']==a.group]
    if not jobs:raise ValueError(a.group)
    job=jobs[0];layers=sorted({j['layer'] for j in jobs});context=a.context or m['context_length']
    root=pathlib.Path(a.cache)/a.group/f'ctx{context}';root.mkdir(parents=True,exist_ok=True)
    corpus_hashes={s:hashlib.sha256((pathlib.Path(a.corpus)/(s+'.jsonl')).read_bytes()).hexdigest() for s in ['train','test']}
    settings=dict(protocol=m['protocol'],model=job['model'],model_revision=job['model_revision'],layers=layers,
        context_length=context,prepend_bos=job['prepend_bos'],dtype='float32',
        calibration_rows=m['calibration_rows'],evaluation_rows=m['evaluation_rows'],corpus_sha256=corpus_hashes)
    fingerprint=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    meta_path=root/'metadata.json'
    if meta_path.exists():
        old=json.loads(meta_path.read_text())
        if old.get('complete') and old.get('fingerprint')==fingerprint:
            print('Cache already complete',root);return
        raise RuntimeError('Existing cache differs/is incomplete; use another cache directory or explicitly remove the incomplete group')
    if not job.get('model_revision'):raise RuntimeError('Model revision must be pinned')
    torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(job['model'],revision=job['model_revision'])
    # Eager attention preserves Gemma-2/3 attention-logit soft-capping.
    attn=a.attn or ('eager' if 'gemma' in job['model'].lower() else 'sdpa')
    print(json.dumps({'model':job['model'],'attn_implementation':attn}),flush=True)
    model=AutoModel.from_pretrained(job['model'],revision=job['model_revision'],torch_dtype=torch.float32,
                                    attn_implementation=attn,low_cpu_mem_usage=True).to(a.device).eval()
    blocks=get_blocks(model)
    capture={};handles=[]
    for layer in layers:
        def hook(module,args,out,l=layer):capture[l]=out[0] if isinstance(out,tuple) else out
        handles.append(blocks[layer].register_forward_hook(hook))
    meta={**settings,'fingerprint':fingerprint,'complete':False,'splits':{},'special_token_ids':tok.all_special_ids}
    meta_path.write_text(json.dumps(meta,indent=2))
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:pad=0
    started=time.time()
    for split,target in [('train',m['calibration_rows']),('test',m['evaluation_rows'])]:
        folder=root/split;folder.mkdir(exist_ok=True);arrays={};n=0
        aux={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',shape=(target,),dtype=dt)
             for k,dt in [('doc','int32'),('pos','int32'),('token','int32'),('special','bool')]}
        for seqs,labels,docs in token_batches(tok,pathlib.Path(a.corpus)/(split+'.jsonl'),context,job['prepend_bos'],a.batch_size):
            ids=torch.full((len(seqs),context),pad,dtype=torch.long,device=a.device)
            mask=torch.zeros_like(ids)
            for i,seq in enumerate(seqs):ids[i,:len(seq)]=torch.tensor(seq,device=a.device);mask[i,:len(seq)]=1
            capture.clear()
            with torch.inference_mode():model(input_ids=ids,attention_mask=mask,use_cache=False)
            valid=mask.bool().flatten();take=min(int(valid.sum()),target-n)
            tokens=ids.flatten()[valid][:take].cpu().numpy()
            pos=np.broadcast_to(np.arange(context),(len(seqs),context)).reshape(-1)[valid.cpu().numpy()][:take]
            doc=np.repeat(labels,context)[valid.cpu().numpy()][:take]
            for layer in layers:
                acts=capture[layer].reshape(-1,capture[layer].shape[-1])[valid][:take].float().cpu().numpy()
                if not np.isfinite(acts).all():raise RuntimeError('Nonfinite model activations')
                if layer not in arrays:arrays[layer]=np.lib.format.open_memmap(folder/f'layer_{layer}.npy',mode='w+',shape=(target,acts.shape[1]),dtype='float32')
                arrays[layer][n:n+take]=acts
            for k,x in [('doc',doc),('pos',pos),('token',tokens),('special',np.isin(tokens,tok.all_special_ids))]:aux[k][n:n+take]=x
            n+=take
            if n>=target:break
        if n<target:raise RuntimeError(f'{split}: only {n}/{target} rows; add corpus documents, do not silently lower the target')
        for arr in [*arrays.values(),*aux.values()]:arr.flush()
        meta['splits'][split]={'rows':n,'n_documents':int(len(np.unique(aux['doc']))),
                              'document_ids':[d['doc_id'] for d in docs]}
        meta_path.write_text(json.dumps(meta,indent=2))
        print(json.dumps({'group':a.group,'split':split,'rows':n,'seconds':round(time.time()-started)}),flush=True)
    for h in handles:h.remove()
    meta['complete']=True;temp=root/'metadata.tmp';temp.write_text(json.dumps(meta,indent=2));temp.replace(meta_path)


if __name__=='__main__':main()
