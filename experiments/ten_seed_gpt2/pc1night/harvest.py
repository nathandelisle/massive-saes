"""Build the required row-aligned stores from article JSONL and a pinned HF model.

Article JSONL: {article_id, split: train|calibration|selection|test, text|rows}.
The caller assigns whole articles before this runs. See make_articles.py.
Native BOS remains in the LM context; row masks select SAE exposure separately.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .io import read,write,npz,loadnp,atomic,key,sha
from .data import SPLITS
from .lm import LM

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-config',required=True);p.add_argument('--articles',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--ctx',type=int,default=128)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--train-tokens',type=int,default=5000000)
    p.add_argument('--heldout-sequences',type=int,default=512);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cfg=read(a.model_config);protocol=key(dict(cfg=cfg,article_sha=sha(a.articles),ctx=a.ctx,batch=a.batch,train_tokens=a.train_tokens,heldout_sequences=a.heldout_sequences))
    if (out/'harvest_protocol.json').exists() and read(out/'harvest_protocol.json')['id']!=protocol:raise ValueError('Harvest protocol changed')
    write(out/'harvest_protocol.json',dict(id=protocol,settings=vars(a)))
    docs=[json.loads(l) for l in Path(a.articles).read_text().splitlines() if l.strip()];groups={s:[] for s in SPLITS};ids=set();content=set()
    for doc in docs:
        text=doc.get('text','\n'.join(doc.get('rows',[])));h=key(text)
        if str(doc['article_id']) in ids or h in content:raise ValueError('Duplicate article identity or content')
        ids.add(str(doc['article_id']));content.add(h);groups[doc['split']].append(doc)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg['lm']['model_path'],revision=cfg['lm']['revision'])
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    if tok.pad_token_id is None:raise ValueError('No pad/eos token')
    cfg['lm'].update(bos_token_id=tok.bos_token_id,eos_token_id=tok.eos_token_id)
    lm=LM(cfg,a.device);site=dict(name=cfg['name'],lm=cfg['lm'],min_articles=30,splits={},training=cfg.get('training',{}))
    for split in SPLITS:
        completed=out/(split+'_DONE.json')
        if completed.exists():site['splits'][split]=read(completed);continue
        docs=sorted(groups[split],key=lambda d:key(['article-order',d['article_id']]))
        if len(docs)<30:raise ValueError(f'{split} has {len(docs)} articles; need >=30')
        # Round-robin over article chunks: long documents cannot consume every eval slot.
        sequences=[];chunks=[]
        for doc in docs:
            text=doc.get('text','\n'.join(doc.get('rows',[])));token_ids=tok(text,add_special_tokens=False)['input_ids']
            size=a.ctx-tok.num_special_tokens_to_add(pair=False)
            chunked=[tok.build_inputs_with_special_tokens(token_ids[i:i+size]) for i in range(0,len(token_ids),size) if len(token_ids[i:i+size])>=2]
            chunks.append((doc['article_id'],chunked))
        nt=0;limit=a.train_tokens if split=='train' else a.heldout_sequences*a.ctx
        for index in range(max(len(c) for _,c in chunks)):
            for art,c in chunks:
                if index>=len(c):continue
                seq=c[index][:a.ctx];sequences.append((str(art),seq));nt+=len(seq)
                if nt>=limit:break
            if nt>=limit:break
        if len(set(art for art,_ in sequences))<30:raise ValueError('Token budget excludes too many articles')
        records=[]
        for start in range(0,len(sequences),a.batch):
            f=out/'harvest_units'/split/f'{start:07d}.npz'
            if not f.exists():
                seqs=sequences[start:start+a.batch];ii=np.full((len(seqs),a.ctx),tok.pad_token_id,np.int64);mm=np.zeros_like(ii,bool)
                for i,(_,s) in enumerate(seqs):ii[i,:len(s)]=s;mm[i,:len(s)]=True
                _,x=lm.forward(torch.tensor(ii,device=a.device),torch.tensor(mm,device=a.device));x=x.float().cpu().numpy()
                rr,pp=np.nonzero(mm);xx=x[rr,pp]
                npz(f,x=xx,input_ids=ii,attention_mask=mm,sequence_articles=np.array([s[0] for s in seqs]),
                    article=np.array([seqs[r][0] for r in rr]),position=pp,token_id=ii[rr,pp],
                    is_bos=np.zeros(len(rr),bool) if tok.bos_token_id is None else ii[rr,pp]==tok.bos_token_id,
                    is_eos=np.zeros(len(rr),bool) if tok.eos_token_id is None else ii[rr,pp]==tok.eos_token_id,
                    sequence_index=rr+start,token_index=pp)
            records.append(f)
            if start%128==0:print(f'[pc1night-harvest] {cfg["name"]} {split} seq={start}/{len(sequences)}',flush=True)
        # Materialize one memmap; keep only small metadata in RAM.
        n=sum(len(s) for _,s in sequences);first=loadnp(records[0]);d=first['x'].shape[1]
        tmp=out/(split+'.partial.npy');dest=np.lib.format.open_memmap(tmp,mode='w+',dtype='float32',shape=(n,d));offset=0;meta={};tokenmeta={}
        for f in records:
            unit=loadnp(f);l=len(unit['x']);dest[offset:offset+l]=unit.pop('x');offset+=l
            for k,v in unit.items():(tokenmeta if k in ('input_ids','attention_mask','sequence_articles') else meta).setdefault(k,[]).append(v)
        if offset!=n:raise AssertionError('Harvest row mismatch')
        dest.flush();del dest;tmp.replace(out/(split+'.npy'))
        npz(out/(split+'_meta.npz'),**{k:np.concatenate(v) for k,v in meta.items()})
        seqarts=np.concatenate(tokenmeta['sequence_articles']);npz(out/(split+'_tokens.npz'),input_ids=np.concatenate(tokenmeta['input_ids']),
            attention_mask=np.concatenate(tokenmeta['attention_mask']),article_ids=seqarts,
            sequence_ids=np.array([key([split,i,art]) for i,art in enumerate(seqarts)]))
        spec=dict(x=split+'.npy',meta=split+'_meta.npz',tokens=split+'_tokens.npz');write(completed,spec);site['splits'][split]=spec
    write(out/'site.json',site);print('[pc1night-harvest] complete',flush=True)

if __name__=='__main__':main()
