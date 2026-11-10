"""Prepare disjoint WikiText-103 document pools; no model or SAE required."""
import argparse, hashlib, json, pathlib, re


def main():
    from datasets import load_dataset
    p=argparse.ArgumentParser();p.add_argument('--out',default='corpus');p.add_argument('--max-docs',type=int,default=256)
    a=p.parse_args();root=pathlib.Path(a.out);root.mkdir(parents=True,exist_ok=True)
    all_hashes=set();meta={}
    for split in ['train','test']:
        data=load_dataset('Salesforce/wikitext','wikitext-103-raw-v1',split=split,streaming=True)
        docs=[];buf=[]
        def flush():
            text='\n'.join(buf).strip()
            if len(text)<256:return
            sha=hashlib.sha256(text.encode()).hexdigest()
            if sha in all_hashes:return
            all_hashes.add(sha);docs.append({'doc_id':sha[:20],'sha256':sha,'text':text,'split':split})
        for row in data:
            t=row['text'].strip()
            if re.fullmatch(r'= [^=].*[^=] =',t):
                flush();buf=[]
                if len(docs)>=a.max_docs:break
            if t:buf.append(t)
        else:flush()
        path=root/(split+'.jsonl')
        path.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs))
        meta[split]={'documents':len(docs),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if len(docs)<30:raise RuntimeError(f'Only {len(docs)} documents for {split}; inspect article parsing')
    (root/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
