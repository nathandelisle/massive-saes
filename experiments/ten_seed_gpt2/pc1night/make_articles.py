"""Whole-article split assignment, independently of activation outcomes."""
import argparse,json,sys
from pathlib import Path
from .io import key,atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-revision',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    from datasets import load_dataset
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'legacy_bridge'))
    from pc1next.data import wikitext_articles
    if Path(a.out).exists():raise FileExistsError(a.out)
    records=[];contents=set()
    for source in ('train','validation','test'):
        ds=load_dataset('Salesforce/wikitext','wikitext-103-v1',split=source,revision=a.dataset_revision)
        docs=list(wikitext_articles(ds,'wikitext-103-v1/'+source));docs.sort(key=lambda d:key(['20260908',d['article_id']]))
        for i,d in enumerate(docs):
            text='\n'.join(d['rows']);h=key(text)
            if h in contents:continue
            contents.add(h);split=('calibration' if i%2==0 else 'selection') if source=='validation' else source
            records.append(dict(article_id=d['article_id'],split=split,text=text,source=source,dataset_revision=a.dataset_revision))
    counts={s:sum(r['split']==s for r in records) for s in ('train','calibration','selection','test')}
    if min(counts.values())<30:raise ValueError(f'Need >=30 articles per split, found {counts}; supply another preregistered article corpus if necessary')
    atomic(a.out,lambda f:f.write(('\n'.join(json.dumps(r) for r in records)+'\n').encode()));print(counts)

if __name__=='__main__':main()
