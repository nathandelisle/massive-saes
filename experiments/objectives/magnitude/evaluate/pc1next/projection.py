"""Exact cross-seed agreement from frozen aggregate outputs, without loading an LLM.

Manifest: {"pc1": "pc1.npy", "article_ids": "articles.npy", "cases": [
 {"name":"real_standard", "runs":[
  {"seed":42,"y":"s42_group_output.npy","row_ids":"s42_row_ids.npy"},
  {"seed":137,"y":"s137_group_output.npy","row_ids":"s137_row_ids.npy"}]}]}

Each y is [valid_tokens, d_model], sum of the frozen group's z_j d_j, NO bias.
Paths are relative to the manifest. Article IDs refer to source articles, not rows.
Add separate cases for each existing matched-control draw, with stable draw IDs.
Writes one JSON per case/pair atomically; skips only hash-identical completed units.
Requires numpy. --smoke checks three analytic counterexamples in under a minute.
"""
import argparse
import hashlib
import itertools
import json
import tempfile
from pathlib import Path

import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def divide(a,b):
    return float(a/b) if b>1e-24 else None


def centered_stats(aa,bb,ab,sa,sb,n):
    va=aa-float(sa@sa)/n;vb=bb-float(sb@sb)/n
    cov=ab-float(sa@sb)/n
    # Zero-variation populations have undefined centered agreement.
    tol=1e-12*max(aa,bb,1.)
    den=va+vb
    return dict(centered_disagreement=divide(max(den-2*cov,0.),den) if den>tol else None,
                centered_inner_product_similarity=divide(cov,np.sqrt(max(va*vb,0.)))
                if min(va,vb)>tol else None,
                variance_a=max(va,0.),variance_b=max(vb,0.))


def compare_outputs(a,b,v,articles=None,chunk=1024):
    if a.ndim!=2 or a.shape!=b.shape or len(a)<2:raise ValueError('Aligned nonempty [tokens, channels] outputs required')
    n,d=a.shape;v=np.asarray(v,dtype=np.float64)
    if v.shape!=(d,) or not np.isfinite(v).all() or np.linalg.norm(v)==0:raise ValueError('Invalid PC1')
    v=v/np.linalg.norm(v)
    if articles is not None and np.asarray(articles).shape!=(n,):raise ValueError('One article ID per token required')
    total_a=total_b=perp_a=perp_b=perp_ab=full_diff=perp_diff=parallel_diff=0.
    sa=np.zeros(d);sb=np.zeros(d);by_article={}
    for start in range(0,n,chunk):
        stop=min(start+chunk,n)
        x=np.asarray(a[start:stop],dtype=np.float64);y=np.asarray(b[start:stop],dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite aggregate outputs')
        xp=x-(x@v)[:,None]*v;yp=y-(y@v)[:,None]*v
        delta=x-y;deltap=xp-yp
        total_a+=np.square(x).sum();total_b+=np.square(y).sum()
        perp_a+=np.square(xp).sum();perp_b+=np.square(yp).sum();perp_ab+=(xp*yp).sum()
        full_diff+=np.square(delta).sum();perp_diff+=np.square(deltap).sum()
        parallel_diff+=np.square(delta@v).sum();sa+=xp.sum(0);sb+=yp.sum(0)
        if articles is not None:
            ids=np.asarray(articles[start:stop])
            for key in np.unique(ids):
                mask=ids==key;u=xp[mask];w=yp[mask]
                stat=by_article.setdefault(str(key),dict(n=0,aa=0.,bb=0.,ab=0.,sa=np.zeros(d),sb=np.zeros(d)))
                stat['n']+=len(u);stat['aa']+=np.square(u).sum();stat['bb']+=np.square(w).sum()
                stat['ab']+=(u*w).sum();stat['sa']+=u.sum(0);stat['sb']+=w.sum(0)
    den=perp_a+perp_b
    qmean=(perp_a/total_a+perp_b/total_b)/2 if min(total_a,total_b)>1e-24 else None
    mse=divide(2*full_diff,total_a+total_b)
    result=dict(n_tokens=n,n_channels=d,total_normalized_mse=mse,
        exact_offaxis_disagreement=divide(perp_diff,den),total_numerator_upper_bound=divide(full_diff,den),
        legacy_offaxis_proxy=divide(mse,2*qmean) if qmean is not None else None,
        parallel_disagreement_contribution_to_bound=divide(parallel_diff,den),
        offaxis_energy_fraction=divide(den,total_a+total_b),
        mean_vector_fraction_of_offaxis_energy=divide((sa@sa+sb@sb)/n,den),
        decomposition_relative_error=divide(abs(full_diff-perp_diff-parallel_diff),max(full_diff,1.)),
        evaluation_centered=centered_stats(perp_a,perp_b,perp_ab,sa,sb,n))
    if by_article:
        ss=[centered_stats(x['aa'],x['bb'],x['ab'],x['sa'],x['sb'],x['n']) for x in by_article.values()]
        valid=[x for x in ss if x['centered_disagreement'] is not None]
        dv=sum(x['variance_a']+x['variance_b'] for x in valid)
        num=sum(x['centered_disagreement']*(x['variance_a']+x['variance_b']) for x in valid)
        result['within_article_centered']=dict(n_articles=len(by_article),n_nondegenerate_articles=len(valid),
            disagreement=divide(num,dv),note='Variance-weighted after separate within-article centering; descriptive, not a confidence interval.')
    return result


def smoke():
    rng=np.random.default_rng(37);n=120;v=np.array([1.,0.,0.])
    a=rng.normal(size=(n,3));b=a.copy();b[:,0]+=5
    r=compare_outputs(a,b,v)
    assert r['exact_offaxis_disagreement']<1e-20 and r['total_numerator_upper_bound']>1
    assert r['evaluation_centered']['centered_inner_product_similarity']>.999999
    # A shared large offset can hide unrelated fluctuations; centering reveals it.
    a=rng.normal(size=(n,3));b=rng.normal(size=(n,3));a[:,1]+=100;b[:,1]+=100
    r=compare_outputs(a,b,v)
    assert r['exact_offaxis_disagreement']<.001
    assert r['evaluation_centered']['centered_disagreement']>.5
    # Article means alone can create apparent token-varying agreement globally.
    ids=np.repeat(np.arange(4),n//4);a=rng.normal(size=(n,3));b=rng.normal(size=(n,3))
    a[:,1]+=50*ids;b[:,1]+=50*ids
    r=compare_outputs(a,b,v,ids)
    assert r['evaluation_centered']['centered_disagreement']<.01
    assert r['within_article_centered']['disagreement']>.5
    # Direct projections and Pythagoras must agree across chunk sizes.
    assert r['decomposition_relative_error']<1e-12
    s=compare_outputs(a,b,v,ids,chunk=17)
    assert np.isclose(r['exact_offaxis_disagreement'],s['exact_offaxis_disagreement'])
    print('[pc1-projected] smoke passed: projection, shared means, article means, chunk invariance',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--chunk',type=int,default=1024);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.smoke:smoke();return
    if args.manifest is None or args.out is None or args.chunk<=0:p.error('--manifest, --out and positive --chunk required')
    base=args.manifest.resolve().parent;manifest=json.loads(args.manifest.read_text())
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    path=lambda s:(base/s).resolve()
    vpath=path(manifest['pc1']);v=np.load(vpath,allow_pickle=False)
    artpath=path(manifest['article_ids']) if manifest.get('article_ids') else None
    articles=np.load(artpath,allow_pickle=False) if artpath else None
    hashed={}
    def fingerprint(f):
        if f not in hashed:hashed[f]=sha256(f)
        return hashed[f]
    common={str(f):fingerprint(f) for f in [args.manifest.resolve(),Path(__file__).resolve(),vpath]+([artpath] if artpath else [])}
    for ci,case in enumerate(manifest['cases']):
        runs=case['runs']
        if len({r['seed'] for r in runs})!=len(runs):raise ValueError('Duplicate seed in case')
        for i,j in itertools.combinations(range(len(runs)),2):
            aa,bb=runs[i],runs[j];files=[path(x[k]) for x in (aa,bb) for k in ('y','row_ids')]
            provenance={**common,**{str(f):fingerprint(f) for f in files}}
            target=out/f'case{ci:03d}_pair{i:02d}_{j:02d}.json'
            if target.exists():
                prev=json.loads(target.read_text())
                if prev.get('input_sha256')!=provenance:raise ValueError(f'Changed inputs for existing unit: {target}')
                print('[pc1-projected] skip',target.name,flush=True);continue
            a,aid,b,bid=[np.load(f,mmap_mode='r',allow_pickle=False) for f in files]
            if aid.ndim!=1 or len(aid)!=len(a) or not np.array_equal(aid,bid):raise ValueError('Token alignment not certified')
            if len(np.unique(aid))!=len(aid):raise ValueError('row_ids must uniquely identify token positions')
            result=compare_outputs(a,b,v,articles,args.chunk)
            payload=dict(case=case['name'],seeds=[aa['seed'],bb['seed']],metrics=result,input_sha256=provenance)
            temp=target.with_suffix('.json.tmp');temp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');temp.replace(target)
            print('[pc1-projected]',case['name'],aa['seed'],bb['seed'],
                  'exact_perp',result['exact_offaxis_disagreement'],'bound',result['total_numerator_upper_bound'],flush=True)


if __name__=='__main__':main()
