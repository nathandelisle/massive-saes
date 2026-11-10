"""Exact ablation ranks, named matching conventions, paired seed/article uncertainty."""
from pathlib import Path
import itertools
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from .io import atomic_json, atomic_npz, rng_for, ratio
from .io import load_json


def top_ids(score, alive, n):
    ids=np.flatnonzero(alive)
    # Stable tie break is feature index; signed CE is never clipped to zero.
    return ids[np.lexsort((ids,-score[ids]))[:min(n,len(ids))]]


def max_cosine(a,b,source_alive,target_alive,chunk=256):
    a=a/np.linalg.norm(a,axis=1,keepdims=True);b=b/np.linalg.norm(b,axis=1,keepdims=True)
    target=np.flatnonzero(target_alive)
    if not len(target):raise ValueError('No matching candidates')
    best=np.full(len(a),np.nan); index=np.full(len(a),-1,dtype=int)
    source=np.flatnonzero(source_alive)
    for start in range(0,len(source),chunk):
        ix=source[start:start+chunk];c=a[ix]@b[target].T;j=c.argmax(1)
        best[ix]=c[np.arange(len(ix)),j];index[ix]=target[j]
    return best,index


def conventions(cosines,threshold):
    """Axes [reference seed, source feature]. No absent-seed imputation."""
    above=cosines>=threshold
    return dict(pairwise=float(above.mean()),mean_cosine=float((cosines.mean(0)>=threshold).mean()),
        any_other_seed=float(above.any(0).mean()),every_other_seed=float(above.all(0).mean()),
        no_other_seed_count=int((~above.any(0)).sum()))


def one_to_one_fraction(a,b,threshold):
    """Maximize the number above threshold, using cosine only to break ties.

    This avoids giving many source atoms credit for one canonical target atom.
    It is a matching diagnostic on the selected source set, not full-dictionary
    identifiability or a statement that selected atoms encode distinct concepts.
    """
    a=a/np.linalg.norm(a,axis=1,keepdims=True);b=b/np.linalg.norm(b,axis=1,keepdims=True)
    cos=a@b.T;weights=(cos>=threshold).astype(float)+cos/(4*max(len(a),1))
    i,j=linear_sum_assignment(-weights)
    return float((cos[i,j]>=threshold).sum()/len(a))


def estimate(table,metric,baseline,weights=None,equal_article=False):
    bi=list(table['baselines']).index(baseline)
    num=table[metric][bi];den=table['count']
    w=np.ones(len(num)) if weights is None else weights
    if equal_article:
        x=np.divide(num,den,out=np.zeros_like(num),where=den>0)
        return (x*w[:,None]).sum(0)/w.sum()
    return np.divide((num*w[:,None]).sum(0),(den*w[:,None]).sum(0),
        out=np.zeros(num.shape[1]),where=(den*w[:,None]).sum(0)>0)


def load_tables(cfg,out):
    records={}
    for run in cfg['runs']:
        name=run['name'];root=Path(out)
        paths=[root/'calibration'/(name+'.npz')]+[root/'rank_tables'/name/(s+'.npz') for s in ('selection','test')]
        if not all(p.is_file() for p in paths):continue
        loaded=[]
        for p in paths:
            with np.load(p) as n:loaded.append({k:n[k] for k in n.files})
        records[name]=dict(run=run,cal=loaded[0],selection=loaded[1],test=loaded[2])
    return records


def analyze(cfg,out,pid):
    records=load_tables(cfg,out); missing=[r['name'] for r in cfg['runs'] if r['name'] not in records]
    groups={}
    for r in cfg['runs']:groups.setdefault((r['condition'],float(r['beta'])),[]).append(r['name'])
    pairs={}; summaries=[]; individual=[]
    thresholds=cfg.get('cosine_thresholds',[0.5,0.7,0.9]); topns=cfg.get('top_ns',[10,50,100,200])
    baselines=cfg.get('baselines',['model','splice'])
    for group,names in groups.items():
        if any(n not in records for n in names):continue
        if len(names)<2:raise ValueError('Need >=2 seeds for sharing')
        for a,b in itertools.permutations(names,2):
            ca,cb=records[a]['cal'],records[b]['cal']
            path=Path(out)/'matching'/(a+'__'+b+'.npz')
            if path.exists():
                with np.load(path) as n:
                    if str(n['protocol_id'])!=pid:raise ValueError('Stale matching')
                    pairs[(a,b)]={k:n[k] for k in n.files}
            else:
                best,indices=max_cosine(ca['decoder'],cb['decoder'],ca['alive'],cb['alive'])
                allbest,_=max_cosine(ca['decoder'],cb['decoder'],ca['alive'],np.ones(len(cb['alive']),bool))
                freshbest,_=max_cosine(ca['decoder'],cb['decoder'],ca['alive'],cb['fresh_alive'])
                pairs[(a,b)]=dict(protocol_id=pid,best=best,indices=indices,all_candidate_best=allbest,fresh_candidate_best=freshbest)
                atomic_npz(path,**pairs[(a,b)])
        for split,baseline,metric,n,threshold in itertools.product(('selection','test'),baselines,('dce_sum','kl_sum'),topns,thresholds):
            seed_rows=[]
            for name in names:
                rec=records[name];alive=rec['cal']['alive'];score=estimate(rec[split],metric,baseline)
                chosen=top_ids(score,alive,n);cos=np.stack([pairs[(name,b)]['best'][chosen] for b in names if b!=name])
                cv=conventions(cos,threshold)
                dict_cv=conventions(np.stack([pairs[(name,b)]['best'][alive] for b in names if b!=name]),threshold)
                proxy=np.divide(rec[split]['z2_sum'].sum(0),rec[split]['count'].sum(0),out=np.zeros(len(alive)),where=rec[split]['count'].sum(0)>0)
                # Decoder norms are retained; energy is E[z^2] ||d||^2, not invariably E[z^2].
                proxy=proxy*np.square(np.linalg.norm(rec['cal']['decoder'],axis=1))
                proxy_top=top_ids(proxy,alive,n)
                alive_ids=np.flatnonzero(alive)
                rho=spearmanr(score[alive_ids],proxy[alive_ids]).statistic if len(alive_ids)>2 else np.nan
                sel_score=estimate(rec['selection'],metric,baseline);sel_top=top_ids(sel_score,alive,n)
                held_score=estimate(rec['test'],metric,baseline)
                row=dict(run=name,seed=rec['run']['seed'],split=split,baseline=baseline,metric=metric,top_n=n,
                    actual_n=len(chosen),threshold=threshold,sharing=cv,dictwide=dict_cv,
                    selected_ids=chosen.tolist(),nonpositive_in_top=int((score[chosen]<=0).sum()),
                    proxy_overlap=len(set(chosen)&set(proxy_top))/len(chosen),proxy_rank_spearman=rho,
                    selection_test_top_overlap=len(set(sel_top)&set(top_ids(held_score,alive,n)))/len(sel_top),
                    heldout_effect_selection_top=float(held_score[sel_top].mean()),
                    equal_article_top_overlap=len(set(chosen)&set(top_ids(estimate(rec[split],metric,baseline,equal_article=True),alive,n)))/len(chosen),
                    top_pc1_alignment=float(rec['cal']['alignment'][chosen].mean()),
                    all_candidates_pairwise=float(np.mean([pairs[(name,b)]['all_candidate_best'][chosen]>=threshold for b in names if b!=name])),
                    fresh_candidates_pairwise=float(np.mean([pairs[(name,b)]['fresh_candidate_best'][chosen]>=threshold for b in names if b!=name])))
                row['mutual_nearest_pairwise']=float(np.mean([
                    (pairs[(name,b)]['best'][chosen]>=threshold)&
                    (pairs[(b,name)]['indices'][pairs[(name,b)]['indices'][chosen]]==chosen)
                    for b in names if b!=name]))
                row['top_near_parallel_pc1_fraction']=float(np.mean(rec['cal']['alignment'][chosen]>=.9))
                row['one_to_one_pairwise']=None
                if metric=='dce_sum' and n==cfg.get('primary_top_n',50) and threshold==cfg.get('primary_threshold',.7):
                    row['one_to_one_pairwise']=float(np.mean([
                        one_to_one_fraction(rec['cal']['decoder'][chosen],records[b]['cal']['decoder'][records[b]['cal']['alive']],threshold)
                        for b in names if b!=name]))
                individual.append(row);seed_rows.append(row)
            summaries.append(dict(condition=group[0],beta=group[1],split=split,baseline=baseline,metric=metric,
                top_n=n,threshold=threshold,n_seeds=len(seed_rows),
                pairwise_sharing=float(np.mean([r['sharing']['pairwise'] for r in seed_rows])),
                dictwide_sharing=float(np.mean([r['dictwide']['pairwise'] for r in seed_rows])),
                proxy_overlap=float(np.mean([r['proxy_overlap'] for r in seed_rows])),
                no_other_seed_counts=[r['sharing']['no_other_seed_count'] for r in seed_rows]))
    contrasts=[]
    # Paired bootstrap resamples training seeds as graph vertices. Identical original
    # seeds are never treated as independent matching targets or allowed to self-match.
    beta_values=sorted({float(r['beta']) for r in cfg['runs']})
    if len(beta_values)>1:
        contrasts=paired_contrasts(cfg,records,pairs,groups,beta_values,out)
    fidelity_rows=[]
    for run in cfg['runs']:
        for split in ('selection','test'):
            p=Path(out)/'fidelity'/run['name']/(split+'.json')
            if p.exists():fidelity_rows.append(dict(condition=run['condition'],beta=run['beta'],seed=run['seed'],**load_json(p)))
    report=dict(protocol_id=pid,complete=not missing,missing_runs=missing,summaries=summaries,fidelity=fidelity_rows,
        per_seed=individual,paired_contrasts=contrasts,
        metric_statement='Signed single-feature zero-ablation CE primary; KL separate. Top-50 energy is a proxy audit only.',
        uncertainty_statement='Seed pairs share vertices. Paired bootstrap excludes self comparisons; only five training seeds limits precision.')
    atomic_json(Path(out)/'BRIDGE_RESULTS.json',report)
    print(f"[pc1-analysis] complete={not missing} runs={len(records)}/{len(cfg['runs'])} rows={len(summaries)}",flush=True)
    return report


def paired_contrasts(cfg,records,pairs,groups,betas,out):
    results=[];by_key={(r['condition'],float(r['beta']),r['seed']):r['name'] for r in cfg['runs']}
    seeds=sorted({r['seed'] for r in cfg['runs']});ns=len(seeds)
    endpoints=[]
    for cond in sorted({r['condition'] for r in cfg['runs']}):
        for b in betas[1:]:endpoints.append(((cond,betas[0]),(cond,b),'beta_effect'))
    if {'standard','bypass'}<=set(c for c,b in groups):
        for b in betas:endpoints.append((('standard',b),('bypass',b),'bypass_effect'))
    draws=cfg.get('bootstrap_draws',1000);topn=cfg.get('primary_top_n',50);thr=cfg.get('primary_threshold',0.7)
    for left,right,kind in endpoints:
        names=[[by_key.get((*g,s)) for s in seeds] for g in (left,right)]
        if any(n is None or n not in records for row in names for n in row):continue
        for baseline in cfg.get('baselines',['model','splice']):
            for split in ('selection','test'):
                arts=records[names[0][0]][split]['article_ids']
                if not all(np.array_equal(arts,records[n][split]['article_ids']) for row in names for n in row):raise ValueError('Article axes do not match across paired cells')
                def value(row,freq,w,anchor=None):
                    total=den=0.
                    for i,a in enumerate(row):
                        if not freq[i]:continue
                        score=estimate(records[a][split],'dce_sum',baseline,w)
                        selected=top_ids(score,records[a]['cal']['alive'],topn)
                        if anchor is not None:
                            aa=anchor[i];selected=top_ids(estimate(records[aa][split],'dce_sum',baseline,w),records[aa]['cal']['alive'],topn)
                            # Descendants of fixed initialized indices, not semantic atom identities.
                            selected=selected[records[a]['cal']['alive'][selected]]
                            if not len(selected):continue
                        for j,b in enumerate(row):
                            if i==j or not freq[j]:continue
                            weight=freq[i]*freq[j];total+=weight*np.mean(pairs[(a,b)]['best'][selected]>=thr);den+=weight
                    return total/den if den else np.nan
                ones=np.ones(ns);w=np.ones(len(arts));point=value(names[1],ones,w)-value(names[0],ones,w)
                loso=[]
                for i in range(ns):
                    f=ones.copy();f[i]=0
                    if f.sum()>=2:loso.append(value(names[1],f,w)-value(names[0],f,w))
                boot=[];rng=rng_for('paired-bootstrap',left,right,baseline,split)
                allow_article=cfg.get('split_policy','article')=='article'
                for _ in range(draws):
                    f=rng.multinomial(ns,np.full(ns,1/ns))
                    if (f>0).sum()<2:continue
                    aw=rng.multinomial(len(arts),np.full(len(arts),1/len(arts))) if allow_article else w
                    boot.append(value(names[1],f,aw)-value(names[0],f,aw))
                same_condition=left[0]==right[0]
                fixed_change=value(names[1],ones,w,anchor=names[0])-value(names[0],ones,w) if same_condition else None
                survivor_fractions=[]
                if same_condition:
                    for a,b in zip(*names):
                        anchor=top_ids(estimate(records[a][split],'dce_sum',baseline),records[a]['cal']['alive'],topn)
                        survivor_fractions.append(float(records[b]['cal']['alive'][anchor].mean()))
                results.append(dict(kind=kind,left=left,right=right,baseline=baseline,ranking_split=split,
                    top_n=topn,threshold=thr,delta_pairwise_sharing=point,
                    interval_95_percentile=np.quantile(boot,[.025,.975]).tolist() if boot else None,
                    bootstrap_valid_draws=len(boot),article_resampled=allow_article,n_articles=len(arts),
                    leave_one_seed_out=loso,fixed_initial_index_delta=fixed_change,fixed_index_alive_survivor_fractions=survivor_fractions,
                    fixed_index_note='Condition on beta-zero indices remaining calibration-alive; this is a selection diagnostic, not a proof of semantic identity.'))
    return results
