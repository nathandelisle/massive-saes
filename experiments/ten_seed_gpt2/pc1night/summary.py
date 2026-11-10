"""Coverage-first synthesis. Paired contrasts and vertex bootstrap, no pair pseudoreplication."""
from pathlib import Path
import itertools
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from .io import read,write,npz,loadnp,rng,ratio,sha

def matching(a,b):
    A=a['decoder'];B=b['decoder'][b['alive']];out=np.full(len(A),np.nan)
    A=A/np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-30);B=B/np.maximum(np.linalg.norm(B,axis=1,keepdims=True),1e-30)
    if len(B):
        for start in range(0,len(A),128):out[start:start+128]=(A[start:start+128]@B.T).max(1)
    return out

def seed_bootstrap_matrix(M,draws=2000,seed=101):
    n=len(M);r=rng(seed,'vertices');points=[]
    for _ in range(draws):
        w=r.multinomial(n,np.ones(n)/n);weight=w[:,None]*w[None,:];np.fill_diagonal(weight,0)
        ok=np.isfinite(M);den=weight[ok].sum()
        if den:points.append(float((weight[ok]*M[ok]).sum()/den))
    return np.quantile(points,[.025,.975]).tolist() if points else None

def paired(values,draws=2000):
    x=np.asarray(values,float);n=len(x)
    if not n:return dict(n=0)
    r=rng('paired',values);b=x[r.integers(n,size=(draws,n))].mean(1)
    if n<=16:
        perm=[float(np.mean(x*np.array(s))) for s in itertools.product((-1,1),repeat=n)]
        p=float(np.mean(np.abs(perm)>=abs(x.mean())-1e-12))
    else:p=None
    return dict(n=n,mean=float(x.mean()),median=float(np.median(x)),values=x,
        interval95=np.quantile(b,[.025,.975]),exact_signflip_p=p,
        caveat='Small n limits inference; a null interval is not an equivalence test.')

def holm(rows):
    tested=[(i,r['exact_signflip_p']) for i,r in enumerate(rows) if r.get('exact_signflip_p') is not None]
    tested.sort(key=lambda v:v[1]);prev=0
    for rank,(i,p) in enumerate(tested):
        prev=max(prev,min(1,p*(len(tested)-rank)));rows[i]['holm_p']=prev

def collect(plan,out,draws=2000):
    out=Path(out);rows={};missing=[];groups={};bad=[]
    for job in plan['jobs']:
        p=out/'fits'/job['id'];marker=p/'complete.json'
        if not marker.exists():missing.append(job['id']);continue
        c=read(marker)
        if any(not (p/f).is_file() or sha(p/f)!=h for f,h in c['hashes'].items()):bad.append(job['id']);continue
        m=read(p/'metrics.json');rows[job['id']]=m;groups.setdefault((job['site'],job['arm']['name']),[]).append(job)
    group_reports=[];contrast=[];matrices={}
    for (site,arm),jobs in groups.items():
        jobs=sorted(jobs,key=lambda j:j['seed']);arrays=[loadnp(out/'fits'/j['id']/'arrays.npz') for j in jobs]
        n=len(jobs);metrics={k:np.full((n,n),np.nan) for k in ('dictionary_sharing','structural_sharing','tiles_sharing','canonical_sharing','group_nmse','group_offaxis_nmse')}
        for i,j in itertools.permutations(range(n),2):
            a,b=arrays[i],arrays[j];cos=matching(a,b)
            for label,mask in [('dictionary',a['alive'])]+[(g,a['mask_'+g]) for g in ('structural','tiles','canonical')]:
                if mask.any():metrics[label+'_sharing'][i,j]=np.mean(cos[mask]>=.7)
            if not np.array_equal(a['test_rows'],b['test_rows']):raise ValueError('Group comparison rows differ')
            ga=a['group_structural'].astype(float);gb=b['group_structural'].astype(float)
            r=ratio(np.square(ga-gb).sum(),.5*(np.square(ga).sum()+np.square(gb).sum()))
            if r is not None:metrics['group_nmse'][i,j]=r
            # Infer a common v from reference artifact, not a decoder principal component.
            ref=loadnp(out/'references'/site/'reference.npz');v=ref['pc1'];qa=ga-(ga@v)[:,None]*v;qb=gb-(gb@v)[:,None]*v
            r=ratio(np.square(qa-qb).sum(),.5*(np.square(qa).sum()+np.square(qb).sum()))
            if r is not None:metrics['group_offaxis_nmse'][i,j]=r
        matrices[(site,arm)]=(jobs,metrics)
        item=dict(site=site,arm=arm,seeds=[j['seed'] for j in jobs],n_seeds=n,metrics={})
        for k,M in metrics.items():
            item['metrics'][k]=dict(mean=float(np.nanmean(M)) if np.isfinite(M).any() else None,
                interval95=seed_bootstrap_matrix(M,draws),matrix=M)
        item['seed_cohorts']={}
        for cohort,seeds in plan['config'].get('seed_cohorts',{}).items():
            ii=[i for i,j in enumerate(jobs) if j['seed'] in seeds]
            item['seed_cohorts'][cohort]=dict(seeds=[jobs[i]['seed'] for i in ii],metrics={})
            for k,M in metrics.items():
                sub=M[np.ix_(ii,ii)]
                item['seed_cohorts'][cohort]['metrics'][k]=float(np.nanmean(sub)) if np.isfinite(sub).any() else None
        group_reports.append(item)
    for site in sorted({j['site'] for j in plan['jobs']}):
        base={m['job']['seed']:m for m in rows.values() if m['job']['site']==site and m['job']['arm']['name']=='baseline'}
        for arm in sorted({j['arm']['name'] for j in plan['jobs'] if j['site']==site and j['arm']['name']!='baseline'}):
            alt={m['job']['seed']:m for m in rows.values() if m['job']['site']==site and m['job']['arm']['name']==arm};seeds=sorted(set(base)&set(alt))
            getters={'perp_fvu':lambda m:m['fvu']['perp'],'total_fvu':lambda m:m['fvu']['total'],
                'l0':lambda m:m['mean_l0'],'canonical_count':lambda m:m['allocations']['canonical']['n_atoms'],
                'cancellation':lambda m:m['allocations']['structural']['cells']['all']['cancellation'],
                'refinement_gain':lambda m:m['code_refinement']['relative_reduction']}
            if base and next(iter(base.values()))['job']['kind']=='synthetic':getters['truth_r2']=lambda m:m['planted_truth']['mean_heldout_coefficient_r2']
            for metric,fn in getters.items():
                vals=[fn(alt[s])-fn(base[s]) for s in seeds if fn(alt[s]) is not None and fn(base[s]) is not None]
                contrast.append(dict(site=site,arm=arm,metric=metric,primary=arm in plan['primary_contrasts'],**paired(vals,draws)))
            if (site,'baseline') in matrices and (site,arm) in matrices:
                bj,bm=matrices[(site,'baseline')];aj,am=matrices[(site,arm)]
                bi=[i for s in seeds for i,j in enumerate(bj) if j['seed']==s];ai=[i for s in seeds for i,j in enumerate(aj) if j['seed']==s]
                for metric in bm:
                    diff=am[metric][np.ix_(ai,ai)]-bm[metric][np.ix_(bi,bi)]
                    contrast.append(dict(site=site,arm=arm,metric=metric,n=len(seeds),primary=arm in plan['primary_contrasts'],
                        mean=float(np.nanmean(diff)) if np.isfinite(diff).any() else None,interval95=seed_bootstrap_matrix(diff,draws),
                        method='paired seed-vertex bootstrap; no self-matches'))
    holm(contrast)
    synthetic_surface=[]
    for (site,arm),jobs in groups.items():
        if jobs[0]['kind']!='synthetic':continue
        mm=[rows[j['id']] for j in jobs]
        synthetic_surface.append(dict(site=site,arm=arm,design=jobs[0].get('synthetic'),seeds=[j['seed'] for j in jobs],
            truth_r2=[m['planted_truth']['mean_heldout_coefficient_r2'] for m in mm],
            canonical_count=[m['allocations']['canonical']['n_atoms'] for m in mm],
            content_fvu=[m['fvu']['perp'] for m in mm],
            manipulation_fraction_changed=[m['manipulation']['fraction_coefficients_changed'] for m in mm]))
    report=dict(expected=plan['expected'],finished=len(rows),missing=missing,corrupt=bad,complete=not missing and not bad,
        per_group=group_reports,contrasts=contrast,
        synthetic_surface=synthetic_surface,
        cautions=['No clinical-style binary verdicts from exploratory thresholds.',
            'Every failed and unlaunched cell remains in coverage.',
            'Groups with incomplete seeds are provisional; do not freeze manuscript numbers.',
            'Per-arm groups are an outcome; frozen-index and matching diagnostics answer different questions.'])
    write(out/'SUMMARY.json',report)
    lines=['# PC1 overnight campaign',f"\nCompleted {len(rows)} / {plan['expected']} registered fits. Corrupt: {len(bad)}.",
        '\nStatus: '+('COMPLETE' if report['complete'] else 'INCOMPLETE -- pending cells remain registered'),
        '\n| Site | Arm | Seeds | Dictionary sharing | Structural sharing |', '|---|---|---:|---:|---:|']
    for r in group_reports:
        vals=[r['metrics'][k]['mean'] for k in ('dictionary_sharing','structural_sharing')]
        ss=['--' if v is None else f'{v:.3f}' for v in vals]
        lines.append(f"| {r['site']} | {r['arm']} | {r['n_seeds']} | {ss[0]} | {ss[1]} |")
    lines+=['\nActual ablation ranks live in LM_SUMMARY.json; energy rankings never substitute for them.',
        'Inspect paired contrasts, held-out content error, and code-refinement gaps before attributing a mechanism.']
    (out/'MORNING_README.md').write_text('\n'.join(lines)+'\n')
    return report
