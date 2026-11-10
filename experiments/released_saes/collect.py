"""Collect all measured views, bootstrap documents, export only reviewed plot rows."""
import argparse,csv,json,pathlib
import numpy as np
from metrics import article_bootstrap


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with pathlib.Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='manifest.json');p.add_argument('--results',default='results')
    p.add_argument('--out',default='tables');p.add_argument('--bootstrap',type=int,default=2000)
    a=p.parse_args();m=json.loads(pathlib.Path(a.manifest).read_text());jobs={j['job_id']:j for j in m['jobs']}
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[];selected=[];seen=set();complete=set();errors=[]
    for path in sorted(pathlib.Path(a.results).rglob('*.json')):
        if path.name.endswith('.error.json'):
            e=json.loads(path.read_text());errors.append({'job_id':e['job']['job_id'],'error':e['error'],'source':str(path)});continue
        r=json.loads(path.read_text())
        if r.get('status')!='ok':continue
        if r['protocol']!=m['protocol']:raise ValueError('Mixed protocol '+str(path))
        job=jobs[r['job']['job_id']]
        for key in ['repo_revision','model_revision','layer','repo_path']:
            if r['job'][key]!=job[key]:raise ValueError('Checkpoint/manifest mismatch '+str(path))
        cache=r['cache_metadata']
        for key in ['protocol','calibration_rows','evaluation_rows']:
            if cache[key]!=m[key]:raise ValueError('Cache/manifest protocol mismatch: '+key+' '+str(path))
        if (cache['model']!=job['model'] or cache['model_revision']!=job['model_revision'] or
            cache['prepend_bos']!=job['prepend_bos'] or cache['context_length']!=r['context_length'] or
            not cache['complete']):raise ValueError('Cache/model convention mismatch '+str(path))
        if r['context_length']==m['context_length']:complete.add(job['job_id'])
        for row in r['rows']:
            if row['population']=='all' and row['n_rows']!=m['evaluation_rows']:
                raise ValueError('Incomplete evaluation population '+str(path))
            key=(job['job_id'],r['context_length'],row['space'],row['population'])
            if key in seen:raise ValueError('Duplicate measurement: '+str(key))
            seen.add(key)
            source=path.parent/row['statistics_file'];g=np.load(source)
            ci=article_bootstrap(g['group_stats'],g['axis'],draws=a.bootstrap)
            flat={'job_id':job['job_id'],'family':job['family'],'model':job['model'],'layer':job['layer'],
                  'sae_id':job['sae_id'],'repo_revision':job['repo_revision'],'model_revision':job['model_revision'],
                  'protocol':r['protocol'],'context_length':r['context_length'],'space':row['space'],
                  'population':row['population'],'n_documents':row['n_documents'],
                  'cache_fingerprint':r['cache_metadata']['fingerprint'],'code_sha256':r['code_sha256'],
                  'source':str(path),'n_features':r['n_features'],'n_alive_any_eval_token':r['n_alive_any_eval_token'],
                  **{k:row[k] for k in ['n_rows','fvu_total','fvu_pc1','fvu_content','content_over_total','w1_eval','residual_energy_on_pc1','l0']}}
            for k in ['fvu_total','fvu_content','content_over_total','w1_eval']:
                if k in ci:flat[k+'_lo95'],flat[k+'_hi95']=ci[k]
            rows.append(flat)
            requested=job.get('primary_space')
            # Constant global rescaling leaves these metrics unchanged; no extra point.
            if requested=='sae_input' and r['adapter_smoke']['normalization']=='none':requested='raw'
            if (job.get('conventions_reviewed') and requested==row['space'] and
                job.get('primary_population')==row['population'] and r['context_length']==m['context_length']):
                if not job.get('convention_source'):raise ValueError('Reviewed conventions need a source')
                selected.append({**flat,'convention_source':job['convention_source'],
                                 'evaluation_interpretation':job.get('evaluation_interpretation','')})
    csv_write(out/'all_measured_views.csv',rows)
    csv_write(out/'figure_released_fvu.csv',selected)
    csv_write(out/'attempt_errors.csv',errors)
    expected={j['job_id'] for j in jobs.values() if j['tier']=='core'}
    report={'core_expected':len(expected),'core_completed':len(expected&complete),
            'core_missing':sorted(expected-complete),'figure_checkpoint_count':len(selected),
            'figure_pending_convention_review':sorted((expected&complete)-{r['job_id'] for r in selected}),
            'note':'Repeated views, widths, and L0s are not independent model replications. Errors are retained even if a retry succeeded.'}
    (out/'completion.json').write_text(json.dumps(report,indent=2))
    family=[]
    for fam in sorted({r['family'] for r in selected}):
        rs=[r for r in selected if r['family']==fam];vals=np.array([r['content_over_total'] for r in rs],float)
        family.append({'family':fam,'checkpoints':len(rs),'model_layer_sites':len({(r['model'],r['layer']) for r in rs}),
                       'median_ratio':float(np.median(vals)),**{f'n_ratio_gt_{t}':int((vals>t).sum()) for t in [2,5,10]}})
    csv_write(out/'family_summary.csv',family);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
