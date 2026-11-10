import argparse,os,time,traceback,platform
from pathlib import Path
import numpy as np
from .io import read,write,loadnp,seal,sha
from .data import resolve_site,verify_site,prepare

def preparation(plan,out):
    out=Path(out);frozen=out/'INPUTS.json'
    if frozen.exists():return read(frozen)
    sites={};hashes={}
    for s in plan['config']['sites']:
        site=resolve_site(s['manifest']);stores,h=verify_site(site)
        if site['name']!=s['name']:raise ValueError('Site name mismatch')
        sites[site['name']]=site;hashes[site['name']]=h
        prepare(site,out/'references'/site['name'])
    for job in plan['jobs']:
        if job['kind']!='synthetic' or job['site'] in sites:continue
        from .synthetic import generate
        p=generate(out/'synthetic_data',job['site'],job['synthetic'],plan['config'].get('synthetic_rows',32768))
        site=resolve_site(p);_,h=verify_site(site);sites[site['name']]=site;hashes[site['name']]=h;prepare(site,out/'references'/site['name'])
    # Paths remain local in execution map; portable identities live in the protocol.
    pc=dict(plan);pc['config']=dict(plan['config']);pc['config']['sites']=[{'name':s['name']} for s in plan['config']['sites']]
    pid=seal(out,dict(plan=pc,input_hashes=hashes))
    result=dict(protocol=pid,sites=sites,hashes=hashes);write(frozen,result);return result

def coverage(plan,out):
    rows=[];out=Path(out)
    for j in plan['jobs']:
        p=out/'fits'/j['id'];state='complete' if (p/'complete.json').exists() else 'pending'
        if state=='pending' and (p/'latest.pt').exists():state='checkpointed'
        if state=='pending' and (out/'failures'/(j['id']+'.json')).exists():state='failed'
        rows.append(dict(id=j['id'],state=state))
    result=dict(expected=len(rows),counts={s:sum(r['state']==s for r in rows) for s in ('complete','pending','checkpointed','failed')},rows=rows)
    write(out/'COVERAGE.json',result);return result

def main():
    p=argparse.ArgumentParser(description='PC1 overnight selection-mechanism campaign')
    p.add_argument('stage',choices=['prepare','train','rank','causal','collect','status','smoke'])
    p.add_argument('--plan');p.add_argument('--out',default=os.environ.get('PC1_NIGHT_OUT','results/pc1night'))
    p.add_argument('--device',default=os.environ.get('DEVICE','cuda'));p.add_argument('--shard',type=int,default=int(os.environ.get('SHARD',0)))
    p.add_argument('--nshards',type=int,default=int(os.environ.get('NSHARDS',1)));p.add_argument('--only',default='*')
    p.add_argument('--max-seconds',type=float,default=float(os.environ.get('MAX_SECONDS',43200)))
    p.add_argument('--pilot-fits',type=int,default=0,help='Timing only; retains FULL registered coverage')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.stage=='smoke':
        from .smoke import run
        run(out);return
    if not a.plan:p.error('--plan is required')
    plan=read(a.plan)
    if not 0<=a.shard<a.nshards:p.error('Require 0<=SHARD<NSHARDS')
    if a.stage=='prepare':
        preparation(plan,out);print('[pc1night-prepare] complete',flush=True);return
    if a.stage=='status':print(coverage(plan,out)['counts']);return
    if not (out/'INPUTS.json').exists():raise FileNotFoundError('Run prepare before launching workers')
    inp=read(out/'INPUTS.json');pid=inp['protocol']
    if a.stage=='collect':
        from .summary import collect
        from .lm import summarize
        from .causal import summarize as causal_summary
        r=collect(plan,out,plan['config'].get('bootstrap_draws',2000));lm=summarize(plan,out);ca=causal_summary(plan,out);coverage(plan,out)
        print(f"[pc1night-collect] fits={r['finished']}/{r['expected']} ranks={lm['finished']}/{lm['expected']} causal={ca['finished']}/{ca['expected']}",flush=True)
        write(out/'CAMPAIGN_STATUS.json',dict(complete=r['complete'] and lm['complete'] and ca['complete'],
            fits=dict(done=r['finished'],expected=r['expected']),ranks=dict(done=lm['finished'],expected=lm['expected']),
            causal=dict(done=ca['finished'],expected=ca['expected'])))
        raise SystemExit(0 if r['complete'] and lm['complete'] and ca['complete'] else 2)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch,fnmatch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS',4)))
    torch.use_deterministic_algorithms(True)
    if a.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic();finished=[];failed=[];blocked=[];checked=set();lm=None;lm_site=None
    for i,job in enumerate(plan['jobs']):
        if i%a.nshards!=a.shard or not fnmatch.fnmatchcase(job['id'],a.only):continue
        if a.stage in ('rank','causal') and (job['kind']!='real' or job['arm']['name'] not in plan['config'].get('rank_arms',['baseline','beta_0.3','no_first','canonical_init','clip_restore'])):continue
        if time.monotonic()-started>=a.max_seconds:break
        site=inp['sites'][job['site']]
        try:
            if job['site'] not in checked:
                _,actual=verify_site(site)
                if actual!=inp['hashes'][job['site']]:raise ValueError('Input bytes changed since preparation')
                # Code/plan hash check independent of machine paths.
                pc=dict(plan);pc['config']=dict(plan['config']);pc['config']['sites']=[{'name':s['name']} for s in plan['config']['sites']]
                if seal(out,dict(plan=pc,input_hashes=inp['hashes']))!=pid:raise ValueError('Protocol mismatch')
                checked.add(job['site'])
            ref=loadnp(out/'references'/job['site']/'reference.npz')
            if a.stage=='train':
                from .train import execute
                result=execute(job,site,ref,plan['config'],out,pid,a.device)
            else:
                from .lm import LM,execute
                if a.stage=='causal':
                    from .causal import execute
                if not (out/'fits'/job['id']/'final.pt').exists():blocked.append(job['id']);continue
                if not site.get('lm'):blocked.append(job['id']);continue
                if lm_site!=job['site']:
                    if lm is not None:del lm
                    if torch.cuda.is_available():torch.cuda.empty_cache()
                    lm=LM(site,a.device);lm_site=job['site']
                result=execute(job,site,ref,plan['config'],out,a.device,lm)
            if result['status'].startswith('blocked'):blocked.append(job['id'])
            else:finished.append(job['id'])
            if a.pilot_fits and len(finished)>=a.pilot_fits:break
        except Exception as exc:
            failed.append(job['id']);trace=traceback.format_exc();print(trace,flush=True)
            write(out/'failures'/(job['id']+'.json'),dict(stage=a.stage,error=str(exc),traceback=trace,protocol=pid))
    state=dict(stage=a.stage,shard=a.shard,nshards=a.nshards,finished=finished,failed=failed,blocked=blocked,
        seconds=time.monotonic()-started,pilot=bool(a.pilot_fits),torch=torch.__version__,python=platform.python_version(),
        statement='Worker exit does NOT certify global coverage. Collector is authoritative.')
    write(out/'workers'/f'{a.stage}_{a.shard}.json',state);coverage(plan,out)
    if failed or blocked:raise SystemExit(2)

if __name__=='__main__':main()
