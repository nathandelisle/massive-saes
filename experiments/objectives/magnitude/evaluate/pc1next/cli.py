"""Entry points for additive PC1 experiments. No implicit training or cloud launch."""
import argparse
import fnmatch
import importlib.metadata
import os
from pathlib import Path
import time
import traceback
import numpy as np
from .io import atomic_json,digest,load_json,seal
from .data import check_splits


def validate(cfg):
    required=['model_path','block_path','block_index','mean_path','pc1_path','tokens','runs']
    for k in required:
        if k not in cfg:raise ValueError('Missing config field: '+k)
    names=[r['name'] for r in cfg['runs']]
    if len(names)!=len(set(names)):raise ValueError('Duplicate run names')
    if len({str(Path(r['checkpoint']).resolve()) for r in cfg['runs']})!=len(cfg['runs']):raise ValueError('Multiple cells alias the same checkpoint path')
    keys=[(r['condition'],float(r['beta']),r['seed']) for r in cfg['runs']]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate condition/beta/seed cells')
    conditions=sorted({k[0] for k in keys});betas=sorted({k[1] for k in keys});seeds=sorted({k[2] for k in keys})
    expected={(c,b,s) for c in conditions for b in betas for s in seeds}
    if set(keys)!=expected:raise ValueError('Incomplete registered grid: explicitly include missing checkpoint cells, do not silently drop them')
    for r in cfg['runs']:
        if any(x in r['name'] for x in ('/','\\','..')):raise ValueError('Unsafe run name')
        if not Path(r['checkpoint']).is_file():raise FileNotFoundError('Missing registered checkpoint: '+r['checkpoint'])
    for k in ('sequence_batch','feature_batch'):
        if cfg.get(k,2)<1:raise ValueError(k+' must be positive')
    if not cfg.get('toy_model'):
        revision=cfg.get('revision','')
        if not Path(cfg['model_path']).is_dir() and (len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision.lower())):
            raise ValueError('Remote model revision must be a pinned 40-character commit')
    if not set(cfg.get('baselines',['model','splice']))<= {'model','splice'}:raise ValueError('Unknown ablation baseline')
    return cfg


def prepare(cfg,out):
    validate(cfg);data,audit=check_splits(cfg)
    cfg=dict(cfg);cfg['checkpoint_hashes']={r['name']:digest(r['checkpoint']) for r in cfg['runs']}
    if Path(cfg['model_path']).is_dir():
        cfg['local_model_files']={str(p.relative_to(cfg['model_path'])):digest(p)
            for p in sorted(Path(cfg['model_path']).rglob('*')) if p.is_file() and p.suffix in ('.safetensors','.bin','.json')}
    cfg['runtime_versions']={k:importlib.metadata.version(k).split('+')[0] for k in ('torch','numpy','transformers')}
    pid=seal(out,cfg);atomic_json(Path(out)/'DATA_AUDIT.json',audit)
    return cfg,data,pid


def preflight(be,sae,run,data):
    import torch
    ids,mask=be.tensors(data['calibration'],slice(0,min(2,len(data['calibration']['ids']))))
    logits,x=be.forward(ids,mask);roundtrip,_=be.forward(ids,mask,x)
    if not torch.equal(logits,roundtrip):raise AssertionError('Replacing block output by itself changed logits')
    z,hat=be.encode(sae,x,run)
    j=int(torch.argmax(z.double().square().sum((0,1))))
    zz=z.clone();zz[...,j]=0
    predicted=sae.decode(z)-z[...,j,None]*sae.W_dec[j]
    actual=sae.decode(zz)
    if not torch.allclose(predicted,actual,rtol=1e-4,atol=1e-4):raise AssertionError('Decoder is not the assumed affine decoder')
    score,_=be.forward(ids,mask,hat-z[...,j,None]*sae.W_dec[j])
    return dict(identity_max_logit_difference=float((logits-roundtrip).abs().max()),
        decoder_edit_max_error=float((predicted-actual).abs().max()),
        hook_dtype=str(x.dtype),max_abs_hook=float(x.abs().max()),top_energy_single_ablation_finite=bool(torch.isfinite(score).all()),
        scope='Implementation preflight, not a scientific result or exhaustive numerical safety guarantee')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['preflight','cache','rank','collect','geometry','interventions','surgery','interchange'])
    p.add_argument('--config',required=True);p.add_argument('--out',default=os.environ.get('PC1_NEXT_OUT'))
    p.add_argument('--shard',type=int,default=int(os.environ.get('SHARD','0')))
    p.add_argument('--nshards',type=int,default=int(os.environ.get('NSHARDS','1')))
    p.add_argument('--only',default='*',help='Run-name glob; excluded cells remain registered in coverage')
    p.add_argument('--pilot-units',type=int,help='Rank timing only; stops after this many new tiles in one cell, never reports full completion')
    args=p.parse_args();out=Path(args.out or 'results/pc1_next');out.mkdir(parents=True,exist_ok=True)
    if not 0<=args.shard<args.nshards:p.error('Invalid shard')
    cfg,data,pid=prepare(load_json(args.config),out)
    if args.stage=='collect':
        from .rank import collect
        from .analysis import analyze
        coverage=[]
        for run in cfg['runs']:
            cp=out/'calibration'/(run['name']+'.npz')
            if not cp.exists():coverage.append(dict(run=run['name'],complete=False,missing=['calibration']));continue
            with np.load(cp) as n:cal={k:n[k] for k in n.files}
            coverage.append(collect(cfg,run,data,cal,out,pid))
        report=analyze(cfg,out,pid);complete=all(r['complete'] for r in coverage) and report['complete']
        atomic_json(out/'CAMPAIGN_STATUS.json',dict(stage='bridge',complete=complete,coverage=coverage,expected_runs=len(cfg['runs'])))
        raise SystemExit(0 if complete else 2)
    if args.stage=='geometry':
        from .geometry import execute
        report=execute(cfg,out,data,pid);raise SystemExit(0 if report['complete'] else 2)
    from .backend import Backend
    from .cache import calibrate,make_cache,fidelity
    be=Backend(cfg);failed=[];finished=[];started=time.monotonic()
    for i,run in enumerate(cfg['runs']):
        if i%args.nshards!=args.shard or not fnmatch.fnmatchcase(run['name'],args.only):continue
        if args.stage in ('interventions','surgery','interchange') and float(run['beta']) not in cfg.get('mechanism_betas',[0.,.3]):continue
        try:
            sae=be.load_sae(run)
            if args.stage=='preflight':
                result=preflight(be,sae,run,data)
                atomic_json(out/'preflight'/(run['name']+'.json'),result)
            else:
                cal=calibrate(be,sae,run,data['calibration'],out,pid,cfg.get('sequence_batch',2),cfg)
                if args.stage=='cache':
                    for split in ('selection','test'):
                        make_cache(be,sae,run,split,data[split],out,pid,cfg.get('sequence_batch',2))
                        fidelity(out,run,split,data[split],pid,cfg.get('sequence_batch',2))
                    result=dict(complete=True)
                elif args.stage=='rank':
                    from .rank import execute
                    result=execute(be,sae,cfg,run,data,cal,out,pid,limit=args.pilot_units)
                elif args.stage=='interventions':
                    from .interventions import execute
                    result=execute(be,sae,cfg,run,data,cal,out,pid)
                elif args.stage=='surgery':
                    from .reparameterize import execute
                    result=execute(be,sae,cfg,run,data,cal,out,pid)
                else:
                    from .interchange import execute
                    result=execute(be,cfg,run,data,out,pid)
            status='pilot' if args.pilot_units else ('complete' if result.get('complete',True) else 'incomplete')
            atomic_json(out/'states'/args.stage/(run['name']+'.json'),dict(status=status,result=result))
            if status=='incomplete':failed.append(run['name'])
            else:finished.append(run['name'])
            print(f'[pc1-{args.stage}] {run["name"]} status={status}',flush=True)
            del sae
            if args.pilot_units:break
        except Exception as exc:
            trace=traceback.format_exc();print(trace,flush=True);failed.append(run['name'])
            atomic_json(out/'failures'/args.stage/(run['name']+f'_{time.time_ns()}.json'),dict(run=run,error=str(exc),traceback=trace))
            atomic_json(out/'states'/args.stage/(run['name']+'.json'),dict(status='failed',error=str(exc)))
    summary=dict(stage=args.stage,shard=args.shard,nshards=args.nshards,finished=finished,failed=failed,
        pilot=args.pilot_units is not None,seconds=time.monotonic()-started,
        scope='Worker completion only; collect must verify the full registered grid')
    atomic_json(out/'workers'/f'{args.stage}_{args.shard}.json',summary)
    if failed:raise SystemExit(2)


if __name__=='__main__':main()
