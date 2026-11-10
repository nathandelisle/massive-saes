"""All seed pairs within each objective cell, using existing cosine primitives.
Reports alive-candidate and all-candidate matching separately; projected matching
excludes nearly-zero projections. No feature-by-feature independence assumption.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from common import signed_matches, sha256, start_run, finish_run, write_json


def direction_metrics(d, db, alive, balive, group, v, block=512):
    co_all, _ = signed_matches(d,db,block)
    co = np.full(len(d),np.nan)
    if len(balive): co,_ = signed_matches(d,db[balive],block)
    dp = d-np.outer(d@v,v); bp = db-np.outer(db@v,v)
    valid = np.linalg.norm(dp,axis=1)>1e-3*np.maximum(np.linalg.norm(d,axis=1),1e-12)
    bvalid = np.linalg.norm(bp,axis=1)>1e-3*np.maximum(np.linalg.norm(db,axis=1),1e-12)
    projected = np.full(len(d),np.nan)
    candidates = balive[bvalid[balive]]
    if len(candidates): projected,_ = signed_matches(dp,bp[candidates],block)
    def rate(values, ids, threshold):
        ids=np.asarray(ids,int); good=ids[np.isfinite(values[ids])]
        return float((values[good]>threshold).mean()) if len(good) else None
    return dict(raw_sharing=rate(co,alive,.7),raw_sharing_all_candidates=rate(co_all,alive,.7),
        group_raw_sharing=rate(co,group,.7),projected_sharing=rate(projected,alive[valid[alive]],.7),
        group_projected_sharing=rate(projected,group[valid[group]],.7),n_group_atoms=len(group),
        threshold_sweep={str(t):dict(dictionary=rate(co,alive,t),group=rate(co,group,t)) for t in [.5,.7,.9]})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grid',required=True)
    p.add_argument('--evaluation',required=True);p.add_argument('--out',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--block',type=int,default=512)
    args=p.parse_args();grid=Path(args.grid);evaluation=Path(args.evaluation)
    data=json.loads((evaluation/'summary.json').read_text())
    if not data['complete']:raise ValueError('Finish evaluating all registered cells first')
    cfg=json.loads((grid/'manifest.json').read_text())['config']
    config={k:v for k,v in vars(args).items() if k!='resume'}
    inputs=[evaluation/'summary.json',cfg['pc1_path']]
    if args.resume:
        out=Path(args.out);meta=json.loads((out/'manifest.json').read_text())
        if meta['config']!=config or any(sha256(i['path'])!=i['sha256'] for i in meta['inputs']):
            raise ValueError('Seed-comparison inputs changed')
        for dependency in ['compare_grid_seeds.py','common.py']:
            if meta['code_sha256'].get(dependency)!=sha256(Path(__file__).parent/dependency):raise ValueError('Comparison code changed; use a new output version')
    else:out,meta=start_run(args.out,config,inputs)
    v=np.load(cfg['pc1_path']).astype(float);v/=np.linalg.norm(v);buckets={};pairs=[]
    for r in data['runs']:buckets.setdefault((r['condition'],r['k'],r['beta']),[]).append(r)
    import torch
    for key,runs in buckets.items():
        for a,b in itertools.combinations(runs,2):
            unit=out/'pairs'/(a['name']+'__'+b['name']+'.json')
            if unit.exists():pairs.extend(json.loads(unit.read_text()));continue
            arrays=[]
            for r in [a,b]:
                path=grid/r['name']/'sae.pt'
                if sha256(path)!=r['checkpoint_sha256']:raise ValueError('Checkpoint changed')
                state=torch.load(path,map_location='cpu',weights_only=True)['state']
                arrays.append(state['W_dec'].numpy().astype(float))
            records=[]
            for source,target,d,db in [(a,b,*arrays),(b,a,*arrays[::-1])]:
                result=direction_metrics(d,db,np.array(source['calibration_alive_ids'],int),
                    np.array(target['calibration_alive_ids'],int),np.array(source['structural_group_ids'],int),v,args.block)
                records.append(dict(reference=source['name'],comparison=target['name'],**result))
            write_json(unit,records);pairs.extend(records)
        print(f'[pc1-E4-pairs] {key} seeds={len(runs)}',flush=True)
    write_json(out/'summary.json',dict(runs=data['runs'],pairs=pairs,
        completed_cells=data['completed_cells'],expected_cells=data['expected_cells'],complete=True,
        matching='Signed cosine, calibration-alive candidates primary; all-candidate sensitivity retained'))
    finish_run(out,meta)


if __name__=='__main__':main()
