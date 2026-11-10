"""Build a verified checkpoint inventory from the paired grid or existing site runs."""
import argparse
from pathlib import Path
from .io import atomic_json,digest,load_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site',required=True);p.add_argument('--grid-root',help='Contains training/, evaluation/cells/, training_config.json')
    p.add_argument('--checkpoint-root',help='Override checkpoint directory for existing-site mode')
    p.add_argument('--v3-path',required=True);p.add_argument('--mean',required=True);p.add_argument('--pc1',required=True)
    p.add_argument('--calibration',required=True);p.add_argument('--selection',required=True);p.add_argument('--test',required=True)
    p.add_argument('--dtype',default='bfloat16',choices=['float32','float16','bfloat16']);p.add_argument('--device',default='cuda')
    p.add_argument('--legacy-reason');p.add_argument('--out',required=True)
    args=p.parse_args();site=load_json(args.site)
    cfg={k:site[k] for k in ('model_path','revision','block_path','block_index','attn_implementation','model_key') if k in site}
    cfg.update(local_files_only=False,v3_path=str(Path(args.v3_path).resolve()),mean_path=str(Path(args.mean).resolve()),pc1_path=str(Path(args.pc1).resolve()),
        tokens={s:str(Path(getattr(args,s)).resolve()) for s in ('calibration','selection','test')},dtype=args.dtype,device=args.device,
        sequence_batch=2,feature_batch=8,baselines=['model','splice'],primary_baseline='model',
        min_firing_rate=0.,group_multiplier=5.,use_frozen_group=True,top_ns=[10,50,100,200],cosine_thresholds=[.5,.7,.9],
        primary_top_n=50,primary_threshold=.7,bootstrap_draws=1000,null_draws=100,resample_draws=3,
        intervention_budgets=[12,50],doses=[.01,.03,.1,.3,1.],mechanism_betas=[0.,.3],surgery_pairs=12,
        split_policy='legacy' if args.legacy_reason else 'article',legacy_reason=args.legacy_reason,runs=[])
    if args.grid_root:
        root=Path(args.grid_root).resolve();train=load_json(root/'training_config.json')
        cfg['paired_training_record']=train;cfg['paired_training_record_sha256']=digest(root/'training_config.json')
        if len(train['k_values'])!=1:raise ValueError('Build a separate config per sparsity k')
        k=train['k_values'][0]
        for cond in train['conditions']:
            for seed in train['seeds']:
                for beta in train['betas']:
                    name=f'{cond}_k{k}_s{seed}_beta{beta:g}'
                    checkpoint=root/'training'/name/'sae.pt';evaluation=root/'evaluation/cells'/(name+'.json')
                    if not checkpoint.is_file() or not evaluation.is_file():raise FileNotFoundError('Missing registered cell: '+name)
                    ev=load_json(evaluation)
                    if digest(checkpoint)!=ev['checkpoint_sha256']:raise ValueError('Checkpoint hash differs from grid: '+name)
                    cfg['runs'].append(dict(name=name,condition=cond,seed=seed,beta=beta,checkpoint=str(checkpoint),
                        checkpoint_sha256=ev['checkpoint_sha256'],frozen_ids_path=str(evaluation),
                        historical_calibration_tokens=ev.get('n_calibration'),historical_test_tokens=ev.get('n_test')))
        cfg['cohort_statement']='Paired trained checkpoints and their original calibration-alive masks retained. Supplied token-cache hashes define the new evaluation cohort; counts alone do not establish old-cohort identity.'
    else:
        for old in site['runs']:
            cp=Path(args.checkpoint_root)/Path(old['checkpoint']).name if args.checkpoint_root else Path(old['checkpoint'])
            if not cp.is_file():raise FileNotFoundError(cp)
            cfg['runs'].append(dict(name=old['name'],seed=old['seed'],condition=old['condition'],beta=0.,checkpoint=str(cp.resolve()),checkpoint_sha256=digest(cp)))
        cfg['cohort_statement']='Existing standard/bypass seed replication. No beta effect can be estimated from beta-zero cells.'
    target=Path(args.out)
    if target.exists():raise FileExistsError('Configuration already exists; write a new version')
    atomic_json(target,cfg);print(f'[pc1-config] registered={len(cfg["runs"])} output={target}',flush=True)


if __name__=='__main__':main()
