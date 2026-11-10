"""Complete, finite registered design; never infer coverage from running instances."""
import argparse,itertools
from pathlib import Path
from .io import read,write

def real_arms():
    a=[dict(name='baseline')]
    a += [dict(name=f'beta_{b:g}',beta=b) for b in (.03,.1,.3)]
    a += [dict(name='bypass',transform='bypass'),dict(name='no_first',population='after_first'),
        dict(name='native_tokens',population='native'),dict(name='clip_restore',transform='clip_restore'),
        dict(name='shuffle',transform='shuffle'),dict(name='shuffle_position',transform='shuffle_position'),
        dict(name='tail_downweight',weighting='tail_downweight'),dict(name='importance_corrected',sampling='importance'),
        dict(name='tail_first',order='tail_first'),dict(name='tail_last',order='tail_last'),
        dict(name='axis_half',transform='scale',axis_scale=.5),dict(name='axis_double',transform='scale',axis_scale=2.),
        dict(name='loss_pc1_0.1',loss_axis=.1),dict(name='loss_pc1_10',loss_axis=10.),
        dict(name='k_half',k_factor=.5),dict(name='k_double',k_factor=2.),
        dict(name='width_half',width_factor=.5),dict(name='width_double',width_factor=2.),
        dict(name='canonical_init',init='canonical'),dict(name='canonical_pinned',init='pinned'),
        dict(name='tilted_init',init='tilted'),dict(name='bias_fixed',bias='fixed'),
        dict(name='encoder_perp',encoder='perp'),dict(name='signed_codes',mode='signed'),
        dict(name='rank_one_only',transform='rank_one'),
        dict(name='beta_late',beta=.3,beta_schedule='late'),dict(name='beta_removed',beta=.3,beta_schedule='removed')]
    return a

def make_plan(config):
    seeds=config.get('seeds',[42,137,0,1,2]);jobs=[]
    selected=config.get('arms')
    arms=[a for a in real_arms() if selected is None or a['name'] in selected]
    if selected and set(selected)!={a['name'] for a in arms}:raise ValueError('Unknown arm')
    for site in config['sites']:
        for arm,seed in itertools.product(arms,seeds):
            jobs.append(dict(id=f"{site['name']}__{arm['name']}__s{seed}",site=site['name'],kind='real',arm=arm,seed=seed))
    if config.get('synthetic',True):
        synth_arms=[a for a in real_arms() if a['name'] in ('baseline','beta_0.3','clip_restore','shuffle','canonical_init','signed_codes')]
        for w,p,dep,support in itertools.product((.9,.99,.999),(.001,.01,.1,.5),(0.,.8),(4,16)):
            name=f'syn_w{w:g}_p{p:g}_dep{dep:g}_s{support}'
            for arm,seed in itertools.product(synth_arms,seeds):
                jobs.append(dict(id=f"{name}__{arm['name']}__s{seed}",site=name,kind='synthetic',arm=arm,seed=seed,
                    synthetic=dict(w1=w,tail_p=p,dependence=dep,content_support=support)))
    names=[j['id'] for j in jobs]
    if len(names)!=len(set(names)):raise ValueError('Duplicate experiment IDs')
    return dict(config=config,jobs=jobs,expected=len(jobs),real=sum(j['kind']=='real' for j in jobs),
        synthetic=sum(j['kind']=='synthetic' for j in jobs),
        primary_contrasts=['no_first','clip_restore','shuffle','tail_downweight','importance_corrected','canonical_init','beta_0.3'],
        interpretation='One randomized seed is a replicate; seed pairs, atoms, and tokens are not independent training replicates.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=read(a.config);base=Path(a.config).resolve().parent
    for s in cfg['sites']:
        if not Path(s['manifest']).is_absolute():s['manifest']=str((base/s['manifest']).resolve())
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    plan=make_plan(cfg);write(out,plan);print(f"[pc1night-plan] {plan['real']} real + {plan['synthetic']} synthetic = {plan['expected']} fits",flush=True)

if __name__=='__main__':main()
