"""Freeze exact IDs from the bundled registry; select without seeing outcomes."""
import argparse, collections, hashlib, json, pathlib, re
import yaml

ROOT=pathlib.Path(__file__).resolve().parent


def build():
    catalog=ROOT/'provenance/pretrained_saes.yaml'
    reg=yaml.safe_load(catalog.read_text())
    jobs=[]
    def add(group, family, model, release, ids, tier='core', prepend_bos=False):
        rs=reg[release]; available={s['id']:s for s in rs['saes']}
        for sid in ids:
            if sid not in available:raise ValueError((release,sid))
            s=available[sid]
            match=re.search(r'(?:layer_|layer|blocks\.|^l)(\d+)',sid)
            if not match:raise ValueError(sid)
            layer=int(match[1])
            key=hashlib.sha256((release+'|'+sid).encode()).hexdigest()[:12]
            jobs.append(dict(job_id=group+'-'+key,group=group,family=family,model=model,
                layer=layer,hook='resid_post',loader='saelens',release=release,sae_id=sid,
                repo_id=rs['repo_id'],repo_path=s['path'],catalog_l0=s.get('l0'),
                tier=tier,prepend_bos=prepend_bos,registry_entry=s,
                conversion_func=rs.get('conversion_func'),
                primary_space=None,primary_population=None,conventions_reviewed=False))
    # Three preselected depths, two sizes/sparsities: depth replication is valuable;
    # dozens of copies at one depth do not establish broad prevalence.
    for w in ['8x','32x']:
        add('llama31_8b','Llama Scope','meta-llama/Llama-3.1-8B',
            'llama_scope_lxr_'+w,[f'l{i}r_{w}' for i in [6,16,27]],prepend_bos=True)
    selection=[]
    for group,release,model,requested in [
        ('gemma2_2b','gemma-scope-2b-pt-res','google/gemma-2-2b',[6,12,22]),
        ('gemma2_9b','gemma-scope-9b-pt-res','google/gemma-2-9b',[9,21,35])]:
        candidates=reg[release]['saes']
        chosen=[]
        for layer in requested:
            for width in ['16k', '131k' if group=='gemma2_9b' else '65k']:
                prefix=f'layer_{layer}/width_{width}/average_l0_'
                opts=[s for s in candidates if s['id'].startswith(prefix)]
                for target in [50,100]:
                    if not opts:
                        selection.append(dict(group=group,layer=layer,width=width,
                                              target_l0=target,status='unavailable'))
                        continue
                    best=min(opts,key=lambda s:(abs(float(s.get('l0',s['id'].split('_')[-1]))-target),s['id']))
                    chosen.append(best['id'])
                    selection.append(dict(group=group,layer=layer,width=width,
                        target_l0=target,selected=best['id'],status='selected'))
        add(group,'Gemma Scope',model,release,sorted(set(chosen)),prepend_bos=True)
    release='gemma-scope-2-1b-pt-res'
    # Official subset offers layers 7,13,17,22; use early/middle/late 7,13,22.
    ids=[f'layer_{l}_width_{w}_l0_{s}' for l in [7,13,22]
         for w in ['16k','65k'] for s in ['small','medium']]
    add('gemma3_1b','Gemma Scope 2','google/gemma-3-1b-pt',release,ids,prepend_bos=True)
    for l0 in [50,100]:
        add('qwen3_8b','Qwen Scope','Qwen/Qwen3-8B-Base',
            f'qwen-scope-3-8b-base-w64k-l{l0}',[f'layer{i}' for i in [7,18,30]])
    for width in ['32k','128k']:
        add('gpt2','OpenAI GPT-2','openai-community/gpt2',
            'gpt2-small-resid-post-v5-'+width,['blocks.8.hook_resid_post'])
    # External to the SAELens registry; use the actual Sparsify release loader.
    jobs.append(dict(job_id='pythia160m-anchor-l6',group='pythia160m',family='EleutherAI Pythia',
        model='EleutherAI/pythia-160m',layer=6,hook='resid_post',loader='sparsify',
        repo_id='EleutherAI/sae-pythia-160m-32k',repo_path='layers.6',sae_id='layers.6',
        tier='core',prepend_bos=False,primary_space=None,primary_population=None,
        conventions_reviewed=False))
    add('qwen25_7b_it','Qwen2.5 filtered BatchTopK','Qwen/Qwen2.5-7B-Instruct',
        'qwen2.5-7b-instruct-andyrdt',
        [f'resid_post_layer_{l}_trainer_1' for l in [7,15,27]],tier='optional')
    seen=set()
    for j in jobs:
        key=(j['repo_id'],j['repo_path'])
        if key in seen:raise ValueError('Duplicate physical checkpoint '+str(key))
        seen.add(key)
    counts=dict(collections.Counter(j['group'] for j in jobs if j['tier']=='core'))
    m=dict(protocol='pc1_released_audit_v1_20260920',registry_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
           selection_rule='fixed depths/widths; nearest released L0 to 50/100 for Gemma Scope; no outcome selection',
           context_length=128,calibration_rows=50000,evaluation_rows=100000,
           primary_corpus='wikitext103',calibration_split='train',evaluation_split='test',
           bootstrap_draws=2000,core_counts=counts,jobs=jobs,selection_log=selection)
    (ROOT/'manifest.json').write_text(json.dumps(m,indent=2))
    print(json.dumps({'core':sum(counts.values()),'by_group':counts,
                      'optional':sum(j['tier']=='optional' for j in jobs)},indent=2))


if __name__=='__main__':build()
