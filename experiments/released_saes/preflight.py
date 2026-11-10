"""Static preflight works without GPUs; --runtime checks the pinned GPU environment."""
import argparse,hashlib,importlib.metadata,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',action='store_true');a=p.parse_args()
    m=json.loads((ROOT/'manifest.json').read_text());pins=json.loads((ROOT/'repository_pins.json').read_text())
    registry=ROOT/'provenance/pretrained_saes.yaml'
    assert hashlib.sha256(registry.read_bytes()).hexdigest()==m['registry_sha256']
    ids=set();physical=set()
    for j in m['jobs']:
        assert j['job_id'] not in ids;ids.add(j['job_id'])
        key=(j['repo_id'],j['repo_revision'],j['repo_path']);assert key not in physical;physical.add(key)
        assert j['checkpoint_path_verified'] and j['resolved_files']
        assert pins[j['repo_id']]['sha']==j['repo_revision']
        assert pins[j['model']]['sha']==j['model_revision']
    result={'static':'pass','checkpoint_count':len(ids),'core_count':sum(j['tier']=='core' for j in m['jobs'])}
    if a.runtime:
        import torch,sae_lens
        from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
        directory=get_pretrained_saes_directory()
        for j in m['jobs']:
            if j['loader']=='saelens':
                info=directory[j['release']]
                assert info.repo_id==j['repo_id'] and info.saes_map[j['sae_id']]==j['repo_path']
        assert torch.cuda.is_available(),'No CUDA GPU'
        result.update(runtime='pass',gpu=torch.cuda.get_device_name(0),
                      versions={k:importlib.metadata.version(k) for k in ['torch','transformers','sae-lens','datasets','huggingface-hub']})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
