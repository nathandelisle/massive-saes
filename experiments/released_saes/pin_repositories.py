"""Resolve HF commit IDs only; downloads no weights. Credentials stay out of logs."""
import argparse,concurrent.futures,json,os,pathlib,urllib.request,urllib.parse


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default='manifest.json');p.add_argument('--refresh',action='store_true');a=p.parse_args()
    path=pathlib.Path(a.manifest);m=json.loads(path.read_text())
    repos=sorted({j[k] for j in m['jobs'] for k in ['model','repo_id']})
    oldpath=path.parent/'repository_pins.json'
    old=json.loads(oldpath.read_text()) if oldpath.exists() and not a.refresh else {}
    def fetch(repo):
        if old.get(repo,{}).get('status')=='ok':return repo,old[repo]
        try:
            headers={'User-Agent':'PC1-prevalence-audit'}
            if os.environ.get('HF_TOKEN'):headers['Authorization']='Bearer '+os.environ['HF_TOKEN']
            req=urllib.request.Request('https://huggingface.co/api/models/'+urllib.parse.quote(repo,safe='/'),headers=headers)
            info=json.loads(urllib.request.urlopen(req,timeout=20).read())
            return repo,dict(sha=info['sha'],canonical_id=info.get('id'),
                            siblings=[x['rfilename'] for x in info.get('siblings',[])],status='ok')
        except Exception as e:return repo,dict(status='error',error=type(e).__name__+': '+str(e))
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:pins=dict(pool.map(fetch,repos))
    for j in m['jobs']:
        j['repo_revision']=pins[j['repo_id']].get('sha')
        j['model_revision']=pins[j['model']].get('sha')
        pin=pins[j['repo_id']]
        stem=j['repo_path']
        j['resolved_files']=[x for x in pin.get('siblings',[]) if x==stem or x.startswith(stem.rstrip('/')+'/')]
        j['checkpoint_path_verified']=bool(j['resolved_files'])
    (path.parent/'repository_pins.json').write_text(json.dumps(pins,indent=2))
    path.write_text(json.dumps(m,indent=2))
    print(json.dumps({k:{'status':v['status'],'sha':v.get('sha')} for k,v in pins.items()},indent=2))
    print('Resolved checkpoint paths:',sum(j['checkpoint_path_verified'] for j in m['jobs']),'/',len(m['jobs']))


if __name__=='__main__':main()
