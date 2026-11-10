"""Download pinned SAE configuration files."""
import concurrent.futures,json,os,pathlib,urllib.request,urllib.parse
ROOT=pathlib.Path(__file__).resolve().parent


def main():
    m=json.loads((ROOT/'manifest.json').read_text());out=ROOT/'provenance/release_configs';out.mkdir(exist_ok=True)
    def run(j):
        files=[f for f in j['resolved_files'] if pathlib.Path(f).name in ['cfg.json','config.json','hyperparams.json']]
        result={'job_id':j['job_id'],'repo':j['repo_id'],'revision':j['repo_revision'],'configs':[]}
        for f in files:
            try:
                headers={'User-Agent':'PC1-prevalence-audit'}
                if os.environ.get('HF_TOKEN'):headers['Authorization']='Bearer '+os.environ['HF_TOKEN']
                url=f"https://huggingface.co/{j['repo_id']}/resolve/{j['repo_revision']}/"+urllib.parse.quote(f,safe='/')
                data=json.loads(urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=20).read())
                result['configs'].append({'path':f,'data':data})
            except Exception as e:result['configs'].append({'path':f,'error':type(e).__name__+': '+str(e)})
        (out/(j['job_id']+'.json')).write_text(json.dumps(result,indent=2));return len(result['configs'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:counts=list(pool.map(run,m['jobs']))
    print('Inspected',len(counts),'checkpoints;',sum(counts),'config file requests; no weights downloaded')


if __name__=='__main__':main()
