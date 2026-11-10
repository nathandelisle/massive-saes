"""Linux worker for an existing GPU instance. Does not provision or delete instances."""
import argparse,json,os,pathlib,signal,subprocess,sys,time


def main():
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--manifest',default='manifest.json')
    p.add_argument('--cache',default='cache');p.add_argument('--results',default='results');p.add_argument('--corpus',default='corpus')
    p.add_argument('--device',default='cuda:0');p.add_argument('--hours',type=float,default=18)
    p.add_argument('--deadline-unix',type=float);p.add_argument('--context',type=int)
    p.add_argument('--s3-output',help='Optional destination for periodic results-only sync; caches remain local')
    p.add_argument('--smoke',action='store_true',help='Run one checkpoint after harvesting; does not launch full group')
    p.add_argument('--position-filter',type=int,default=0)
    a=p.parse_args();deadline=a.deadline_unix or time.time()+a.hours*3600
    if a.s3_output and not a.s3_output.startswith('s3://'):raise ValueError('Expected s3:// destination')
    logroot=pathlib.Path(a.results)/'logs';logroot.mkdir(parents=True,exist_ok=True)
    def sync():
        if a.s3_output:
            try:subprocess.run(['aws','s3','sync',a.results,a.s3_output,'--only-show-errors'],check=True,timeout=120)
            except Exception as e:print('Sync failed; local results preserved:',type(e).__name__,flush=True)
    common=['--manifest',a.manifest,'--group',a.group,'--cache',a.cache,'--device',a.device]
    if a.context:common+=['--context',str(a.context)]
    commands=[['harvest.py',*common,'--corpus',a.corpus],
              ['audit.py',*common,'--out',a.results,'--deadline-unix',str(deadline),
               *(['--position-filter',str(a.position_filter)] if a.position_filter else [])]]
    if a.smoke:commands[-1]+=['--limit-jobs','1']
    for cmd in commands:
        if time.time()>=deadline:raise SystemExit('Deadline reached')
        with (logroot/f'{a.group}-{cmd[0]}.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-u',*cmd],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            last=0
            try:
                while process.poll() is None:
                    if time.time()>deadline:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                        raise TimeoutError('Worker deadline reached')
                    if time.time()-last>=60:sync();last=time.time()
                    time.sleep(2)
                if process.returncode:raise RuntimeError(f'{cmd[0]} exited {process.returncode}; inspect worker log')
            finally:sync()
    print('Worker finished; collect.py reports individual checkpoint errors/missing cells',flush=True)


if __name__=='__main__':main()
