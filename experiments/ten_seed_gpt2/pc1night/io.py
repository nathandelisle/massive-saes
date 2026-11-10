import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def key(x): return hashlib.sha256(json.dumps(clean(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def atomic(p,writer):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=p.parent,prefix=p.name+'.',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as f: writer(f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def write(p,x): atomic(p,lambda f:f.write((json.dumps(clean(x),indent=2,allow_nan=False)+'\n').encode()))
def npz(p,**x):atomic(p,lambda f:np.savez_compressed(f,**x))
def save_torch(p,x):
    import torch
    atomic(p,lambda f:torch.save(x,f))
def rng(*s):return np.random.default_rng(int(key(s)[:16],16))
def seal(out,manifest):
    code={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    m=dict(manifest=manifest,code=code); pid=key(m);p=Path(out)/'PROTOCOL.json'
    if p.exists() and read(p)['id']!=pid:raise ValueError('Protocol changed. Use a NEW output prefix.')
    if not p.exists():write(p,dict(id=pid,**m))
    return pid
def ratio(a,b):return float(a/b) if b>1e-25 else None
def loadnp(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
