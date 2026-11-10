"""Raw activation stores and train-fitted interventions. No inferred BOS masks."""
import os
from pathlib import Path
import numpy as np
from .io import read,write,npz,rng,sha,ratio

SPLITS=('train','calibration','selection','test')

class Store:
    def __init__(self,spec):
        self.spec=spec;self.x=np.load(spec['x'],mmap_mode='r')
        with np.load(spec['meta'],allow_pickle=False) as z:self.meta={k:z[k] for k in z.files}
        needed=('article','position','is_bos','is_eos','token_id')
        for k in needed:
            if k not in self.meta:raise ValueError('Missing explicit row metadata '+k)
            if self.meta[k].shape!=(len(self.x),):raise ValueError('Row metadata mismatch '+k)
        for k in ('is_bos','is_eos'):
            if self.meta[k].dtype.kind not in 'biu' or not np.isin(self.meta[k],[0,1]).all():raise ValueError('Expected explicit boolean '+k)
        for k in ('position','token_id'):
            if self.meta[k].dtype.kind not in 'iu' or (self.meta[k]<0).any():raise ValueError('Expected nonnegative integer '+k)
        if np.any(self.meta['article'].astype(str)==''):raise ValueError('Empty article identity')
        if self.x.ndim!=2 or len(self.x)<2:raise ValueError('Expected raw x[N,D], no padding rows')
        self.meta['is_bos']=self.meta['is_bos'].astype(bool);self.meta['is_eos']=self.meta['is_eos'].astype(bool)
        self.first=self.meta['position']==0
        self.native=~(self.meta['is_bos']|self.meta['is_eos'])
        self.ordinary=self.native&~self.first
    def masks(self):
        return {'all':np.ones(len(self.x),bool),'first':self.first,'after_first':~self.first,
                'native':self.native,'ordinary':self.ordinary,'bos':self.meta['is_bos']}

def resolve_site(path):
    p=Path(path).resolve();s=read(p)
    for sp in SPLITS:
        for field in ('x','meta','tokens'):
            if field in s['splits'][sp]:
                f=Path(s['splits'][sp][field]);s['splits'][sp][field]=str(f if f.is_absolute() else p.parent/f)
    for f in ('mean','pc1'):
        if s.get(f):s[f]=str((p.parent/s[f]).resolve()) if not Path(s[f]).is_absolute() else s[f]
    return s

def verify_site(site):
    stores={sp:Store(site['splits'][sp]) for sp in SPLITS};hashes={};arts={}
    for sp,st in stores.items():
        arts[sp]=set(st.meta['article'].astype(str));hashes[sp]={k:sha(st.spec[k]) for k in ('x','meta')}
        if 'tokens' in st.spec:hashes[sp]['tokens']=sha(st.spec['tokens'])
        if len(arts[sp])<site.get('min_articles',30):raise ValueError(f'{sp}: too few articles ({len(arts[sp])}); do not pretend rows are articles')
        if st.x.shape[1]!=stores['train'].x.shape[1]:raise ValueError('Dimension mismatch')
    for i,a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            if arts[a]&arts[b]:raise ValueError('Article leakage '+a+'/'+b)
    for f in ('mean','pc1'):
        if site.get(f):hashes[f]=sha(site[f])
    return stores,hashes

def prepare(site,out):
    """Fixed reference axis; every transform fits training data only."""
    out=Path(out);st=Store(site['splits']['train']);d=st.x.shape[1]
    # Full training mean, streamed in double precision.
    n=len(st.x);mu=np.zeros(d);second=0.
    for start in range(0,n,8192):
        b=np.asarray(st.x[start:start+8192],dtype=np.float64)
        if not np.isfinite(b).all():raise FloatingPointError('Nonfinite raw activations')
        mu+=b.sum(0);second+=np.square(b).sum()
    mu/=n
    if site.get('mean'):mu=np.load(site['mean']).astype(np.float64)
    if site.get('pc1'):
        if site.get('axis_fit_split')!='train':raise ValueError('External axis requires axis_fit_split=train provenance')
        v=np.load(site['pc1']).astype(np.float64);fit='external_train_axis'
    else:
        # Deterministic sample with matrix-free covariance power iteration.
        ix=np.sort(rng('pca',site['name']).choice(n,min(n,site.get('pca_rows',100000)),False))
        b=np.asarray(st.x[ix],dtype=np.float64);b-=b.mean(0)
        v=rng('pc1-init',site['name']).normal(size=d);v/=np.linalg.norm(v)
        for _ in range(100):
            new=b.T@(b@v);new/=np.linalg.norm(new)
            if abs(new@v)>1-1e-12:v=new;break
            v=new
        cv=b.T@(b@v)/len(b);eig=float(v@cv)
        fit=dict(rows=len(b),relative_eigen_residual=float(np.linalg.norm(cv-eig*v)/max(eig,1e-30)))
        del b
    if mu.shape!=(d,) or v.shape!=(d,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-20:raise ValueError('Bad axis')
    v/=np.linalg.norm(v)
    # Stable sign convention used only for presentation.
    if v[np.argmax(abs(v))]<0:v=-v
    coef=np.empty(n);total=0.;perp=0.
    for start in range(0,n,8192):
        b=np.asarray(st.x[start:start+8192],dtype=np.float64)-mu;a=b@v
        coef[start:start+len(b)]=a;total+=np.square(b).sum();perp+=np.square(b-a[:,None]*v).sum()
    scale=float(np.sqrt(total/n));q=float(np.quantile(abs(coef),.99))
    clipped=np.clip(coef,-q,q);cm=float(clipped.mean());cs=float(clipped.std());ast=float(coef.std())
    # Global scalar only. Train objective uses average ||error||^2 in these units.
    weights=np.minimum(1.,q*q/np.maximum(coef*coef,1e-30))
    weights/=weights.mean();sample_p=np.abs(coef)/(np.mean(abs(coef))+1e-30)+1
    sample_p/=sample_p.sum()
    npz(out/'reference.npz',mean=mu,pc1=v,scale=scale,coef=coef,weights=weights,
        sample_p=sample_p,clip_q=q,clip_mean=cm,clip_std=cs,coef_mean=coef.mean(),coef_std=ast)
    write(out/'reference.json',dict(axis_fit=fit,n_train=n,scale=scale,w1=1-perp/total,
        first_energy_share=ratio(np.square(coef[st.first]).sum(),np.square(coef).sum()),
        first_frequency=st.first.mean(),note='w1 relative to fixed train mean; centered evaluation variance reported separately'))
    return dict(mean=mu,pc1=v,scale=scale,coef=coef,weights=weights,sample_p=sample_p,
                clip_q=q,clip_mean=cm,clip_std=cs,coef_mean=coef.mean(),coef_std=ast)

class TrainingView:
    def __init__(self,store,ref,arm,order_seed=20260908):
        self.st=store;self.ref=ref;self.arm=arm;self.seed=order_seed
        mask=np.ones(len(store.x),bool)
        if arm.get('population')=='after_first':mask=~store.first
        if arm.get('population')=='native':mask=store.native
        if not mask.any():raise ValueError('Empty train population')
        self.eligible=np.flatnonzero(mask);self.coef=np.asarray(ref['coef'])
        self.donor=None
        if arm.get('transform') in ('shuffle','shuffle_position'):
            self.donor=np.arange(len(store.x))
            groups=[self.eligible] if arm['transform']=='shuffle' else [self.eligible[store.meta['position'][self.eligible]==p] for p in np.unique(store.meta['position'][self.eligible])]
            for g in groups:self.donor[g]=rng('donor',order_seed,int(g[0])).permutation(g)
        self._block=None;self._indices=None;self._rows=None;self._cache=None
    def indices(self,step,batch,steps):
        """32-batch superblocks have identical multisets even under curricula."""
        block=step//32
        if block!=self._block:
            count=min(32,steps-block*32)*batch;r=rng('stream',self.seed,block)
            if self.arm.get('sampling')=='importance':
                p=self.ref['sample_p'][self.eligible];p=p/p.sum()
                ix=r.choice(self.eligible,count,True,p=p)
            else:ix=r.choice(self.eligible,count,True)
            order=self.arm.get('order','random')
            if order in ('tail_first','tail_last'):
                ix=ix[np.argsort(np.abs(self.coef[ix]),kind='stable')]
                if order=='tail_first':ix=ix[::-1]
            self._block=block;self._indices=ix
            self._rows=None;self._cache=None
            if os.environ.get('PC1NIGHT_BLOCK_CACHE')=='1':
                # The superblock's whole index multiset is already known here, so the
                # 32*batch rows can be read ONCE in sorted order instead of as 32
                # separate fancy-index reads of `batch` random rows each. Rows are
                # 3-14 KB and the train stores are 14-67 GB against 16 GB of RAM, so
                # the per-step read is real scattered disk I/O, not page cache.
                # Bit-identical by construction: the same rows are returned in the
                # same positions, so `batch` sees byte-identical input either way.
                self._rows=np.unique(ix)
                self._cache=np.asarray(self.st.x[self._rows])
        start=(step%32)*batch;return self._indices[start:start+batch]
    def rows(self,ix):
        """Store rows for ix, from the superblock cache when it covers them."""
        if self._cache is not None:
            at=np.searchsorted(self._rows,ix)
            if at.max(initial=-1)<len(self._rows) and np.array_equal(self._rows[at],ix):
                return self._cache[at]
        return self.st.x[ix]
    def batch(self,ix):
        b=(np.asarray(self.rows(ix),dtype=np.float32)-self.ref['mean']).astype(np.float32)
        a=self.coef[ix];t=self.arm.get('transform','identity');new=a.copy()
        if t=='clip_restore':
            new=(np.clip(a,-self.ref['clip_q'],self.ref['clip_q'])-self.ref['clip_mean'])*self.ref['coef_std']/max(self.ref['clip_std'],1e-30)+self.ref['coef_mean']
        elif t in ('shuffle','shuffle_position'):new=self.coef[self.donor[ix]]
        elif t=='scale':new=a*self.arm['axis_scale']
        elif t=='rank_one':b=a[:,None]*self.ref['pc1']
        elif t not in ('identity','bypass'):raise ValueError('Unknown transform '+t)
        b=b+(new-a)[:,None]*self.ref['pc1']
        if t=='bypass':b=b-(b@self.ref['pc1'])[:,None]*self.ref['pc1']
        weights=np.ones(len(ix))
        if self.arm.get('weighting')=='tail_downweight':weights=self.ref['weights'][ix]
        if self.arm.get('sampling')=='importance':
            pp=self.ref['sample_p'];weights=pp[self.eligible].sum()/(len(self.eligible)*pp[ix])
        return (b/self.ref['scale']).astype(np.float32),weights.astype(np.float32)

def evaluate_input(x,ref,arm):
    q=(x-ref['mean'])/ref['scale'];a=q@ref['pc1']
    if arm.get('transform')=='bypass':q=q-a[...,None]*ref['pc1']
    return q.astype(np.float32),a
