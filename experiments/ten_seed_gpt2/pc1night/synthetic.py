"""Planted sparse content; tail frequency, dependence, and variance controlled separately."""
from pathlib import Path
import numpy as np
from .io import rng,write,npz,atomic
from .data import SPLITS

def generate(root,name,spec,n=32768,d=64):
    root=Path(root)/name;target=root/'site.json'
    if target.exists():return target
    root.mkdir(parents=True,exist_ok=True);r=rng('ground_truth',name)
    Q=np.linalg.qr(r.normal(size=(d,d)))[0];v=Q[:,0];D=Q[:,1:25].T
    site=dict(name=name,min_articles=8,axis_fit_split='train',mean='mean.npy',pc1='pc1.npy',splits={},synthetic=spec)
    for split in SPLITS:
        r=rng('synthetic_data',name,split);nn=n if split=='train' else max(1024,n//4)
        z=np.zeros((nn,len(D)));support=spec['content_support']
        chosen=np.argsort(r.random((nn,len(D))),axis=1)[:,:support]
        np.put_along_axis(z,chosen,r.exponential(1.,chosen.shape),axis=1);res=z@D
        # A Gaussian-copula score changes coupling without changing event count.
        zrank=np.argsort(np.argsort(z[:,0]+r.random(nn)*1e-8))/nn
        from scipy.special import ndtri
        score=spec['dependence']*ndtri(np.clip(zrank,.00001,.99999))+np.sqrt(1-spec['dependence']**2)*r.normal(size=nn)
        event=np.zeros(nn,bool);event[np.argsort(score)[-max(1,int(round(spec['tail_p']*nn))):]]=True
        a=event.astype(float);a-=a.mean()
        # Make the planted axis an actual covariance eigenvector even when event
        # occurrence depends on content. Retain nonlinear association, remove the
        # linear cross-covariance that would otherwise rotate the true PC1.
        zc=z-z.mean(0);a-=zc@np.linalg.lstsq(zc,a,rcond=None)[0];a-=a.mean();a/=a.std()
        content_var=np.var(res,axis=0).sum();a*=np.sqrt(spec['w1']/(1-spec['w1'])*content_var)
        x=res+a[:,None]*v;pos=np.arange(nn)%128
        # Roles intentionally independent of event in the primary synthetic experiment.
        meta=dict(article=np.array([f'{name}_{split}_a{i//128}' for i in range(nn)]),position=pos,
            token_id=r.integers(0,1000,nn),is_bos=pos==0,is_eos=np.zeros(nn,bool))
        atomic(root/(split+'.npy'),lambda f:np.save(f,x.astype(np.float32)))
        npz(root/(split+'_meta.npz'),**meta);npz(root/(split+'_truth.npz'),coefficients=z,event=event,pc1_coefficient=a,decoder=D)
        site['splits'][split]=dict(x=split+'.npy',meta=split+'_meta.npz')
        if split=='train':
            atomic(root/'mean.npy',lambda f:np.save(f,x.mean(0)));atomic(root/'pc1.npy',lambda f:np.save(f,v))
    site['synthetic_convention']='tail_p is the generating event frequency before linear decorrelation; dependence is a copula parameter. Scalar/content cross-covariance is removed before setting w1, making the planted axis a true empirical PC1.'
    write(target,site);return target
