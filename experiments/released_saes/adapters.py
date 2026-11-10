"""Use release converters; capture actual normalization functions, not guessed math."""
import contextlib,hashlib,json,pathlib,sys
import numpy as np


@contextlib.contextmanager
def pinned_downloads(repo_id, revision, records):
    import huggingface_hub
    original=huggingface_hub.hf_hub_download
    def download(*args,**kw):
        repo=kw.get('repo_id',args[0] if args else None)
        if repo==repo_id:
            prior=kw.get('revision')
            if prior not in [None,'main',revision]:raise RuntimeError('Conflicting checkpoint revision')
            kw['revision']=revision
        path=original(*args,**kw)
        if repo==repo_id:
            item={'filename':kw.get('filename',args[1] if len(args)>1 else None),
                  'subfolder':kw.get('subfolder'),'revision':revision}
            records.append(item)
        return path
    patched=[]
    for module in list(sys.modules.values()):
        if module is None:continue
        name=getattr(module,'__name__','')
        if not (name.startswith('sae_lens') or name=='huggingface_hub'):continue
        if getattr(module,'hf_hub_download',None) is original:
            patched.append((module,original));setattr(module,'hf_hub_download',download)
    try:yield
    finally:
        for module,fn in patched:setattr(module,'hf_hub_download',fn)


class Adapter:
    def __init__(self,job,device):
        import torch
        import sae_lens
        from sae_lens import SAE
        from sae_lens.loading.pretrained_sae_loaders import NAMED_PRETRAINED_SAE_LOADERS
        self.torch=torch;self.device=device;self.downloads=[]
        if not job.get('repo_revision'):raise RuntimeError('Checkpoint revision must be pinned')
        with pinned_downloads(job['repo_id'],job['repo_revision'],self.downloads):
            if job['loader']=='sparsify':
                # Fetch cfg.json at the same revision as the weights.
                import huggingface_hub
                cfg_file=huggingface_hub.hf_hub_download(job['repo_id'],filename=f"{job['sae_id']}/cfg.json",
                                                         revision=job['repo_revision'])
                self.downloads.append({'filename':f"{job['sae_id']}/cfg.json",'subfolder':None,
                                       'revision':job['repo_revision'],'note':'prefetched config for sparsify loader'})
                if not pathlib.Path(cfg_file).exists():raise RuntimeError('sparsify config missing after download')
                # This checkpoint stores W_dec as [d_sae, d_in].
                _base=NAMED_PRETRAINED_SAE_LOADERS['sparsify']
                # The native Pythia encoder subtracts b_dec before encoding.
                _CENTER={('EleutherAI/sae-pythia-160m-32k','2046768ae0c8cb69a2e8ed64f2eafb9f8c5fa294')}
                def _oriented(*args,**kw):
                    cfg_dict,state_dict,sparsity=_base(*args,**kw)
                    d_in,d_sae=cfg_dict.get('d_in'),cfg_dict.get('d_sae')
                    W=state_dict.get('W_dec')
                    if W is not None and d_in and d_sae and d_in!=d_sae and tuple(W.shape)==(d_in,d_sae):
                        state_dict['W_dec']=W.T.contiguous()
                        self.wdec_orientation_corrected=True
                    if (job['repo_id'],job['repo_revision']) in _CENTER:
                        self.encoder_centering_before=bool(cfg_dict.get('apply_b_dec_to_input'))
                        cfg_dict['apply_b_dec_to_input']=True
                        self.encoder_centering_corrected=True
                    return cfg_dict,state_dict,sparsity
                self.wdec_orientation_corrected=False
                self.encoder_centering_corrected=False
                self.encoder_centering_before=None
                self.sae,self.cfg,_=SAE.from_pretrained_with_cfg_and_sparsity(
                    release=job['repo_id'],sae_id=job['sae_id'],device=device,dtype='float32',
                    converter=_oriented)
            else:
                self.sae,self.cfg,_=SAE.from_pretrained_with_cfg_and_sparsity(
                    release=job['release'],sae_id=job['sae_id'],device=device,dtype='float32')
        self.sae.eval()
        if getattr(self.sae,'use_error_term',False):raise RuntimeError('Error-term passthrough would invalidate reconstruction')
        self.norm=getattr(self.sae.cfg,'normalize_activations','none')
        if self.norm not in ['none','layer_norm','constant_norm_rescale']:
            raise RuntimeError(f'Unresolved normalization {self.norm}; inspect converter, do not rescale empirically')
        self.version=getattr(sae_lens,'__version__','unknown')
        self.n_features=self.sae.W_dec.shape[0]
        self.batch_in=None;self.batch_out=None
        self.input_fn=self.sae.run_time_activation_norm_fn_in
        self.output_fn=self.sae.run_time_activation_norm_fn_out
        def fin(x):
            value=self.input_fn(x);self.batch_in=value.detach().clone();return value
        def fout(x):
            self.batch_out=x.detach().clone();return self.output_fn(x)
        self.sae.run_time_activation_norm_fn_in=fin
        self.sae.run_time_activation_norm_fn_out=fout

    def transform_input(self,x):
        """Call the exact runtime transform and clear its temporary state."""
        t=self.torch.as_tensor(np.asarray(x).copy(),dtype=self.torch.float32,device=self.device)
        with self.torch.inference_mode():
            y=self.input_fn(t).detach().clone()
            self.output_fn(y)
        return y.cpu().numpy()

    def reconstruct(self,x):
        t=self.torch.as_tensor(np.asarray(x).copy(),dtype=self.torch.float32,device=self.device)
        self.batch_in=None;self.batch_out=None
        with self.torch.inference_mode():
            z=self.sae.encode(t)
            h=self.sae.decode(z)
            dense=z.to_dense() if z.is_sparse else z
            alive=(dense!=0).any(0).cpu().numpy()
            l0=(dense!=0).sum(-1).cpu().numpy()
        if self.batch_in is None or self.batch_out is None:
            raise RuntimeError('Converter bypasses expected normalization hooks; add a reviewed adapter')
        return {'raw':(t.cpu().numpy(),h.cpu().numpy()),
                'sae_input':(self.batch_in.cpu().numpy(),self.batch_out.cpu().numpy()),
                'alive':alive,'l0':l0}

    def smoke(self,x):
        """Check captured encode/decode equals full SAE call on real activations."""
        a=self.reconstruct(x)
        t=self.torch.as_tensor(np.asarray(x).copy(),dtype=self.torch.float32,device=self.device)
        with self.torch.inference_mode():b=self.sae(t).cpu().numpy()
        rel=float(np.linalg.norm(b-a['raw'][1])/max(np.linalg.norm(b),1e-20))
        if not np.isfinite(b).all() or rel>1e-5:raise RuntimeError(f'Forward/reconstruction mismatch {rel}')
        return {'full_forward_relative_error':rel,'n_probe_rows':len(x),
                'normalization':self.norm,'saelens_version':self.version,
                'wdec_orientation_corrected':bool(getattr(self,'wdec_orientation_corrected',False)),
                'encoder_centering_corrected':bool(getattr(self,'encoder_centering_corrected',False)),
                'apply_b_dec_to_input_as_loaded':getattr(self,'encoder_centering_before',None),
                'apply_b_dec_to_input_used':bool(getattr(self.sae.cfg,'apply_b_dec_to_input',False)),
                'note':'Compares encode/decode with the full forward call of the converted SAE.'}


def fit_axis_gpu(x,device,max_iter=150):
    """Float64 power iteration; records convergence instead of assuming dominance."""
    import torch
    with torch.inference_mode():
        a=torch.as_tensor(np.asarray(x).copy(),device=device,dtype=torch.float64)
        mu=a.mean(0);a-=mu
        gen=torch.Generator(device=device);gen.manual_seed(1729)
        v=torch.randn(a.shape[1],device=device,dtype=torch.float64,generator=gen);v/=v.norm()
        for i in range(max_iter):
            cv=a.T@(a@v)/len(a)
            if cv.norm()==0:raise RuntimeError('Zero calibration variance')
            w=cv/cv.norm();change=min(float((w-v).norm()),float((w+v).norm()));v=w
            if change<1e-8:break
        cv=a.T@(a@v)/len(a);lam=v@cv
        residual=float((cv-lam*v).norm()/lam)
        solver='power_iteration'
        if residual>=1e-5:
            cov=a.T@a/len(a)
            eigen,vec=torch.linalg.eigh(cov)
            v=vec[:,-1];lam=eigen[-1]
            residual=float((cov@v-lam*v).norm()/lam)
            solver='covariance_eigh_fallback'
        info={'iterations':i+1,'relative_eigen_residual':residual,
              'w1_calibration':float(lam/a.square().sum(1).mean()),'converged':residual<1e-5,'solver':solver}
        v*=1 if v[v.abs().argmax()]>=0 else -1
        return mu.cpu().numpy(),v.cpu().numpy(),info
