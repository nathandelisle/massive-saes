"""Explicit experimental SAE, separate from the frozen v3 headline protocol.

TopK(ReLU((x-b)W_enc+b_enc)); unit decoder rows, tied initialization only.
Raw inputs are globally rescaled by a TRAIN-only scalar, never per-token.
"""
import torch
from torch import nn
import torch.nn.functional as F

class SAE(nn.Module):
    def __init__(self,d,m,k,v,seed=0,mode='topk',init='random',bias='learned',encoder='full'):
        super().__init__(); self.k=k;self.mode=mode;self.encoder_mode=encoder
        g=torch.Generator(device='cpu').manual_seed(int(seed))
        dec=F.normalize(torch.randn(m,d,generator=g),dim=1)
        self.W_dec=nn.Parameter(dec); self.W_enc=nn.Parameter(dec.T.clone())
        self.b_enc=nn.Parameter(torch.zeros(m));self.b_dec=nn.Parameter(torch.zeros(d),requires_grad=bias=='learned')
        self.register_buffer('v',torch.as_tensor(v,dtype=torch.float32))
        self.register_buffer('pinned',torch.zeros(m,dtype=torch.bool))
        if init in ('canonical','pinned'):
            with torch.no_grad():
                self.W_dec[0]=self.v;self.W_dec[1]=-self.v
                self.W_enc[:,0]=self.v;self.W_enc[:,1]=-self.v
                if init=='pinned':self.pinned[:2]=True
        if init=='tilted':
            with torch.no_grad():
                u=dec[2]-torch.dot(dec[2],self.v)*self.v;u=F.normalize(u,dim=0)
                for j,sg in enumerate((1.,-1.)):
                    for off in (1.,-1.):
                        ix=2*j+int(off<0)
                        self.W_dec[ix]=F.normalize(sg*self.v+off*u,dim=0)
                        self.W_enc[:,ix]=sg*self.v/(2**.5)

    def pre(self,x):
        q=x-self.b_dec
        if self.encoder_mode=='perp':q=q-(q@self.v)[...,None]*self.v
        return q@self.W_enc+self.b_enc
    def encode(self,x):
        p=self.pre(x)
        if self.mode=='signed':
            _,i=p.abs().topk(self.k,dim=-1);return torch.zeros_like(p).scatter(-1,i,p.gather(-1,i))
        p=F.relu(p)
        if self.mode=='relu':return p
        a,i=p.topk(self.k,dim=-1);return torch.zeros_like(p).scatter(-1,i,a)
    def decode(self,z):return z@self.W_dec+self.b_dec
    def forward(self,x):
        z=self.encode(x);return self.decode(z),z
    @torch.no_grad()
    def constrain(self):
        self.W_dec.copy_(F.normalize(self.W_dec,dim=1))
        if self.pinned.any():
            self.W_dec[0]=self.v;self.W_dec[1]=-self.v
            self.W_enc[:,0]=self.v;self.W_enc[:,1]=-self.v;self.b_enc[:2]=0
    # no_grad is required, not cosmetic: W_dec requires grad, so building these products
    # in grad mode attaches an autograd graph to W_dec.grad and retains one m x d fp32
    # tensor per step (9.01 MiB at m=3072, d=768), OOM at step ~2460 of 20000.
    @torch.no_grad()
    def tangent_grad(self):
        if self.W_dec.grad is not None:
            self.W_dec.grad-= (self.W_dec.grad*self.W_dec).sum(1,keepdim=True)*self.W_dec
            self.W_dec.grad[self.pinned]=0
        if self.pinned.any():
            self.W_enc.grad[:,self.pinned]=0;self.b_enc.grad[self.pinned]=0

def loss_parts(model,x,weights,beta,loss_axis=1.):
    hat,z=model(x);err=hat-x;p=err@model.v
    mse=err.square().sum(-1)+(loss_axis-1)*p.square()
    l1=z.abs().sum(-1)
    return (weights*(mse+beta*l1)).mean(),hat,z,mse,l1

def make(cfg,v):
    return SAE(cfg['d'],cfg['m'],cfg['k'],v,cfg['seed'],cfg.get('mode','topk'),
               cfg.get('init','random'),cfg.get('bias','learned'),cfg.get('encoder','full'))
