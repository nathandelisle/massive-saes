"""Train and evaluate PC1 controls on exported activation caches.

This is a reference cached-activation harness, not a transformer hook library.
It reports RECONSTRUCTION ablation, never calls it causal importance, and does
not claim interpretability from decoder stability. Read PROTOCOL.txt first.
"""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rotation_test import digest, max_cosine


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def blocks(widths):
    bounds = np.cumsum([0]+list(widths))
    return [slice(int(a),int(b)) for a,b in zip(bounds[:-1],bounds[1:])]


def check_cache(root, task):
    root = Path(root)
    meta = json.loads((root/"metadata.json").read_text())
    if meta["task"] != task: raise ValueError("Task/cache mismatch")
    if not meta.get("cache_id") or not meta.get("source"):
        raise ValueError("metadata needs a unique cache_id and source/provenance")
    if meta.get("coordinates") != "raw_uncentered": raise ValueError("Expected raw, uncentered activations")
    data, docsets = {}, []
    for split in ("train", "validation", "test"):
        y = np.load(root/split/"y.npy", mmap_mode="r", allow_pickle=False)
        x = y if task == "crosscoder" else np.load(root/split/"x.npy", mmap_mode="r", allow_pickle=False)
        doc = np.load(root/split/"doc_id.npy", allow_pickle=False)
        if y.ndim != 2 or x.ndim != 2 or len(x)!=len(y) or doc.shape!=(len(y),) or not len(y):
            raise ValueError("Invalid cache shapes")
        if y.shape[1] != sum(meta["output_widths"]): raise ValueError("Bad output widths")
        data[split] = (x,y,doc); docsets.append(set(doc.tolist()))
    if any(docsets[i]&docsets[j] for i in range(3) for j in range(i)):
        raise ValueError("Article IDs overlap between train/validation/test")
    return meta,data


def eig_pc(a):
    a = np.asarray(a, np.float64)
    cov = a.T@a / len(a)
    ev, u = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    return u[:,order[0]], np.maximum(ev[order],0)


def fit_preprocessing(x, y, widths, scale_mode, fit_tokens, seed):
    """Fit everything on a declared random TRAIN sample, never on eval splits."""
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(y), min(fit_tokens,len(y)), replace=False)
    xf, yf = np.asarray(x[ix],np.float64), np.asarray(y[ix],np.float64)
    if not np.isfinite(xf).all() or not np.isfinite(yf).all(): raise ValueError("Nonfinite training activations")
    xm, ym = xf.mean(0), yf.mean(0)
    xc, yc = xf-xm, yf-ym
    xs = max(float(np.sqrt(np.mean(np.square(xc).sum(1)))),1e-12)
    ranges = blocks(widths)
    if scale_mode == "per_stream":
        scales = [max(float(np.sqrt(np.mean(np.square(yc[:,s]).sum(1)))),1e-12) for s in ranges]
    elif scale_mode == "global":
        global_scale = max(float(np.sqrt(np.mean(np.square(yc).sum(1)))),1e-12)
        scales = [global_scale]*len(widths)
    else: raise ValueError("scale_mode must be global or per_stream")
    ys = np.concatenate([np.repeat(s,w) for s,w in zip(scales,widths)])
    yn = yc/ys
    pcs, spectrum = [], []
    for s in ranges:
        v, ev = eig_pc(yn[:,s]); b=np.zeros(y.shape[1]); b[s]=v
        pcs.append(b); spectrum.append(ev)
    pcblock = np.column_stack(pcs)
    joint, joint_ev = eig_pc(yn)
    alphas = yn@pcblock
    # Quantiles are for a declared amplitude-tail diagnostic, not a definition
    # of massive activations. Use the paper's token criterion in the LM adapter.
    tail_threshold = float(np.quantile(np.linalg.norm(alphas,axis=1),.99))
    return {"xmean":xm,"xscale":np.array(xs),"ymean":ym,"yscale":ys,
            "block_pc":pcblock,"joint_pc":joint[:,None],
            "pc1_shares":np.array([e[0]/e.sum() if e.sum()>0 else 0 for e in spectrum]),
            "joint_spectrum":joint_ev,"coefficient_covariance":alphas.T@alphas/len(alphas),
            "tail_threshold":np.array(tail_threshold),"fit_indices":ix}


def random_blocks(widths, rng):
    out=[]
    for s in blocks(widths):
        v=np.zeros(sum(widths)); q=rng.normal(size=s.stop-s.start)
        v[s]=q/np.linalg.norm(q); out.append(v)
    return np.column_stack(out)


def fit_ridge(x, y, prep, projection, ridge):
    ix=prep["fit_indices"]
    xn=(np.asarray(x[ix],float)-prep["xmean"])/prep["xscale"]
    yn=(np.asarray(y[ix],float)-prep["ymean"])/prep["yscale"]
    a=np.column_stack([xn,np.ones(len(xn))]); target=yn@projection
    penalty=np.eye(a.shape[1])*ridge; penalty[-1,-1]=0
    # Ridge parameter is in average-loss units; only intercept is unpenalized.
    w=np.linalg.solve(a.T@a/len(a)+penalty, a.T@target/len(a))
    return w


class Coder(nn.Module):
    def __init__(self, config, prep, input_dim):
        super().__init__(); self.config=config
        self.task=config["task"]; self.variant=config["variant"]
        self.widths=config["output_widths"]; self.ranges=blocks(self.widths)
        output_dim=sum(self.widths); f=config["features"]
        for name in ("xmean","xscale","ymean","yscale","block_pc","joint_pc","tail_threshold"):
            self.register_buffer(name,torch.tensor(prep[name],dtype=torch.float32))
        rng=np.random.default_rng(config.get("direction_seed",101))
        project=np.zeros((output_dim,0))
        if self.task=="transcoder":
            if self.variant in ("rank1","oracle","sae_bypass"): project=prep["block_pc"]
            elif self.variant=="random1": project=random_blocks(self.widths,rng)
            elif self.variant not in ("plain","skip","sae"): raise ValueError("Unknown transcoder control")
        else:
            if self.variant=="block_pc": project=prep["block_pc"]
            elif self.variant=="joint_pc": project=prep["joint_pc"]
            elif self.variant=="random_block": project=random_blocks(self.widths,rng)
            elif self.variant!="plain": raise ValueError("Unknown crosscoder variant")
        self.register_buffer("projection",torch.tensor(project,dtype=torch.float32))
        self.sae = self.task=="crosscoder" or self.variant in ("sae","sae_bypass")
        dim=output_dim if self.sae else input_dim
        self.encoder=nn.Linear(dim,f)
        self.decoder=nn.Parameter(torch.randn(f,output_dim))
        self.bias=nn.Parameter(torch.zeros(output_dim))
        nn.init.zeros_(self.encoder.bias)
        self.skip=nn.Linear(input_dim,output_dim) if self.variant=="skip" else None
        if self.skip is not None: nn.init.zeros_(self.skip.weight); nn.init.zeros_(self.skip.bias)
        rank=project.shape[1]
        self.register_buffer("ridge_weight",torch.zeros(input_dim+1,rank))
        self.project_decoder()

    def q(self,a):
        return a-(a@self.projection)@self.projection.T

    @torch.no_grad()
    def project_decoder(self):
        self.decoder.copy_(self.q(self.decoder))
        n=self.decoder.norm(dim=1,keepdim=True)
        if torch.any(n<1e-12): raise ValueError("Degenerate projected decoder")
        self.decoder.div_(n)
        self.bias.copy_(self.q(self.bias))

    def forward(self,x,y=None):
        xn=(x-self.xmean)/self.xscale
        yn=None if y is None else (y-self.ymean)/self.yscale
        if self.sae:
            if yn is None: raise ValueError("Auto/crosscoder needs its reconstruction input")
            encoder_in=self.q(yn)
        else: encoder_in=xn
        pre=self.encoder(encoder_in)
        z=pre.relu()
        if self.config["activation"]=="topk":
            values,indices=z.topk(self.config["k"],dim=1)
            z=torch.zeros_like(z).scatter(1,indices,values)
        elif self.config["activation"]!="relu_l1": raise ValueError("Unknown activation")
        sparse=z@self.decoder+self.q(self.bias)
        dense=torch.zeros_like(sparse)
        if self.sae or self.variant=="oracle":
            if yn is None: raise ValueError("Oracle requires target and is NOT a predictive transcoder")
            dense=(yn@self.projection)@self.projection.T
        elif self.variant in ("rank1","random1"):
            aug=torch.cat([xn,torch.ones(len(xn),1,device=xn.device)],dim=1)
            dense=(aug@self.ridge_weight)@self.projection.T
        elif self.skip is not None: dense=self.skip(xn)
        return sparse+dense,z,dense

    def penalty(self,z):
        if self.config["activation"]!="relu_l1": return z.sum()*0
        if self.config["penalty"]=="block_sum":
            norms=sum(self.decoder[:,s].norm(dim=1) for s in self.ranges)
        elif self.config["penalty"]=="joint": norms=self.decoder.norm(dim=1)
        else: raise ValueError("Unknown L1 penalty")
        return self.config["l1"]*(z*norms).sum(1).mean()


def tensor_batch(array, index, device):
    return torch.as_tensor(np.array(array[index],dtype=np.float32,copy=True),device=device)


@torch.no_grad()
def evaluate(model, data, batch, device, details=False):
    model.eval(); x,y,docs=data; f=model.config["features"]; d=sum(model.widths)
    sums=np.zeros(2+len(model.widths)*2)
    norm_sums=np.zeros((2,d)); z2=np.zeros(f); hz=np.zeros((f,d)); alive=np.zeros(f,bool)
    freq=np.zeros(f); l0=pen=0.; p=model.block_pc
    perdoc={}; denseerr=densevar=0.; tail_budget=tail_count=0.
    # Geometry measured in the explicit objective coordinates; native block metrics below.
    D=model.decoder.detach().double().cpu().numpy()
    alignment=np.square(D@model.block_pc.double().cpu().numpy()).sum(1)/np.square(D).sum(1)
    # Freeze this geometric rule; thresholds are not selected to maximize a result.
    # For full reporting, CC should also re-evaluate at the paper's canonical alive-set median.
    for i in range(0,len(x),batch):
        sl=slice(i,i+batch); tx=tensor_batch(x,sl,device); ty=tensor_batch(y,sl,device)
        pred,z,dense=model(tx,ty)
        yn=(ty-model.ymean)/model.yscale; en=pred-yn
        e=en*model.yscale; yc=ty-model.ymean
        eq=e-(e@p)@p.T
        e2=e.square().sum(1); eq2=eq.square().sum(1)
        rows=torch.stack([e2,eq2],1).double().cpu().numpy()
        for doc,row in zip(docs[sl].tolist(),rows):
            if doc not in perdoc: perdoc[doc]=[0,0.,0.]
            perdoc[doc][0]+=1; perdoc[doc][1]+=float(row[0]); perdoc[doc][2]+=float(row[1])
        sums[:2]+=rows.sum(0)
        for j,s in enumerate(model.ranges):
            sums[2+2*j:4+2*j]+=[e[:,s].double().square().sum().item(),eq[:,s].double().square().sum().item()]
        yy=ty.double().cpu().numpy(); norm_sums[0]+=yy.sum(0); norm_sums[1]+=(yy*yy).sum(0)
        zz=z.double().cpu().numpy(); ee=en.double().cpu().numpy()
        z2+=(zz*zz).sum(0); hz+=zz.T@ee; freq+=(zz>0).sum(0); alive|=(zz>0).any(0)
        l0+=(zz>0).sum(); pen+=model.penalty(z).item()*len(z)
        alpha=yn@p; predicted_alpha=dense@p
        denseerr+=(predicted_alpha-alpha).double().square().sum().item()
        densevar+=alpha.double().square().sum().item()
    # True test variance about TEST mean, with PC directions fitted on TRAIN.
    mean=norm_sums[0]/len(y)
    var_total=norm_sums[1].sum()-len(y)*np.square(mean).sum()
    var_pc=0.
    variance_block=np.zeros((len(model.widths),2))
    pcs=model.block_pc.detach().double().cpu().numpy()
    for i in range(0,len(y),batch):
        yy=np.asarray(y[i:i+batch],float)-mean
        q=yy-(yy@pcs)@pcs.T
        var_pc+=np.square(yy@pcs).sum()
        for j,s in enumerate(model.ranges): variance_block[j]+=[np.square(yy[:,s]).sum(),np.square(q[:,s]).sum()]
    var_content=max(0.,var_total-var_pc)
    # Ablating z_j subtracts z_j*d_j: MINUS 2 e*d_j*z_j, plus its energy.
    score=(-2*np.sum(D*hz,1)+np.square(D).sum(1)*z2)/len(y)
    median=float(np.median(alignment[alive])) if alive.any() else 0
    cluster=(alignment>5*median)&alive
    for i in range(0,len(x),batch):
        sl=slice(i,i+batch); ty=tensor_batch(y,sl,device)
        _,z,_=model(tensor_batch(x,sl,device),ty)
        yn=(ty-model.ymean)/model.yscale
        mask=(yn@p).norm(dim=1)>model.tail_threshold
        if mask.any():
            zz=z[mask]; numer=(zz[:,torch.as_tensor(cluster,device=device)]>0).sum(1)
            den=(zz>0).sum(1).clamp_min(1)
            tail_budget+=(numer/den).sum().item(); tail_count+=len(zz)
    def ratio(a,b): return float(a/b) if b>1e-20 else None
    metrics={"n_tokens":len(y),"n_documents":len(perdoc),"total_mse_native":float(sums[0]/len(y)),
             "content_mse_native":float(sums[1]/len(y)),"total_fvu_native":ratio(sums[0],var_total),
             "content_fvu_native":ratio(sums[1],var_content),"mean_l0":float(l0/len(y)),
             "alive_features":int(alive.sum()),"mean_penalty":float(pen/len(y)),
             "dense_branch_pc_mse_over_train_centered_energy":ratio(denseerr,densevar),
             "pc_tail_cluster_active_fraction":ratio(tail_budget,tail_count),
             "pc_tail_token_count":int(tail_count),"block_metrics":[]}
    for j in range(len(model.widths)):
        metrics["block_metrics"].append({"block":j,"fvu":ratio(sums[2+2*j],variance_block[j,0]),
                         "content_fvu":ratio(sums[3+2*j],variance_block[j,1])})
    native_d=D*model.yscale.cpu().numpy()
    native_alignment=np.square(native_d@pcs).sum(1)/np.square(native_d).sum(1)
    features={"decoder_objective":D,"decoder_native":native_d,
              "frequency":freq/len(y),"activation_second_moment":z2/len(y),
              "reconstruction_zero_ablation_objective":score,"pc_subspace_alignment":alignment,
              "pc_subspace_alignment_native":native_alignment,
              "pc_cluster":cluster,"alive":alive,"output_widths":np.array(model.widths),
              "block_decoder_norms_objective":np.stack([np.linalg.norm(D[:,s],axis=1) for s in model.ranges],1),
              "block_decoder_norms_native":np.stack([np.linalg.norm((D*model.yscale.cpu().numpy())[:,s],axis=1) for s in model.ranges],1)}
    return metrics,features,perdoc


@torch.no_grad()
def validation_objective(model,data,batch,device,max_tokens):
    # Fixed validation subset; no test-set model selection.
    x,y,_=data; n=min(max_tokens,len(x)); total=0.
    for i in range(0,n,batch):
        sl=slice(i,min(i+batch,n)); tx=tensor_batch(x,sl,device); ty=tensor_batch(y,sl,device)
        pred,z,_=model(tx,ty); target=(ty-model.ymean)/model.yscale
        obj=(pred-target).square().sum(1).mean()+model.penalty(z)
        total+=obj.item()*len(tx)
    return total/n


def train(args):
    config=json.loads(args.config.read_text()); seed_all(config["seed"])
    meta,data=check_cache(args.cache,config["task"])
    config["output_widths"]=meta["output_widths"]
    if config["task"]=="transcoder" and len(config["output_widths"])!=1:
        raise ValueError("This transcoder harness expects one MLP-output target")
    if config["features"]<config["k"] or config["k"]<1: raise ValueError("Bad TopK")
    args.out.mkdir(parents=True,exist_ok=False)
    x,y,_=data["train"]
    prep=fit_preprocessing(x,y,config["output_widths"],config["scale_mode"],config["fit_tokens"],config.get("data_seed",101))
    np.savez_compressed(args.out/"preprocessing.npz",**prep)
    device=torch.device(args.device); model=Coder(config,prep,x.shape[1]).to(device)
    if config["variant"] in ("rank1","random1"):
        if config["ridge"]<=0: raise ValueError("Use a declared positive ridge penalty")
        w=fit_ridge(x,y,prep,model.projection.cpu().numpy(),config["ridge"])
        model.ridge_weight.copy_(torch.tensor(w,dtype=torch.float32,device=device))
    optimizer=torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    rng=np.random.default_rng(config.get("data_seed",101)); history=[]
    best=float("inf"); start=time.monotonic()
    for step in range(1,config["steps"]+1):
        model.train(); ix=rng.integers(0,len(x),config["batch_size"])
        tx=tensor_batch(x,ix,device); ty=tensor_batch(y,ix,device)
        if not torch.isfinite(tx).all() or not torch.isfinite(ty).all(): raise ValueError("Nonfinite cache batch")
        pred,z,_=model(tx,ty); target=(ty-model.ymean)/model.yscale
        mse=(pred-target).square().sum(1).mean(); loss=mse+model.penalty(z)
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite loss")
        optimizer.zero_grad(); loss.backward(); optimizer.step(); model.project_decoder()
        if step%config["evaluate_every"]==0 or step==config["steps"]:
            val=validation_objective(model,data["validation"],config["eval_batch_size"],device,config["validation_tokens"])
            row={"step":step,"train_mse":mse.item(),"validation_objective":val,
                 "elapsed_seconds":time.monotonic()-start,"mean_l0_batch":float((z>0).sum(1).float().mean())}
            history.append(row); print(json.dumps(row),flush=True)
            if val<best:
                best=val
                torch.save({"state":model.state_dict(),"config":config,"selected_step":step},args.out/"best.pt")
    torch.save({"state":model.state_dict(),"config":config,"selected_step":config["steps"]},args.out/"last.pt")
    (args.out/"history.json").write_text(json.dumps(history,indent=2))
    (args.out/"config.json").write_text(json.dumps(config,indent=2))
    (args.out/"cache_metadata.json").write_text(json.dumps(meta,indent=2))
    (args.out/"run_status.txt").write_text("Training finished. Test set NOT evaluated. Lock config and checkpoints before running evaluate.\n")


def evaluate_cli(args):
    checkpoint=torch.load(args.run/args.checkpoint,map_location="cpu",weights_only=True)
    config=checkpoint["config"]; meta,data=check_cache(args.cache,config["task"])
    if meta != json.loads((args.run/"cache_metadata.json").read_text()):
        raise ValueError("Evaluation cache metadata differs from training cache")
    prep=dict(np.load(args.run/"preprocessing.npz",allow_pickle=False))
    device=torch.device(args.device); model=Coder(config,prep,data["train"][0].shape[1]).to(device)
    model.load_state_dict(checkpoint["state"])
    metrics,features,perdoc=evaluate(model,data[args.split],config["eval_batch_size"],device,True)
    metrics.update(checkpoint_sha256=digest(args.run/args.checkpoint),selected_step=checkpoint["selected_step"],
                   config=config,split=args.split,
                   caution="Feature ablation is reconstruction MSE, not downstream causal loss.")
    stem=args.run/f"{args.split}_{Path(args.checkpoint).stem}"
    Path(str(stem)+"_metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False))
    np.savez_compressed(str(stem)+"_features.npz",**features)
    Path(str(stem)+"_documents.json").write_text(json.dumps(perdoc,indent=2))
    print(json.dumps(metrics,indent=2,allow_nan=False))


def compare(args):
    a=dict(np.load(args.a,allow_pickle=False)); b=dict(np.load(args.b,allow_pickle=False))
    aa=a["alive"].astype(bool); bb=b["alive"].astype(bool)
    if not aa.any() or not bb.any(): raise ValueError("No alive features")
    out={}
    for space in ("objective","native"):
        qa=a[f"decoder_{space}"][aa]; qb=b[f"decoder_{space}"][bb]
        c=max_cosine(qa,qb,absolute=False); ca=max_cosine(qa,qb,absolute=True)
        ia=np.flatnonzero(aa)
        top=np.argsort(a["reconstruction_zero_ablation_objective"][ia])[-min(50,len(ia)):]
        out[space]={"dictionary_shared_signed":float((c>.7).mean()),
                    "dictionary_shared_absolute":float((ca>.7).mean()),
                    "top50_reconstruction_ablation_shared_signed":float((c[top]>.7).mean()),
                    "mean_max_cosine":float(c.mean())}
    out["warning"]="Not a causal importance inversion test. Compare both directions and all seed pairs."
    args.out.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))


def diagnose(args):
    """Training geometry and validation predictability; never inspect test loss."""
    meta,data=check_cache(args.cache,args.task)
    x,y,_=data["train"]; xv,yv,_=data["validation"]
    prep=fit_preprocessing(x,y,meta["output_widths"],args.scale_mode,args.fit_tokens,101)
    args.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.out/"preprocessing.npz",**prep)
    ev=prep["joint_spectrum"]
    report={"scope":"training geometry; validation prediction; no test outcomes",
            "metadata":meta,"pc1_shares_by_stream":prep["pc1_shares"].tolist(),
            "joint_variance_fractions_top10":(ev[:10]/ev.sum()).tolist() if ev.sum()>0 else [],
            "pc_coefficient_second_moment":prep["coefficient_covariance"].tolist()}
    if args.task=="transcoder":
        if len(meta["output_widths"])!=1: raise ValueError("Expected one MLP output")
        candidates=[]
        # Validation-only model selection; final held-out R2 belongs in the locked run.
        for ridge in (1e-6,1e-4,1e-2):
            w=fit_ridge(x,y,prep,prep["block_pc"],ridge)
            err=sum_y=sum_y2=n=0.
            for i in range(0,len(yv),1024):
                tx=(np.asarray(xv[i:i+1024],float)-prep["xmean"])/prep["xscale"]
                target=((np.asarray(yv[i:i+1024],float)-prep["ymean"])/prep["yscale"])@prep["block_pc"]
                pred=np.column_stack([tx,np.ones(len(tx))])@w
                err+=np.square(pred-target).sum(); sum_y+=target.sum(); sum_y2+=np.square(target).sum(); n+=len(tx)
            variance=sum_y2-sum_y*sum_y/n
            candidates.append({"ridge":ridge,"validation_mse":float(err/n),
                               "validation_r2":float(1-err/variance) if variance>1e-20 else None})
        report["ridge_candidates"]=candidates
        report["selected_ridge"]=min(candidates,key=lambda r:r["validation_mse"])["ridge"]
    (args.out/"diagnostics.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    t=sub.add_parser("train"); t.add_argument("--cache",type=Path,required=True); t.add_argument("--config",type=Path,required=True)
    t.add_argument("--out",type=Path,required=True); t.add_argument("--device",default="cpu"); t.set_defaults(func=train)
    e=sub.add_parser("evaluate"); e.add_argument("--cache",type=Path,required=True); e.add_argument("--run",type=Path,required=True)
    e.add_argument("--checkpoint",default="best.pt",choices=["best.pt","last.pt"])
    e.add_argument("--split",choices=["validation","test"],required=True); e.add_argument("--device",default="cpu"); e.set_defaults(func=evaluate_cli)
    c=sub.add_parser("compare"); c.add_argument("--a",type=Path,required=True); c.add_argument("--b",type=Path,required=True)
    c.add_argument("--out",type=Path,required=True); c.set_defaults(func=compare)
    d=sub.add_parser("diagnose"); d.add_argument("--cache",type=Path,required=True); d.add_argument("--out",type=Path,required=True)
    d.add_argument("--task",choices=["transcoder","crosscoder"],required=True); d.add_argument("--fit-tokens",type=int,default=50000)
    d.add_argument("--scale-mode",choices=["global","per_stream"],default="global"); d.set_defaults(func=diagnose)
    args=p.parse_args(); args.func(args)


if __name__=="__main__": main()
