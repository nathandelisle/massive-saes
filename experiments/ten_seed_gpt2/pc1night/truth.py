from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .data import evaluate_input
from .diagnostics import sample_rows,infer
from .io import loadnp

def evaluate_truth(model,cal,test,ref,arm,site,n):
    ci=sample_rows(cal,n);ti=sample_rows(test,n)
    cq,_=evaluate_input(np.asarray(cal.x[ci],float),ref,arm);tq,_=evaluate_input(np.asarray(test.x[ti],float),ref,arm)
    _,cz,_=infer(model,cq);_,tz,_=infer(model,tq)
    root=Path(cal.spec['x']).parent;c=loadnp(root/'calibration_truth.npz');t=loadnp(root/'test_truth.npz')
    ym=c['coefficients'][ci].mean(0);xm=cz.mean(0);xs=cz.std(0);alive=xs>1e-6
    cx=(cz[:,alive]-xm[alive])/xs[alive];tx=(tz[:,alive]-xm[alive])/xs[alive]
    # Fixed alpha, no held-out tuning. Centered regularized predictor.
    alpha=1.;W=np.linalg.solve(cx.T@cx+alpha*np.eye(cx.shape[1]),cx.T@(c['coefficients'][ci]-ym))
    y=t['coefficients'][ti];pred=tx@W+ym
    r2=1-np.square(y-pred).sum(0)/np.maximum(np.square(y-y.mean(0)).sum(0),1e-30)
    D=model.W_dec.detach().cpu().numpy();D=D/np.linalg.norm(D,axis=1,keepdims=True);truth=c['decoder']
    cos=truth@D.T;ii,jj=linear_sum_assignment(-cos)
    return dict(mean_heldout_coefficient_r2=float(r2.mean()),per_feature_r2=r2,
        signed_truth_max_cosine=cos.max(1),one_to_one_truth_cosine=cos[ii,jj],ridge_alpha=alpha,
        n_cal=len(ci),n_test=len(ti),interpretation='Linear recoverability from the full code; not proof individual latents equal planted features')
