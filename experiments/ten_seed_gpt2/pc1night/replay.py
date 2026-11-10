"""Frozen-code decoder replay: exact linear identifiability, with held-out checking.

This does not claim reparameterized codes remain representable by the original
TopK encoder. It diagnoses the reconstruction constraint on a frozen code group.
"""
import numpy as np
from .io import ratio,rng

def replay(z_cal,z_test,decoder,cal_energy,max_members=128,residual_test=None,reference_axis=None):
    ids=np.argsort(cal_energy)[::-1][:max_members];A=np.asarray(z_cal[:,ids],float);B=np.asarray(z_test[:,ids],float);D=np.asarray(decoder[ids],float)
    if not len(ids):return dict(n_members=0,curves=[])
    G=A.T@A/max(len(A),1);H=B.T@B/max(len(B),1);ev,V=np.linalg.eigh(G);ev=np.maximum(ev,0)
    top=max(float(ev.max()),1e-30);I=np.eye(len(ids));curves=[]
    calden=float((D*(G@D)).sum());testden=float((D*(H@D)).sum());dn=D/np.maximum(np.linalg.norm(D,axis=1,keepdims=True),1e-30)
    cross=None if residual_test is None else B.T@np.asarray(residual_test,float)/len(B)
    original_mse=None if residual_test is None else float(np.square(residual_test).sum()/len(B))
    v=None if reference_axis is None else np.asarray(reference_axis,float)
    original_perp=None if v is None or residual_test is None else float(np.mean(np.square(residual_test).sum(1)-np.square(np.asarray(residual_test)@v)))
    for threshold in (1e-10,1e-8,1e-6,1e-4,.01):
        keep=ev>top*threshold;P=V[:,keep]@V[:,keep].T
        for seed in (0,1,2):
            R=rng('decoder_replay',seed).normal(size=D.shape);R/=np.maximum(np.linalg.norm(R,axis=1,keepdims=True),1e-30)
            new=P@D+(I-P)@R;delta=new-D;norm=np.linalg.norm(new,axis=1);unit=new/np.maximum(norm[:,None],1e-30)
            # Quadratic forms avoid materializing another N x d prediction cloud.
            calerr=max(0.,float((delta*(G@delta)).sum()));testerr=max(0.,float((delta*(H@delta)).sum()))
            change=None if cross is None else 2*float((delta*cross).sum())+testerr
            perp_change=None
            if change is not None and v is not None:
                dp=delta@v;parallel_change=2*float(dp@(cross@v))+float(dp@H@dp);perp_change=change-parallel_change
            curves.append(dict(relative_eigen_cut=threshold,seed=seed,retained_rank=int(keep.sum()),discarded_dimensions=int((~keep).sum()),
                calibration_relative_aggregate_error=ratio(calerr,calden),heldout_relative_aggregate_error=ratio(testerr,testden),
                perturbation_energy_relative_to_full_sae_error=None if original_mse is None else ratio(testerr,original_mse),
                full_sae_mse_change=None if cross is None else 2*float((delta*cross).sum())+testerr,
                full_sae_relative_mse_change=None if cross is None else ratio(2*float((delta*cross).sum())+testerr,original_mse),
                full_sae_perp_relative_mse_change=None if perp_change is None else ratio(perp_change,original_perp),
                mean_best_signed_cosine_to_original_members=float((unit@dn.T).max(1).mean()),
                original_member_sharing_0_7=float(np.mean((unit@dn.T).max(1)>=.7)),
                raw_decoder_norm_range=[float(norm.min()),float(norm.max())],
                coefficient_l1_after_unit_normalization= float(np.mean(np.sum(abs(A)*norm,axis=1))),
                original_coefficient_l1=float(np.mean(np.sum(abs(A),axis=1)))))
    return dict(n_members=len(ids),relative_member_indices=ids,calibration_rows=len(A),test_rows=len(B),original_full_sae_mse=original_mse,curves=curves,
        statement='Decoder-only linear replay of a calibration-selected group. Unit-normalizing rows requires reciprocal code rescaling; original TopK encoder closure is not asserted.')
