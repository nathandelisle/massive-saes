"""Record what an intervention actually changed, before interpreting its outcomes."""
import numpy as np
from .data import Store,TrainingView
from .diagnostics import concentration
from .io import rng,ratio

def audit(site,ref,arm):
    st=Store(site['splits']['train']);view=TrainingView(st,ref,arm);ix=view.eligible;a=ref['coef'][ix].copy();new=a.copy()
    t=arm.get('transform','identity')
    if t=='clip_restore':new=(np.clip(a,-ref['clip_q'],ref['clip_q'])-ref['clip_mean'])*ref['coef_std']/max(ref['clip_std'],1e-30)+ref['coef_mean']
    if t in ('shuffle','shuffle_position'):new=ref['coef'][view.donor[ix]]
    if t=='scale':new=a*arm['axis_scale']
    if t=='bypass':new=np.zeros_like(a)
    changed=np.mean(abs(new-a)>1e-6*max(np.std(a),1e-30))
    w=ref['weights'][ix] if arm.get('weighting')=='tail_downweight' else np.ones(len(ix))
    sample=rng('manipulation_sample',site['name']).choice(ix,min(4096,len(ix)),False);b,ww=view.batch(sample)
    v=ref['pc1'];ac=b@v;res=b-ac[:,None]*v;center=ac-ac.mean();rc=res-res.mean(0)
    cross=np.mean(center[:,None]*rc,axis=0)
    xc=b-b.mean(0);pv=v.copy()
    for _ in range(20):
        pv=xc.T@(xc@pv);pv/=max(np.linalg.norm(pv),1e-30)
    pvar=np.var(xc@pv);fullvar=np.var(xc,axis=0).sum()
    return dict(n_train=len(ix),source_n=len(st.x),first_fraction=float(st.first[ix].mean()),actual_bos_fraction=float(st.meta['is_bos'][ix].mean()),
        original_scalar=concentration(a),transformed_scalar=concentration(new),fraction_coefficients_changed=float(changed),
        realized_variance_ratio=ratio(np.var(new),np.var(a)),
        weighted_pc1_second_moment=ratio((w*new*new).sum(),w.sum()),weight_effective_rows=ratio(w.sum()**2,(w*w).sum()),
        empirical_dependence_crosscov_norm=float(np.linalg.norm(cross)),
        sample_refitted_pc1_cosine=float(abs(pv@v)),sample_refitted_pc1_variance_fraction=ratio(pvar,fullvar),
        empirical_refaxis_variance_fraction=ratio(np.var(ac),np.var(b,axis=0).sum()),
        dependence_sample_n=len(sample),
        interpretation='A null outcome is uninformative about a manipulation that barely changed the relevant statistic; inspect this record first.')
