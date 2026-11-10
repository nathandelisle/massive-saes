"""Recompute the reported tile counts and sharing from the saved alignment ranges and feature arrays."""
import json, sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / "results" / "examples" / "sec4_tiles"
E1 = json.load(open(ROOT / "cliff.json"))
ORDER=["pythia-70m-deduped","pythia-160m","gpt2","pythia-1.4b","gemma-2-2b",
       "qwen-2.5-7b","pythia-6.9b","llama-3.1-8b","gemma-2-9b"]
THR=0.7
rows={}
print(f"{'model':<20} {'band':>16} {'n':>5} {'%alive':>7} {'%pos imp':>9} {'shareM':>7} {'shareP':>7} {'dictM':>6} {'enrich(imp/size)':>17}")
tb=ta=0; pis=[]
for m in ORDER:
    c=E1["cells"][m+"_standard"]; lo,hi=c["cliff_cos2"],c["band_hi_cos2"]
    z=np.load(ROOT / "a5_features_standard" / f"features_{m}_standard.npz"); al=z["alive"].astype(bool); c2=z["pc1_cos2"]; imp=z["imp_zero"]
    mc=z["max_cos"]; mcm=mc.mean(0); pf=(mc>=THR).mean(0)
    band=al&(c2>=lo)&(c2<hi); fin=np.isfinite(imp); pos=al&fin&(imp>0)
    den=float(imp[pos].sum()); sh=float(imp[band&pos].sum()/den)
    base=float((mcm[al]>=THR).mean()); fa=band.sum()/al.sum()
    pis.append(100*sh); tb+=int(band.sum()); ta+=int(al.sum())
    rows[m]=dict(lo=lo,hi=hi,n=int(band.sum()),frac_alive=float(fa),
                 share_pos_imp=sh,sharing_mean=float((mcm[band]>=THR).mean()),
                 sharing_pair=float(pf[band].mean()),dictwide=base,
                 enrichment=float(sh/fa))
    print(f"{m:<20} [{lo:.0e},{hi:.0e}) {band.sum():5d} {100*fa:6.2f}% {100*sh:8.1f}% "
          f"{(mcm[band]>=THR).mean():7.3f} {pf[band].mean():7.3f} {base:6.3f} {sh/fa:16.0f}x")
print(f"\npooled {tb}/{ta} = {100*tb/ta:.2f}%   band n {min(r['n'] for r in rows.values())}-{max(r['n'] for r in rows.values())}   "
      f"positive-importance share {min(pis):.1f}%-{max(pis):.1f}% (median {np.median(pis):.1f}%)")
orig = json.load(open(ROOT / "sec4_band_e1edges_ORIGINAL_2026-08-28.json"))
same = all(abs(orig[m][k]-rows[m][k]) < 1e-12 for m in rows for k in rows[m])
print("reproduces the 2026-08-28 output exactly:", same)
sys.exit(0 if same else 1)
