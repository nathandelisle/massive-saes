"""Summarize gradient and integrated-gradient numerical checks."""
import csv
import json
import sys
from pathlib import Path

PACK, OUT = Path(sys.argv[1]), Path(sys.argv[2])
RAW = PACK / "raw"
rows = []
for d in sorted(RAW.glob("rx-imp-*/imp/*/*")):
    job = d.parts[-4]
    if "-h" in job.rsplit("-s", 1)[-1] or job.endswith(("-h0", "-h1", "-h2", "-h3", "-h4", "-h5", "-h6")):
        continue
    g = json.loads((d / "gates.json").read_text()) if (d / "gates.json").exists() else {}
    fd = (g.get("gate9") or {}).get("fd")
    igc = d / "rank_ig_convergence.json"
    ig = json.loads(igc.read_text()) if igc.exists() else None
    cond, seed = d.name.rsplit("_s", 1)
    r = dict(model=d.parts[-2], condition=cond, seed=int(seed), job=job,
             gates_json=bool(g), gate8_saved="gate8" in g)
    if fd:
        r.update(fd_saved=True, fd_max_rel=fd.get("max_rel"), fd_pass_recorded=fd.get("pass_"),
                 fd_checks=json.dumps([[c["pos"], c["feature"], round(c["autograd"], 6), round(c["finite_diff"], 6)]
                                       for c in fd.get("checks", [])]))
    else:
        r.update(fd_saved=False)
    if ig:
        h = ig if isinstance(ig, list) else ig.get("ig_convergence", ig)
        last = h[-1] if isinstance(h, list) and h else {}
        r.update(ig_steps_final=last.get("steps"), ig_completeness_rel_error=last.get("completeness_rel_error"),
                 ig_top50_overlap_prev=last.get("top50_overlap_prev"),
                 ig_converged_engine_rule=bool((last.get("completeness_rel_error") or 1) < 0.01 and
                                               (last.get("top50_overlap_prev") or 0) >= 0.96),
                 ig_ladder=json.dumps(h))
    rows.append(r)
keys = list(dict.fromkeys(k for x in rows for k in x))
tmp = OUT / "gradient_validation_production.csv.tmp"
with open(tmp, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
tmp.replace(OUT / "gradient_validation_production.csv")
n = len(rows)
print(n, "dictionaries; fd saved", sum(r["fd_saved"] for r in rows), "; fd pass", sum(bool(r.get("fd_pass_recorded")) for r in rows),
      "; gate8 saved", sum(r["gate8_saved"] for r in rows), "; IG converged", sum(bool(r.get("ig_converged_engine_rule")) for r in rows),
      "of", sum("ig_steps_final" in r for r in rows))
