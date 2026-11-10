"""Recompute the mean number of active features using the recorded token counts."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

RAW = Path(sys.argv[1]) / "raw"
OUT = Path(sys.argv[2])
FIX = 100000 / 106496
rows = []
for f in sorted(RAW.glob("rx-train-*/train/*/*/summary.json")):
    s = json.loads(f.read_text())
    for cond, c in (s.get("cells") or {}).items():
        rep = np.array(c.get("l0_measured") or [], float)
        if not len(rep):
            continue
        cor = rep * FIX
        fvu = [x for x in (c.get("fvu_final_log") or []) if x is not None]
        bs = c.get("by_source") or []
        rows.append(dict(model=s["model"], kind=s["kind"], layer=s.get("layer"), arch=c.get("arch"), condition=cond,
                         kwargs=json.dumps(c.get("kwargs")),
                         l0_reported_mean=float(rep.mean()), l0_corrected_mean=float(cor.mean()),
                         l0_corrected_min=float(cor.min()), l0_corrected_max=float(cor.max()),
                         within_pm5_of_64_corrected=bool(((cor >= 60.8) & (cor <= 67.2)).all()),
                         within_pm10_of_64_corrected=bool(((cor >= 57.6) & (cor <= 70.4)).all()),
                         old_flag_l0_matched_pm5_on_reported=c.get("l0_matched_pm5"),
                         fvu_mean=float(np.mean(fvu)) if fvu else None, fvu_min=min(fvu) if fvu else None,
                         fvu_max=max(fvu) if fvu else None,
                         top50_shared=c.get("top50_shared_mean_over_sources"),
                         top50_shared_sd_over_sources=c.get("top50_shared_sd_over_sources"),
                         dictwide=float(np.mean([b["dictwide_mean"] for b in bs])) if bs else None,
                         n_alive_mean=float(np.mean([b["n_alive"] for b in bs])) if bs else None,
                         calibration_ok=c.get("calibration_ok"),
                         alive_convention="historical: reported rate > 1e-3 == true rate > 9.39e-4 (first 106,496 rows)"))
keys = list(dict.fromkeys(k for r in rows for k in r))
tmp = OUT / "train_l0_corrected.csv.tmp"
with open(tmp, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
tmp.replace(OUT / "train_l0_corrected.csv")
print(len(rows), "cells")
