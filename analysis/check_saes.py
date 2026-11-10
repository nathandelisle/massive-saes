"""Download three representative checkpoints at the revisions in checkpoints/repositories.json and compare with the saved fixtures. Requires torch, huggingface_hub, and repository access."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pc1_sae import load_release_sae, load_preprocessing, Splice  # noqa: E402

man = json.loads((ROOT / "checkpoints" / "repositories.json").read_text())
core = man["core"]
bad = 0
for fx in sorted((ROOT / "analysis" / "fixtures").glob("*.npz")):
    model, cond, seed = fx.stem.rsplit("_", 2)
    f = np.load(fx)
    sae, cfg = load_release_sae(core["repo"], "%s/%s/%s" % (model, cond, seed), revision=core["revision"])
    assert cfg["sha256"] == str(f["sha256"]), "checkpoint differs from the fixture's"
    mean, v1 = load_preprocessing(core["repo"], model, revision=core["revision"])
    sp = Splice(sae, cond, mean, v1)
    h = torch.as_tensor(f["h"])
    z, c = sp.encode(h)
    xh = sp.decode(z, c)
    zr, xr = torch.as_tensor(f["z"]), torch.as_tensor(f["x_hat"])
    dz, dx = float((z - zr).abs().max()), float((xh - xr).abs().max())
    same_support = bool(((z != 0) == (zr != 0)).all())          # identical active features
    rel = dx / float(xr.abs().max())                             # float32 BLAS differences across torch builds ~1e-7
    l0 = float((z != 0).sum(-1).float().mean())
    ok = same_support and rel < 1e-5 and dz / float(zr.abs().max()) < 1e-5 and abs(l0 - cfg.get("k", l0)) < 1e-9
    bad += not ok
    print("%-22s %-9s %s  sha256 ok  same active set %s  max|dz| %.1e  max|dx_hat| %.1e (rel %.1e)  L0 %.1f  %s"
          % (model, cond, seed, same_support, dz, dx, rel, l0, "OK" if ok else "FAIL"))
print("\n%d failures" % bad)
sys.exit(1 if bad else 0)
