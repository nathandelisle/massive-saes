"""Recompute the two-seed Llama-70B sharing results from the original and converted feature arrays."""
from pathlib import Path
import numpy as np
D = Path(__file__).resolve().parents[1] / "results" / "examples" / "llama70b"


def top50(imp, ok):
    ix = np.flatnonzero(ok & np.isfinite(imp))
    return ix[np.lexsort((ix, -imp[ix]))][:50]


for cond, k in (("standard", "k0"), ("bypass", "k1")):
    z = np.load(D / "original_modal_sae-final-v1" / ("%s_features.npz" % k))
    al = z["freq"].astype(float) > 1e-3
    sh = (z["avg_match"].astype(np.float32) >= 0.7).astype(float)
    t = top50(z["importance"].astype(float), al)
    c = np.load(D / "converted_v3_schema" / ("features_llama-3.1-70b_%s.npz" % cond))
    shc = (c["max_cos"] >= 0.7).mean(0)
    tc = top50(c["imp_zero"], c["alive"].astype(bool))
    print("%-9s original: top-50 %.3f  dict %.3f   converted: top-50 %.3f  dict %.3f" %
          (cond, sh[t].mean(), sh[al].mean(), shc[tc].mean(), shc[c["alive"].astype(bool)].mean()))

    expected = 0.38 if cond == "standard" else 0.84
    assert np.isclose(sh[t].mean(), expected), f"Unexpected {cond} top-50 sharing"
    assert np.isclose(shc[tc].mean(), expected), f"Converted {cond} array differs"
