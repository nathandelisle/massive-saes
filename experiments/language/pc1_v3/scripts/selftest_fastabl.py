"""Check feature selection and activation locations."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v3.common import fastabl                                     # noqa: E402

FAIL = []


def check(name, cond, extra=""):
    print(f"{'ok  ' if cond else 'FAIL'}  {name} {extra}")
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------- screen logic
d = 512
rng = np.random.default_rng(0)
ez2 = np.exp(rng.normal(0, 3.0, d))
alive = np.ones(d, bool)
alive[:17] = False
# dCE ~ (1/2) kappa E[z^2] with kappa in a narrow band, as measured
kappa = np.exp(rng.normal(0, 0.15, d))
dce = {int(j): float(0.5 * kappa[j] * ez2[j]) for j in np.flatnonzero(alive)}

rep = fastabl.screen_report(dce, ez2, alive, ks=(20, 50), ms=(100, 200, 400))
check("screen spearman is high when dCE = .5 k E[z^2]",
      rep["spearman_logez2_vs_dce"] > 0.98,
      f"rho={rep['spearman_logez2_vs_dce']:.4f}")
check("recall of true top-50 reaches 1.0 by m=400",
      rep["recall"]["m400_top50"] == 1.0, str(rep["recall"]))
check("recall is monotone in m",
      rep["recall"]["m100_top50"] <= rep["recall"]["m200_top50"]
      <= rep["recall"]["m400_top50"])
check("n_alive counted correctly", rep["n_alive"] == int(alive.sum()))

cand = fastabl.screen_candidates(ez2, alive, 50)
check("screen never returns a dead feature", not (~alive[cand]).any())
check("screen returns exactly m", len(cand) == 50)
check("screen is capped by n_alive",
      len(fastabl.screen_candidates(ez2, alive, 10_000)) == int(alive.sum()))

# a screen that is uncorrelated with importance must show poor recall, or the
# recall statistic itself is broken and would rubber-stamp anything
shuf = ez2.copy()
rng.shuffle(shuf)
bad = fastabl.screen_report(dce, shuf, alive, ks=(50,), ms=(100,))
check("recall detects a BAD screen", bad["recall"]["m100_top50"] < 0.6,
      f"recall={bad['recall']['m100_top50']:.2f}")

# --------------------------------------------------------------- dense arrays
a = fastabl.to_array(d, dce)
check("unevaluated features are -inf, never 0",
      np.isneginf(a[:17]).all() and np.isfinite(a[17:]).all())
check("-inf can never win an argmax", int(np.argmax(a)) >= 17)

# -------------------------------------------------------------------- spearman
check("spearman monotone = 1", abs(fastabl.spearman([1, 2, 3, 4],
                                                    [2, 4, 6, 9]) - 1) < 1e-12)
check("spearman antitone = -1", abs(fastabl.spearman([1, 2, 3],
                                                     [3, 2, 1]) + 1) < 1e-12)
check("spearman ignores non-finite",
      abs(fastabl.spearman([1, 2, 3, np.nan], [1, 2, 3, 0]) - 1) < 1e-12)

# ------------------------------------------------------------------ chunk size
c_small = fastabl.tuned_chunk(None, 50304)
c_big = fastabl.tuned_chunk(None, 152064)
check("chunk >= default 8 for both vocabs", min(c_small, c_big) >= 8,
      f"pythia={c_small} qwen={c_big}")
check("bigger vocab does not get a bigger chunk", c_big <= c_small)
check("fp32 logits stay under ~7GB",
      c_big * 128 * 152064 * 4 / 1e9 < 7.0,
      f"{c_big * 128 * 152064 * 4 / 1e9:.2f}GB")

# NOTE: the layer-convention check moved to selftest_fidelity.py, where it is
# an exact tensor-identity test against v2's capture rather than a fingerprint
# match against the (differently-indexed) layer-profile tables.

print()
if FAIL:
    print(f"{len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("all fastabl/gate0 unit checks pass")
