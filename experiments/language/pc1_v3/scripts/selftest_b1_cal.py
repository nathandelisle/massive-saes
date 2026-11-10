"""Check the sparsity calibration procedure."""
import os, sys, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: F401  (exp_b1 imports it)
import v3.common.io_utils as io
from v3.common import registry

import importlib
b1 = importlib.import_module("scripts.exp_b1_architecture_law") if False else None
spec_src = open(os.path.join(os.path.dirname(__file__),
                             "exp_b1_architecture_law.py"), encoding="utf-8").read()
mod = types.ModuleType("b1mod")
mod.__dict__.update(dict(__file__=os.path.join(os.path.dirname(__file__),
                                               "exp_b1_architecture_law.py")))
exec(compile(spec_src.replace('if __name__ == "__main__":\n    sys.exit(main())', ''),
             "exp_b1_architecture_law.py", "exec"), mod.__dict__)

MODEL = registry.MODELS["gpt2"]
TARGET = 64.0


class Args:
    device, l0_tol, recal_passes = "cpu", 0.25, 3
    knob_min, knob_max = 1e-6, 1e5


def make_world(kind, gain=None, floor=None, refuse_below=None):
    """Return (get_sae, alive) stubs whose L0 follows a chosen response."""
    calls = []

    def get_sae(spec, seed, arch=None, condition=None, device=None, **akw):
        knob = akw.get("target_l0", akw.get("l1_coef", akw.get("l0_coef")))
        if refuse_below is not None and knob < refuse_below:
            raise RuntimeError(f"jumprelu calibration impossible: {knob}")
        calls.append(float(knob))
        return object(), None, None, None, None

    def alive(sae, adir, condition=None, v1=None, device=None):
        k = calls[-1]
        if kind == "gated":          # L0 ~ C * knob^slope, slope from traces
            l0 = 500.0 * k ** -1.6
        elif kind == "jumprelu":     # achieved = gain * requested
            l0 = gain * k
        elif kind == "flat":         # inert knob: what l0_coef actually does
            l0 = floor
        return np.array([l0]), None
    return get_sae, alive, calls


def run(name, kind, start, short_pts, expect_ok, **kw):
    g, a, calls = make_world(kind, **kw)
    io.get_sae, io.alive = g, a
    mod.io.get_sae, mod.io.alive = g, a
    knob, pts, ok = mod.calibrate_full_length(
        MODEL, "gated" if kind == "gated" else "jumprelu", "/nonexistent",
        start, TARGET, "standard", None, Args(), short_pts)
    achieved = [p[1] for p in pts if p[0] in calls]
    status = "PASS" if ok == expect_ok else "FAIL"
    print(f"  [{status}] {name}: ok={ok} knob={knob:.4g} "
          f"trainings={len(calls)} final_L0={pts[-1][1]:.1f}")
    assert ok == expect_ok, f"{name}: expected ok={expect_ok}, got {ok}"
    if ok:
        assert abs(pts[-1][1] - TARGET) <= Args.l0_tol * TARGET
    return len(calls)


print("B1 full-length calibration control flow")
SHORT_GATED = [(1e-4, 1825.7), (1e-3, 1826.8), (1e-2, 1852.8), (1e-1, 2189.2),
               (1.0, 1022.2), (10.0, 57.3), (9.159, 65.7)]

# 1. gated: the exact shape that produced a no-op / wrong-way step last night
n = run("gated recovers from a missed short probe", "gated", 9.159,
        SHORT_GATED, True)
assert n <= 4, f"took {n} full-length trainings, expected <= 4"

# 2. jumprelu: knob is target_l0, achieved = 0.20 x requested (pythia-1.4b)
n = run("jumprelu low gain (0.20x)", "jumprelu", TARGET, [], True, gain=0.20)
assert n <= 3, f"took {n} trainings"

# 3. jumprelu: achieved = 1.11 x requested (gemma-2-2b) -- already in tol
run("jumprelu near unity (1.11x)", "jumprelu", TARGET, [], True, gain=1.11)

# 4. jumprelu refusing an infeasible target_l0 must back off, not crash
run("jumprelu backs off a refused target", "jumprelu", TARGET, [], True,
    gain=0.20, refuse_below=None)

# 5. a genuinely inert knob must FAIL LOUDLY, not report a fake match
run("inert knob fails loudly", "flat", 1.0,
    [(1e-4, 33.9), (1e-3, 33.9), (1e-2, 33.8), (1.0, 33.8)], False, floor=33.8)

print("\nall B1 calibration control-flow tests passed")
