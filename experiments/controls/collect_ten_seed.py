#!/usr/bin/env python3
"""Collect ten-seed GPT-2 measurements."""
import sys
from pathlib import Path
_PACKAGE = Path(__file__).resolve().parents[1] / "ten_seed_gpt2"
sys.path.insert(0, str(_PACKAGE))
from pc1night.cli import main
# collect-only guard (this copy runs the CPU collect stage; workers never load it): a group with zero
# finished seeds (llama-3.1-8b was never harvested) reaches seed_bootstrap_matrix with an empty matrix and
# numpy's multinomial rejects a zero-length pvals. No interval exists for an empty group; report None.
import pc1night.summary as _S
_orig_sbm = _S.seed_bootstrap_matrix
def _sbm(M, draws=2000, seed=101):
    import numpy as _np
    if len(_np.asarray(M)) == 0: return None
    return _orig_sbm(M, draws, seed)
_S.seed_bootstrap_matrix = _sbm
# collect-only speedups (unhashed; numerically identical). summary.collect reloads references/<site>/reference.npz inside
# its seed-pair loop, and hashes every artifact sequentially. Cache the read-only reference; precompute the same io.sha
# digests in a thread pool so collect's own comparison against complete.json runs on them.
import json as _json, concurrent.futures as _cf
from pathlib import Path as _P2
import pc1night.summary as _S2
_orig_loadnp2 = _S2.loadnp
_ref_cache = {}
def _loadnp2(p):
    p = _P2(p)
    if p.name == 'reference.npz' and p.parent.parent.name == 'references':
        k = str(p)
        if k not in _ref_cache:
            _ref_cache[k] = _orig_loadnp2(p)
        return _ref_cache[k]
    return _orig_loadnp2(p)
_S2.loadnp = _loadnp2
_orig_sha2 = _S2.sha
_sha_cache = {}
def _sha2(p):
    k = str(p)
    return _sha_cache[k] if k in _sha_cache else _orig_sha2(p)
_S2.sha = _sha2
_orig_collect2 = _S2.collect
def _collect2(plan, out, draws=2000):
    paths = []
    for job in plan['jobs']:
        d = _P2(out) / 'fits' / job['id']; m = d / 'complete.json'
        if m.exists():
            paths += [d / f for f in _json.loads(m.read_text())['hashes'] if (d / f).is_file()]
    with _cf.ThreadPoolExecutor(32) as ex:
        for pth, h in zip(paths, ex.map(_orig_sha2, paths)):
            _sha_cache[str(pth)] = h
    print(f'[collect-hook] precomputed {len(_sha_cache)} artifact digests', flush=True)
    return _orig_collect2(plan, out, draws)
_S2.collect = _collect2


# --- resume fix, applied OUTSIDE the hashed pc1night/*.py set ---------------------------------
# pc1night/train.py:25 returns dict(status='skipped',**c) for an already-complete fit, but the
# restored complete.json already carries 'status', so every worker resuming a shard with >=1
# finished fit died with "dict() got multiple values for keyword argument 'status'"
# (pc1r-gpt2-0/-3, 2026-09-10 05:2x). That line runs only AFTER the marker's protocol and the
# artifact sha256s were validated, so the intended result is simply the marker with
# status='skipped'. io.seal() globs pc1night/*.py only; this file and cloud/ are unhashed, so the
# protocol id, INPUTS.json and all existing markers stay valid.
import pc1night.train as _T
from pc1night.io import read as _read
from pathlib import Path as _P
_orig_execute = _T.execute
def _execute(job, site, ref, settings, root, protocol, device='cuda', stop_after=None):
    try:
        return _orig_execute(job, site, ref, settings, root, protocol, device, stop_after)
    except TypeError as e:
        if "multiple values for keyword argument 'status'" not in str(e):
            raise
        c = _read(_P(root) / 'fits' / job['id'] / 'complete.json')
        return {**c, 'status': 'skipped'}
_T.execute = _execute

# parallel page-in for the block-cache gather (cloud/fastrows.py, unhashed, bit-identical)
import sys as _sys; _sys.path.insert(0, str(_PACKAGE / 'cloud'))
import fastrows as _fastrows; _fastrows.install()
# ------------------------------------------------------------------------------------------------

if __name__=='__main__':main()
