"""Stage B (transcoders) and Stage C (crosscoders) jobs: caches -> diagnose -> train arms x seeds -> test eval once
-> cross-seed compare -> (B only) native CE with the transcoder spliced in place of the MLP.

Training / evaluation / compare are the SUPPLIED harness (ext/cached_coders.py, byte-identical to
code/ext_reference/cached_coders.py), run as subprocesses. Test split evaluated once per run, best.pt (selected on
validation by the harness). Configs: the supplied examples (8192 features, TopK k=64, 20k steps, batch 4096, lr 3e-4);
rank1/random1 ridge = the value `diagnose` selects on validation.

B arms  (transcoder):  T0 plain, T1 rank1, T2 random1, T3 skip, T4 oracle (diagnostic, needs y), S0 sae, S1 sae_bypass
C arms  (crosscoder):  X0 plain/global, X1 plain/per_stream, X2 block_pc, X3 joint_pc, X4 random_block,
                       + ReLU-L1 plain with block_sum vs joint penalties (seed 42, 3 l1 values)
C data:  C0 identical [r12|r12], C1 scale_only [r12|r12 with PC1 coefficient x2], C3a cross-layer [r12|r16],
         C3b revision pair [r12 @ step71000 | r12 @ main]
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from rx import common, modelio as M, partitions as P
from ext import cache as C

HERE = Path(__file__).resolve().parent
CC = HERE / "cached_coders.py"
BASE = dict(activation="topk", features=8192, k=64, data_seed=101, scale_mode="global", penalty="joint", l1=0.0,
            ridge=1e-5, direction_seed=101, fit_tokens=50000, steps=20000, batch_size=4096, learning_rate=3e-4,
            evaluate_every=500, eval_batch_size=1024, validation_tokens=50000)
B_ARMS = dict(T0=dict(variant="plain"), T1=dict(variant="rank1"), T2=dict(variant="random1"), T3=dict(variant="skip"),
              T4=dict(variant="oracle"), S0=dict(variant="sae"), S1=dict(variant="sae_bypass"))
C_ARMS = dict(X0=dict(variant="plain"), X1=dict(variant="plain", scale_mode="per_stream"),
              X2=dict(variant="block_pc"), X3=dict(variant="joint_pc"), X4=dict(variant="random_block"))
C_L1 = {"L1%s_%g" % (pen, l1): dict(variant="plain", activation="relu_l1", penalty=pen, l1=l1)
        for pen in ("block_sum", "joint") for l1 in (1e-3, 3e-3, 1e-2)}


def sh(args, log):
    with open(log, "a") as f:
        r = subprocess.run([sys.executable] + [str(a) for a in args], stdout=f, stderr=subprocess.STDOUT,
                           cwd=str(HERE.parent))
    return r.returncode


def run_one(cache, out, cfg, device="cuda"):
    out = Path(out)
    if (out / "test_best_metrics.json").exists():
        return "cached"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not (out / "last.pt").exists():
        import shutil
        shutil.rmtree(out)
    cfgp = out.parent / (out.name + ".config.json")
    cfgp.write_text(json.dumps(cfg, indent=1))
    log = out.parent / (out.name + ".log")
    if not (out / "last.pt").exists():
        if sh([CC, "train", "--cache", cache, "--config", cfgp, "--out", out, "--device", device], log):
            return "train_failed"
    if sh([CC, "evaluate", "--cache", cache, "--run", out, "--split", "test", "--checkpoint", "best.pt",
           "--device", device], log):
        return "eval_failed"
    return "ok"


def diagnose(cache, out, task):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not (out / "diagnostics.json").exists():
        if out.exists():
            import shutil
            shutil.rmtree(out)
        sh([CC, "diagnose", "--cache", cache, "--out", out, "--task", task], str(out) + ".log")
    return json.loads((out / "diagnostics.json").read_text())


def compare_all(runs, outdir):
    """All ordered seed pairs per arm (reference convention: seed 42 vs others is a subset)."""
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (arm, seeds) in runs.items():
        for sa, sb in itertools.permutations(seeds, 2):
            fa, fb = seeds[sa] / "test_best_features.npz", seeds[sb] / "test_best_features.npz"
            if not (fa.exists() and fb.exists()):
                continue
            o = outdir / ("%s_s%s_vs_s%s.json" % (arm, sa, sb))
            if not o.exists():
                sh([CC, "compare", "--a", fa, "--b", fb, "--out", o], str(outdir / "compare.log"))
            if o.exists():
                rows.append(dict(arm=arm, a=sa, b=sb, **json.loads(o.read_text())))
    common.write_json(outdir / "compare_summary.json", rows)
    return rows


def pool(jobs, par):
    res = {}
    with ThreadPoolExecutor(par) as ex:
        futs = {ex.submit(run_one, *j): j[1] for j in jobs}
        for f, name in futs.items():
            res[str(name)] = f.result()
            common.log("coders", "%s -> %s" % (name, res[str(name)]))
    return res


# ---------------------------------------------------------------- native CE for transcoders
@torch.no_grad()
def native_ce(model, spec, layer, runs, outp):
    sys.path.insert(0, str(HERE))
    import cached_coders as cc
    tok_part = P.Partition("valid", native_ce.tok)
    coders = {}
    for name, rd in runs.items():
        ck = Path(rd) / "best.pt"
        if not ck.exists():
            continue
        ckpt = torch.load(ck, map_location="cpu", weights_only=True)
        prep = dict(np.load(Path(rd) / "preprocessing.npz", allow_pickle=False))
        cm = cc.Coder(ckpt["config"], prep, spec.d_model).to(DEV).eval()
        cm.load_state_dict(ckpt["state"])
        coders[name] = cm
    b = M.blocks(model, spec)[layer]
    tgt = b.post_feedforward_layernorm if spec.family == "gemma" else b.mlp
    state = {"x": None, "coder": None}
    h1 = b.mlp.register_forward_pre_hook(lambda m, a: state.__setitem__("x", a[0]))

    def post(m, i, o):
        if state["coder"] is None:
            return o
        y = o[0] if isinstance(o, tuple) else o
        sh_ = y.shape
        x = state["x"].reshape(-1, sh_[-1]).float()
        cm = state["coder"]
        pred, _, _ = cm(x, y.reshape(-1, sh_[-1]).float())
        r = (pred * cm.yscale + cm.ymean).reshape(sh_).to(y.dtype)
        return (r,) + tuple(o[1:]) if isinstance(o, tuple) else r
    h2 = tgt.register_forward_hook(post)
    sums = {}
    try:
        for sel, ids, am in tok_part.batches(max_tokens=4096):
            for name in ["clean"] + list(coders):
                state["coder"] = coders.get(name)
                lg = model(ids.to(M.input_device(model)), attention_mask=am.to(M.input_device(model))).logits
                nll, valid = M.token_nll(lg, ids, am)
                for bi, dd in enumerate(tok_part.doc[sel]):
                    s = sums.setdefault(name, {}).setdefault(int(dd), [0, 0.0])
                    s[0] += int(valid[bi].sum()); s[1] += float(nll[bi][valid[bi]].double().sum())
    finally:
        h1.remove(); h2.remove()
    docs = sorted(sums["clean"])
    n = np.array([sums["clean"][d][0] for d in docs], float)
    S = {k: np.array([v[d][1] for d in docs]) for k, v in sums.items()}
    rng = np.random.default_rng(812)
    B = [rng.integers(0, len(docs), len(docs)) for _ in range(2000)]
    out = dict(layer=layer, articles=len(docs), tokens=int(n.sum()), ce_clean=float(S["clean"].sum() / n.sum()), runs={})
    for k in coders:
        dlt = S[k] - S["clean"]
        bs = [dlt[i].sum() / n[i].sum() for i in B]
        out["runs"][k] = dict(ce=float(S[k].sum() / n.sum()), ce_increase=float(dlt.sum() / n.sum()),
                              ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
    common.write_json(outp, out)
    np.savez(str(outp).replace(".json", "_docsums.npz"), doc=np.array(docs), n=n, **S)
    return out


# ---------------------------------------------------------------- jobs
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(spec):
    return M.load(spec, device=DEV, dtype=None if DEV == "cuda" else torch.float32)


def job_B(a):
    spec = M.spec_of(a.model)
    layers = [int(x) for x in a.layers.split(",")]
    croot = Path(common.WORK) / "caches" / a.model
    res = common.RESULTS / "extB" / a.model
    if not all((croot / ("mlp%d" % l) / "metadata.json").exists() for l in layers):
        model, tok = load(spec)
        C.build(model, spec, tok, "mlp", layers, croot, a.train_tokens, a.model)
        del model
        torch.cuda.empty_cache()
    for l in layers:
        (res / ("L%d" % l)).mkdir(parents=True, exist_ok=True)
        (res / ("L%d" % l) / "cache_metadata.json").write_text((croot / ("mlp%d" % l) / "metadata.json").read_text())
    seeds = [int(s) for s in a.seeds.split(",")]
    arms = a.arms.split(",")
    jobs, runs = [], {}
    for l in layers:
        diag = diagnose(croot / ("mlp%d" % l), res / ("L%d" % l) / "diagnose", "transcoder")
        for arm in arms:
            for s in seeds:
                cfg = dict(BASE, task="transcoder", seed=s, **B_ARMS[arm])
                if arm in ("T1", "T2"):
                    cfg["ridge"] = diag["selected_ridge"]
                rd = res / ("L%d" % l) / ("%s_s%d" % (arm, s))
                jobs.append((croot / ("mlp%d" % l), rd, cfg, DEV))
                runs.setdefault(l, {}).setdefault(arm, {})[s] = rd
    status = pool(jobs, a.par)
    common.write_json(res / "run_status.json", status)
    for l in layers:
        compare_all(runs[l], res / ("L%d" % l) / "compare")
    model, tok = load(spec)
    native_ce.tok = tok
    for l in layers:
        flat = {"%s_s%d" % (arm, s): rd for arm, ss in runs[l].items() for s, rd in ss.items()}
        native_ce(model, spec, l, flat, res / ("L%d" % l) / "native_ce.json")
    return dict(model=a.model, layers=layers, status=status)


def job_C(a):
    spec = M.spec_of(a.model)
    croot = Path(common.WORK) / "caches" / a.model
    res = common.RESULTS / "extC" / a.model
    conds = a.conds.split(",")
    la, lb = [int(x) for x in (a.layers or "12,16").split(",")]
    need_main = [la] + ([lb] if "C3a" in conds else [])
    if not all((croot / ("resid%d" % l) / "metadata.json").exists() for l in need_main):
        model, tok = load(spec)
        C.build(model, spec, tok, "resid", need_main, croot, a.train_tokens, a.model)
        del model
        torch.cuda.empty_cache()
    rev = Path(common.WORK) / "caches" / ("%s@%s" % (a.model, a.revision))
    if "C3b" in conds and not (rev / ("resid%d" % la) / "metadata.json").exists():
        C.main(["--model", a.model, "--kind", "resid", "--layers", str(la), "--root", str(rev),
                      "--train-tokens", str(a.train_tokens), "--revision", a.revision])
        torch.cuda.empty_cache()
    pairs = {}
    r12 = croot / ("resid%d" % la)
    for c in conds:
        out = Path(common.WORK) / "caches" / ("pair_%s_%s" % (a.model, c))
        pairs[c] = out
        if (out / "metadata.json").exists():
            continue
        if c == "C0":
            C.pair(r12, r12, out, label="C0_identical")
        elif c == "C1":
            tr = np.load(r12 / "train" / "r.npy", mmap_mode="r")
            ix = np.random.default_rng(101).choice(len(tr), min(200000, len(tr)), replace=False)
            X = np.asarray(tr[np.sort(ix)], np.float64)
            mu = X.mean(0)
            ev, ec = np.linalg.eigh(np.cov(X, rowvar=False))
            v = ec[:, -1]
            np.savez(Path(common.WORK) / "caches" / "C1_train_pc.npz", mu=mu, v=v, frac=ev[-1] / ev.sum())

            def tf(B, mu=mu.astype(np.float32), v=v.astype(np.float32)):
                return B + ((B - mu) @ v)[:, None] * v[None, :]          # PC1 coefficient x2, all else identical
            C.pair(r12, r12, out, transform=tf, label="C1_scale_only_pc1x2_trainPC")
        elif c == "C3a":
            C.pair(r12, croot / ("resid%d" % lb), out, label="C3a_crosslayer_%d_%d" % (la, lb))
        elif c == "C3b":
            C.pair(rev / ("resid%d" % la), r12, out, label="C3b_revision_%s_vs_main_L%d" % (a.revision, la))
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs, runs = [], {}
    for c in conds:
        diagnose(pairs[c], res / c / "diagnose", "crosscoder")
        for arm, kw in C_ARMS.items():
            for s in seeds:
                rd = res / c / ("%s_s%d" % (arm, s))
                jobs.append((pairs[c], rd, dict(BASE, task="crosscoder", seed=s, **kw), DEV))
                runs.setdefault(c, {}).setdefault(arm, {})[s] = rd
        for arm, kw in C_L1.items():
            rd = res / c / ("%s_s42" % arm)
            jobs.append((pairs[c], rd, dict(BASE, task="crosscoder", seed=42, **kw), DEV))
    status = pool(jobs, a.par)
    common.write_json(res / ("run_status_%s.json" % "_".join(conds)), status)
    for c in conds:
        compare_all(runs[c], res / c / "compare")
    return dict(model=a.model, conds=conds, status=status)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["B", "C"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default="")
    ap.add_argument("--arms", default="T0,T1,T2,T3,T4,S0,S1")
    ap.add_argument("--seeds", default="42,137,0,1,2")
    ap.add_argument("--conds", default="C0,C1")
    ap.add_argument("--revision", default="step71000")
    ap.add_argument("--train-tokens", type=int, default=3_000_000)
    ap.add_argument("--par", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--features", type=int, default=8192)
    a = ap.parse_args(argv)
    BASE.update(steps=a.steps, features=a.features, evaluate_every=min(500, a.steps))
    return job_B(a) if a.stage == "B" else job_C(a)
