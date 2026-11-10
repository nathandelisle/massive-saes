"""Qwen-2.5-7B Gated seed 0 fails reproducibly under the original configuration (FVU rises 7x between steps 4k and 7k,
then the dictionary collapses to ~465 live atoms). This script (a) characterizes the blow-up and (b) trains the cell
under a minimally corrected protocol.

--role diag0 : 'gated_diag' = the ORIGINAL Gated objective and trainer, numerically identical, but logging every 100
               steps: each loss term, the gradient norm per parameter group, the batch's largest row norm and the
               fraction of massive rows. Seed 0 only.
--role fix --seed S [--cell] : 'gated_clip' = the original Gated objective and trainer plus ADAPTIVE gradient-norm
               clipping: after 200 warm-up steps, if the global gradient norm exceeds 5x its running average
               (EMA 0.99 of clipped norms), it is rescaled to 5x that average. A healthy run should almost never clip,
               so seeds that trained normally are expected to be unchanged. Clip events are logged. The seed-42 box
               waits for seeds 137 and 0 and runs the ORIGINAL run_cell on the three corrected seeds.
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

import gi_common as G
from gi_common import io, registry, S

import sys
sys.path.insert(0, G.HERE)
import exp_b1_architecture_law as B1                                     # noqa: E402
import qwen_diag as QD                                                  # noqa: E402
from v3.common.data import load_model_and_tokenizer, eval_sequences     # noqa: E402
from v3.common.fastabl import tuned_chunk                               # noqa: E402

SPEC = registry.MODELS["qwen-2.5-7b"]
COEF = 5.64369
CLIP_MULT, EMA, WARMUP = 5.0, 0.99, 200
DIAG = {}


class GatedDiagSAE(S.GatedSAE):
    arch = "gated_diag"


class GatedClipSAE(S.GatedSAE):
    arch = "gated_clip"


S.ARCHS["gated_diag"] = GatedDiagSAE
S.ARCHS["gated_clip"] = GatedClipSAE
_orig_train = S.train_sae


def train_custom(batches, d_model, d_sae, arch="topk", seed=0, lr=3e-4, device="cuda", log_every=1000, verbose=True, tag="",
                 recal_every=100, recal_until_frac=1.0, total_steps=0, **arch_kwargs):
    """Verbatim copy of v3 saes.train_sae for Gated (same seed, init, loss order, Adam, renorm), plus diagnostics and,
    for 'gated_clip', the adaptive clip."""
    if arch not in ("gated_diag", "gated_clip"):
        return _orig_train(batches, d_model, d_sae, arch=arch, seed=seed, lr=lr, device=device, log_every=log_every,
                           verbose=verbose, tag=tag, recal_every=recal_every, recal_until_frac=recal_until_frac,
                           total_steps=total_steps, **arch_kwargs)
    t0 = time.time()
    torch.manual_seed(seed)
    sae = S.ARCHS[arch](d_model, d_sae, seed=seed, **arch_kwargs).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    log, diag = [], []
    ema, clips = None, 0
    sae.train()
    for step, x in enumerate(batches):
        recon, z = sae(x)
        mse = F.mse_loss(recon, x, reduction="none").sum(-1).mean()
        pen = sae.sparsity_loss(z, None)
        aux = sae.aux_loss(x)
        loss = mse + pen
        loss = loss + aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        want_diag = step % 100 == 0
        if want_diag or arch == "gated_clip":
            gn = float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in sae.parameters() if p.grad is not None)))
        clipped = False
        if arch == "gated_clip":
            if ema is not None and step >= WARMUP and gn > CLIP_MULT * ema:
                scale = CLIP_MULT * ema / gn
                for p in sae.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
                clipped = True
                clips += 1
            g_used = min(gn, CLIP_MULT * ema) if (ema is not None and step >= WARMUP) else gn
            ema = g_used if ema is None else EMA * ema + (1 - EMA) * g_used
        if want_diag or clipped:
            with torch.no_grad():
                norms = x.norm(dim=1)
                rec = dict(step=step, mse=float(mse), gate_penalty=float(pen), aux=float(aux), grad_norm=gn, clipped=clipped,
                           batch_max_row_norm=float(norms.max()), batch_frac_rows_gt_10x_median=float((norms > 10 * norms.median()).float().mean()),
                           l0=float((z != 0).float().sum(-1).mean()), gates_open_frac=float((sae._pi_gate > 0).float().mean()),
                           grad_by_param={n: float(p.grad.norm()) for n, p in sae.named_parameters() if p.grad is not None},
                           ema=ema)
            if want_diag:
                diag.append(rec)
            elif clipped and len(diag) < 5000:
                diag.append(rec)
        opt.step()
        sae.renorm_decoder()
        if step % log_every == 0 or step < 5:
            with torch.no_grad():
                var = x.pow(2).sum(-1).mean()
                fvu = (mse / var).item()
                l0 = (z != 0).float().sum(-1).mean().item()
            log.append(dict(step=step, mse=mse.item(), fvu=fvu, l0=l0))
            if verbose and step % log_every == 0:
                print(f"      [sae {tag}] step {step:6d}  fvu={fvu:.4f}  L0={l0:.1f}  clips={clips}  {(time.time()-t0)/60:.1f}m", flush=True)
            if not torch.isfinite(mse):
                raise RuntimeError("non-finite loss")
    sae.eval()
    DIAG[(arch, seed)] = dict(diag=diag, clips=clips)
    return sae, log


S.train_sae = train_custom
io.train_sae = train_custom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["diag0", "fix"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell", action="store_true")
    ap.add_argument("--sibling-prefix", default="")
    ap.add_argument("--wait-min", type=float, default=120)
    a = ap.parse_args()
    st = G.store(SPEC)
    X_test = G.harvest_split(SPEC, "test", 200_000)
    arch = "gated_diag" if a.role == "diag0" else "gated_clip"
    s = 0 if a.role == "diag0" else a.seed
    sae, _v, _m, _a, extra = io.get_sae(SPEC, s, arch=arch, condition="standard", device="cuda", l1_coef=COEF)
    cfg = torch.load(io.sae_cache_path(SPEC, s, arch=arch, condition="standard", l1_coef=COEF), map_location="cpu", weights_only=False)["cfg"]
    r = QD.diagnose(sae, extra, cfg, s, st, X_test, arch)
    d = DIAG.get((arch, s), {})
    r["clip_events"] = d.get("clips")
    r["diag_every_100_steps"] = d.get("diag")
    r["protocol"] = (dict(clip="adaptive: rescale global grad norm to 5x its EMA(0.99) when exceeded, after 200 steps")
                     if arch == "gated_clip" else dict(clip="none (original trainer, diagnostics only)"))
    G.dump(r, f"qwen_{a.role}_s{s}.json")
    print(f"[{G.now()}] {arch} s{s}: L0 exact {r['l0_exact']:.2f} alive {r['n_alive']} never {r['n_never_fired']} fvu {r['fvu_total']:.5f} "
          f"content {r['fvu_content']:.4f} held-out content {r['heldout_fvu_content']:.4f} clips {r['clip_events']}", flush=True)
    del sae
    torch.cuda.empty_cache()
    if not (a.role == "fix" and a.cell and s == 42):
        return
    need = {t: G.cache_name(SPEC, t, "gated_clip", l1_coef=COEF) for t in (137, 0)}
    deadline = time.time() + 60 * a.wait_min
    while True:
        missing = [t for t, h in need.items() if not os.path.exists(os.path.join(io.ROOT, "saes", h))]
        for t in list(missing):
            key = f"{a.sibling_prefix}{t}/saes/{need[t]}"
            if G.s3_exists(key):
                G.fetch(key, os.path.join(io.ROOT, "saes", need[t]))
                print(f"[{G.now()}] fetched corrected sibling seed {t}", flush=True)
        missing = [t for t, h in need.items() if not os.path.exists(os.path.join(io.ROOT, "saes", h))]
        if not missing or time.time() > deadline:
            break
        time.sleep(60)
    args = argparse.Namespace(
        models=["qwen-2.5-7b"], archs=["gated_clip"], eval_seqs=registry.EVAL_SEQS_IMPORTANCE, topk=50, chunk=0, screen=1500,
        knob_min=1e-6, knob_max=1e5, recal_passes=0, target_l0=64.0, l0_tol=0.25, cal_steps=4000, force=True,
        seeds=[42, 137, 0], allow_uncalibrated=False, device="cuda", progress=500)
    model, tok = load_model_and_tokenizer(SPEC, device="cuda")
    seqs, am = eval_sequences(tok, n_seqs=args.eval_seqs)
    args.chunk = tuned_chunk(SPEC, vocab_size=int(model.config.vocab_size))
    cell = B1.run_cell(model, SPEC, seqs, st["adir"], st["evecs"], "gated_clip", dict(l1_coef=COEF), "standard", args, am=am)
    shutil.copy(io.path("results", B1.EXP, f"feat_{SPEC.key}_gated_clip_standard.npz"),
                io.path("results", "gated_investigation", "feat_qwen-2.5-7b_gated_clip_standard.npz"))
    per_seed = []
    for t in args.seeds:
        sae, _v, _m, _a, ex = io.get_sae(SPEC, t, arch="gated_clip", condition="standard", device="cuda", l1_coef=COEF)
        f, _ = G.store_fidelity(sae, st)
        per_seed.append(dict(seed=t, **f, **G.heldout_fidelity(sae, X_test, st["mean"], st["v1"]), train_log_tail=G.log_tail(ex)))
        del sae
        torch.cuda.empty_cache()
    G.dump(dict(role="fix_cell", coef=COEF, seeds=args.seeds, cell=cell, per_seed=per_seed, siblings_trained_here=missing),
           "qwen_fix_cell.json")
    print(f"[{G.now()}] corrected cell: top-50 {cell['top50_mean']:.3f} +/- {cell['top50_std']:.3f} dict-wide {cell['dictwide_mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
