"""Measure total and content reconstruction error, matched coefficient counts, and feature allocation."""
import argparse
import os
import time

import numpy as np
import torch

import gi_common as G
from gi_common import io, registry

PREFIX = {"gpt2": "b1v2-gpt2", "pythia-160m": "b1v2-small", "pythia-1.4b": "b1v2-p14b", "pythia-6.9b": "b1v2-p69",
          "llama-3.1-8b": "b1v2-l8b", "gemma-2-2b": "b1v2-gemma", "gemma-2-9b": "b1v2-g9b", "qwen-2.5-7b": "b1v2-qwen"}


@torch.no_grad()
def fidelity_cond(sae, X_iter, v1, condition):
    """Exact L0 / alive / total / content FVU, feeding a bypass SAE its projected input and restoring PC1."""
    v = torch.tensor(v1, dtype=torch.float32, device="cuda")
    vd = v.double()
    tot = rows = 0
    fire = None
    err = var = perr = pvar = 0.0
    for x in X_iter:                                    # x is already mean-centered
        coef = x @ v
        x_in = x - coef[:, None] * v[None, :] if condition == "bypass" else x
        hat, z = sae(x_in)
        if condition == "bypass":
            hat = hat + coef[:, None] * v[None, :]      # restore the PC1 component exactly
        nz = z != 0
        tot += int(nz.sum()); rows += x.shape[0]
        fire = nz.double().sum(0) if fire is None else fire + nz.double().sum(0)
        e = (hat - x).double(); xd = x.double()
        err += float(e.square().sum()); var += float(xd.square().sum())
        ep = e - (e @ vd)[:, None] * vd[None, :]; xp = xd - (xd @ vd)[:, None] * vd[None, :]
        perr += float(ep.square().sum()); pvar += float(xp.square().sum())
    rate = (fire / rows).cpu().numpy()
    return dict(l0_exact=tot / rows, rows=rows, n_alive=int((rate > registry.ALIVE_FIRING_RATE).sum()),
                n_never_fired=int((rate == 0).sum()), fvu_total_not_comparable_across_conditions=err / var,
                fvu_content=perr / pvar, w1_rows=1 - pvar / var), rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", default="topk")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 0, 1, 2])
    a = ap.parse_args()
    spec = registry.MODELS[a.model]
    t0 = time.time()
    st = G.store(spec)
    X = G.harvest_split(spec, "test", 200_000)
    mean_t = torch.tensor(st["mean"], dtype=torch.float32, device="cuda")
    out = dict(model=a.model, arch=a.arch, prefix=PREFIX[a.model], w1_store=st["w1"], runs=[],
               heldout=dict(corpus="wikitext-103", split="test", n_tokens=int(len(X))))
    for condition in ("standard", "bypass"):
        for s in a.seeds:
            path = io.sae_cache_path(spec, s, arch=a.arch, condition=condition, k=64)
            key = f"{PREFIX[a.model]}/saes/{os.path.basename(path)}"
            if not G.s3_exists(key):
                print(f"[{G.now()}] {condition} s{s}: {key} absent; skipped", flush=True)
                continue
            p = G.fetch(key, os.path.join(io.ROOT, "sec6_saes", os.path.basename(key)))
            sae, extra, cfg = G.load_ckpt(p)
            store_fid, _rate = fidelity_cond(sae, G._rows(st), st["v1"], condition)

            def held():
                for i in range(0, len(X), 8192):
                    yield torch.tensor(X[i:i + 8192], device="cuda") - mean_t
            ho, _r = fidelity_cond(sae, held(), st["v1"], condition)
            arch_rates, arch_alive = io.alive(sae, st["adir"], condition=condition, v1=st["v1"], device="cuda")
            rec = dict(condition=condition, seed=s, checkpoint=os.path.basename(key), checkpoint_sha256=G.sha256(p),
                       cfg_in_checkpoint=cfg, seed_in_checkpoint=(extra.get("cfg") or {}).get("seed"),
                       condition_in_checkpoint=(extra.get("cfg") or {}).get("condition"),
                       l0_archived_counter=float(arch_rates.sum()), n_alive_archived_counter=int(arch_alive.sum()),
                       train_log_tail=G.log_tail(extra), **{f"store_{k}": v for k, v in store_fid.items()},
                       **{f"heldout_{k}": v for k, v in ho.items()})
            out["runs"].append(rec)
            print(f"[{G.now()}] {a.model} {a.arch} {condition} s{s}: L0 {store_fid['l0_exact']:.2f} "
                  f"(counter {rec['l0_archived_counter']:.2f}) alive {store_fid['n_alive']} store content "
                  f"{store_fid['fvu_content']:.4f} held-out content {ho['fvu_content']:.4f}", flush=True)
            del sae
            torch.cuda.empty_cache()
            G.dump(out, f"sec6_fidelity_{a.model}.json")
    out["minutes"] = (time.time() - t0) / 60
    G.dump(out, f"sec6_fidelity_{a.model}.json")


if __name__ == "__main__":
    main()
