"""Select feature candidates for the original subject--verb agreement experiments."""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

V3 = os.environ.get("PC1_V3", str(pathlib.Path(__file__).resolve().parents[1] / "language" / "pc1_v3"))
HERE = pathlib.Path(__file__).resolve().parent
BUCKET = os.environ.get("S3_BUCKET", "ssup-research-079415246777")
SCREEN = 1500                    # the paper's screen, restored
EVAL_SEQS = 64                   # registry.EVAL_SEQS_IMPORTANCE
PC1_COS2_EXCLUDE = 0.01
ARCHIVE_SRC = "c3-small/results/a5_metric_battery"


def log(*a):
    print("[rank]", *a, flush=True)


def fetch_archive(model, condition, cache="cache_features"):
    p = pathlib.Path(cache)
    p.mkdir(parents=True, exist_ok=True)
    dest = p / ("features_%s_%s.npz" % (model, condition))
    if not dest.exists():
        r = subprocess.run(["aws", "s3", "cp", "s3://%s/%s/%s" % (BUCKET, ARCHIVE_SRC, dest.name),
                            str(dest), "--only-show-errors"], timeout=1200)
        if r.returncode != 0:
            return None
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 0])
    p.add_argument("--out", default="protocol/pools_v2")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--screen", type=int, default=SCREEN)
    p.add_argument("--eval-seqs", type=int, default=EVAL_SEQS)
    p.add_argument("--sizes", nargs="+", type=int, default=[10, 50, 100])
    a = p.parse_args()

    import torch
    sys.path.insert(0, V3)
    sys.path.insert(0, str(HERE))
    from v3.common import registry
    from v3.common.data import load_model_and_tokenizer, eval_sequences
    from v3.common.io_utils import ensure_acts
    from v3.common.pca import estimate_pcs
    from v3.common.saes import load_sae
    from v3.common.fastabl import z_stats, screen_candidates, ablation_over, tuned_chunk
    from checkpoint_utils import config_hash, fetch_sae

    spec = registry.MODELS[a.model]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False

    adir = ensure_acts(spec, device=dev)
    _e, evecs, mu_np = estimate_pcs(adir)
    v1_np = evecs[:, 0]
    torch.cuda.empty_cache()

    model, tok = load_model_and_tokenizer(spec, device=dev)
    model.eval()
    # Independent evaluation text -- NOT task prompts. This is the whole point of a task-independent
    # ranking: the pools must be fixed before any task is seen.
    seqs, am = eval_sequences(tok, n_seqs=a.eval_seqs)
    chunk = tuned_chunk(spec)
    log("%s: %d eval seqs, screen %d, chunk %d" % (a.model, len(seqs), a.screen, chunk))

    mu = torch.tensor(mu_np, dtype=torch.float32, device=dev)
    v1 = torch.tensor(v1_np, dtype=torch.float32, device=dev)

    result = dict(model=a.model, hf_id=spec.hf_id, layer=spec.target_layer, screen=a.screen,
                  eval_seqs=a.eval_seqs, pc1_cos2_exclude=PC1_COS2_EXCLUDE,
                  metric="zero-ablation delta-CE of the SAE-spliced model (ablated minus non-ablated), "
                         "observed tokens, independent eval corpus -- the paper's protocol",
                  seeds={}, archive_check={}, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    for seed in a.seeds:
        result["seeds"][str(seed)] = {}
        for cond in ("standard", "bypass"):
            t0 = time.time()
            cfg = dict(model=spec.key, seed=seed, arch="topk", condition=cond, corpus="wikitext",
                       bos_free=False, layer=spec.target_layer, steps=registry.STEPS,
                       batch=registry.BATCH_SIZE, d_sae=spec.d_sae, k=64.0)
            h = config_hash(cfg)
            sae, _ = load_sae(str(fetch_sae(h, "saes/%s.pt" % h)), device=dev)
            sae.eval()

            st = z_stats(model, spec, seqs, sae, mu, v1=(v1 if cond == "bypass" else None),
                         mode=cond, chunk=chunk, am=am)
            alive = st["rate"] > registry.ALIVE_FIRING_RATE
            cand = screen_candidates(st["ez2"], alive, a.screen)
            log("  seed %d %s: %d alive, %d screened" % (seed, cond, int(alive.sum()), len(cand)))

            imp, base_ce = ablation_over(model, spec, seqs, sae, mu, cand,
                                         v1=(v1 if cond == "bypass" else None), mode=cond,
                                         metric="zero", chunk=chunk, am=am,
                                         on_partial=lambda n, N, o, b: log("    %d/%d" % (n, N)),
                                         partial_every=500)
            ids = np.array(sorted(imp, key=lambda j: -imp[j]), dtype=int)
            scores = np.array([imp[int(j)] for j in ids], dtype=float)

            D = sae.W_dec.detach().float()
            c2 = ((D / D.norm(dim=1, keepdim=True)) @ v1).pow(2).cpu().numpy()

            pools = {}
            for n in a.sizes:
                pools[str(n)] = dict(
                    standard=[int(x) for x in ids[:n]] if cond == "standard" else None,
                    posthoc=([int(x) for x in ids if c2[int(x)] < PC1_COS2_EXCLUDE][:n]
                             if cond == "standard" else None),
                    bypass=[int(x) for x in ids[:n]] if cond == "bypass" else None)
            np.savez_compressed(out / ("rank_%s_%s_s%d.npz" % (a.model, cond, seed)),
                                ranked_ids=ids, delta_ce=scores, ez2=st["ez2"], rate=st["rate"],
                                alive=alive, pc1_cos2=c2, screened=cand)
            result["seeds"][str(seed)][cond] = dict(
                sae_hash=h, n_alive=int(alive.sum()), n_screened=int(len(cand)),
                baseline_spliced_ce=base_ce, seconds=time.time() - t0,
                n_excluded_posthoc=int(((c2 >= PC1_COS2_EXCLUDE) & alive).sum()),
                frac_alive_excluded_posthoc=float(((c2 >= PC1_COS2_EXCLUDE) & alive).sum()
                                                  / max(int(alive.sum()), 1)),
                pools={k: {m: v for m, v in d.items() if v is not None} for k, d in pools.items()},
                top50_scores=[float(x) for x in scores[:50]])
            log("  seed %d %s done in %.1f min (base CE %.4f)"
                % (seed, cond, (time.time() - t0) / 60, base_ce))

            # --- identity check against the archived array, by FEATURE ID ---------------------
            if seed == 42:
                arch = fetch_archive(a.model, cond)
                if arch is not None:
                    z = np.load(arch, allow_pickle=True)
                    a_imp, a_alive = z["imp_zero"], z["alive"]
                    a_score = a_imp.astype(np.float64).copy()
                    a_score[~a_alive] = -np.inf
                    a_top = np.argsort(a_score)[::-1][:50]
                    mine = ids[:50]
                    ov = len(set(int(x) for x in a_top) & set(int(x) for x in mine))
                    # rank correlation over the intersection of measured candidates
                    common = [int(j) for j in ids if np.isfinite(a_score[int(j)])]
                    if len(common) > 10:
                        mr = {int(j): r for r, j in enumerate(ids)}
                        ar = {int(j): r for r, j in enumerate(np.argsort(a_score)[::-1])}
                        xs = np.array([mr[j] for j in common], float)
                        ys = np.array([ar[j] for j in common], float)
                        rho = float(np.corrcoef(np.argsort(np.argsort(xs)),
                                                np.argsort(np.argsort(ys)))[0, 1])
                    else:
                        rho = None
                    result["archive_check"]["%s" % cond] = dict(
                        top50_id_overlap=ov, top50_of=50, spearman_over_common=rho,
                        n_common=len(common),
                        status=("IDENTICAL" if ov == 50 else
                                "CLOSE" if ov >= 40 else "MISMATCH"),
                        note=("Archived array is the source used for the paper's Section 6 pools. "
                              "Overlap is by feature ID, not by a similar sharing rate."))
                    log("  ARCHIVE CHECK %s: %d/50 IDs match, spearman %s -> %s"
                        % (cond, ov, rho, result["archive_check"][cond]["status"]))
            del sae
            torch.cuda.empty_cache()

    (out / ("pools_v2_%s.json" % a.model)).write_text(json.dumps(result, indent=1, default=str))
    log("wrote pools_v2_%s.json" % a.model)


if __name__ == "__main__":
    main()
