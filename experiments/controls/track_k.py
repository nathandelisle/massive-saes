"""Measure PC1 variance after excluding first positions and special tokens."""
import argparse
import csv
import pathlib
import time

import numpy as np

from rcommon import add_v3, log, write_json

SITES = {
    "gpt2": dict(resid="transformer.h", attn="transformer.h.{i}.attn",
                 mlp="transformer.h.{i}.mlp"),
    "pythia": dict(resid="gpt_neox.layers", attn="gpt_neox.layers.{i}.attention",
                   mlp="gpt_neox.layers.{i}.mlp"),
    "qwen": dict(resid="model.layers", attn="model.layers.{i}.self_attn",
                 mlp="model.layers.{i}.mlp"),
    "llama": dict(resid="model.layers", attn="model.layers.{i}.self_attn",
                  mlp="model.layers.{i}.mlp"),
    "gemma": dict(resid="model.layers", attn="model.layers.{i}.self_attn",
                  mlp="model.layers.{i}.mlp"),
}


def getmod(model, path):
    m = model
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def stats(X, first_mask, special_mask, massive_q=0.99):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(0)
    Xc = X - mean[None, :]
    C = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    lam = np.linalg.eigvalsh(C)[::-1]
    n = np.linalg.norm(X, axis=1)
    nc = np.linalg.norm(Xc, axis=1)
    # uncentered energy on the leading uncentered direction
    G = (X.T @ X) / max(len(X), 1)
    ulam = np.linalg.eigvalsh(G)[::-1]
    ev = lam / lam.sum()
    thr = np.quantile(n, massive_q)
    _, evecs = np.linalg.eigh(C)
    v1 = evecs[:, -1]
    coef = Xc @ v1
    return dict(
        n_rows=int(len(X)), d=int(X.shape[1]),
        w1_centered=float(ev[0]), w2_centered=float(ev[1]),
        uncentered_energy_frac=float(ulam[0] / ulam.sum()),
        effective_rank=float(np.exp(-(ev[ev > 0] * np.log(ev[ev > 0])).sum())),
        participation_ratio=float((lam.sum() ** 2) / (lam ** 2).sum()),
        norm_p50=float(np.quantile(n, .5)), norm_p90=float(np.quantile(n, .9)),
        norm_p99=float(np.quantile(n, .99)), norm_max=float(n.max()),
        max_over_median_norm=float(n.max() / max(np.quantile(n, .5), 1e-12)),
        centered_norm_p50=float(np.quantile(nc, .5)),
        massive_threshold_q99=float(thr),
        frac_first=float(first_mask.mean()), frac_special=float(special_mask.mean()),
        frac_massive=float((n > thr).mean()),
        pc1_coef_p01=float(np.quantile(coef, .01)), pc1_coef_p50=float(np.quantile(coef, .5)),
        pc1_coef_p99=float(np.quantile(coef, .99)),
        pc1_coef_absmax=float(np.abs(coef).max()),
        cos_pc1_mean_vector=float(abs(v1 @ (mean / (np.linalg.norm(mean) + 1e-12)))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="results/track_k")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-seqs", type=int, default=128)
    p.add_argument("--max-rows", type=int, default=60000)
    a = p.parse_args()

    import torch
    add_v3()
    from v3.common import registry
    from v3.common.data import load_model_and_tokenizer, get_blocks, corpus_texts
    from v3.common.registry import block_index

    spec = registry.MODELS[a.model]
    dev = a.device
    out = pathlib.Path(a.out)
    model, tok = load_model_and_tokenizer(spec, device=dev)
    model.eval()
    fam = SITES[spec.family]
    li = block_index(spec)
    rows = []
    res = dict(model=a.model, family=spec.family, layer=spec.target_layer,
               block_index=li, n_seqs=a.n_seqs,
               scope=("geometry only; no SAE is trained or evaluated here, so nothing "
                      "in this file speaks to a feature-ranking inversion at an "
                      "untrained site"),
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    texts = [t for t in corpus_texts("wikitext", "validation") if t.strip()][:a.n_seqs * 4]
    sp_ids = set(int(i) for i in (tok.all_special_ids or []))
    adds_bos = bool(getattr(tok, "add_bos_token", False)) or (
        len(tok("hello")["input_ids"]) > len(tok("hello", add_special_tokens=False)["input_ids"]))
    res["tokenizer_adds_bos"] = adds_bos

    for ctx in (128, 512):
        for bos in ((True, False) if adds_bos else (True,)):
            enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True,
                      max_length=ctx, add_special_tokens=bos)
            ids, amask = enc["input_ids"][:a.n_seqs], enc["attention_mask"][:a.n_seqs]
            grab = {}
            hooks = []

            def mk(name):
                def h(m, i, o):
                    t = o[0] if isinstance(o, tuple) else o
                    grab.setdefault(name, []).append(t.detach().float().cpu().numpy())
                return h
            blocks = get_blocks(model, spec.block_path)
            hooks.append(blocks[li].register_forward_hook(mk("resid")))
            for nm in ("attn", "mlp"):
                try:
                    hooks.append(getmod(model, fam[nm].format(i=li))
                                 .register_forward_hook(mk(nm)))
                except Exception as e:
                    res.setdefault("site_errors", {})[nm] = str(e)
            with torch.no_grad():
                for i in range(0, ids.shape[0], 8):
                    model(ids[i:i + 8].to(dev), attention_mask=amask[i:i + 8].to(dev))
            for h in hooks:
                h.remove()

            real = amask.numpy().astype(bool)
            firstm = np.zeros(ids.shape, bool)
            firstm[:, 0] = True
            spm = np.isin(ids.numpy(), list(sp_ids)) if sp_ids else np.zeros(ids.shape, bool)
            for site, chunks in grab.items():
                X = np.concatenate([c.reshape(-1, c.shape[-1]) for c in chunks], 0)
                keep = real.reshape(-1)
                if X.shape[0] != keep.shape[0]:
                    keep = keep[:X.shape[0]]
                views = dict(
                    all=keep,
                    no_special=keep & ~spm.reshape(-1)[:len(keep)] & ~firstm.reshape(-1)[:len(keep)],
                    after_first=keep & ~firstm.reshape(-1)[:len(keep)])
                for vname, m in views.items():
                    idx = np.where(m)[0]
                    if len(idx) > a.max_rows:
                        idx = np.random.default_rng(0).choice(idx, a.max_rows, replace=False)
                    if len(idx) < 512:
                        continue
                    st = stats(X[idx], firstm.reshape(-1)[idx], spm.reshape(-1)[idx])
                    rows.append(dict(model=a.model, site=site, context_length=ctx,
                                     bos_prepended=bos, view=vname,
                                     comparison_kind=("row exclusion" if vname != "all"
                                                      else "none"), **st))
                    log("trkK", "%s ctx%d bos=%s %-11s %-11s w1=%.4f eff_rank=%.1f "
                                "max/med=%.1f"
                        % (a.model, ctx, bos, site, vname, st["w1_centered"],
                           st["effective_rank"], st["max_over_median_norm"]))
            del grab

    out.mkdir(parents=True, exist_ok=True)
    f = out / ("track_k_%s.csv" % a.model)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with f.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    res["n_rows"] = len(rows)
    write_json(out / ("track_k_%s.json" % a.model), res)
    log("trkK", "DONE %s: %d rows" % (a.model, len(rows)))


if __name__ == "__main__":
    main()
