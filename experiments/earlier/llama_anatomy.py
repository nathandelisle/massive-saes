"""
llama_anatomy.py — settles WHY Llama escapes the inversion despite 96%+ PC1.

Hypothesis (v2, after localization_check): Llama's PC1 tiles are just as
non-identifiable as Qwen's, but their OOD damage is CONCENTRATED AT THE BOS
POSITION, so average-loss importance is small and they never invade the
top-50. Qwen's tiles damage every token and flood the ranking.

Per model this script measures, for the PC1-aligned cloud (|cos|>TAU) and,
for reference, the top-50-by-importance set:

  A. IMPORTANCE PLACEMENT
     - median importance percentile of cloud atoms (within alive dict)
     - fraction of cloud atoms in the top-50 / top-1% of the ranking
     - fraction of the model's ACTUAL top-50 that is cloud (the invasion rate)

  B. LOCALIZATION OF THE SIGNAL
     - firing rate quartiles of cloud atoms
     - BOS mass: fraction of each cloud atom's total activation mass on
       position 0 (median across cloud). Llama-pred: ~1. Qwen-pred: ~1/128.

  C. LOCALIZATION OF THE DAMAGE (the causal core)
     - ablate the single highest-importance cloud atom, measure the loss
       increase PER POSITION: report loss delta at position 0 vs. mean over
       positions 1..127. Llama-pred: damage at pos 0 only. Qwen-pred: flat.
     - same for a top-50 semantic (non-cloud) feature as control.

  D. IDENTIFIABILITY (replicates localization_check, kept for one JSON)
     - shared>0.7 fraction and median cross-seed cos of cloud atoms.

Models: the two poles (Llama-8B, Qwen-7B) + Pythia-1.4B as the delimiter case.
Run:  modal run llama_anatomy.py            (~10-15 min/model, A100-80GB)
Out:  printed per-model block + /data/final_out/{safe}_anatomy.json
"""

import json
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.1", "transformers>=4.36", "datasets", "numpy", "accelerate")
)
app = modal.App("sae-llama-anatomy", image=image)
volume = modal.Volume.from_name("sae-final-v1", create_if_missing=True)
HF = modal.Secret.from_name("huggingface")
OUT = "/data/final_out"

SEED_A, SEED_B = 42, 137
TAU = 0.3
SEQ_LEN = 128
N_TOKENS_V1 = 200_000
N_EVAL_SEQ = 96          # sequences for per-position damage measurement

MODELS = [
    {"name": "meta-llama/Llama-3.1-8B", "layer": 16},
    {"name": "Qwen/Qwen2.5-7B",         "layer": 14},
    {"name": "EleutherAI/pythia-1.4b",  "layer": 12},
]


def _topk_sae():
    import torch, torch.nn as nn, torch.nn.functional as F
    class TopKSAE(nn.Module):
        def __init__(self, d_model, d_sae, k=64, seed=42):
            super().__init__(); torch.manual_seed(seed)
            self.encoder = nn.Linear(d_model, d_sae, bias=True)
            self.decoder = nn.Linear(d_sae, d_model, bias=True)
        def encode(self, x):
            import torch
            pre = self.encoder(x - self.decoder.bias)
            vals, idx = pre.topk(64, dim=-1)
            z = torch.zeros_like(pre); z.scatter_(-1, idx, torch.relu(vals)); return z
        def decode(self, z): return self.decoder(z)
    return TopKSAE


def _layer(model, name, i):
    if hasattr(model, "gpt_neox"): return model.gpt_neox.layers[i]
    if hasattr(model, "model") and hasattr(model.model, "layers"): return model.model.layers[i]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"): return model.transformer.h[i]
    raise ValueError(name)


@app.function(gpu="A100-80GB", timeout=7200, memory=131_072, cpu=8,
              secrets=[HF], volumes={"/data": volume})
def anatomy(cfg):
    import os, torch, numpy as np
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    TopK = _topk_sae()

    name, layer = cfg["name"], cfg["layer"]; safe = name.replace("/", "_")
    sd = f"/data/{safe}"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
    model.eval()
    d_model = model.config.hidden_size; d_sae = min(d_model * 4, 32768)
    lyr = _layer(model, name, layer); dev = next(lyr.parameters()).device

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    texts = [t["text"] for t in ds if len(t["text"].strip()) > 200]

    # ---------- v1 ----------
    acts, got, ti, cap = [], 0, 0, {}
    cb = 8 if d_model <= 2560 else 4
    def cap_hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o; cap["h"] = h.detach()
    hd = lyr.register_forward_hook(cap_hook)
    while got < N_TOKENS_V1 and ti < len(texts):
        b = texts[ti:ti+cb]; ti += cb
        enc = tok(b, return_tensors="pt", max_length=SEQ_LEN,
                  truncation=True, padding="max_length").to(dev)
        with torch.no_grad(): model(**enc)
        m = enc["attention_mask"].bool()
        acts.append(cap["h"][m].float().cpu()); got += int(m.sum())
    hd.remove()
    X = torch.cat(acts, 0)[:N_TOKENS_V1].double(); X = X - X.mean(0)
    w, V = torch.linalg.eigh((X.T @ X) / (len(X) - 1))
    v1 = V[:, -1].float().to(dev)
    del X, acts; torch.cuda.empty_cache()

    # ---------- SAEs, features, cloud ----------
    feat = np.load(f"{sd}/k0_features.npz")
    imp, freq = feat["importance"], feat["freq"]
    alive = freq > 0.001
    def load(seed):
        s = TopK(d_model, d_sae).to(dev)
        s.load_state_dict(torch.load(f"{sd}/sae_k0_seed{seed}.pt", map_location=dev))
        s.eval(); return s
    saeA, saeB = load(SEED_A), load(SEED_B)
    decA = F.normalize(saeA.decoder.weight.data.T, dim=1)
    decB = F.normalize(saeB.decoder.weight.data.T, dim=1)
    best_cos = (decA @ decB.T).max(dim=1).values.cpu().numpy()
    pc1_cos = (decA @ v1).abs().cpu().numpy()
    cloud = alive & (pc1_cos > TAU)
    cidx = np.where(cloud)[0]
    aidx = np.where(alive)[0]
    top50 = set(aidx[np.argsort(imp[aidx])[-50:]].tolist())

    # A. importance placement
    ranks = {int(f): float((imp[aidx] < imp[f]).mean()) for f in cidx}
    med_pct = float(np.median(list(ranks.values()))) if len(cidx) else float("nan")
    in_top50 = float(np.mean([f in top50 for f in cidx])) if len(cidx) else 0.0
    p99 = np.percentile(imp[aidx], 99)
    in_p99 = float(np.mean(imp[cidx] >= p99)) if len(cidx) else 0.0
    invasion = float(np.mean([f in set(cidx.tolist()) for f in top50]))

    # B. localization: BOS mass of cloud activations on eval sequences
    enc = tok(texts[2000:2000+N_EVAL_SEQ], return_tensors="pt", max_length=SEQ_LEN,
              truncation=True, padding="max_length").to(dev)
    hd = lyr.register_forward_hook(cap_hook)
    with torch.no_grad(): model(**enc)
    hd.remove()
    H = cap["h"].float()                                  # [B, T, d]
    with torch.no_grad():
        Z = saeA.encode(H.reshape(-1, d_model)).reshape(H.shape[0], H.shape[1], -1)
    if len(cidx):
        zc = Z[:, :, cidx]                                # [B, T, n_cloud]
        mass = zc.abs().sum(dim=(0, 1)) + 1e-9
        bos_mass = zc[:, 0, :].abs().sum(dim=0) / mass    # frac at position 0
        med_bos_mass = float(bos_mass.median().item())
    else:
        med_bos_mass = float("nan")

    # C. per-position damage: ablate strongest cloud atom vs a semantic control
    def spliced_loss_per_pos(ablate_feat=None):
        """Run with SAE spliced at `layer`; return per-position CE loss [T-1]."""
        def splice(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            hf = h.float()
            z = saeA.encode(hf.reshape(-1, d_model))
            if ablate_feat is not None:
                z[:, ablate_feat] = 0.0
            rec = saeA.decode(z).reshape(hf.shape).to(h.dtype)
            return (rec,) + o[1:] if isinstance(o, tuple) else rec
        hk = lyr.register_forward_hook(splice)
        with torch.no_grad():
            logits = model(**enc).logits.float()
        hk.remove()
        ids = enc["input_ids"]; msk = enc["attention_mask"]
        lp = F.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)      # [B, T-1]
        valid = (msk[:, 1:] > 0).float()
        per_pos = (nll * valid).sum(0) / valid.sum(0).clamp(min=1)  # [T-1]
        return per_pos.cpu().numpy()

    base = spliced_loss_per_pos(None)
    res_damage = {}
    if len(cidx):
        star = int(cidx[np.argmax(imp[cidx])])            # strongest cloud atom
        d_cloud = spliced_loss_per_pos(star) - base
        res_damage["cloud_feat"] = star
        res_damage["cloud_delta_pos0"] = float(d_cloud[0])
        res_damage["cloud_delta_rest_mean"] = float(d_cloud[1:].mean())
        res_damage["cloud_ratio_pos0_over_rest"] = float(
            d_cloud[0] / (abs(d_cloud[1:].mean()) + 1e-6))
    sem_candidates = [f for f in sorted(top50, key=lambda f: -imp[f]) if f not in set(cidx.tolist())]
    if sem_candidates:
        ctrl = int(sem_candidates[0])
        d_sem = spliced_loss_per_pos(ctrl) - base
        res_damage["sem_feat"] = ctrl
        res_damage["sem_delta_pos0"] = float(d_sem[0])
        res_damage["sem_delta_rest_mean"] = float(d_sem[1:].mean())

    out = {
        "model": name, "tau": TAU, "n_cloud": int(len(cidx)),
        "A_importance": {"median_percentile": med_pct, "frac_in_top50": in_top50,
                          "frac_in_top1pct": in_p99, "top50_invasion_by_cloud": invasion},
        "B_localization": {"rate_quartiles": [float(np.percentile(freq[cidx], q))
                                              for q in (25, 50, 75)] if len(cidx) else [],
                            "median_bos_mass_frac": med_bos_mass},
        "C_damage": res_damage,
        "D_identifiability": {"shared_frac": float((best_cos[cidx] > 0.7).mean()) if len(cidx) else None,
                               "median_cross_seed_cos": float(np.median(best_cos[cidx])) if len(cidx) else None},
    }
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{safe}_anatomy.json", "w") as f: json.dump(out, f, indent=2)
    volume.commit()

    print("=" * 76)
    print(f"MODEL {name}   cloud = |cos(dec, v1)| > {TAU}:  n={len(cidx)}")
    print(f"A. importance: median pct={med_pct:.2f}  in-top50={in_top50:.0%}  "
          f"in-top1%={in_p99:.0%}  | top-50 invaded by cloud: {invasion:.0%}")
    if len(cidx):
        q = out["B_localization"]["rate_quartiles"]
        print(f"B. firing rate [{q[0]:.4f},{q[1]:.4f},{q[2]:.4f}]  "
              f"BOS mass frac (median): {med_bos_mass:.2f}   (1/T={1/SEQ_LEN:.4f})")
    if "cloud_delta_pos0" in res_damage:
        print(f"C. ablate strongest cloud atom #{res_damage['cloud_feat']}: "
              f"dLoss pos0={res_damage['cloud_delta_pos0']:+.3f}  "
              f"rest={res_damage['cloud_delta_rest_mean']:+.3f}  "
              f"ratio={res_damage['cloud_ratio_pos0_over_rest']:.1f}x")
    if "sem_delta_pos0" in res_damage:
        print(f"   control semantic #{res_damage['sem_feat']}: "
              f"pos0={res_damage['sem_delta_pos0']:+.3f}  rest={res_damage['sem_delta_rest_mean']:+.3f}")
    print(f"D. identifiability: shared>0.7={out['D_identifiability']['shared_frac']}  "
          f"med cos={out['D_identifiability']['median_cross_seed_cos']}")
    print("=" * 76)
    return out


@app.local_entrypoint()
def main():
    hs = [(c["name"], anatomy.spawn(c)) for c in MODELS]
    print(f"Spawned {len(hs)}.")
    for nm, h in hs:
        try: h.get(); print("done:", nm)
        except KeyboardInterrupt: print("detached; jobs continue"); break
        except Exception as e: print(nm, ":", e)
