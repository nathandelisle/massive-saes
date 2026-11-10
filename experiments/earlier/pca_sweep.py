"""
PCA Bypass Experiment: Causal Control
=======================================

Bypass the top-k PCs during SAE training and evaluation. If raw top-50 importance-sharing jumps from ~9% to ~80%+
without any post-hoc exclusion, the dense subspace causally confounds
importance metrics.

Design:
  - Pythia-160M, layer 6 (cheap, fast, validates concept)
  - Sweep k = [0, 1, 2, 5, 10, 20, 50] PCs bypassed
  - For each k: train 2 SAEs on PCA-residual, compute matching + importance
  - PCA bypass hook: dense component passes through exactly, only
    sparse residual goes through SAE
  - Report: FVE, CE overhead, raw top-50/100 shared rate (NO exclusion)
  - Also: match k=10 SAE features against original SAE's clean features

Usage: modal run --detach pca_bypass.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1", "transformers>=4.36", "datasets",
        "numpy", "matplotlib", "scipy", "tqdm",
    )
)

app = modal.App("sae-pca-bypass", image=image)
volume = modal.Volume.from_name("sae-experiment-data", create_if_missing=True)

SAE_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k=64, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
            self.encoder.weight.data = self.decoder.weight.data.T.clone()

    def encode(self, x):
        pre_act = self.encoder(x - self.decoder.bias)
        topk_vals, topk_idx = pre_act.topk(self.k, dim=-1)
        z = torch.zeros_like(pre_act)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

def train_sae(activations, d_model, d_sae, seed, k=64, n_steps=20000,
              batch_size=4096, lr=3e-4, device="cuda"):
    sae = TopKSAE(d_model, d_sae, k=k, seed=seed).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    n = activations.shape[0]
    for step in range(n_steps):
        idx = torch.randint(0, n, (batch_size,))
        batch = activations[idx].to(device)
        x_hat, z = sae(batch)
        loss = F.mse_loss(x_hat, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sae.normalize_decoder()
        if step % 5000 == 0:
            fve = 1 - loss.item() / batch.var().item()
            n_dead = (z.sum(dim=0) == 0).sum().item()
            print(f"    Step {step}: mse={loss.item():.6f} FVE={fve:.4f} dead={n_dead}")
    return sae
'''


@app.function(gpu="H100", timeout=10800, cpu=8, memory=32768, volumes={"/data": volume})
def run_pca_bypass():
    import torch
    import torch.nn.functional as F
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from tqdm import tqdm
    from scipy import stats
    from scipy.stats import binomtest
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json

    exec(SAE_CODE, globals())

    device = "cuda"
    MODEL_NAME = "EleutherAI/pythia-160m"
    TARGET_LAYER = 6
    D_SAE = 4096
    K_SPARSE = 64
    SEEDS = [42, 137]
    K_PCS_SWEEP = [0, 1, 2, 5, 10, 20, 50]

    # ================================================================
    # Phase 1: Load model, collect activations, compute PCA
    # ================================================================
    print("=" * 70)
    print("Phase 1: Setup")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    d_model = model.config.hidden_size

    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    all_activations = []
    tokens_collected = 0
    text_buffer = []

    for item in dataset:
        if item["text"].strip():
            text_buffer.append(item["text"])
        if len(text_buffer) >= 32:
            inputs = tokenizer(
                text_buffer[:32], return_tensors="pt",
                max_length=128, truncation=True, padding="max_length",
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            acts = outputs.hidden_states[TARGET_LAYER + 1]
            mask = inputs["attention_mask"].bool()
            all_activations.append(acts[mask].cpu())
            tokens_collected += acts[mask].shape[0]
            text_buffer = text_buffer[32:]
            if tokens_collected >= 1_000_000:
                break

    activations = torch.cat(all_activations, dim=0)[:1_000_000].float()
    act_mean = activations.mean(dim=0)
    activations_centered = activations - act_mean
    print(f"  Collected {activations.shape[0]:,} activations")

    # Full PCA
    print("  Computing PCA...")
    pca_sample = activations_centered[:50000].numpy()
    cov = (pca_sample.T @ pca_sample) / (len(pca_sample) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx_sorted = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sorted]
    eigenvectors = eigenvectors[:, idx_sorted]  # columns are PCs

    cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
    print(f"  PC1: {eigenvalues[0]/eigenvalues.sum()*100:.1f}%")
    print(f"  Top 5: {cumvar[4]*100:.1f}%")
    print(f"  Top 10: {cumvar[9]*100:.1f}%")
    print(f"  Top 50: {cumvar[49]*100:.1f}%")

    # Prepare eval set
    eval_dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    eval_texts = [item["text"] for item in eval_dataset if item["text"].strip()][:64]
    eval_inputs = tokenizer(
        eval_texts, return_tensors="pt",
        max_length=256, truncation=True, padding="max_length",
    ).to(device)
    eval_labels = eval_inputs["input_ids"].clone()
    eval_labels[eval_inputs["attention_mask"] == 0] = -100

    with torch.no_grad():
        orig_loss = model(**eval_inputs, labels=eval_labels).loss.item()
    print(f"  Original model loss: {orig_loss:.4f}")

    # ================================================================
    # Phase 2: Sweep over k PCs bypassed
    # ================================================================
    results = []

    for k_pcs in K_PCS_SWEEP:
        print(f"\n{'='*70}")
        print(f"k_pcs = {k_pcs} PCs bypassed")
        print(f"{'='*70}")

        # Build projection matrix for top-k PCs
        if k_pcs > 0:
            # P = V_k V_k^T projects onto top-k PC subspace
            V_k = torch.tensor(eigenvectors[:, :k_pcs], dtype=torch.float32).to(device)
            # For training: project out top-k PCs from centered activations
            V_k_cpu = torch.tensor(eigenvectors[:, :k_pcs], dtype=torch.float32)
            projections = activations_centered @ V_k_cpu  # (n, k)
            residuals = activations_centered - projections @ V_k_cpu.T  # (n, d_model)
            var_removed = eigenvalues[:k_pcs].sum() / eigenvalues.sum()
            print(f"  Variance removed: {var_removed*100:.1f}%")
            print(f"  Residual norm: {residuals.norm(dim=-1).mean():.2f} "
                  f"(original: {activations_centered.norm(dim=-1).mean():.2f})")
        else:
            residuals = activations_centered
            V_k = None

        # Train 2 SAEs on residuals
        saes = {}
        for seed in SEEDS:
            print(f"\n  Training SAE seed={seed} on {'residual' if k_pcs > 0 else 'full'} activations...")
            sae = train_sae(residuals, d_model, D_SAE, seed=seed, k=K_SPARSE, device=device)
            saes[seed] = sae

        # Matching
        dec_a = F.normalize(saes[SEEDS[0]].decoder.weight.data.T, dim=1)
        dec_b = F.normalize(saes[SEEDS[1]].decoder.weight.data.T, dim=1)
        cos_sim = dec_a @ dec_b.T
        match_quality = cos_sim.max(dim=1)[0].cpu().numpy()

        # Feature stats
        with torch.no_grad():
            sample = residuals[:100000].to(device)
            z_sample = saes[SEEDS[0]].encode(sample)
            freq = (z_sample > 0).float().mean(dim=0).cpu().numpy()

        alive = freq > 0.001
        n_alive = int(alive.sum())
        baseline_shared = (match_quality[alive] > 0.7).sum() / max(alive.sum(), 1)
        print(f"\n  Alive: {n_alive}/{D_SAE}")
        print(f"  Baseline shared: {baseline_shared*100:.1f}%")

        # Importance with PCA bypass hook
        def compute_loss_bypass(sae, ablate_feature=None):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                x = hidden.float() - act_mean.to(hidden.device)

                if V_k is not None:
                    # Dense component: project onto top-k PCs, pass through exactly
                    x_pc = x @ V_k @ V_k.T  # (batch, seq, d_model)
                    x_resid = x - x_pc
                else:
                    x_pc = torch.zeros_like(x)
                    x_resid = x

                # Sparse component: SAE encode/decode
                z = sae.encode(x_resid)
                if ablate_feature is not None:
                    z[:, :, ablate_feature] = 0.0
                x_resid_hat = sae.decode(z)

                # Recombine: dense (exact) + sparse (SAE) + mean
                reconstructed = act_mean.to(hidden.device) + x_pc + x_resid_hat
                reconstructed = reconstructed.to(hidden.dtype)

                if isinstance(output, tuple):
                    return (reconstructed,) + output[1:]
                return reconstructed

            hook = model.gpt_neox.layers[TARGET_LAYER].register_forward_hook(hook_fn)
            with torch.no_grad():
                loss = model(**eval_inputs, labels=eval_labels).loss.item()
            hook.remove()
            return loss

        baseline_loss = compute_loss_bypass(saes[SEEDS[0]])
        ce_overhead = baseline_loss - orig_loss
        print(f"  CE overhead (with bypass): {ce_overhead:.4f}")

        # Compute FVE on residuals
        with torch.no_grad():
            sample_resid = residuals[:50000].to(device)
            recon, _ = saes[SEEDS[0]](sample_resid)
            fve = 1 - F.mse_loss(recon, sample_resid).item() / sample_resid.var().item()
        print(f"  Residual FVE: {fve:.4f}")

        # Ablation importance (only for alive features)
        alive_indices = np.where(alive)[0]
        importance = np.zeros(D_SAE)
        print(f"  Ablating {len(alive_indices)} features...")
        for feat_idx in tqdm(alive_indices, desc=f"  k={k_pcs}"):
            importance[feat_idx] = compute_loss_bypass(
                saes[SEEDS[0]], ablate_feature=int(feat_idx)
            ) - baseline_loss

        # Raw top-K sharing (NO exclusion!)
        r_raw, p_raw = stats.spearmanr(importance[alive], match_quality[alive])

        top_k_sharing = {}
        for top_k in [20, 50, 100, 200, 500]:
            if top_k > alive.sum():
                continue
            topk_idx = np.argsort(importance[alive])[-top_k:]
            n_shared = int((match_quality[alive][topk_idx] > 0.7).sum())
            rate = n_shared / top_k
            bp = binomtest(n_shared, top_k, baseline_shared, alternative='greater').pvalue
            top_k_sharing[top_k] = {"rate": float(rate), "p": float(bp)}

        t50 = top_k_sharing.get(50, {}).get("rate", 0)
        t100 = top_k_sharing.get(100, {}).get("rate", 0)

        print(f"\n  === RAW RESULTS (no exclusion) ===")
        print(f"  Spearman r: {r_raw:.4f} (p={p_raw:.2e})")
        print(f"  Raw top-50 shared: {t50*100:.1f}% (baseline: {baseline_shared*100:.1f}%)")
        print(f"  Raw top-100 shared: {t100*100:.1f}%")
        for k, v in top_k_sharing.items():
            print(f"    Top-{k}: {v['rate']*100:.1f}% (p={v['p']:.2e})")

        result = {
            "k_pcs": k_pcs,
            "var_removed": float(eigenvalues[:k_pcs].sum() / eigenvalues.sum()) if k_pcs > 0 else 0,
            "n_alive": n_alive,
            "baseline_shared": float(baseline_shared),
            "ce_overhead": float(ce_overhead),
            "residual_fve": float(fve),
            "spearman_r": float(r_raw),
            "top_k_sharing": {str(k): v for k, v in top_k_sharing.items()},
        }
        results.append(result)

    # ================================================================
    # Phase 3: Summary plot
    # ================================================================
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("PCA Bypass: Does Removing Dense Structure Fix Raw Importance?", fontsize=13)

    ks = [r["k_pcs"] for r in results]
    t50s = [r["top_k_sharing"].get("50", {}).get("rate", 0) * 100 for r in results]
    baselines = [r["baseline_shared"] * 100 for r in results]
    overheads = [r["ce_overhead"] for r in results]
    fves = [r["residual_fve"] for r in results]
    rs = [r["spearman_r"] for r in results]

    # Panel 1: Raw top-50 sharing vs k PCs bypassed
    ax = axes[0]
    ax.plot(ks, t50s, "o-", color="steelblue", markersize=8, linewidth=2, label="Raw top-50 shared")
    ax.plot(ks, baselines, "s--", color="gray", markersize=5, alpha=0.5, label="Baseline")
    ax.set_xlabel("PCs bypassed (k)")
    ax.set_ylabel("% features shared across seeds")
    ax.set_title("Raw Top-50 Sharing Rate\n(no post-hoc exclusion)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    # Mark the "no bypass" and "clean" reference points
    ax.axhline(y=86, color="steelblue", ls=":", alpha=0.3, label="Post-hoc clean (86%)")
    ax.annotate("post-hoc\nclean result", xy=(max(ks)*0.7, 87), fontsize=8, color="steelblue", alpha=0.5)

    # Panel 2: CE overhead
    ax = axes[1]
    ax.plot(ks, overheads, "o-", color="salmon", markersize=8, linewidth=2)
    ax.set_xlabel("PCs bypassed (k)")
    ax.set_ylabel("CE overhead (loss increase)")
    ax.set_title("Reconstruction Quality")
    ax2 = ax.twinx()
    ax2.plot(ks, fves, "s--", color="steelblue", markersize=5, alpha=0.7)
    ax2.set_ylabel("Residual FVE", color="steelblue")

    # Panel 3: Spearman r
    ax = axes[2]
    ax.plot(ks, rs, "o-", color="steelblue", markersize=8, linewidth=2)
    ax.set_xlabel("PCs bypassed (k)")
    ax.set_ylabel("Spearman r (importance vs match)")
    ax.set_title("Raw Importance-Consistency\nCorrelation")
    ax.axhline(y=0, color="gray", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("/data/pca_bypass_results.png", dpi=150, bbox_inches="tight")

    # Summary table
    print(f"\n  {'k':>4} {'var_removed':>12} {'CE_overhead':>12} {'FVE':>8} "
          f"{'r':>8} {'top50':>8} {'baseline':>10}")
    print(f"  {'-'*70}")
    for r in results:
        t50_r = r["top_k_sharing"].get("50", {}).get("rate", 0)
        print(f"  {r['k_pcs']:>4} {r['var_removed']*100:>11.1f}% {r['ce_overhead']:>12.4f} "
              f"{r['residual_fve']:>8.4f} {r['spearman_r']:>8.4f} "
              f"{t50_r*100:>7.1f}% {r['baseline_shared']*100:>9.1f}%")

    with open("/data/pca_bypass_results.json", "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()
    print("\n  Saved results")
    print("  DONE")
    return results


@app.local_entrypoint()
def main():
    import json
    results = run_pca_bypass.remote()
    print("\nSummary:")
    for r in results:
        t50 = r["top_k_sharing"].get("50", {}).get("rate", 0)
        print(f"  k={r['k_pcs']:>3}: top50={t50*100:.1f}%, "
              f"overhead={r['ce_overhead']:.4f}, r={r['spearman_r']:.4f}")
