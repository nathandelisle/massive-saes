import modal
import json

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1", "transformers>=4.36", "datasets",
        "numpy", "scipy", "tqdm", "accelerate",
    )
)

app = modal.App("sae-llama70b", image=image)
volume = modal.Volume.from_name("sae-final-v1", create_if_missing=True)

SEEDS = [42, 137]
N_TOKENS = 5_000_000
N_STEPS = 20_000
K_SPARSE = 64
LR = 3e-4
BATCH_SIZE = 4096
EVAL_SEQS = 64
EVAL_LEN = 128
MODEL_NAME = "meta-llama/Llama-3.1-70B"
TARGET_LAYER = 40


@app.function(
    gpu="H200:2",
    timeout=86400,
    cpu=16,
    memory=262144,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/data": volume},
    image=image,
)
def run():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from tqdm import tqdm
    from scipy import stats
    import gc, os

    safe_name = MODEL_NAME.replace("/", "_")
    save_dir = f"/data/{safe_name}"
    os.makedirs(save_dir, exist_ok=True)

    checkpoints_exist = all(
        os.path.exists(f"{save_dir}/sae_k{k}_seed{s}.pt")
        for k in [0, 1] for s in SEEDS
    )
    pca_exists = os.path.exists(f"{save_dir}/pca.npz")

    print("=" * 70)
    print(f"MODEL: {MODEL_NAME}")
    print(f"Layer: {TARGET_LAYER}")
    print(f"Checkpoints found: {checkpoints_exist}")
    print(f"PCA found: {pca_exists}")
    print("=" * 70)

    class TopKSAE(nn.Module):
        def __init__(self, d_model, d_sae, k=64, seed=42):
            super().__init__()
            torch.manual_seed(seed)
            self.d_model = d_model
            self.d_sae = d_sae
            self.k = k
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

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    d_model = model.config.hidden_size
    d_sae = min(d_model * 4, 32768)
    print(f"d_model={d_model}, d_sae={d_sae}")

    def get_layer_module(layer_idx):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]
        raise ValueError(f"Unknown architecture for {MODEL_NAME}")

    layer_module = get_layer_module(TARGET_LAYER)
    layer_device = next(layer_module.parameters()).device
    print(f"Layer module: {type(layer_module).__name__} on {layer_device}")

    print("\nCollecting activations...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = [t["text"] for t in dataset if t["text"].strip()]

    all_acts = []
    collected = 0
    text_idx = 0

    while collected < N_TOKENS and text_idx < len(texts):
        batch = texts[text_idx:text_idx + 4]
        text_idx += 4
        if not batch:
            break
        inputs = tokenizer(
            batch, return_tensors="pt",
            max_length=128, truncation=True, padding="max_length",
        )
        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        acts = outputs.hidden_states[TARGET_LAYER + 1].float()
        mask = inputs["attention_mask"].to(acts.device).bool()
        all_acts.append(acts[mask].half().cpu())
        collected += acts[mask].shape[0]
        del outputs, acts
        torch.cuda.empty_cache()
        if collected % 500000 < 4 * 128:
            print(f"  {collected:,} tokens...")

    activations = torch.cat(all_acts, dim=0)[:N_TOKENS]
    del all_acts
    gc.collect()
    act_mean = torch.zeros(activations.shape[1], dtype=torch.float32)
    chunk_size = 500000
    for i in range(0, len(activations), chunk_size):
        act_mean += activations[i:i+chunk_size].float().sum(dim=0)
    act_mean /= len(activations)
    activations_centered = activations - act_mean.half()
    del activations
    gc.collect()
    print(f"Collected {activations_centered.shape[0]:,} activations ({activations_centered.nbytes / 1e9:.1f} GB)")

    if pca_exists:
        print("\nLoading PCA from disk...")
        pca_data = np.load(f"{save_dir}/pca.npz")
        eigenvalues = pca_data["eigenvalues"]
        eigenvectors = pca_data["eigenvectors"]
        pc1 = eigenvectors[:, 0]
        pc1_var = float(pca_data["pc1_var"])
    else:
        print("\nComputing PCA...")
        sample_n = min(50000, len(activations_centered))
        sample = activations_centered[:sample_n].numpy().astype(np.float64)
        cov = (sample.T @ sample) / (sample_n - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        pc1 = eigenvectors[:, 0]
        pc1_var = eigenvalues[0] / eigenvalues.sum()
        np.savez(f"{save_dir}/pca.npz",
                 eigenvalues=eigenvalues[:50],
                 eigenvectors=eigenvectors[:, :50],
                 mean=act_mean.numpy(),
                 pc1_var=pc1_var)
    print(f"PC1: {pc1_var*100:.1f}% variance")

    print("\nPreparing eval data...")
    eval_dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    eval_texts = [t["text"] for t in eval_dataset if t["text"].strip()][:EVAL_SEQS]
    eval_inputs = tokenizer(
        eval_texts, return_tensors="pt",
        max_length=EVAL_LEN, truncation=True, padding="max_length",
    )
    input_device = next(model.parameters()).device
    eval_inputs = {k: v.to(input_device) for k, v in eval_inputs.items()}
    eval_labels = eval_inputs["input_ids"].clone()
    eval_labels[eval_inputs["attention_mask"] == 0] = -100

    act_mean_dev = act_mean.to(layer_device)
    pc1_t = torch.tensor(pc1, dtype=torch.float32).to(layer_device)
    eval_batch = 4

    results = {
        "model": MODEL_NAME,
        "layer": TARGET_LAYER,
        "d_model": d_model,
        "d_sae": d_sae,
        "pc1_var": float(pc1_var),
        "n_tokens": N_TOKENS,
        "n_steps": N_STEPS,
        "n_eval_seqs": len(eval_texts),
    }

    for k_pcs in [0, 1]:
        print(f"\n{'='*70}")
        print(f"CONDITION k={k_pcs}")
        print(f"{'='*70}")

        if k_pcs > 0:
            V_k = torch.tensor(eigenvectors[:, :k_pcs], dtype=torch.float16)
            V_k_dev = torch.tensor(eigenvectors[:, :k_pcs], dtype=torch.float32).to(layer_device)
            proj = activations_centered @ V_k
            train_data = activations_centered - proj @ V_k.T
            del proj
            var_removed = eigenvalues[:k_pcs].sum() / eigenvalues.sum()
            print(f"Variance removed: {var_removed*100:.1f}%")
        else:
            V_k_dev = None
            train_data = activations_centered

        checkpoint_a = f"{save_dir}/sae_k{k_pcs}_seed{SEEDS[0]}.pt"
        checkpoint_b = f"{save_dir}/sae_k{k_pcs}_seed{SEEDS[1]}.pt"

        if os.path.exists(checkpoint_a) and os.path.exists(checkpoint_b):
            print(f"\nLoading SAE checkpoints from disk...")
            saes = {}
            for seed in SEEDS:
                sae = TopKSAE(d_model, d_sae, k=K_SPARSE, seed=seed).to(layer_device)
                sae.load_state_dict(torch.load(f"{save_dir}/sae_k{k_pcs}_seed{seed}.pt", map_location=layer_device))
                saes[seed] = sae
                print(f"  Loaded seed {seed}")
        else:
            saes = {}
            for seed in SEEDS:
                print(f"\nTraining SAE seed {seed}...")
                sae = TopKSAE(d_model, d_sae, k=K_SPARSE, seed=seed).to(layer_device)
                optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
                n = train_data.shape[0]
                for step in range(N_STEPS):
                    batch_idx = torch.randint(0, n, (BATCH_SIZE,))
                    batch = train_data[batch_idx].to(layer_device).float()
                    x_hat, z = sae(batch)
                    loss = F.mse_loss(x_hat, batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    sae.normalize_decoder()
                    if step % 5000 == 0:
                        fve = 1 - loss.item() / batch.var().item()
                        print(f"  Step {step}: mse={loss.item():.6f} FVE={fve:.4f}")
                saes[seed] = sae
                torch.save(sae.state_dict(), f"{save_dir}/sae_k{k_pcs}_seed{seed}.pt")
            volume.commit()

        ref_sae = saes[SEEDS[0]]

        print("\nComputing feature stats...")
        with torch.no_grad():
            freq_chunks = []
            for i in range(0, min(100000, len(train_data)), 10000):
                chunk = train_data[i:i+10000].to(layer_device).float()
                freq_chunks.append((ref_sae.encode(chunk) > 0).float().cpu())
            freq = torch.cat(freq_chunks, dim=0).mean(dim=0).numpy()
        alive = freq > 0.001
        n_alive = int(alive.sum())
        print(f"Alive: {n_alive}/{d_sae}")

        dec_ref = F.normalize(ref_sae.decoder.weight.data.T, dim=1)
        pc1_cos2 = ((dec_ref @ pc1_t).cpu().numpy()) ** 2

        print("\nComputing matching...")
        match_results = {}
        for other_seed in SEEDS[1:]:
            dec_other = F.normalize(saes[other_seed].decoder.weight.data.T, dim=1)
            chunk = 2048
            max_cos = np.zeros(d_sae)
            for i in range(0, d_sae, chunk):
                end = min(i + chunk, d_sae)
                sim = (dec_ref[i:end] @ dec_other.T).max(dim=1)[0].cpu().numpy()
                max_cos[i:end] = sim
            shared_rate = (max_cos[alive] > 0.7).sum() / max(alive.sum(), 1)
            match_results[other_seed] = {
                "max_cos": max_cos,
                "shared_rate": float(shared_rate),
            }
            print(f"  vs seed {other_seed}: {shared_rate*100:.1f}% shared")

        avg_match = np.mean([m["max_cos"] for m in match_results.values()], axis=0)
        baseline_shared = np.mean([m["shared_rate"] for m in match_results.values()])
        print(f"Baseline shared: {baseline_shared*100:.1f}%")

        alive_idx = np.where(alive)[0]
        print(f"\nFull ablation: {len(alive_idx)} features")

        def compute_loss_batched(ablate_feature=None):
            total_loss = 0.0
            total_tokens = 0
            for bi in range(0, EVAL_SEQS, eval_batch):
                be = min(bi + eval_batch, EVAL_SEQS)
                batch_inputs = {k: v[bi:be] for k, v in eval_inputs.items()}
                batch_labels = eval_labels[bi:be]

                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    x = hidden.float() - act_mean_dev
                    if V_k_dev is not None:
                        pc_comp = x @ V_k_dev @ V_k_dev.T
                        x_resid = x - pc_comp
                    else:
                        pc_comp = None
                        x_resid = x
                    z = ref_sae.encode(x_resid)
                    if ablate_feature is not None:
                        z[:, :, ablate_feature] = 0.0
                    x_hat = ref_sae.decode(z)
                    if pc_comp is not None:
                        x_hat = x_hat + pc_comp
                    reconstructed = (x_hat + act_mean_dev).to(hidden.dtype)
                    if isinstance(output, tuple):
                        return (reconstructed,) + output[1:]
                    return reconstructed

                hook = get_layer_module(TARGET_LAYER).register_forward_hook(hook_fn)
                with torch.no_grad():
                    loss = model(**batch_inputs, labels=batch_labels).loss
                hook.remove()
                n_tokens = (batch_labels != -100).sum().item()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens
            return total_loss / total_tokens

        baseline_loss = compute_loss_batched()
        print(f"SAE baseline loss: {baseline_loss:.4f}")

        importance = np.zeros(d_sae)
        for fi in tqdm(alive_idx, desc=f"k={k_pcs} ablation"):
            importance[fi] = compute_loss_batched(int(fi)) - baseline_loss

        print("\nComputing statistics...")
        median_cos2 = np.median(pc1_cos2[alive])
        clean = alive & (pc1_cos2 <= 5 * median_cos2)

        ablated_mask = np.zeros(d_sae, dtype=bool)
        ablated_mask[alive_idx] = True
        ablated_and_alive = alive & ablated_mask

        per_pair_top50 = []
        per_pair_top50_clean = []
        per_pair_r = []
        for other_seed in SEEDS[1:]:
            mq = match_results[other_seed]["max_cos"]

            ranked = np.where(ablated_and_alive)[0]
            top50_feat = ranked[np.argsort(importance[ranked])[-min(50, len(ranked)):]]
            t50 = (mq[top50_feat] > 0.7).sum() / len(top50_feat)
            per_pair_top50.append(float(t50))

            clean_and_ablated = clean & ablated_mask
            if clean_and_ablated.sum() >= 50:
                ranked_clean = np.where(clean_and_ablated)[0]
                top50_clean = ranked_clean[np.argsort(importance[ranked_clean])[-50:]]
                t50_clean = (mq[top50_clean] > 0.7).sum() / 50
                per_pair_top50_clean.append(float(t50_clean))

            if ablated_and_alive.sum() > 10:
                r, _ = stats.spearmanr(importance[ablated_and_alive], mq[ablated_and_alive])
                per_pair_r.append(float(r))

        topk_raw = {}
        topk_clean = {}
        for kv in [20, 50, 100, 200, 500]:
            ranked = np.where(ablated_and_alive)[0]
            if kv > len(ranked):
                continue
            ti = ranked[np.argsort(importance[ranked])[-kv:]]
            topk_raw[str(kv)] = float((avg_match[ti] > 0.7).sum() / kv)

            clean_and_ablated = clean & ablated_mask
            if clean_and_ablated.sum() >= kv:
                ci = np.where(clean_and_ablated)[0]
                ci = ci[np.argsort(importance[ci])[-kv:]]
                topk_clean[str(kv)] = float((avg_match[ci] > 0.7).sum() / kv)

        condition_results = {
            "n_alive": n_alive,
            "n_ablated": int(ablated_mask.sum()),
            "baseline_shared_mean": float(baseline_shared),
            "raw_top50_mean": float(np.mean(per_pair_top50)),
            "raw_top50_per_pair": per_pair_top50,
            "clean_top50_mean": float(np.mean(per_pair_top50_clean)) if per_pair_top50_clean else None,
            "spearman_r_mean": float(np.mean(per_pair_r)) if per_pair_r else None,
            "topk_raw": topk_raw,
            "topk_clean": topk_clean,
            "ce_baseline": float(baseline_loss),
            "n_excluded": int(alive.sum() - clean.sum()),
        }
        results[f"k{k_pcs}"] = condition_results

        print(f"\n--- RESULTS k={k_pcs} ---")
        print(f"Baseline shared: {baseline_shared*100:.1f}%")
        print(f"Raw top-50: {np.mean(per_pair_top50)*100:.1f}%")
        if per_pair_top50_clean:
            print(f"Clean top-50: {np.mean(per_pair_top50_clean)*100:.1f}%")
        if per_pair_r:
            print(f"Spearman r: {np.mean(per_pair_r):.4f}")
        print(f"Top-k raw: {topk_raw}")
        print(f"CE baseline: {baseline_loss:.4f}")

        np.savez(f"{save_dir}/k{k_pcs}_features.npz",
                 importance=importance, avg_match=avg_match,
                 freq=freq, pc1_cos2=pc1_cos2)
        volume.commit()

    with open(f"{save_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    print(f"\n{'='*70}")
    print(f"SUMMARY: {MODEL_NAME}")
    print(f"{'='*70}")
    print(f"PC1: {pc1_var*100:.1f}%")
    print(f"k=0 raw top-50: {results['k0']['raw_top50_mean']*100:.1f}%")
    print(f"k=1 raw top-50: {results['k1']['raw_top50_mean']*100:.1f}%")
    print(f"Lift: +{(results['k1']['raw_top50_mean'] - results['k0']['raw_top50_mean'])*100:.0f}pp")
    print("DONE")


@app.local_entrypoint()
def main():
    run.remote()
