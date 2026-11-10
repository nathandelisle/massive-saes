"""Model registry for the v3 experiment suite.

Matches arXiv-v2 Table 8 exactly (model set, target layer = floor(L/2), d_model,
d_sae = 4*d_model). Adding a model = adding an entry here; every script consumes
this registry so protocol drift across experiments is impossible.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str            # short name used in results files
    hf_id: str
    family: str
    n_layers: int
    target_layer: int   # middle residual-stream layer, floor(L/2), 1-indexed block
    d_model: int
    d_sae: int
    block_path: str     # attribute path to the ModuleList of transformer blocks
    dtype: str = "float16"
    needs_hf_token: bool = False
    min_gpu_gb: int = 24


MODELS = {
    "pythia-160m": ModelSpec("pythia-160m", "EleutherAI/pythia-160m", "pythia",
                             12, 6, 768, 3072, "gpt_neox.layers"),
    # SFC (Marks et al.) trained its released dictionaries on this exact model,
    # so the D2 case study uses it rather than substituting pythia-160m.
    "pythia-70m-deduped": ModelSpec("pythia-70m-deduped",
                                    "EleutherAI/pythia-70m-deduped", "pythia",
                                    6, 3, 512, 2048, "gpt_neox.layers"),
    "gpt2": ModelSpec("gpt2", "openai-community/gpt2", "gpt2",
                      12, 6, 768, 3072, "transformer.h"),
    "pythia-1.4b": ModelSpec("pythia-1.4b", "EleutherAI/pythia-1.4b", "pythia",
                             24, 12, 2048, 8192, "gpt_neox.layers"),
    # gemma-2 OVERFLOWS in float16: its logit soft-capping and large
    # intermediate magnitudes exceed fp16 range, producing NaN/garbage CE.
    # bfloat16 is required, and soft-capping needs eager attention.
    "gemma-2-2b": ModelSpec("gemma-2-2b", "google/gemma-2-2b", "gemma",
                            26, 13, 2304, 9216, "model.layers",
                            dtype="bfloat16", needs_hf_token=True,
                            min_gpu_gb=24),
    "qwen-2.5-7b": ModelSpec("qwen-2.5-7b", "Qwen/Qwen2.5-7B", "qwen",
                             28, 14, 3584, 14336, "model.layers", min_gpu_gb=40),
    "pythia-6.9b": ModelSpec("pythia-6.9b", "EleutherAI/pythia-6.9b", "pythia",
                             32, 16, 4096, 16384, "gpt_neox.layers", min_gpu_gb=40),
    "llama-3.1-8b": ModelSpec("llama-3.1-8b", "meta-llama/Llama-3.1-8B", "llama",
                              32, 16, 4096, 16384, "model.layers",
                              needs_hf_token=True, min_gpu_gb=40),
    "gemma-2-9b": ModelSpec("gemma-2-9b", "google/gemma-2-9b", "gemma",
                            42, 21, 3584, 14336, "model.layers",
                            dtype="bfloat16", needs_hf_token=True,
                            min_gpu_gb=48),
    # Cost-gated: needs 2x80GB (or 8x48GB with device_map="auto"). Every script
    # that iterates MODELS excludes this unless --include-70b is passed.
    "llama-3.1-70b": ModelSpec("llama-3.1-70b", "meta-llama/Llama-3.1-70B", "llama",
                               80, 40, 8192, 32768, "model.layers",
                               needs_hf_token=True, min_gpu_gb=160),
}

SEEDS_FULL = [42, 137, 0, 1, 2]     # paper's five seeds; 42 is the reference seed
SEEDS_PAIR = [42, 137]              # legacy two-seed cells being upgraded in exp_a3
REFERENCE_SEED = 42

# Training protocol constants — arXiv-v2 Appendix G. Do not change silently;
# every deviation must be a CLI flag so v2 tables remain reproducible.
TRAIN_TOKENS = 5_000_000
SEQ_LEN = 128
BATCH_SIZE = 4096
STEPS = 20_000
LR = 3e-4
TOPK_K = 64
DICT_MULT = 4
PCA_SAMPLE = 50_000          # first 50k centered activations, float64 (v2 App A)
ALIVE_FIRING_RATE = 1e-3     # v2 definition: firing rate > 0.1% on 100k sample
ALIVE_SAMPLE = 100_000
SHARED_COS_THRESHOLD = 0.7
EVAL_SEQS_IMPORTANCE = 64    # v2 Section 4 (v1 used 256; flag --eval-seqs to vary)
DENSE_COS2_MULT = 5.0        # dense cluster: decoder cos^2 with PC1 > 5x median


# ---------------------------------------------------------------------------
# THE LAYER CONVENTION. Defined here, once, because getting it wrong shifts
# every number by one block while looking completely healthy.
#
# v2 (github.com/nathandelisle/pc1-sae-reproducibility, core/final.py) captured
#     acts = outputs.hidden_states[target_layer + 1]
# and hooked
#     get_layer_module(target_layer).register_forward_hook(...)
# Both are the OUTPUT OF BLOCK INDEX `target_layer`. HF's hidden_states[0] is
# the embedding output, so hidden_states[n + 1] == output of blocks[n].
#
# So for spec.target_layer = 12: block index 12, hidden_states[13].
# An earlier draft of this suite used `blocks[target_layer - 1]`, i.e.
# hidden_states[12] — one block early, for every model.
BLOCK_OFFSET = 0


def block_index(spec, layer=None) -> int:
    """Index into the model's ModuleList of blocks whose OUTPUT we capture."""
    return (spec.target_layer if layer is None else layer) + BLOCK_OFFSET


def hidden_state_index(spec, layer=None) -> int:
    """Equivalent index into a `output_hidden_states=True` tuple."""
    return block_index(spec, layer) + 1


def iter_models(names=None, include_70b=False):
    if names:
        for n in names:
            yield MODELS[n]
        return
    for k, m in MODELS.items():
        if k == "llama-3.1-70b" and not include_70b:
            continue
        yield m
