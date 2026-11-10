# License for results and model weights

Numerical results (`results/**`, `index/**`) and the released SAE weights and preprocessing assets (Hugging Face
repositories listed in `RELEASE_MANIFEST.json`) are licensed under the
**Creative Commons Attribution 4.0 International License (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/
Please cite the paper when using them. Code (`code/**`, `verify/**`) is under the MIT License (`LICENSE`).

Third-party terms still apply to the base models whose activations these SAEs were trained on:
Llama 3.1 (Llama 3.1 Community License) for `llama-3.1-*` SAEs, and Gemma (Gemma Terms of Use) for `gemma-2-*` SAEs;
Pythia and Qwen2.5 are Apache-2.0 and GPT-2 is MIT. Evaluation datasets (WikiText-103, SST-2, AG News, Amazon Polarity,
dair-ai/emotion) remain under their own licenses; only labels/derived statistics are included here.
