# Soft MoE & WSP Ablation — turning dense models into domain experts without training

> Ivanna Lab feature (v5.2). Source technique: [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takeyama et al., MIT CSAIL, 2026), validated in a weight-space lab on Qwen3.5-2B, then productized here.

## What this does

Ivanna Lab can turn **any dense model** into a **domain expert** — with **zero gradient training**:

1. **Theme** — you provide 25–30 *domain* prompts (real texts from your domain) and 25–30 *contrast* prompts (household/code/cuisine/law/sport — anything outside the domain), **length-matched** so text length doesn't leak into the direction.
2. **Activation scan** — both sets are run through the dense model; per layer, the *domain direction* is computed as `d̂ = mean(hidden(domain)) − mean(hidden(contrast))`. A hold-out AUC gate (> 0.85) verifies the direction is real, and *causal writers* (which modules in which layers actually write this direction into the residual stream) are identified.
3. **Steer patch** — a rank-one update is baked into the output matrices (`down_proj`/`o_proj`/`out_proj`) of writer-selected layers:
   `W' = W + α·outer(d̂, ī)/‖ī‖²`
   where `ī` is the mean module input on domain prompts. At inference, this shifts the output along `d̂` by ≈ α·‖d̂‖ — amplifying domain behavior. No LoRA, no fine-tuning, no training data curation beyond prompts.
4. **Gates everywhere** — prompt-set validation gate (embedding cross-cosine < 0.5), scan AUC gate, optional eval gate (base vs expert generation on fresh prompts), router accuracy gate. If a gate fails, no patch is applied.
5. **Router** — an embedding-space router (cosine margin between theme and contrast centroids) picks the expert variant instead of the base model at inference time.

The result is a **regular model checkpoint**: chat with it, convert it to GGUF, evaluate it, or publish it to Hugging Face with an auto-generated model card.

## WSP Ablation — removing behavior instead of amplifying it

The same machinery works in reverse. If the theme's domain prompts describe a *behavior you want gone* (e.g., verbose `<think>` reasoning blocks before short answers), the job runs with **goal = ablate**:

`W' = (I − α·d̂d̂ᵀ)W`

The domain direction is *subtracted* from the weights of writer layers — suppressing that behavior at the source. This is the direct descendant of the lab's first experiment (killing a runaway `<think>` habit, ×10 faster inference). Combined with soft MoE it gives a two-step recipe: **ablate what you don't want, amplify what you do**.

## Verified results (Qwen3.5 family, single 4GB GPU)

| Experiment | Base | Scan gate | Patch | Router |
|---|---|---|---|---|
| SMS / aviation (ICAO corpus) | Qwen3.5-4B | **AUC 1.0** (60s) | 22 tensors, layers 19–31 | **100%** |
| HTML authoring (MDN corpus) | Qwen3.5-0.8B | **AUC 1.0** (20s) | auto layers | registered |
| Python language & PEPs | Qwen3.5-0.8B | **AUC 1.0** (18.6s) | auto layers | registered |

Published experts (safetensors, with EN model cards):
- [NikitaLazarev/Qwen3.5-4B-SMS-SoftMoE](https://huggingface.co/NikitaLazarev/Qwen3.5-4B-SMS-SoftMoE)
- [NikitaLazarev/Qwen3.5-0.8B-HTML-SoftMoE](https://huggingface.co/NikitaLazarev/Qwen3.5-0.8B-HTML-SoftMoE)
- [NikitaLazarev/Qwen3.5-0.8B-Python-SoftMoE](https://huggingface.co/NikitaLazarev/Qwen3.5-0.8B-Python-SoftMoE)

## In the UI

The **Experiments & soft MoE** section walks the whole pipeline: create a theme (paste prompt sets) → run the validation gate → build the expert (progress page with per-stage gates, charts: per-layer AUC, causal writers, |d̂| heatmap, |ΔW| per tensor, eval bars) → read the **LLM-generated interpretation** of the run → publish to Hugging Face with one click. Re-running with `goal = ablate` produces the WSP-ablation variant.

## Known limitations (honest)

- Metrics are proxies (lengths, repeats, think-leaks); the final quality decision is a manual spot-check of saved eval texts.
- Validated on one model family (Qwen3.5, hybrid linear attention) and a handful of domains; other architectures work via suffix-matching of output modules, but α/layer defaults should be re-tuned.
- Eval generation for models ≥2.5B on 4GB GPUs is currently skipped (bitsandbytes↔transformers meta-dispatch incompatibility + memory ceiling); experts remain chat-ready (int4) and evaluable through the Evaluation module or chat.
