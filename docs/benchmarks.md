# Benchmarks (v5.3)

Standard LLM benchmarks via [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(EleutherAI) — the industry-standard evaluation harness. Flow mirrors the model
download UX: **catalog → live search → install dataset → pick a model → run →
results + leaderboard**.

## Architecture

```
Benchmark (catalog row)                      BenchmarkRun (job)
  name = lm-eval task id                       model FK → ModelCheckpoint
  curated (top-15) or lm_eval registry         benchmarks M2M
  install state on the row                     limit / num_fewshot / batch_size
  (EmbeddingModel pattern)                     quantized (auto ≥ 1.5B → nf4)
                                               results JSON (raw report)
                                               summary JSON {task: {metric: v}}
```

- **Engine isolation**: lm-eval lives in `/opt/lmeval-venv` inside the GPU
  image (built by `Dockerfile` step 9.5). The main environment keeps its
  pinned stack (transformers 5.12.1 / accelerate 1.6.0 / bnb 0.49.2) untouched.
  The runner is a subprocess: `/opt/lmeval-venv/bin/python -m lm_eval --model hf
  --tasks … --model_args pretrained=<local_path>,…` — streamed into the run log.
- **Quantization**: models ≥ 1.5B automatically get `load_in_4bit=True,
  bnb_4bit_quant_type=nf4` (4 GB VRAM policy, same as the soft MoE scan).
- **Catalog**: `services/benchmarks/catalog.py` holds the curated top-15
  (MMLU, GSM8K, HellaSwag, ARC-Challenge, WinoGrande, TruthfulQA, IFEval,
  GPQA, BIG-Bench Hard, MATH-algebra, MGSM, HumanEval, TriviaQA, PIQA, BoolQ).
  The full 600+ registry is synced from `TaskManager` introspection by
  `refresh_benchmark_catalog_task` (needs the venv; curated-only without it).
- **Install** = prefetching the benchmark's dataset into the HF datasets cache
  (`install_benchmark_task`, CPU worker), so runs don't start cold.

## Usage

1. **Benchmarks** page (sidebar → Training & evaluation → Benchmarks):
   install datasets for the benchmarks you want (GSM8K is the fastest start).
2. **New run**: pick a downloaded checkpoint + installed benchmarks, set a
   limit (start with 100–300; full MMLU is 14k examples), run.
3. Watch the live log on the run page; when it completes you get a metrics
   table and the raw lm-eval JSON.
4. **Leaderboard**: best score per model × benchmark pair across all runs.

## Tasks

| Task | Queue | Purpose |
|---|---|---|
| `refresh_benchmark_catalog_task` | cpu | sync curated + full registry |
| `install_benchmark_task` | cpu | prefetch dataset for one benchmark |
| `run_benchmark_task` | gpu | execute lm-eval in the isolated venv |

All routed in `config/celery.py` **and** `config/settings.py` (dual routes are
intentional — see repo quirk notes).

## 4GB GPU reality (Qwen3.5 hybrid arch)

- Models **< 1.5B** run in bf16, full GPU — fine on a 4 GB card.
- Models **≥ 1.5B** load through the config-injected nf4 copy
  (`_prepare_quantized_model`): lm-eval 0.4.13 reads bnb settings from the
  model config.json only, and transformers 5.x dropped bare bnb CLI kwargs.
  Skip patterns must be **regex-anchored** (`model\.layers\.\d+\.linear_attn\..*`)
  — plain substrings do not match nested hybrid-net children.
- Even then, a **2B hybrid model does not fit 3.63 GiB usable VRAM** during
  generation (bf16 tails of the gated-delta-net layers + KV cache) — batch
  size does not save it. Same limitation family as eval ≥ 2.5B in soft MoE.
  Run such models on a bigger GPU, or benchmark their GGUF builds with a
  llama.cpp-based runner (phase 2 candidate).

## Pilot results (16-17.11.2026, Qwen3.5-0.8B pair)

- **HumanEval (full 164)**: vanilla 0.2256, python-docs soft-MoE expert 0.2256 — identical. The
  docs-direction does not touch code generation; the patch is provably harmless for general coding.
- **code2text_python (BLEU-4, 150 samples)**: vanilla 1.0519, expert 1.0289 — a −2.2% delta,
  within noise for this sample size. Docstring generation is not measurably shifted by the
  docs-direction patch either.
- **uncheatable_eval_github_python (perplexity)**: OOM on 4 GB even at batch 1 — the task builds
  ~1.7 GiB loglikelihood requests in one chunk. Not feasible on this card; rerun on ≥8 GB.
- Takeaway: standard benchmarks don't surface the soft-MoE docs-direction effect. Demonstrating it
  needs domain-native tasks — the phase-2 custom benchmark builder over the lab's own RAG corpora
  (PEP/docs QA) is the right vehicle.

## Notes & limitations

- First rebuild of the GPU image after this feature downloads the lm-eval
  venv (~2–3 GB extra image size) — the price of full dependency isolation.
- GPQA requires accepting dataset access on the HF Hub (gated repo) with the
  account's HF token before install.
- Full runs of big benchmarks (MMLU, TriviaQA) take hours on a single 4 GB
  GPU — use `--limit` subsets for quick signals.
- Phase 2 (planned): an in-house mini-runner + custom benchmark builder UI
  (`BenchmarkRun.engine='custom'` is already reserved in the schema).
