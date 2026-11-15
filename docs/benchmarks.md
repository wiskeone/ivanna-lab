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

## Notes & limitations

- First rebuild of the GPU image after this feature downloads the lm-eval
  venv (~2–3 GB extra image size) — the price of full dependency isolation.
- GPQA requires accepting dataset access on the HF Hub (gated repo) with the
  account's HF token before install.
- Full runs of big benchmarks (MMLU, TriviaQA) take hours on a single 4 GB
  GPU — use `--limit` subsets for quick signals.
- Phase 2 (planned): an in-house mini-runner + custom benchmark builder UI
  (`BenchmarkRun.engine='custom'` is already reserved in the schema).
