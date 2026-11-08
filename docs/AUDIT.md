# Ivanna Lab — product & technical audit (August 2026)

An honest look at what this platform does, who it is for, and where it stands.

## What this project is

A self-hosted web platform that covers the **full "make my own model" loop** in one place:

```
HF Hub / local weights ──► registry & experiments ──► inference / chat ──► comparison
        │                                                            │
        ▼                                                            ▼
  RAG collections (docs → chunks → embeddings → hybrid search) ── RAG-grounded chat
        │
        ▼
  distillation: docs → chunk groups → generated questions → teacher answers → dataset
        │
        ▼
  fine-tuning (instruction / RAFT / document trainers) ──► checkpoints & metrics
        │
        ▼
  merge (mergekit: linear/SLERP/TIES/DARE) ──► quantization ──► GGUF export ──► back to inference
```

Everything runs in Docker (Django 5 + Celery GPU/CPU workers + Postgres + Redis), with a task queue, progress tracking and a monitoring stack.

## Needs it covers (and for whom)

| Need | Covered? | Notes |
|---|---|---|
| "I have domain PDFs and want a model that knows them" | ✅ | The distillation pipeline is the differentiator: documents → auto-generated questions per chunk-group → teacher answers → instruction dataset → fine-tune. No manual dataset writing required. |
| "Train with retrieved context, not just raw text" | ✅ | **RAFT trainer** with distractor chunks, oracle share tuning and KL-regularization against catastrophic forgetting — rare to find in self-hosted tools. |
| "Compare models fairly before/after" | ✅ | Model comparisons with per-question answers, ratings, leaderboards. |
| "Keep my experiments isolated" | ✅ | Experiments group models/checkpoints; version history; export. |
| "Quantize and ship GGUF" | ✅ | llama.cpp-based conversion from the same UI. |
| "RAG chat over my documents" | ✅ | FAISS + BM25 hybrid, reranking toggle, feedback collection, search-quality evaluation. |

**Target user:** a technical individual or small team running their own GPU box who wants the whole loop without stitching together five different tools. Not a multi-tenant SaaS.

## How it compares

- **LLaMA-Factory / axolotl** — stronger fine-tuning configuration surfaces, but no document-to-dataset distillation, no RAG, no model registry/experiments.
- **Open WebUI** — great chat/RAG UX, but no training/merging/evaluation.
- **text-generation-webui** — inference focus.
- Ivanna Lab's niche is the **closed loop**: the evaluation results of one model feed the next fine-tuning round, and RAG artifacts (collections) double as RAFT training sources.

## Honest limitations

- **UI is Russian-first** (English core translations landed; form/admin labels are next).
- **py-layer strings** in admin/forms are still being localized.
- **vLLM** support is a stub (separate Dockerfile, not wired).
- Single-node design; no multi-user quotas beyond trial limits; no HTTPS out of the box (put nginx TLS in front).
- Trainers run on a single GPU (Celery `solo` pool) — no distributed training.
- No automated test suite yet; the Django `check`/template parse validation is done, runtime QA is manual.
- Security posture is "trusted home lab": change default passwords, don't expose helper UIs (pgAdmin/Flower/Grafana) publicly.

## What was fixed before publication

- `NoReverseMatch` on merge-create links in 4 templates; duplicate URL routes.
- `admin/admin` superuser backdoor in `entrypoint.sh` (now `DJANGO_SUPERUSER_*` only).
- Log rotation (10 MB × 5), dead Prometheus targets documented, empty `requirements.txt` restored.
- Secrets/cache volumes excluded from the repository; `.env` never committed.

## Roadmap ideas

1. Finish EN localization (models/forms/admin strings).
2. Wire vLLM as an optional inference backend.
3. Add a minimal test suite for the trainers and RAG services.
4. Dataset preview/editor improvements; evaluation export formats (CSV/Markdown reports).
5. Optional HTTPS profile for nginx; per-user resource quotas.
