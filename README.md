# Ivanna Lab

Self-hosted **LLM model lab**: a web platform for managing models, building datasets, fine-tuning, merging models, running inference and evaluating results — everything around the "make my own model" loop, Dockerized for a single GPU machine.

> Born as a personal toolchain for model experiments; now open-sourced.
> **UI language:** Russian is the source language; core interface strings
> (1300+) are translated to **English** — switch with the 🌐 menu in the navbar.
> Some form/admin labels remain Russian for now.

## What it does

- **Model registry** — import from Hugging Face or local paths; track versions and checkpoints; quantization; GGUF conversion (llama.cpp)
- **Inference & chat** — chat sessions with any registered model, model-vs-model comparison, inference testing
- **Fine-tuning** — instruction / RAFT / document trainers (LoRA-class adapters via transformers+bitsandbytes), templates, task queue with progress
- **Model merging** — mergekit-powered linear / SLERP / TIES / DARE merges with a web UI
- **Datasets & evaluation** — dataset builder, upload/import/export (incl. Hugging Face datasets), evaluation tasks, answer ratings and leaderboards
- **Distillation** — generate Q&A with teacher checkpoints, assemble instruction datasets
- **RAG** — document collections (PDF/DOCX/HTML/CSV/Excel), chunking, FAISS + BM25 hybrid search, RAG-grounded chat
- **Monitoring** — GPU status, Prometheus + Grafana, Flower

## Stack

| Layer | Tech |
|---|---|
| Backend | Django 5 + Django REST Framework, Gunicorn |
| Frontend | Django templates + Bootstrap 5.3 (dark) + jQuery + Plotly |
| Database | PostgreSQL 15 |
| Queue | Redis 7 + Celery 5 (separate **GPU** and **CPU** workers, beat, Flower) |
| AI | transformers, bitsandbytes, mergekit, sentence-transformers, FAISS, llama.cpp (built from source) |
| Monitoring | Prometheus + Grafana + node/redis exporters |
| Runtime | Docker, NVIDIA CUDA 12.8 base image (GPU worker); CPU worker image included |

## Quick start

Requirements: Docker + docker-compose, NVIDIA GPU with container toolkit (a CPU worker image is included; the main web container expects CUDA base but runs CPU-only workloads too).

```bash
# 1. Environment
cp .env.example .env
#    then edit .env (SECRET_KEY, DB_*, REDIS_*, HF_TOKEN, ...)

# 2. Build & run
docker-compose up -d

# 3. Create an admin user (no default superuser is created automatically)
docker exec -it ivanna_web python manage.py createsuperuser

# 4. Open the app
#    http://localhost
```

Stop: `docker-compose down`.

### Service endpoints

| Service | URL |
|---|---|
| Web app | http://localhost |
| Swagger / Redoc | http://localhost/swagger/ , /redoc/ |
| Flower | http://localhost:5555 |
| Grafana | http://localhost:3000 |
| pgAdmin | http://localhost:5050 |
| Prometheus | http://localhost:9090 |

> Change default passwords (`docker-compose.yml`, `.env`) before exposing beyond localhost. The superuser is created only via `DJANGO_SUPERUSER_*` environment variables — there is no built-in default account.

## How it is organized

```
app/
  config/                   Django project (settings, celery, urls)
  ivanna_model_manager/     The platform: models, views, DRF api_views,
                            Celery tasks (tasks*.py), admin, forms,
                            templates (80 pages)
  ivanna_model_manager/services/
                            fine_tuning/ (instruction/RAFT/document trainers),
                            merger (mergekit), faiss_service, bm25_service,
                            rag_service, vector_store_manager, ...
celery_worker/              worker entrypoints
vllm/                       reserved for a future vLLM service (not wired yet)
prometheus/ grafana/ nginx/ scripts/
```

## Notes & limitations

- **UI is in Russian** for now.
- **vLLM** is shipped as a separate `Dockerfile.vllm` and a commented-out compose service — integration is a work in progress, the platform itself talks to models through transformers.
- Large artifacts (HF cache, model weights, datasets, logs) are intentionally git-ignored; they live in mounted volumes (`media/`, `docker_volumes/`, `app/.cache/`, `app/models/`).
- Log files use `RotatingFileHandler` (10 MB × 5).

## License

MIT — see [LICENSE](LICENSE).
