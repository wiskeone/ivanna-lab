# Roadmap

Working plan for future releases. Order is approximate — items move up if someone asks for them.

## v4.x — quality & trust

- [x] **i18n** — templates + Python strings (RU source, EN translation, 🌐 switcher)
- [ ] **Test suite** — pytest for trainers (RAFT dataset assembly, KL setup), mergekit wrapper, RAG services; `manage.py check` in CI
- [ ] **CI** — GitHub Actions: ruff, check, docker build (CPU image)
- [ ] **Prebuilt images** — GHCR: `docker compose pull`
- [ ] **Log hygiene** — structured logging, request IDs for Celery tasks
- [ ] **HTTPS profile for nginx** + `CSRF_TRUSTED_ORIGINS` docs

## v5.0 — inference & scale

- [ ] **vLLM wiring** — optional OpenAI-compatible backend (the `Dockerfile.vllm` stub exists), model selection per session: transformers ↔ vLLM
- [ ] **Multi-GPU** — device selection per worker, big-model sharding notes
- [ ] **Inference presets** — named generation-config profiles reusable across sessions and comparisons
- [ ] **Fine-tune queue improvements** — pause/resume, per-job GPU memory profiles
- [ ] **Dataset editor** — inline row editing with validation for mid-size datasets

## v5.x — ecosystem

- [ ] **Evaluation hub** — benchmark presets (MMLU-subset, domain QA), export reports (CSV/Markdown/PDF)
- [ ] **Model cards** — auto-generated card from registry + evaluation history, publishable to HF
- [ ] **GGUF upload to HF** — push quantized artifacts back to the Hub
- [ ] **Plugin recipes** — one-click domain pipelines (legal, code, medical) built on the distillation flow
- [ ] **REST API keys** — programmatic access to tasks and inference

---

## Дорожная карта (RU)

План приблизительный — приоритеты двигаются по запросам.

**v4.x — качество и доверие:** тест-сьют для трейнеров и RAG-сервисов, CI (ruff + check + сборка), готовые образы в GHCR, структурные логи с request-id, HTTPS-профиль nginx.

**v5.0 — инференс и масштаб:** подключение vLLM как опционального OpenAI-совместимого бэкенда (заготовка `Dockerfile.vllm` уже есть), выбор устройства на воркера, пресеты генерации, пауза/резюме файнтюна, редактор датасетов.

**v5.x — экосистема:** хаб оценок с экспортом отчётов, авто-карточки моделей для Hugging Face, пуш GGUF обратно в Hub, однострочные доменные пайплайны (legal/code/medical) на базе дистилляции, API-ключи.
