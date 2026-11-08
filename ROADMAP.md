# Roadmap

Working plan for future releases. Order is approximate — items move up if someone asks for them.

## v4.x — quality & trust

- [x] **i18n** — templates + Python strings (RU source, EN translation, 🌐 switcher)
- [x] **Test suite** — mergekit config builder (linear/TIES/DARE), validator, resource estimates (v4); trainers next
- [x] **CI** — GitHub Actions: ruff, tests inside CPU image, compose config validation (v4)
- [x] **Prebuilt images** — GHCR: `ghcr.io/wiskeone/ivanna-lab:latest` (docker-publish.yml, GITHUB_TOKEN)
- [ ] **Log hygiene** — structured logging, request IDs for Celery tasks
- [x] **HTTPS profile for nginx** — `nginx/nginx.ssl.conf.template` (v4)

## v5.0 — inference & scale

- [x] **vLLM wiring** — `services/vllm_client.py` (OpenAI-compatible), per-checkpoint `serving_backend` switch (transformers ↔ vLLM), optional `vllm` compose service with GPU profile, client unit tests (v5.0)
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

**v4.x — качество и доверие:** ✅ тесты (mergekit-конфиг), ✅ CI, ✅ HTTPS-профиль, ✅ GHCR-образы. Осталось: трейнеры/RAG-тесты, структурные логи.

**v5.0 — инференс и масштаб:** ✅ vLLM подключён (бэкенд на чекпоинте, compose-профиль). Осталось: выбор GPU на воркера, пресеты генерации, пауза/резюме файнтюна, редактор датасетов.

**v5.x — экосистема:** хаб оценок с экспортом отчётов, авто-карточки моделей для Hugging Face, пуш GGUF обратно в Hub, однострочные доменные пайплайны (legal/code/medical) на базе дистилляции, API-ключи.
