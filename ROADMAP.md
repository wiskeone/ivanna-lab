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

- [x] **vLLM wiring** — `services/vllm_client.py` (OpenAI-compatible), per-checkpoint `serving_backend` switch, optional `vllm` compose service, client unit tests (v5.0). Field-verified on GTX 1650 4GB with Qwen3-1.7B Q8_0 GGUF: working flags `--enforce-eager --gpu-memory-utilization 0.85 --max-model-len 1024` + `PYTORCH_ALLOC_CONF=expandable_segments:True` (CUDA graphs / 0.92+ OOM on 4GB)
- [ ] **Multi-GPU** — device selection per worker, big-model sharding notes
- [x] **Inference presets** — `GenerationPreset` model + CRUD API + 🌐-применение к сессии из настроек чата (v5.0); сравнения — следующая итерация
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

**v5.0 — инференс и масштаб:** ✅ vLLM подключён, ✅ пресеты генерации (модель/API/UI/тесты 9). Осталось: выбор GPU на воркера, пауза/резюме файнтюна, редактор датасетов.

**v5.x — экосистема:** хаб оценок с экспортом отчётов, авто-карточки моделей для Hugging Face, пуш GGUF обратно в Hub, однострочные доменные пайплайны (legal/code/medical) на базе дистилляции, API-ключи.
