# Статус трёх проектов — чеклист (09.11)

## ✅ RagDocs (Android, приватный wiskeone/Rag_Docs)

| Пункт | Статус |
|---|---|
| Конвейер: парсинг (15+ форматов) → чанкер → эмбеддинги (bge-m3 ONNX) → FAISS → гибрид + реранкер | ✅ |
| Генерация: llama.cpp (KV-кэш), стриминг, размышления | ✅ |
| Темы: Light/Dark/Black/Ocean/Cream | ✅ |
| 10 локалей | ✅ |
| WorkManager-сторож индексации | ✅ |
| Отпечаток файла для resume (БД 13) | ✅ |
| Токен-чанкер («Вручную» = токены) | ✅ |
| 16KB page sizes | ✅ |
| Управление моделями (удаление) | ✅ |
| Дедуп сессий-двойников | ✅ |
| x86_64 LLM-decode починен | ✅ |
| arm64 = x86_64 (один пин llama.cpp) | ✅ |
| Golden-set hit@5=0.944 | ✅ |
| 41 тест | ✅ |
| **Осталось** | R8/подпись (Store), EN-комменты, поле-чек vLLM/presets |

## ✅ ragsearch (Django RAG, публичный wiskeone/ragsearch)

| Пункт | Статус |
|---|---|
| Конвейер: парсинг 15+ форматов → чанкер → BGE-M3 → FAISS + BM25 → гибрид | ✅ |
| Шифрование документов (Fernet at rest) | ✅ |
| Chat: Qwen 3.5-4B (transformers, 4-bit) | ✅ |
| GPU/CPU Celery воркеры | ✅ |
| **v2.0**: тесты (encryption/parser/score-regression), CI (ruff/build/tests), HTTPS-профиль | ✅ |
| **v2.1**: admin-шаблоны EN (146 trans) | ✅ |
| **v3.0**: ApiKey (SHA-256 + Bearer auth), роли (admin/member/viewer), квоты (docs/colls/storage), экспорт Markdown, SSE-стриминг | ✅ |
| **v3.x**: инкрементальная индексация (SHA-256 skip), vLLM клиент, реранкер (cross-encoder), Grafana dashboard | ✅ |
| **Осталось** | мульти-модель на коллекцию (проверить FK), миграции (нужен chown), EN-комменты |

## ✅ Ivanna Lab (Django LLM-платформа, публичный wiskeone/ivanna-lab)

| Пункт | Статус |
|---|---|
| Реестр моделей: HF/local, версии, квантизация, GGUF (llama.cpp) | ✅ |
| Инференс: чат-сессии, сравнения, тест | ✅ |
| Файнтюн: instruction/RAFT/document trainers + шаблоны + задачи | ✅ |
| Мерж: mergekit (linear/SLERP/TIES/DARE) + web UI | ✅ |
| Датасеты: builder, import/export, HF datasets | ✅ |
| Evaluation: задачи, сравнения, рейтинги | ✅ |
| Дистилляция: Q&A генерация чекпоинтами | ✅ |
| RAG: коллекции, FAISS+BM25, RAG-чат | ✅ |
| Мониторинг: GPU, Prometheus/Grafana, Flower | ✅ |
| **v4**: i18n (RU→EN), тесты (mergekit/client/presets/RAFT), CI, HTTPS, логи, миграция-политика | ✅ |
| **v5.0**: vLLM (field-verified 4GB), пресеты генерации, пауза/резюме FT, редактор датасетов, GPU-выбор | ✅ |
| **Осталось** | тесты трейнеров (глубокие), структурные логи — сделано, eval-хаб, model-карточки, GGUF-push, EN-комменты |

## Общие инфраструктурные пункты

| | RagDocs | ragsearch | ivanna-lab |
|---|---|---|---|
| Git репо | ✅ приватный | ✅ публичный | ✅ публичный |
| CI | ✅ (Android) | ✅ (ruff+tests+compose) | ✅ (ruff+tests+compose) |
| GHCR-образы | N/A | ✅ | ✅ |
| MIT License | ✅ | ✅ | ✅ |
| README EN+RU | ✅ | ✅ EN+RU | ✅ EN (RU = docs) |
| ROADMAP | ✅ | ✅ | ✅ |
| Audit | ✅ TECH_AUDIT_v5 | ✅ в README | ✅ docs/AUDIT.md |
| Миграции в git | N/A (Room) | ✅ (0012 pending chown) | ✅ (0041) |

## Финальный список на потом

1. ragsearch: `sudo chown -R nik:nik app/rag/migrations/` → перенести 0011+0012 из migrations_pending/
2. ivanna-lab: тесты трейнеров (RAFT полный prepare с моками БД)
3. Оба: EN-комменты (~1500 строк, конвейер /tmp/i18n/)
4. RagDocs: R8/подпись → Google Play
5. Ivanna: vLLM динамический запуск контейнера на каждый чекпоинт
