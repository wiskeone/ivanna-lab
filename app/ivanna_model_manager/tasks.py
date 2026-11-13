from __future__ import absolute_import, unicode_literals

import json
import time
import gc
import logging
import traceback
import os
import threading
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple
from datasets import load_dataset, Dataset

import torch
from celery import shared_task
from django.utils.translation import gettext as _
from django.conf import settings
from django.utils import timezone
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from django.contrib.auth.models import User


from .models import (
    EvaluationResult, ModelCheckpoint, ModelDownloadJob,
    ModelEvaluationTask, ModelMergeJob, FineTuningJob, ModelConversionJob,
    DistillationQuestionSet, DistillationChunkGroup, DistillationQuestion, 
    DistillationAnswerJob, DocumentChunk, DistillationAnswer, 
    HuggingFaceDataset, EvaluationDataset
)

from .services.download_service import download_service
from .services.redis_model_service import redis_model_manager
from .services.quantization_router import QuantizationRouter
from .tasks_rag import (
    _build_faiss_index_impl,
    _create_embeddings_for_collection_impl,
    _update_faiss_index_impl,
    process_document_task,
    create_embeddings_for_collection_task,
    rebuild_embeddings_and_index_task,
    build_faiss_index_task,
    update_faiss_index_task,
    regenerate_document_chunks_task,
)
from .tasks_embedding import download_embedding_model_task

from .services.fine_tuning.document_trainer import DocumentTrainer
from .services.fine_tuning.instruction_trainer import InstructionTrainer
from .services.fine_tuning.raft_trainer import RAFTTrainer
from .services.fine_tuning.evaluator import FineTuningEvaluator

from .services.model_cache import model_cache

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# --------------------------------------------------------------------------
def check_pid_exists(pid):
    """Проверяет, существует ли процесс с данным PID."""
    import os
    import signal
    
    if not pid:
        return False
    
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _load_model_with_quanto(model_path: str, quantization: str, **kwargs):
    """
    Загружает модель с квантизацией на CPU с помощью quanto и принудительно
    размещает её на CPU.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import quanto

    logger.info(f"Loading model with quanto {quantization} on CPU")

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Загружаем модель в float32 на CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        **kwargs
    )

    # Квантизация
    if quantization == 'int4':
        quanto.quantize(model, weights=quanto.qint4, activations=None)
    elif quantization == 'int8':
        quanto.quantize(model, weights=quanto.qint8, activations=None)
    else:
        logger.error(f"Unsupported quanto quantization: {quantization}")
        return model, tokenizer

    quanto.freeze(model)

    # Принудительно размещаем на CPU (хотя уже на CPU, гарантируем)
    model.to('cpu')
    logger.info(f"Model quantized with quanto {quantization} and moved to CPU")

    return model, tokenizer


def _load_model_for_generation(checkpoint, quantization):
    """Вспомогательная функция для загрузки модели в задаче генерации."""
    from .services.quantization_router import QuantizationRouter
    from .services.redis_model_service import redis_model_manager
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    try:
        model_identifier = checkpoint.huggingface_id or checkpoint.name
        
        # Регистрируем блокировку
        if not redis_model_manager.register_model_loading(model_identifier, checkpoint.loader_config):
            return {'success': False, 'error': 'Model is being loaded by another process'}
        
        # Определяем стратегию загрузки
        model_size_gb = checkpoint.size_gb
        quant_config, actual_quant, target_device = QuantizationRouter.get_quantization_config(
            quantization,
            model_size_gb
        )
        
        logger.info(f"Loading model for generation: actual_quant={actual_quant}, device={target_device}")
        
        # Загружаем токенизатор с trust_remote_code=True
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint.local_path,
            trust_remote_code=True,
            use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Загружаем модель
        if isinstance(quant_config, dict) and quant_config.get('method') == 'quanto':
            model, tokenizer = _load_model_with_quanto(
                checkpoint.local_path,
                quantization=quant_config['quantization']
            )
        else:
            model_kwargs = {
                'trust_remote_code': True,  # обязательно для DeepSeek
                'low_cpu_mem_usage': True,
            }
            if target_device == 'cuda':
                model_kwargs['device_map'] = 'auto'
                if quant_config:
                    model_kwargs['quantization_config'] = quant_config
                else:
                    if actual_quant == 'float16':
                        model_kwargs['torch_dtype'] = torch.float16
                    elif actual_quant == 'bfloat16':
                        model_kwargs['torch_dtype'] = torch.bfloat16
            else:
                model_kwargs['torch_dtype'] = torch.float32
            
            model = AutoModelForCausalLM.from_pretrained(checkpoint.local_path, **model_kwargs)
        
        # Регистрируем в Redis
        redis_model_manager.register_local_model(
            model_id=model_identifier,
            model=model,
            tokenizer=tokenizer,
            config=checkpoint.loader_config,
            device_info=target_device,
            quantization=actual_quant
        )
        
        return {'success': True, 'model_identifier': model_identifier}
    except Exception as e:
        logger.error(f"Error loading model for generation: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def _load_model_for_evaluation(checkpoint, quantization):
    """Загружает модель для оценки, возвращает (model, tokenizer) или (None, None) при ошибке."""
    from .services.quantization_router import QuantizationRouter
    from .services.redis_model_service import redis_model_manager
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    try:
        model_identifier = checkpoint.huggingface_id or checkpoint.name

        # Регистрируем блокировку (как в оригинале)
        if not redis_model_manager.register_model_loading(model_identifier, checkpoint.loader_config):
            logger.error(f"Model {model_identifier} is being loaded by another process")
            return None, None

        # Определяем стратегию загрузки с fallback через QuantizationRouter
        model_size_gb = checkpoint.size_gb
        quant_config, actual_quant, target_device = QuantizationRouter.get_quantization_config(
            quantization,
            model_size_gb
        )

        logger.info(f"Loading model for evaluation: actual_quant={actual_quant}, device={target_device}")

        # Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint.local_path,
            trust_remote_code=True,
            use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Загружаем модель (аналогично load_model_to_memory_task, но без пайплайна)
        if isinstance(quant_config, dict) and quant_config.get('method') == 'quanto':
            # Для quanto загружаем на CPU
            model, tokenizer = _load_model_with_quanto(
                checkpoint.local_path,
                quantization=quant_config['quantization']
            )
        else:
            model_kwargs = {
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            }
            if target_device == 'cuda':
                model_kwargs['device_map'] = 'auto'
                if quant_config:
                    model_kwargs['quantization_config'] = quant_config
                else:
                    if actual_quant == 'float16':
                        model_kwargs['torch_dtype'] = torch.float16
                    elif actual_quant == 'bfloat16':
                        model_kwargs['torch_dtype'] = torch.bfloat16
            else:
                model_kwargs['torch_dtype'] = torch.float32

            model = AutoModelForCausalLM.from_pretrained(checkpoint.local_path, **model_kwargs)

        # Регистрируем в Redis (для информации, не используется для генерации)
        redis_model_manager.register_local_model(
            model_id=model_identifier,
            model=model,
            tokenizer=tokenizer,
            config=checkpoint.loader_config,
            device_info=target_device,
            quantization=actual_quant
        )

        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model for evaluation: {e}", exc_info=True)
        return None, None


@shared_task(bind=True, queue='cpu_tasks')
def download_model_task(self, model_id, user_id=None):
    logger.info(f"📥 download_model_task: model_id={model_id}, user_id={user_id}")
    try:
        checkpoint = ModelCheckpoint.objects.get(id=model_id)

        # Получаем токен, если есть пользователь
        token = None
        if user_id:
            try:
                user = User.objects.get(id=user_id)
                token = user.profile.huggingface_token or None
            except (User.DoesNotExist, AttributeError):
                pass

        def progress_callback(progress, step):
            self.update_state(
                state='PROGRESS',
                meta={'progress': progress, 'current_step': step, 'model_name': checkpoint.display_name}
            )

        # Вызываем метод download_model_sync
        checkpoint = download_service.download_model_sync(
            model_id,
            token=token,
            progress_callback=progress_callback
        )
        return {
            'success': True,
            'model_id': model_id,
            'local_path': checkpoint.local_path,
            'status': checkpoint.status,
        }
    except ModelCheckpoint.DoesNotExist:
        return {'success': False, 'error': f'Model {model_id} not found'}
    except Exception as e:
        logger.error(f"Download error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


@shared_task(queue='cpu_tasks')
def cleanup_download_jobs(days: int = 7):
    """Очистка завершённых задач загрузки."""
    cutoff = timezone.now() - timedelta(days=days)
    deleted, _ = ModelDownloadJob.objects.filter(
        created_at__lt=cutoff,
        status__in=['COMPLETED', 'FAILED', 'CANCELLED']
    ).delete()
    logger.info(f"🧹 Cleaned up {deleted} old download jobs")
    return {'success': True, 'deleted': deleted}


# --------------------------------------------------------------------------
# ЗАГРУЗКА МОДЕЛИ В ПАМЯТЬ
# --------------------------------------------------------------------------
@shared_task(bind=True, queue='gpu_tasks')
def load_model_to_memory_task(self, model_id: int, quantization: Optional[str] = None):
    """Загрузка модели в память с каскадной системой fallback'ов."""
    from .models import ModelCheckpoint
    from .services.redis_model_service import redis_model_manager
    from .services.quantization_router import QuantizationRouter
    import os
    import gc
    import time

    logger.info(f"🚀 load_model_to_memory_task: model_id={model_id}, quantization={quantization}")

    def update_progress(progress, step):
        self.update_state(state='PROGRESS', meta={'progress': progress, 'current_step': step})

    try:
        checkpoint = ModelCheckpoint.objects.get(id=model_id, is_active=True)
        model_identifier = checkpoint.huggingface_id or checkpoint.name

        # Проверка, не загружена ли уже
        existing = redis_model_manager.get_model_info(model_identifier)
        if existing and existing.get('loaded'):
            checkpoint.status = 'LOADED'
            checkpoint.save(update_fields=['status'])
            logger.info(f"Model {model_identifier} already loaded in memory")
            return {'success': True, 'message': 'Model already loaded', 'model_id': model_identifier}

        if not checkpoint.local_path or not os.path.exists(checkpoint.local_path):
            raise FileNotFoundError(f"Local path not found: {checkpoint.local_path}")

        # Регистрация блокировки
        if not redis_model_manager.register_model_loading(model_identifier, checkpoint.loader_config):
            status = redis_model_manager.get_loading_status(model_identifier)
            return {'success': False, 'error': 'Already loading in another process', 'status': status}

        update_progress(5, "Подготовка конфигурации")
        redis_model_manager.update_loading_progress(model_identifier, 5, "Подготовка конфигурации")

        # Очищаем кэш перед загрузкой
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_memory, total_memory = torch.cuda.mem_get_info()
            free_gb = free_memory / 1024**3
            logger.info(f"GPU memory before loading: {free_gb:.2f}GB free")

        # Определяем оптимальную стратегию загрузки через роутер
        model_size_gb = checkpoint.size_gb
        quant_config, actual_quant, target_device = QuantizationRouter.get_quantization_config(
            quantization or checkpoint.quantization,
            model_size_gb
        )

        logger.info(f"✅ Selected config: actual_quant={actual_quant}, device={target_device}")

        # Загрузка токенизатора
        update_progress(15, "Загрузка токенизатора")
        redis_model_manager.update_loading_progress(model_identifier, 15, "Загрузка токенизатора")
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint.local_path,
            trust_remote_code=True,
            use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Загрузка модели (с возможностью fallback на CPU)
        update_progress(30, "Загрузка модели")
        redis_model_manager.update_loading_progress(model_identifier, 30, "Загрузка модели")

        model = None
        device_map = None
        skip_pipeline = False   # флаг, чтобы не создавать pipeline при CPU fallback

        # Попытка загрузки с выбранной стратегией
        try:
            if isinstance(quant_config, dict) and quant_config.get('method') == 'quanto':
                model, tokenizer = _load_model_with_quanto(
                    checkpoint.local_path,
                    quantization=quant_config['quantization']
                )
                device_map = None
                target_device = 'cpu'
                skip_pipeline = True
                logger.info(f"Loaded with quanto {quant_config['quantization']} on CPU")
            else:
                model_kwargs = {
                    'trust_remote_code': True,
                    'low_cpu_mem_usage': True,
                }
                
                if target_device == 'cuda':
                    model_kwargs['device_map'] = 'auto'
                    if quant_config:
                        model_kwargs['quantization_config'] = quant_config
                        logger.info(f"Using bitsandbytes {actual_quant} on GPU")
                    else:
                        if actual_quant == 'float16':
                            model_kwargs['torch_dtype'] = torch.float16
                            logger.info("Using float16 on GPU")
                        elif actual_quant == 'bfloat16':
                            model_kwargs['torch_dtype'] = torch.bfloat16
                            logger.info("Using bfloat16 on GPU")
                else:
                    model_kwargs['torch_dtype'] = torch.float32
                    logger.info("Using float32 on CPU")

                model = AutoModelForCausalLM.from_pretrained(checkpoint.local_path, **model_kwargs)
                device_map = model_kwargs.get('device_map')

        except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
            error_msg = str(e).lower()
            # Если ошибка связана с нехваткой памяти на GPU – пробуем CPU fallback
            if 'cuda' in error_msg or 'memory' in error_msg or 'gpu' in error_msg or 'offloaded' in error_msg:
                logger.warning(f"GPU loading failed: {e}. Falling back to CPU with {actual_quant}...")
                try:
                    # Пробуем загрузить на CPU с той же квантизацией (если поддерживается)
                    if actual_quant in ['int4', 'int8']:
                        # Используем quanto
                        model, tokenizer = _load_model_with_quanto(checkpoint.local_path, quantization=actual_quant)
                        logger.info(f"Loaded with quanto {actual_quant} on CPU (fallback)")
                    else:
                        # Для float16/float32 загружаем обычную модель на CPU
                        logger.info(f"Loading full precision model on CPU (fallback)")
                        model = AutoModelForCausalLM.from_pretrained(
                            checkpoint.local_path,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        )
                    target_device = 'cpu'
                    device_map = None
                    skip_pipeline = True
                    logger.info(f"✅ Successfully loaded on CPU with fallback, actual_quant={actual_quant}")
                except Exception as e2:
                    logger.error(f"CPU fallback also failed: {e2}")
                    raise
            else:
                raise

        # Логируем устройство модели
        if hasattr(model, 'device'):
            logger.info(f"Model is on device: {model.device}")
        elif hasattr(model, 'hf_device_map'):
            logger.info(f"Model device map: {model.hf_device_map}")

        update_progress(70, "Создание пайплайна")
        redis_model_manager.update_loading_progress(model_identifier, 70, "Создание пайплайна")

        from transformers import pipeline

        # Создаём pipeline только для GPU-моделей, для CPU пропускаем
        if skip_pipeline or target_device == 'cpu' or (actual_quant and actual_quant.startswith('quanto_')):
            logger.info("Pipeline not created (CPU model or fallback), will use direct generation")
            pipe = None
        else:
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map=device_map
            )

        update_progress(90, "Регистрация модели")
        redis_model_manager.update_loading_progress(model_identifier, 90, "Регистрация модели")

        device_info = target_device
        model_info = redis_model_manager.register_local_model(
            model_id=model_identifier,
            model=model,
            tokenizer=tokenizer,
            config=checkpoint.loader_config,
            device_info=device_info,
            pipeline=pipe,
            quantization=actual_quant
        )

        update_progress(100, "Готово")
        redis_model_manager.update_loading_progress(model_identifier, 100, "Готово")

        # Обновляем статус модели в БД
        checkpoint.status = 'LOADED'
        checkpoint.quantization = quantization or checkpoint.quantization or ''
        checkpoint.loader_config = checkpoint.loader_config or {}
        checkpoint.loader_config['actual_quantization'] = actual_quant
        checkpoint.loader_config['target_device'] = target_device
        checkpoint.loader_config['loaded_at'] = time.time()
        
        checkpoint.save(update_fields=['status', 'quantization', 'loader_config'])
        logger.info(f"✅ Database updated for model {model_identifier}: status=LOADED, actual_quant={actual_quant}, device={target_device}")

        # Финальная очистка кэша
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_memory, total_memory = torch.cuda.mem_get_info()
            free_gb = free_memory / 1024**3
            logger.info(f"GPU memory after loading: {free_gb:.2f}GB free")

        logger.info(f"✅ Model {model_identifier} loaded successfully with {actual_quant} on {target_device}")
        return {'success': True, 'model_id': model_identifier, 'model_info': model_info}

    except Exception as e:
        logger.error(f"❌ Error loading model: {e}", exc_info=True)
        redis_model_manager.register_model_error(model_identifier, str(e))
        try:
            checkpoint = ModelCheckpoint.objects.get(id=model_id)
            checkpoint.status = 'ERROR'
            checkpoint.error_message = str(e)
            checkpoint.save(update_fields=['status', 'error_message'])
        except:
            pass
        return {'success': False, 'error': str(e)}


@shared_task(bind=True, queue='gpu_tasks')
def unload_model_from_memory_task(self, model_id, force: bool = False):
    """
    Выгрузка модели из памяти с принудительной очисткой
    """
    logger.info(f"⏹️ Задача выгрузки модели: {model_id}, force={force}")
    
    try:
        # Определяем идентификатор модели
        if isinstance(model_id, int):
            checkpoint = ModelCheckpoint.objects.get(id=model_id)
            model_identifier = checkpoint.huggingface_id or checkpoint.name
        else:
            model_identifier = model_id
            checkpoint = None
        
        # Выгружаем из памяти
        success = redis_model_manager.unload_model(model_identifier, force=force)
        
        if success and checkpoint:
            # Обновляем статус в БД
            checkpoint.status = 'DOWNLOADED' if checkpoint.local_path else 'PENDING'
            checkpoint.save()
            
            # Очищаем информацию о загрузке
            ModelDownloadJob.objects.filter(
                checkpoint=checkpoint,
                status='DOWNLOADING'
            ).update(status='CANCELLED')
        
        # Дополнительная очистка памяти в Celery worker
        import gc
        gc.collect()
        
        return {
            'success': success,
            'model_id': model_identifier,
            'force': force,
            'memory_freed': True
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка выгрузки: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


@shared_task(bind=True, queue='gpu_tasks', max_retries=3)
def generate_text_task(self, model_id, prompt, session_id=None, use_rag=False, rag_config=None, **kwargs):
    """
    Генерация текста с использованием модели.
    Если модель не загружена локально в этом воркере, загружает её.
    Поддерживает RAG с несколькими коллекциями и полными параметрами.
    """
    from .models import ModelCheckpoint
    from .services.redis_model_service import redis_model_manager
    from .services.rag_service import RAGService
    from django.db.models import Q
    import os
    import time

    logger.info(f"🔧 generate_text_task: model_id={model_id}, session_id={session_id}, use_rag={use_rag}")

    try:
        # Определяем идентификатор модели (может быть числом или строкой)
        if isinstance(model_id, int):
            try:
                checkpoint = ModelCheckpoint.objects.get(id=model_id, is_active=True)
                model_identifier = checkpoint.huggingface_id or checkpoint.name
            except ModelCheckpoint.DoesNotExist:
                return {'success': False, 'error': f'Model {model_id} not found'}
        else:
            model_identifier = model_id
            checkpoint = None

        # v5.0: роутинг бэкенда — vLLM обслуживается внешним сервером,
        # локальная загрузка transformers не нужна
        serving_backend = getattr(checkpoint, 'serving_backend', 'transformers') if checkpoint else 'transformers'
        if serving_backend == 'vllm':
            from .services.vllm_client import VLLMClient
            from django.conf import settings as dj_settings
            vllm = VLLMClient()
            served = getattr(dj_settings, 'VLLM_SERVED_MODEL', '')
            target_model = served or model_identifier
            logger.info(f"🚀 vLLM backend: model={target_model} (checkpoint {checkpoint.id if checkpoint else model_identifier})")

        # Проверяем, загружена ли модель локально
        if serving_backend != 'vllm' and model_identifier not in redis_model_manager.local_models:
            # Попытка загрузить модель
            if checkpoint is None:
                try:
                    checkpoint = ModelCheckpoint.objects.get(
                        Q(huggingface_id=model_identifier) | Q(name=model_identifier),
                        is_active=True
                    )
                except ModelCheckpoint.DoesNotExist:
                    return {'success': False, 'error': f'Model {model_identifier} not found in DB'}

            if serving_backend != 'vllm' and (not checkpoint.local_path or not os.path.exists(checkpoint.local_path)):
                return {'success': False, 'error': 'Model not downloaded locally'}

            # Определяем квантизацию
            quantization = kwargs.get('quantization') or checkpoint.quantization or None

            # Загружаем модель (используем отдельную функцию)
            load_result = _load_model_for_generation(checkpoint, quantization)
            if not load_result.get('success'):
                return load_result

        # Если включён RAG, получаем контекст для текущего сообщения
        final_prompt = prompt
        rag_sources = []
        if use_rag and rag_config and rag_config.get('selected_collections'):
            try:
                rag = RAGService()
                from .models import EmbeddingModel
                if not EmbeddingModel.objects.filter(is_default=True).exists():
                    logger.warning("No default embedding model found, RAG disabled")
                else:
                    collection_ids = rag_config.get('selected_collections', [])
                    top_k = rag_config.get('top_k', 5)
                    semantic_weight = rag_config.get('semantic_weight', 0.7)
                    keyword_weight = rag_config.get('keyword_weight', 0.2)
                    freshness_weight = rag_config.get('freshness_weight', 0.1)
                    use_reranking = rag_config.get('use_reranking', True)

                    # Выполняем поиск
                    search_result = rag.search(
                        query=prompt,  # в идеале нужно извлекать последний вопрос, но для простоты используем весь промпт
                        user_id='system',
                        collection_ids=collection_ids,
                        top_k=top_k,
                        search_type='hybrid',
                        semantic_weight=semantic_weight,
                        keyword_weight=keyword_weight,
                        freshness_weight=freshness_weight,
                        use_reranking=use_reranking
                    )

                    if search_result.get('success') and search_result.get('results'):
                        results = search_result['results'][:top_k]
                        # Собираем контекст и информацию об источниках
                        context_parts = []
                        for r in results:
                            context_parts.append(
                                f"[Документ: {r.get('document_title', 'Без названия')}]\n{r.get('content', '')}"
                            )
                            rag_sources.append({
                                'title': r.get('document_title', ''),
                                'chunk_id': r.get('chunk_id'),
                                'document_id': r.get('document_id'),
                                'relevance': r.get('relevance_score', 0)
                            })
                        context = "\n\n---\n\n".join(context_parts)
                        
                        # Вставляем контекст в начало промпта
                        # Пытаемся определить, есть ли уже системный промпт
                        parts = prompt.split("\n", 1)
                        if len(parts) == 2 and parts[0].startswith("System:"):
                            final_prompt = f"{parts[0]}\n\nКонтекст из документов:\n{context}\n\n{parts[1]}"
                        else:
                            final_prompt = f"Контекст из документов:\n{context}\n\n{prompt}"
                        
                        logger.info(f"RAG context added, {len(results)} chunks")
            except Exception as e:
                logger.error(f"Error getting RAG context: {e}", exc_info=True)

        # Генерируем текст
        if serving_backend == 'vllm':
            from .services.vllm_client import VLLMClient
            result = VLLMClient().generate(model=target_model, prompt=final_prompt, **kwargs)
            if not result.get('success'):
                return result
        else:
            result = redis_model_manager.generate_text(model_identifier, final_prompt, **kwargs)
        
        # Добавляем информацию об использованных RAG-источниках
        if result.get('success') and rag_sources:
            result['rag_sources'] = rag_sources
            logger.info(f"✅ RAG sources added to result: {rag_sources}")  # для отладки
        
        # Если есть session_id, добавляем сообщение в сессию
        if session_id and result.get('success'):
            try:
                from .services.chat_service import chat_service
                metadata = {
                    'tokens': {
                        'input': result.get('input_tokens', 0),
                        'output': result.get('output_tokens', 0),
                        'total': result.get('total_tokens', 0)
                    },
                    'generation_time_ms': result.get('generation_time', 0) * 1000,
                    'model_id': model_id,
                    'device': result.get('device', 'unknown'),
                    'rag_used': use_rag,
                    'rag_sources': rag_sources if use_rag else None,
                }
                chat_service.add_message(session_id, 'assistant', result['response'], metadata)
            except Exception as e:
                logger.error(f"Error adding message to session {session_id}: {e}")
        
        return result

    except Exception as e:
        logger.error(f"Error in generate_text_task: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'error_type': e.__class__.__name__
        }


# --------------------------------------------------------------------------
# СИНХРОНИЗАЦИЯ СТАТУСОВ
# --------------------------------------------------------------------------
@shared_task(queue='cpu_tasks')
def sync_model_statuses_task():
    """Синхронизирует статусы моделей с Redis и проверяет живость воркеров."""
    from .models import ModelCheckpoint

    logger.info("🔄 Starting model status sync")
    synced = 0

    for model in ModelCheckpoint.objects.filter(status='LOADED', is_active=True):
        model_id = model.huggingface_id or model.name
        info = redis_model_manager.get_model_info(model_id)
        if not info or not info.get('loaded'):
            model.status = 'DOWNLOADED' if model.local_path else 'PENDING'
            model.save()
            synced += 1
        else:
            # Проверяем, жив ли процесс, загрузивший модель
            loaded_by = info.get('loaded_by', {})
            pid = loaded_by.get('pid')
            if pid and not check_pid_exists(pid):
                logger.warning(f"Worker with PID {pid} for model {model_id} is dead, unloading")
                redis_model_manager.unload_model(model_id, force=True)
                model.status = 'DOWNLOADED' if model.local_path else 'PENDING'
                model.save()
                synced += 1

    for model in ModelCheckpoint.objects.filter(status='LOADING', is_active=True):
        model_id = model.huggingface_id or model.name
        status = redis_model_manager.get_loading_status(model_id)
        if status.get('status') == 'LOADED':
            model.status = 'LOADED'
            model.save()
            synced += 1
        elif status.get('status') == 'ERROR':
            model.status = 'ERROR'
            model.error_message = status.get('error', '')
            model.save()
            synced += 1

    for model in ModelCheckpoint.objects.filter(status='DOWNLOADING', is_active=True):
        active = ModelDownloadJob.objects.filter(checkpoint=model, status__in=['PENDING', 'DOWNLOADING'])
        if not active.exists():
            model.status = 'PENDING'
            model.save()
            synced += 1

    logger.info(f"✅ Sync completed, {synced} models updated")
    return {'success': True, 'synced': synced}

@shared_task(queue='cpu_tasks')
def refresh_model_ttl():
    """Обновляет TTL для ключей моделей в Redis."""
    try:
        keys = redis_model_manager.redis.keys("model:*:info")
        for key in keys:
            redis_model_manager.redis.expire(key, 3600)
        return {'success': True, 'refreshed': len(keys)}
    except Exception as e:
        logger.error(f"TTL refresh error: {e}")
        return {'success': False, 'error': str(e)}


@shared_task(queue='cpu_tasks')
def cleanup_stale_models():
    """
    Периодическая очистка моделей, которые висят в Redis,
    но не используются (нет heartbeat от воркера).
    """
    from .services.redis_model_service import redis_model_manager
    
    logger.info("🧹 Starting stale models cleanup")
    
    # Получаем все загруженные модели из Redis
    models = redis_model_manager.list_loaded_models()
    cleaned = 0
    
    for model_id in models:
        info = redis_model_manager.get_model_info(model_id)
        if info:
            last_accessed = info.get('last_accessed', 0)
            loaded_by = info.get('loaded_by', {})
            loaded_pid = loaded_by.get('pid')
            
            # Если модель не использовалась более 1 часа
            if time.time() - last_accessed > 3600:
                logger.warning(f"Model {model_id} not accessed for >1h, unloading")
                redis_model_manager.unload_model(model_id, force=True)
                cleaned += 1
                
            # Если воркер с таким PID не существует (для Linux)
            elif loaded_pid and not check_pid_exists(loaded_pid):
                logger.warning(f"Worker with PID {loaded_pid} for model {model_id} is dead, unloading")
                redis_model_manager.unload_model(model_id, force=True)
                cleaned += 1
    
    logger.info(f"✅ Cleaned up {cleaned} stale models")
    return {'cleaned': cleaned}


@shared_task(queue='cpu_tasks')
def cleanup_stale_model_locks():
    """Очистка зависших блокировок моделей."""
    from .services.redis_model_service import redis_model_manager
    
    logger.info("🔓 Cleaning stale model locks")
    cleaned = redis_model_manager.clean_stale_locks()
    logger.info(f"✅ Cleaned {cleaned} stale locks")
    return {'cleaned': cleaned}


# --------------------------------------------------------------------------
# СЛИЯНИЕ МОДЕЛЕЙ
# --------------------------------------------------------------------------
@shared_task(bind=True, queue='gpu_tasks')
def merge_models_task(self, merge_job_id):
    """Задача слияния моделей с отслеживанием прогресса."""
    from .models import ModelMergeJob, ModelCheckpoint
    from .services.merger import merger
    import os
    import shutil
    from django.utils import timezone

    logger = logging.getLogger(__name__)
    logger.info(f"🔗 merge_models_task: job_id={merge_job_id}")

    try:
        merge_job = ModelMergeJob.objects.get(id=merge_job_id)
        if merge_job.status not in ['PENDING', 'VALIDATING', 'FAILED']:
            logger.warning(f"Job {merge_job_id} already in state {merge_job.status}, skipping")
            return {'success': False, 'error': 'Job already running or completed'}

        # Валидация входных данных
        if not merge_job.base_model.local_path or not os.path.exists(merge_job.base_model.local_path):
            raise FileNotFoundError(f"Base model path not found: {merge_job.base_model.local_path}")
        if not merge_job.merge_model.local_path or not os.path.exists(merge_job.merge_model.local_path):
            raise FileNotFoundError(f"Merge model path not found: {merge_job.merge_model.local_path}")

        merge_job.status = 'VALIDATING'
        merge_job.save(update_fields=['status'])

        # Создаём выходную папку
        output_dir = os.path.join(settings.MEDIA_ROOT, 'merged_models')
        os.makedirs(output_dir, exist_ok=True)
        output_name = f"merged_{merge_job.base_model.safe_name}_{merge_job.merge_model.safe_name}_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = os.path.join(output_dir, output_name)

        # Обновляем статус
        merge_job.status = 'RUNNING'
        merge_job.current_step = 'Подготовка конфигурации'
        merge_job.save(update_fields=['status', 'current_step'])

        # Запускаем слияние
        result = merger.merge_models(
            base_model_path=merge_job.base_model.local_path,
            merge_model_path=merge_job.merge_model.local_path,
            output_path=output_path,
            method=merge_job.merge_method.lower(),
            merge_ratio=merge_job.merge_ratio,
            density=merge_job.merge_config.get('density', 0.7),
            **merge_job.merge_config
        )

        if not result['success']:
            raise Exception(result.get('error', 'Unknown error'))

        # Создаём новый чекпоинт для слитой модели
        merged_checkpoint = ModelCheckpoint.objects.create(
            name=f"merged_{merge_job.base_model.name}_{merge_job.merge_model.name}_{int(timezone.now().timestamp())}",
            display_name=f"Слияние {merge_job.base_model.display_name} и {merge_job.merge_model.display_name}",
            model_type=merge_job.base_model.model_type,
            parameter_count=merge_job.base_model.parameter_count,  # предполагаем, что одинаковые
            status='DOWNLOADED',
            local_path=output_path,
            relative_path=f"merged_models/{output_name}",
            is_active=True,
            parent=merge_job.base_model,
            tags=merge_job.base_model.tags + ['merged']
        )

        # Обновляем задачу
        merge_job.status = 'COMPLETED'
        merge_job.output_model = merged_checkpoint
        merge_job.completed_at = timezone.now()
        merge_job.logs = result.get('logs', '')
        merge_job.save()

        logger.info(f"✅ Merge job {merge_job_id} completed. Output model ID: {merged_checkpoint.id}")
        return {'success': True, 'output_model_id': merged_checkpoint.id}

    except Exception as e:
        logger.error(f"❌ Merge job {merge_job_id} failed: {e}", exc_info=True)
        try:
            merge_job = ModelMergeJob.objects.get(id=merge_job_id)
            merge_job.status = 'FAILED'
            merge_job.error_details = {'error': str(e), 'traceback': traceback.format_exc()}
            merge_job.completed_at = timezone.now()
            merge_job.save(update_fields=['status', 'error_details', 'completed_at'])
        except:
            pass
        return {'success': False, 'error': str(e)}


# --------------------------------------------------------------------------
# ЗАДАЧИ ОЦЕНКИ (EVALUATION)
# --------------------------------------------------------------------------
@shared_task(bind=True, queue='gpu_tasks', time_limit=None, soft_time_limit=None)
def run_evaluation_task(self, task_id, use_rag=False, collection_ids=None):
    """
    Запуск оценки модели на датасете с чекпоинтами и без ограничения времени.
    """
    from .models import ModelEvaluationTask, EvaluationResult, DocumentCollection, EmbeddingModel
    from .services.rag_service import RAGService
    from .services.redis_model_service import redis_model_manager
    import time
    import gc
    import signal
    import sys

    logger = logging.getLogger(__name__)
    logger.info(f"📊 run_evaluation_task: {task_id}, use_rag={use_rag}, collection_ids={collection_ids} (unlimited time)")

    lock_key = f"evaluation_task:{task_id}:running"
    if redis_model_manager.redis.get(lock_key):
        logger.warning(f"Task {task_id} is already running, skipping")
        return {'success': False, 'error': 'Task already running'}
    redis_model_manager.redis.setex(lock_key, 3600, '1')

    current_model_idx = 0
    current_question_idx = 0
    processed_models = []

    def signal_handler(signum, frame):
        logger.warning(f"⚠️ Received signal {signum}, saving checkpoint...")
        try:
            task.refresh_from_db()
            task.metadata['checkpoint'] = {
                'last_model_idx': current_model_idx,
                'last_question_idx': current_question_idx,
                'processed_models': processed_models,
                'timestamp': timezone.now().isoformat()
            }
            task.save(update_fields=['metadata'])
            logger.info(f"✅ Checkpoint saved at model {current_model_idx}, question {current_question_idx}")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
        redis_model_manager.redis.delete(lock_key)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    task = None
    for attempt in range(5):
        try:
            task = ModelEvaluationTask.objects.get(id=task_id)
            break
        except ModelEvaluationTask.DoesNotExist:
            if attempt == 4:
                logger.error(f"Task {task_id} not found after 5 attempts")
                redis_model_manager.redis.delete(lock_key)
                return {'success': False, 'error': f'Task {task_id} not found'}
            time.sleep(1)

    if task.status in ['COMPLETED', 'FAILED']:
        logger.warning(f"Task {task_id} already {task.status}, skipping")
        redis_model_manager.redis.delete(lock_key)
        return {'success': False, 'error': f'Task already {task.status}'}

    try:
        checkpoint = task.metadata.get('checkpoint', {})
        if checkpoint:
            logger.info(f"🔄 Resuming from checkpoint: {checkpoint}")
            start_model_idx = checkpoint.get('last_model_idx', 0)
            start_question_idx = checkpoint.get('last_question_idx', 0)
            processed_models = checkpoint.get('processed_models', [])
        else:
            start_model_idx = 0
            start_question_idx = 0
            processed_models = []

        task.status = 'RUNNING'
        task.started_at = task.started_at or timezone.now()
        task.current_step = f'Resuming from model {start_model_idx}, question {start_question_idx}' if checkpoint else 'Инициализация'
        task.save(update_fields=['status', 'started_at', 'current_step', 'metadata'])

        if not checkpoint:
            deleted_count, _ = EvaluationResult.objects.filter(task=task).delete()
            if deleted_count:
                logger.info(f"🗑️ Deleted {deleted_count} old evaluation results")

        dataset = task.dataset
        questions = dataset.questions if dataset else []
        if not questions:
            task.fail_evaluation("No questions")
            redis_model_manager.redis.delete(lock_key)
            return {'success': False, 'error': 'No questions'}

        models = list(task.evaluation_models.all())
        if not models:
            task.fail_evaluation("No models")
            redis_model_manager.redis.delete(lock_key)
            return {'success': False, 'error': 'No models'}

        logger.info(f"📋 Dataset: {dataset.name}, total questions: {len(questions)}, models: {[m.name for m in models]}")
        logger.info(f"▶️ Starting from model {start_model_idx}, question {start_question_idx}")

        generation_config = task.generation_config or {}
        quantization_override = generation_config.get('quantization')

        # Восстанавливаем RAG параметры из конфигурации, если они не переданы явно
        if not use_rag:
            use_rag = generation_config.get('use_rag', False)
        if not collection_ids:
            collection_ids = generation_config.get('rag_collections', [])

        # === RAG инициализация ===
        rag_service = None
        embedding_model_id = None
        collection_id = None
        if use_rag and collection_ids:
            collection_id = collection_ids[0] if isinstance(collection_ids, list) else collection_ids
            try:
                collection = DocumentCollection.objects.get(id=collection_id)
                embedding_model = collection.get_effective_embedding_model()
                if embedding_model:
                    embedding_model_id = embedding_model.id
                    logger.info(f"🔍 Using embedding model: {embedding_model.name} (ID={embedding_model.id})")
                else:
                    logger.warning("Collection has no embedding model, RAG may fail")
                rag_service = RAGService()
                logger.info(f"🔍 RAG service initialized with collection: {collection_id}")
            except DocumentCollection.DoesNotExist:
                logger.error(f"Collection {collection_id} not found, RAG disabled")
                use_rag = False

        results_dict = task.results or {}
        any_model_loaded = False
        total_questions = len(questions)

        current_model_idx = start_model_idx
        current_question_idx = start_question_idx

        for model_idx, model in enumerate(models[start_model_idx:], start_model_idx):
            current_model_idx = model_idx
            current_question_idx = 0

            if model.id in processed_models:
                logger.info(f"⏩ Skipping already processed model {model.name}")
                continue

            model_quant = quantization_override if quantization_override else model.quantization
            model_obj, tokenizer = _load_model_for_evaluation(model, quantization=model_quant)
            if model_obj is None:
                logger.error(f"Failed to load model {model.name}, skipping")
                continue

            any_model_loaded = True
            model_device = next(model_obj.parameters()).device
            logger.info(f"✅ Model {model.name} loaded on {model_device} (quant={model_quant})")

            model_results = {}

            for q_idx, q in enumerate(questions[start_question_idx:], start_question_idx):
                current_question_idx = q_idx

                q_id = q.get('id', q_idx + 1)
                q_text = q.get('text', '')
                if not q_text:
                    continue

                # === RAG: получаем контекст ===
                rag_context = ""
                rag_sources_count = 0
                rag_sources_info = []
                if rag_service and collection_id:
                    logger.info(f"   🔍 RAG search for question {q_id} with collection {collection_id}")
                    try:
                        search_result = rag_service.search(
                            query=q_text,
                            user_id='system',
                            collection_ids=collection_id,
                            top_k=5,
                            embedding_model_id=embedding_model_id
                        )
                        if search_result.get('success') and search_result.get('results'):
                            context_parts = []
                            for r in search_result['results'][:5]:
                                context_parts.append(
                                    f"[Документ: {r.get('document_title', 'Без названия')}]\n{r.get('content', '')}"
                                )
                                rag_sources_info.append({
                                    'title': r.get('document_title', 'Без названия'),
                                    'document_id': r.get('document_id'),
                                    'chunk_id': r.get('chunk_id'),
                                    'relevance': r.get('relevance_score', 0),
                                    'content': r.get('content', ''),
                                    'preview': r.get('content', '')[:200] + ('...' if len(r.get('content', '')) > 200 else '')
                                })
                            rag_context = "\n\n---\n\n".join(context_parts)
                            rag_sources_count = len(search_result['results'])
                            logger.info(f"   ✅ RAG: found {rag_sources_count} chunks for question {q_id}")
                        else:
                            logger.info(f"   ⚠️ RAG: no results for question {q_id}")
                    except Exception as e:
                        logger.error(f"❌ RAG search error for question {q_id}: {e}")
                        rag_context = ""
                else:
                    logger.info(f"   ❌ RAG disabled for question {q_id}, use_rag={use_rag}, collection_ids={collection_ids}")

                # Формируем промпт
                if rag_context:
                    prompt = (
                        f"Контекст из документов:\n{rag_context}\n\n"
                        f"Вопрос: {q_text}\n\n"
                        f"Ответ на основе документов:"
                    )
                else:
                    prompt = q_text

                inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(model_device)

                start_time = time.time()
                max_retries = 2
                retry_count = 0
                response = None
                response_time = 0
                rag_used = rag_context != ""

                while retry_count < max_retries:
                    try:
                        with torch.no_grad():
                            outputs = model_obj.generate(
                                **inputs,
                                max_new_tokens=generation_config.get('max_tokens', 512),
                                temperature=generation_config.get('temperature', 0.7),
                                top_p=generation_config.get('top_p', 0.9),
                                top_k=generation_config.get('top_k', 50),
                                repetition_penalty=generation_config.get('repetition_penalty', 1.0),
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )
                        response_time = time.time() - start_time
                        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        break  # успешно

                    except torch.cuda.OutOfMemoryError as e:
                        logger.error(f"CUDA OOM in evaluation for model {model.name}, question {q_id}: {e}. Moving model to CPU.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        model_obj.to('cpu')
                        # Пытаемся конвертировать в float32, но если модель квантизирована, пропускаем
                        try:
                            model_obj = model_obj.float()
                        except ValueError as float_err:
                            if "not supported for quantized model" in str(float_err):
                                logger.info("Model is quantized, skipping float conversion")
                            else:
                                raise
                        model_device = torch.device('cpu')
                        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(model_device)
                        retry_count += 1
                        continue

                    except RuntimeError as e:
                        err_msg = str(e).lower()
                        if "cublas" in err_msg or "cuda" in err_msg or "cudnn" in err_msg:
                            logger.error(f"CUDA-related error in evaluation for model {model.name}, question {q_id}: {e}. Moving model to CPU and retrying.")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            model_obj.to('cpu')
                            try:
                                model_obj = model_obj.float()
                            except ValueError as float_err:
                                if "not supported for quantized model" in str(float_err):
                                    logger.info("Model is quantized, skipping float conversion")
                                else:
                                    raise
                            model_device = torch.device('cpu')
                            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(model_device)
                            retry_count += 1
                            continue
                        else:
                            raise

                if response is None:
                    response = f"ERROR: {str(e)}"
                    response_time = 0
                    logger.error(f"Failed to generate for question {q_id} after {max_retries} attempts")

                logger.info(f"   ✅ Question {q_id} completed in {response_time:.2f}s (RAG: {rag_used})")

                result_entry = {
                    'response': response,
                    'response_time': response_time,
                    'rag_used': rag_used,
                }
                if rag_used:
                    result_entry['rag_sources_count'] = rag_sources_count
                    result_entry['rag_sources'] = rag_sources_info

                model_results[str(q_id)] = result_entry

                EvaluationResult.objects.update_or_create(
                    task=task,
                    model=model,
                    question_id=q_id,
                    defaults={
                        'question_text': q_text,
                        'model_response': response,
                        'response_time': response_time,
                        'tokens_used': len(tokenizer.encode(response)),
                        'metadata': {
                            'rag_used': rag_used,
                            'rag_sources_count': rag_sources_count if rag_used else 0,
                            'rag_sources': rag_sources_info if rag_used else []
                        },
                        'updated_at': timezone.now()
                    }
                )

                # Чекпоинт после каждого вопроса
                task.metadata['checkpoint'] = {
                    'last_model_idx': model_idx,
                    'last_question_idx': q_idx,
                    'processed_models': processed_models,
                    'timestamp': timezone.now().isoformat()
                }
                task.save(update_fields=['metadata'])

                progress = (model_idx * total_questions + q_idx + 1) / (len(models) * total_questions) * 100
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': progress,
                        'current_step': f'Model {model.name}: question {q_idx+1}/{total_questions}'
                    }
                )
                task.progress = progress
                task.current_step = f'Model {model.name}: question {q_idx+1}/{total_questions}'
                task.save(update_fields=['progress', 'current_step'])

            results_dict[str(model.id)] = model_results
            task.results = results_dict
            task.save(update_fields=['results'])

            processed_models.append(model.id)
            task.metadata['processed_models'] = processed_models
            task.save(update_fields=['metadata'])

            del model_obj
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            start_question_idx = 0

        if not any_model_loaded:
            error_msg = "No models could be loaded for evaluation."
            logger.error(error_msg)
            task.status = 'FAILED'
            task.error_message = error_msg
            task.completed_at = timezone.now()
            task.save()
            redis_model_manager.redis.delete(lock_key)
            return {'success': False, 'error': error_msg}

        if 'checkpoint' in task.metadata:
            del task.metadata['checkpoint']
        task.status = 'COMPLETED'
        task.progress = 100.0
        task.current_step = 'Завершено'
        task.completed_at = timezone.now()
        task.save()

        logger.info(f"🎉 Evaluation task {task_id} completed successfully")
        redis_model_manager.redis.delete(lock_key)
        return {'success': True, 'task_id': task_id, 'rag_used': use_rag}

    except Exception as e:
        logger.error(f"❌ Evaluation task {task_id} failed: {e}", exc_info=True)
        if task:
            try:
                task.metadata['checkpoint'] = {
                    'last_model_idx': current_model_idx,
                    'last_question_idx': current_question_idx,
                    'processed_models': processed_models,
                    'error': str(e),
                    'timestamp': timezone.now().isoformat()
                }
                task.status = 'FAILED'
                task.error_message = str(e)
                task.completed_at = timezone.now()
                task.save()
            except:
                pass
        redis_model_manager.redis.delete(lock_key)
        return {'success': False, 'error': str(e)}


@shared_task(queue='cpu_tasks')
def process_uploaded_dataset(file_path, dataset_id, file_format='auto'):
    """Обработка загруженного файла датасета."""
    from .models import EvaluationDataset
    try:
        dataset = EvaluationDataset.objects.get(id=dataset_id)
        success = dataset.add_questions_from_file(file_path, file_format)
        return {'success': success, 'dataset_id': dataset_id, 'questions_count': dataset.total_questions}
    except Exception as e:
        logger.error(f"Dataset processing error: {e}")
        return {'success': False, 'error': str(e)}


@shared_task(queue='cpu_tasks')
def export_evaluation_results(task_id, export_format='json'):
    """Экспорт результатов оценки (заглушка)."""
    return {'success': True, 'task_id': task_id, 'format': export_format}


# --------------------------------------------------------------------------
# FineTuning
# --------------------------------------------------------------------------
@shared_task(bind=True, queue='gpu_tasks', time_limit=86400)
def run_fine_tuning_task(self, job_id, restart=False):
    """
    Запуск fine-tuning с поддержкой чекпоинтов, fallback на CPU и логированием ошибок.
    """
    logger.info(f"🚀 run_fine_tuning_task: job_id={job_id}, restart={restart}")

    def update_progress(progress, step):
        self.update_state(
            state='PROGRESS',
            meta={'progress': progress, 'current_step': step}
        )

    try:
        job = FineTuningJob.objects.select_related('output_model', 'base_model').get(id=job_id)

        # Если restart=True – очищаем директорию модели (для перезапуска с нуля)
        if restart and job.output_model and job.output_model.local_path:
            import shutil
            output_dir = job.output_model.local_path
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)
                job.add_log("🧹 Директория модели очищена для перезапуска")

        job.status = 'PREPARING'
        job.save()
        job.add_log("Starting fine-tuning job...")

        # Пробуем GPU, при OOM переключаемся на CPU
        device = 'cuda'
        try:
            output_model = _run_training_on_device(job, device)
        except torch.cuda.OutOfMemoryError as e:
            job.add_log(f"⚠️ CUDA out of memory: {e}. Пробуем на CPU...")
            job.add_error(f"CUDA OOM: {e}")  # записываем ошибку, но продолжаем
            torch.cuda.empty_cache()
            device = 'cpu'
            output_model = _run_training_on_device(job, device)
        except Exception as e:
            # Любая другая ошибка – логируем и пробрасываем дальше
            job.add_error(str(e), traceback.format_exc())
            raise

        if output_model:
            output_model.status = 'DOWNLOADED'
            output_model.save()
            job.add_log(f"✅ Модель {output_model.display_name} сохранена и готова к использованию")

        # v5.1: тренер мог вернуть None при паузе — статус уже PAUSED
        if output_model is None:
            job.refresh_from_db()
            if job.status != 'PAUSED':
                job.status = 'PAUSED'
                job.save()
            logger.info(f"⏸ Fine-tuning job {job_id} paused — resume продолжит с чекпоинта")
            return {'success': True, 'status': 'PAUSED'}

        job.status = 'COMPLETED'
        job.save()
        return {'success': True, 'job_id': job_id, 'output_model_id': output_model.id}

    except FineTuningJob.DoesNotExist:
        logger.error(f"FineTuningJob {job_id} not found")
        return {'success': False, 'error': f'Job {job_id} not found'}
    except Exception as e:
        logger.error(f"Error in fine-tuning job {job_id}: {e}", exc_info=True)
        # Пытаемся сохранить ошибку в задаче, если она ещё существует
        try:
            job.add_error(str(e), traceback.format_exc())
            job.fail(str(e), {'traceback': traceback.format_exc()})
        except:
            pass
        return {'success': False, 'error': str(e)}


def _run_training_on_device(job, device):
    """
    Выбирает нужный тренер и запускает обучение на указанном устройстве.
    v5.1: job.gpu_device переопределяет device (0..N = cuda:N, -1 = cpu).
    """
    # v5.1: явный выбор GPU из формы перекрывает авто-определение
    if job.gpu_device is not None:
        if job.gpu_device == -1:
            device = 'cpu'
        elif job.gpu_device >= 0 and torch.cuda.is_available():
            device = f'cuda:{job.gpu_device}'
        logger.info(f"🎯 v5.1: GPU выбор из формы → device={device}")

    if job.method == 'document':
        from .services.fine_tuning.document_trainer import DocumentTrainer
        trainer = DocumentTrainer(job, device=device)
    elif job.method == 'instruction':
        from .services.fine_tuning.instruction_trainer import InstructionTrainer
        trainer = InstructionTrainer(job, device=device)
    elif job.method == 'raft':
        from .services.fine_tuning.raft_trainer import RAFTTrainer
        trainer = RAFTTrainer(job, device=device)
    else:
        raise ValueError(f"Unknown method: {job.method}")
    return trainer.train()


@shared_task(queue='cpu_tasks')
def resume_fine_tuning_task(job_id):
    """v5.1: резюме файнтюна — повторный прогон подхватывает чекпоинт."""
    return run_fine_tuning_task(job_id)


@shared_task(queue='cpu_tasks')
def evaluate_fine_tuned_model_task(job_id, dataset_id):
    """Оценка дообученной модели на датасете"""
    from .models import FineTuningJob, EvaluationDataset
    
    try:
        job = FineTuningJob.objects.get(id=job_id)
        dataset = EvaluationDataset.objects.get(id=dataset_id)
        
        evaluator = FineTuningEvaluator(job)
        metrics = evaluator.evaluate(dataset)
        
        return {
            'success': True,
            'job_id': job_id,
            'dataset_id': dataset_id,
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}

# --------------------------------------------------------------------------
# FineTuning
# --------------------------------------------------------------------------

@shared_task(bind=True, queue='cpu_tasks')
def convert_model_to_gguf_task(self, job_id):
    logger.info(f"🚀 convert_model_to_gguf_task: job_id={job_id}")

    def update_progress(progress, step):
        self.update_state(
            state='PROGRESS',
            meta={'progress': progress, 'current_step': step}
        )

    try:
        job = ModelConversionJob.objects.select_related('source_model').get(id=job_id)
        job.status = 'PREPARING'
        job.save()
        job.add_log("Starting GGUF conversion...")
        update_progress(5, "Подготовка")

        src_model = job.source_model
        if not src_model.local_path or not os.path.exists(src_model.local_path):
            raise FileNotFoundError(f"Source model path not found: {src_model.local_path}")

        # Определяем выходной путь
        output_dir = os.path.join(settings.MEDIA_ROOT, 'converted_models')
        os.makedirs(output_dir, exist_ok=True)
        safe_name = src_model.safe_name
        output_filename = f"{safe_name}_{job.quantization}.gguf"
        output_path = os.path.join(output_dir, output_filename)

        # Путь к скрипту конвертации (можно настроить в settings)
        convert_script = getattr(settings, 'LLAMA_CPP_CONVERT_SCRIPT', '/opt/llama.cpp/convert.py')
        if not os.path.exists(convert_script):
            # Попробуем найти в проекте
            convert_script = os.path.join(settings.BASE_DIR, '..', 'llama.cpp', 'convert.py')

        # Формируем команду
        cmd = [
            'python', convert_script,
            src_model.local_path,
            '--outfile', output_path,
            '--outtype', job.quantization,
        ]
        # Добавляем дополнительные параметры из job.config
        for key, value in job.config.items():
            cmd.append(f'--{key}')
            if value is not None:
                cmd.append(str(value))

        job.status = 'CONVERTING'
        job.save()
        job.add_log(f"Running command: {' '.join(cmd)}")
        update_progress(10, "Конвертация")

        # Запускаем процесс
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Читаем вывод построчно
        for line in process.stdout:
            line = line.strip()
            if line:
                job.add_log(line)
                logger.info(f"[convert] {line}")

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Conversion failed with code {process.returncode}")

        job.add_log("Conversion completed, creating output model entry...")
        update_progress(90, "Сохранение модели")

        # Создаём новый чекпоинт
        new_checkpoint = ModelCheckpoint.objects.create(
            name=f"{src_model.name}_gguf_{job.quantization}",
            display_name=f"{src_model.display_name} (GGUF, {job.quantization})",
            model_type=src_model.model_type,
            huggingface_id=src_model.huggingface_id,
            huggingface_revision=src_model.huggingface_revision,
            loader_config={
                'format': 'gguf',
                'quantization': job.quantization,
            },
            local_path=output_path,
            relative_path=f"converted_models/{output_filename}",
            status='DOWNLOADED',
            parameter_count=src_model.parameter_count,
            quantization='gguf',
            context_length=src_model.context_length,
            vocab_size=src_model.vocab_size,
            license_type=src_model.license_type,
            is_safe=src_model.is_safe,
            safety_score=src_model.safety_score,
            parent=src_model,
            is_active=True,
            tags=src_model.tags + ['gguf']
        )

        job.complete(new_checkpoint)
        update_progress(100, "Готово")
        return {'success': True, 'job_id': job_id, 'output_model_id': new_checkpoint.id}

    except ModelConversionJob.DoesNotExist:
        logger.error(f"ConversionJob {job_id} not found")
        return {'success': False, 'error': f'Job {job_id} not found'}
    except Exception as e:
        logger.error(f"Conversion error: {e}", exc_info=True)
        job.fail(str(e), {'traceback': traceback.format_exc()})
        return {'success': False, 'error': str(e)}


# --------------------------------------------------------------------------
# ЗАДАЧИ RAG (реализации в tasks_rag.py, здесь только импорт)
# --------------------------------------------------------------------------


@shared_task(bind=True, queue='gpu_tasks', max_retries=3)
def create_embeddings_for_collection_task(self, collection_id):
    return _create_embeddings_for_collection_impl(self, collection_id)


@shared_task(queue='cpu_tasks')
def build_faiss_index_task(collection_id, index_type='flat'):
    return _build_faiss_index_impl(collection_id, index_type)


@shared_task(queue='cpu_tasks')
def update_faiss_index_task(collection_id):
    return _update_faiss_index_impl(collection_id)


@shared_task(bind=True, queue='cpu_tasks')
def regenerate_document_chunks_task(self, document_id):
    return _regenerate_document_chunks_impl(self, document_id)


# --------------------------------------------------------------------------
# ПЕРИОДИЧЕСКИЕ ЗАДАЧИ
# --------------------------------------------------------------------------
@shared_task(queue='cpu_tasks')
def health_check_task():
    """Проверка здоровья системы."""
    from .models import ModelCheckpoint, DocumentCollection, ModelDownloadJob
    try:
        redis_model_manager.redis.ping()
        redis_ok = True
    except:
        redis_ok = False

    return {
        'success': True,
        'timestamp': timezone.now().isoformat(),
        'total_models': ModelCheckpoint.objects.filter(is_active=True).count(),
        'loaded_models': ModelCheckpoint.objects.filter(status='LOADED', is_active=True).count(),
        'active_downloads': ModelDownloadJob.objects.filter(status__in=['PENDING', 'DOWNLOADING']).count(),
        'total_collections': DocumentCollection.objects.filter(is_active=True).count(),
        'redis_connected': redis_ok,
    }


@shared_task(queue='cpu_tasks')
def check_system_health_task():
    """Алиас для health_check_task."""
    return health_check_task()


@shared_task(queue='cpu_tasks')
def update_collection_statistics_task():
    """Обновление статистики всех активных коллекций."""
    from .models import DocumentCollection
    for col in DocumentCollection.objects.filter(is_active=True):
        col.update_statistics()
    return {'success': True}


@shared_task(queue='cpu_tasks')
def validate_documents_task():
    """Валидация документов (заглушка)."""
    return {'success': True}


@shared_task(queue='cpu_tasks')
def cleanup_vector_index_task():
    """Очистка устаревших индексов (заглушка)."""
    return {'success': True}


@shared_task(queue='cpu_tasks')
def generate_system_report_task():
    """Генерация отчёта (заглушка)."""
    return {'success': True}


@shared_task(queue='cpu_tasks')
def monitor_processing_tasks():
    """Мониторинг задач обработки (заглушка)."""
    return {'success': True}


@shared_task(queue='cpu_tasks')
def auto_clean_memory():
    """Автоматическая очистка памяти каждые 30 минут"""
    import gc
    import torch
    
    logger.info("🧹 Автоматическая очистка памяти")
    
    # Проверяем, есть ли загруженные модели
    loaded_models = redis_model_manager.list_loaded_models()
    
    if not loaded_models:
        # Если моделей нет - чистим всё
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("✅ Память очищена (нет активных моделей)")
    else:
        logger.info(f"⏸️ Пропускаем очистку: загружено {len(loaded_models)} моделей")
    
    return {'cleaned': len(loaded_models) == 0}


# --------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ СИНХРОННОЙ ЗАГРУЗКИ УЧИТЕЛЯ
# --------------------------------------------------------------------------
def _load_teacher_model_sync(teacher_model_id, quantization=None):
    """Синхронная загрузка модели-учителя с ожиданием, если она загружается в другом процессе."""
    from .models import ModelCheckpoint
    from .tasks import _load_model_for_generation
    import time

    teacher = ModelCheckpoint.objects.get(id=teacher_model_id)
    model_identifier = teacher.huggingface_id or teacher.name

    # Если модель уже загружена в этом процессе, возвращаем её
    if model_identifier in redis_model_manager.local_models:
        return teacher

    # Проверяем, не загружается ли модель в другом процессе
    loading_status = redis_model_manager.get_loading_status(model_identifier)
    if loading_status.get('status') == 'LOADING':
        # Ждём до 30 секунд (6 попыток по 5 секунд)
        for attempt in range(6):
            time.sleep(5)
            # Проверяем, не появилась ли модель в локальном кэше
            if model_identifier in redis_model_manager.local_models:
                return teacher
            # Проверяем статус снова
            loading_status = redis_model_manager.get_loading_status(model_identifier)
            if loading_status.get('status') == 'LOADED':
                # Модель загружена в другом процессе, но не в этом – нужно загрузить локально
                break
            if loading_status.get('status') == 'FAILED':
                break
        # Если после ожидания модель всё ещё не загружена локально, пробуем загрузить
        if model_identifier not in redis_model_manager.local_models:
            load_result = _load_model_for_generation(teacher, quantization)
            if not load_result.get('success'):
                raise Exception(f"Не удалось загрузить модель учителя: {load_result.get('error')}")
    else:
        # Модель не загружается, загружаем
        load_result = _load_model_for_generation(teacher, quantization)
        if not load_result.get('success'):
            raise Exception(f"Не удалось загрузить модель учителя: {load_result.get('error')}")

    return teacher


# --------------------------------------------------------------------------
# ЗАДАЧА ГЕНЕРАЦИИ ВОПРОСОВ ДЛЯ ВСЕХ ГРУПП НАБОРА
# --------------------------------------------------------------------------
@shared_task(bind=True, queue='gpu_tasks', max_retries=3)
def generate_questions_for_set_task(self, question_set_id, teacher_model_id, quantization=None):
    logger.info(f"🔍 generate_questions_for_set_task: set_id={question_set_id}, teacher={teacher_model_id}")

    try:
        qset = DistillationQuestionSet.objects.get(id=question_set_id)
        if qset.status == 'COMPLETED':
            return {'success': True, 'message': 'Already completed'}

        # Загружаем учителя
        teacher = _load_teacher_model_sync(teacher_model_id, quantization)
        model_identifier = teacher.huggingface_id or teacher.name

        # Параметры генерации
        config = qset.default_generation_config or {}
        prompt_template = (
            "На основе текста сгенерируй {num_questions} вопроса. Каждый вопрос пиши с новой строки, только вопрос, без пояснений.\n\n"
            "Пример:\n"
            "Текст: Вода кипит при 100 градусах Цельсия.\n"
            "Вопросы:\n"
            "При какой температуре кипит вода?\n"
            "Что происходит с водой при нагревании до 100°C?\n\n"
            "Текст:\n{context}\n\n"
            "Вопросы:"
        )
        max_tokens = config.get('max_tokens', 512)
        temperature = config.get('temperature', 0.7)
        top_p = config.get('top_p', 0.9)
        top_k = config.get('top_k', 50)
        repetition_penalty = config.get('repetition_penalty', 1.0)

        # Получаем группы в порядке order
        groups = qset.groups.all().order_by('order')
        total_groups = groups.count()
        if total_groups == 0:
            raise ValueError("Нет групп для обработки")

        # Восстанавливаем чекпоинт (последняя обработанная группа)
        checkpoint = qset.checkpoint_data or {}
        start_group_idx = checkpoint.get('last_group_idx', 0)

        qset.status = 'RUNNING'
        qset.save(update_fields=['status'])

        # Кэш для чанков (чтобы не дергать БД на каждый вопрос)
        chunk_cache = {}

        for idx, group in enumerate(groups[start_group_idx:], start=start_group_idx):
            # Загружаем чанки для группы
            missing = [cid for cid in group.chunk_ids if cid not in chunk_cache]
            if missing:
                chunks = DocumentChunk.objects.filter(id__in=missing, is_active=True)
                for ch in chunks:
                    chunk_cache[ch.id] = ch.content

            # Формируем контекст из чанков в порядке их следования
            context_parts = []
            for cid in group.chunk_ids:
                if cid in chunk_cache:
                    context_parts.append(chunk_cache[cid])
            context = "\n\n---\n\n".join(context_parts)

            # Получаем количество вопросов для этой группы (может быть переопределено)
            num_questions = group.num_questions
            prompt = prompt_template.format(context=context, num_questions=num_questions)

            # Генерируем вопросы
            result = redis_model_manager.generate_text(
                model_id=model_identifier,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )

            if result.get('success'):
                raw_questions = result['response'].strip().split('\n')
                questions = [q.strip() for q in raw_questions if q.strip()][:num_questions]
                created = 0
                for q_text in questions:
                    DistillationQuestion.objects.create(
                        group=group,
                        question_text=q_text,
                        metadata={
                            'teacher': teacher_model_id,
                            'config': config,
                            'context_chunk_ids': group.chunk_ids
                        }
                    )
                    created += 1
                group.logs = f"Сгенерировано {created} вопросов"
                group.save(update_fields=['logs'])
                qset.logs += f"Группа {idx+1}/{total_groups}: {created} вопросов\n"
            else:
                error = result.get('error', 'Unknown error')
                group.logs = f"Ошибка: {error}"
                group.save(update_fields=['logs'])
                qset.logs += f"Группа {idx+1}/{total_groups} ошибка: {error}\n"

            # Сохраняем чекпоинт после каждой группы
            qset.checkpoint_data = {
                'last_group_idx': idx,
                'total_groups': total_groups,
                'teacher_model_id': teacher_model_id,
                'quantization': quantization,
                'timestamp': timezone.now().isoformat()
            }
            qset.last_processed = timezone.now()
            progress = (idx + 1) / total_groups * 100
            qset.progress = progress
            qset.current_step = f'Группа {idx+1}/{total_groups}'
            qset.save(update_fields=['progress', 'current_step', 'logs', 'checkpoint_data', 'last_processed'])

            self.update_state(state='PROGRESS', meta={'progress': progress, 'current_step': qset.current_step})

        # Завершение
        qset.status = 'COMPLETED'
        qset.progress = 100.0
        qset.current_step = 'Завершено'
        qset.checkpoint_data = {}  # очищаем чекпоинт
        qset.save(update_fields=['status', 'progress', 'current_step', 'checkpoint_data'])
        logger.info(f"✅ Генерация вопросов для набора {question_set_id} завершена")
        return {'success': True, 'question_set_id': question_set_id, 'total_questions': DistillationQuestion.objects.filter(group__question_set=qset).count()}

    except Exception as e:
        logger.error(f"Ошибка в generate_questions_for_set_task: {e}", exc_info=True)
        try:
            qset = DistillationQuestionSet.objects.get(id=question_set_id)
            qset.status = 'FAILED'
            qset.logs += f"\nОшибка: {str(e)}\n"
            qset.save(update_fields=['status', 'logs'])
        except:
            pass
        raise self.retry(exc=e, countdown=60) from e


# --------------------------------------------------------------------------
# ЗАДАЧА ГЕНЕРАЦИИ ОТВЕТОВ ДЛЯ ВОПРОСОВ
# --------------------------------------------------------------------------
@shared_task(bind=True, queue='gpu_tasks', max_retries=3)
def generate_answers_for_questions_task(self, answer_job_id, teacher_model_id, quantization=None):
    logger.info(f"📝 generate_answers_for_questions_task: job_id={answer_job_id}, teacher={teacher_model_id}")

    try:
        job = DistillationAnswerJob.objects.get(id=answer_job_id)
        if job.status == 'COMPLETED':
            return {'success': True, 'message': 'Already completed'}

        teacher = _load_teacher_model_sync(teacher_model_id, quantization)
        model_identifier = teacher.huggingface_id or teacher.name

        config = job.generation_config or {}
        prompt_template = config.get('prompt_template', (
            "Ответь на вопрос, используя только информацию из предоставленного контекста.\n\n"
            "Контекст:\n{context}\n\n"
            "Вопрос: {question}\n\nОтвет:"
        ))
        max_tokens = config.get('max_tokens', 1024)
        temperature = config.get('temperature', 0.6)
        top_p = config.get('top_p', 0.9)
        top_k = config.get('top_k', 50)
        repetition_penalty = config.get('repetition_penalty', 1.0)

        # Определяем список вопросов
        if job.question_ids:
            questions = DistillationQuestion.objects.filter(id__in=job.question_ids)
        elif job.question_set:
            questions = DistillationQuestion.objects.filter(group__question_set=job.question_set)
        else:
            raise ValueError("Не указаны вопросы для обработки")

        total = questions.count()
        if total == 0:
            job.status = 'COMPLETED'
            job.save()
            return {'success': True, 'message': 'No questions'}

        if job.planned_count == 0:
            job.planned_count = total
            job.save(update_fields=['planned_count'])

        # Чекпоинт
        start_index = 0
        if job.checkpoint_data and 'last_processed_question_index' in job.checkpoint_data:
            start_index = job.checkpoint_data['last_processed_question_index'] + 1
            job.logs += f"Возобновление с вопроса {start_index+1}/{total}\n"
            job.save()

        job.status = 'RUNNING'
        job.save(update_fields=['status'])

        chunk_cache = {}

        for idx, q in enumerate(questions.order_by('id')[start_index:], start=start_index):
            # Получаем контекст из группы чанков
            group = q.group
            context = ""
            if group and group.chunk_ids:
                missing = [cid for cid in group.chunk_ids if cid not in chunk_cache]
                if missing:
                    chunks = DocumentChunk.objects.filter(id__in=missing, is_active=True)
                    for ch in chunks:
                        chunk_cache[ch.id] = ch.content
                context_parts = [chunk_cache[cid] for cid in group.chunk_ids if cid in chunk_cache]
                if context_parts:
                    context = "\n\n---\n\n".join(context_parts)

            prompt = prompt_template.format(context=context, question=q.question_text)

            result = redis_model_manager.generate_text(
                model_id=model_identifier,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )

            if result.get('success'):
                DistillationAnswer.objects.create(
                    question=q,
                    answer_job=job,
                    answer_text=result['response']
                )
                processed = DistillationAnswer.objects.filter(answer_job=job).count()
                job.processed_count = processed
                job.logs += f"Вопрос {idx+1}/{total} обработан\n"
            else:
                error = result.get('error', 'Unknown error')
                job.logs += f"Вопрос {idx+1}/{total} ошибка: {error}\n"

            # Сохраняем чекпоинт
            job.checkpoint_data = {
                'last_processed_question_index': idx,
                'total_questions': total,
                'timestamp': timezone.now().isoformat()
            }
            job.last_processed = timezone.now()
            progress = processed / job.planned_count * 100 if job.planned_count > 0 else 0
            job.progress = progress
            job.current_step = f'Вопрос {processed}/{job.planned_count}'
            job.save(update_fields=['progress', 'current_step', 'processed_count', 'logs', 'checkpoint_data', 'last_processed'])

            self.update_state(state='PROGRESS', meta={'progress': progress, 'current_step': job.current_step})

        job.status = 'COMPLETED'
        job.progress = 100.0
        job.current_step = 'Завершено'
        job.checkpoint_data = {}
        job.save(update_fields=['status', 'progress', 'current_step', 'checkpoint_data'])
        logger.info(f"✅ Генерация ответов для задачи {answer_job_id} завершена, обработано {job.processed_count}/{job.planned_count}")
        return {'success': True, 'answer_job_id': answer_job_id, 'processed': job.processed_count, 'total': job.planned_count}

    except Exception as e:
        logger.error(f"Ошибка в generate_answers_for_questions_task: {e}", exc_info=True)
        try:
            job = DistillationAnswerJob.objects.get(id=answer_job_id)
            job.status = 'FAILED'
            job.logs += f"\nОшибка: {str(e)}\n"
            job.save(update_fields=['status', 'logs'])
        except:
            pass
        raise self.retry(exc=e, countdown=60) from e

@shared_task(bind=True, queue='gpu_tasks')
def regenerate_answer_task(self, question_id, answer_job_id, teacher_model_id, quantization, generation_config):
    logger.info(f"🔄 Regenerating answer for question {question_id} in job {answer_job_id}")
    try:
        question = DistillationQuestion.objects.get(id=question_id)
        job = DistillationAnswerJob.objects.get(id=answer_job_id)

        # Загружаем модель-учитель с повторными попытками
        teacher = _load_teacher_model_sync(teacher_model_id, quantization)
        model_identifier = teacher.huggingface_id or teacher.name

        # Формируем контекст
        context = ""
        group = question.group
        if group and group.chunk_ids:
            chunks = DocumentChunk.objects.filter(id__in=group.chunk_ids, is_active=True)
            context = "\n\n---\n\n".join([chunk.content for chunk in chunks])

        gen_config = generation_config or job.generation_config or {}
        
        # Обработка prompt_template
        prompt_template = gen_config.get('prompt_template')
        if not prompt_template:
            prompt_template = (
                "Ответь на вопрос, используя только информацию из предоставленного контекста.\n\n"
                "Контекст:\n{context}\n\n"
                "Вопрос: {question}\n\nОтвет:"
            )
        
        max_tokens = gen_config.get('max_tokens', 1024)
        temperature = gen_config.get('temperature', 0.6)
        top_p = gen_config.get('top_p', 0.9)
        top_k = gen_config.get('top_k', 50)
        repetition_penalty = gen_config.get('repetition_penalty', 1.0)

        prompt = prompt_template.format(context=context, question=question.question_text)

        result = redis_model_manager.generate_text(
            model_id=model_identifier,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )

        if result.get('success'):
            new_answer = DistillationAnswer.objects.create(
                question=question,
                answer_job=job,
                answer_text=result['response']
            )
            logger.info(f"✅ Answer regenerated for question {question_id}")
            return {'success': True, 'answer_id': new_answer.id, 'answer_text': new_answer.answer_text}
        else:
            error = result.get('error', 'Unknown error')
            logger.error(f"Failed to regenerate answer: {error}")
            return {'success': False, 'error': error}
    except Exception as e:
        logger.error(f"Error in regenerate_answer_task: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


# --------------------------------------------------------------------------
# ЗАДАЧА ЗАГРУЗКИ ДАТАСЕТА С ХФ
# --------------------------------------------------------------------------
@shared_task(bind=True, queue='cpu_tasks')
def import_huggingface_dataset_task(self, dataset_id):
    logger.info(f"📥 Importing Hugging Face dataset: {dataset_id}")
    try:
        hf_dataset = HuggingFaceDataset.objects.get(id=dataset_id)
        hf_dataset.status = 'DOWNLOADING'
        hf_dataset.save()

        # Загрузка датасета
        load_kwargs = {'split': hf_dataset.split}
        if hf_dataset.config:
            load_kwargs['name'] = hf_dataset.config
        dataset = load_dataset(hf_dataset.repo_id, **load_kwargs)

        # Преобразование в список инструкций
        items = []

        # Пытаемся определить структуру
        if isinstance(dataset, dict):
            # Может быть несколько сплитов, берём первый
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]

        # Проверяем доступные поля
        sample = dataset[0]
        available_fields = list(sample.keys())
        logger.info(f"Available fields in dataset: {available_fields}")

        # Поиск полей для инструкции и ответа
        instruction_field = None
        output_field = None

        # Стандартные имена
        for field in available_fields:
            field_lower = field.lower()
            if field_lower in ['instruction', 'prompt', 'question', 'input']:
                instruction_field = field
            elif field_lower in ['output', 'response', 'answer', 'completion']:
                output_field = field

        # Если есть поле 'conversations' (список сообщений)
        if 'conversations' in available_fields:
            for item in dataset:
                conversations = item['conversations']
                # Ищем последнее сообщение ассистента
                last_assistant = None
                last_user = None
                for msg in reversed(conversations):
                    if msg.get('role') == 'assistant' and not last_assistant:
                        last_assistant = msg.get('content', '')
                    if msg.get('role') == 'user' and not last_user:
                        last_user = msg.get('content', '')
                    if last_assistant and last_user:
                        break
                if last_user and last_assistant:
                    items.append({'instruction': last_user, 'output': last_assistant})
            if items:
                logger.info(f"Extracted {len(items)} items from conversations field")

        # Если есть поле 'sample' (диалог в текстовом формате или список)
        elif 'sample' in available_fields:
            for item in dataset:
                sample_text = item['sample']
                # Преобразование в строку, если не строка
                if isinstance(sample_text, list):
                    # Проверяем, не является ли список диалогом
                    if all(isinstance(s, dict) and 'role' in s and 'content' in s for s in sample_text):
                        # Это похоже на conversations
                        last_assistant = None
                        last_user = None
                        for msg in reversed(sample_text):
                            if msg.get('role') == 'assistant' and not last_assistant:
                                last_assistant = msg.get('content', '')
                            if msg.get('role') == 'user' and not last_user:
                                last_user = msg.get('content', '')
                            if last_assistant and last_user:
                                break
                        if last_user and last_assistant:
                            items.append({'instruction': last_user, 'output': last_assistant})
                            continue
                    else:
                        # Обычный список строк – склеиваем
                        sample_text = '\n'.join([str(s) for s in sample_text])
                elif not isinstance(sample_text, str):
                    sample_text = str(sample_text)

                # Если после преобразования это строка, парсим
                if isinstance(sample_text, str):
                    # Пытаемся извлечь последний вопрос и ответ
                    import re
                    assistant_patterns = [
                        r'Assistant:\s*(.+?)(?=\n\s*(?:User:|Q:|$))',
                        r'A:\s*(.+?)(?=\n\s*(?:Q:|User:|$))',
                        r'Ответ:\s*(.+?)(?=\n\s*(?:Вопрос:|$))',
                    ]
                    user_patterns = [
                        r'User:\s*(.+?)(?=\n\s*(?:Assistant:|A:|$))',
                        r'Q:\s*(.+?)(?=\n\s*(?:A:|$))',
                        r'Вопрос:\s*(.+?)(?=\n\s*(?:Ответ:|$))',
                    ]
                    assistant = None
                    user = None
                    for pat in assistant_patterns:
                        match = re.search(pat, sample_text, re.DOTALL)
                        if match:
                            assistant = match.group(1).strip()
                            break
                    for pat in user_patterns:
                        match = re.search(pat, sample_text, re.DOTALL)
                        if match:
                            user = match.group(1).strip()
                            break
                    if user and assistant:
                        items.append({'instruction': user, 'output': assistant})
                        continue
                    # Если не удалось, пробуем разбить по строкам и взять последние две непустые строки
                    lines = [l.strip() for l in sample_text.split('\n') if l.strip()]
                    if len(lines) >= 2:
                        last_two = lines[-2:]
                        items.append({'instruction': last_two[0], 'output': last_two[1]})
            if items:
                logger.info(f"Extracted {len(items)} items from sample field")

        # Если есть стандартные поля
        elif instruction_field and output_field:
            for item in dataset:
                instruction = item.get(instruction_field, '').strip()
                output = item.get(output_field, '').strip()
                if instruction and output:
                    items.append({'instruction': instruction, 'output': output})
            logger.info(f"Extracted {len(items)} items from {instruction_field}/{output_field}")

        # Если ничего не нашлось, выводим подробное сообщение об ошибке
        if not items:
            # Выводим пример первого элемента для отладки
            first_sample = next(iter(dataset)) if dataset else {}
            logger.error(f"Cannot extract instruction-output pairs. First sample: {first_sample}")
            raise ValueError(f"Не удалось определить поля инструкции и ответа. Доступные поля: {available_fields}. Пример: {first_sample}")

        # Сохраняем в JSONL
        dataset_name = hf_dataset.name or hf_dataset.repo_id.replace('/', '_')
        file_name = f"hf_{dataset_name}_{hf_dataset.split}.jsonl"
        file_path = os.path.join(settings.MEDIA_ROOT, 'evaluation_datasets', file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            for item in items:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

        # Создаём EvaluationDataset
        eval_dataset = EvaluationDataset.objects.create(
            name=dataset_name,
            description=hf_dataset.description or f"Импортировано из {hf_dataset.repo_id}",
            file_path=file_path,
            source_type='API',
            created_by=hf_dataset.created_by,
            total_questions=len(items)  # только количество, без самих данных
        )
        eval_dataset.save()

        hf_dataset.status = 'COMPLETED'
        hf_dataset.total_entries = len(items)
        hf_dataset.file_path = file_path
        hf_dataset.save()

        logger.info(f"✅ Imported {len(items)} items from {hf_dataset.repo_id}")
        return {'success': True, 'dataset_id': eval_dataset.id, 'items': len(items)}

    except Exception as e:
        logger.error(f"Error importing dataset: {e}", exc_info=True)
        try:
            hf_dataset = HuggingFaceDataset.objects.get(id=dataset_id)
            hf_dataset.status = 'FAILED'
            hf_dataset.error_message = str(e)
            hf_dataset.save()
        except:
            pass
        return {'success': False, 'error': str(e)}