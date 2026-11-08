from __future__ import absolute_import, unicode_literals

import json
import logging
import multiprocessing
import os
from pathlib import Path

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from kombu.serialization import register

# Устанавливаем метод запуска процессов для избежания проблем с fork
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch

# Кастомный JSON encoder для Path объектов
class PathAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
        except ImportError:
            pass
        return super().default(obj)

def pathaware_dumps(obj):
    return json.dumps(obj, cls=PathAwareJSONEncoder)

def pathaware_loads(obj):
    return json.loads(obj)

register(
    'pathaware_json',
    pathaware_dumps,
    pathaware_loads,
    content_type='application/json',
    content_encoding='utf-8'
)

# Устанавливаем переменную окружения Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

# Инициализация Django
import django
django.setup()

# Создаём экземпляр Celery
app = Celery('config')

# Загружаем конфигурацию из Django settings (namespace CELERY)
app.config_from_object('django.conf:settings', namespace='CELERY')

# Автоматически находим задачи в установленных приложениях
app.autodiscover_tasks()

# Конфигурация очередей
app.conf.task_queues = {
    'gpu_tasks': {
        'exchange': 'gpu_tasks',
        'routing_key': 'gpu_tasks',
    },
    'cpu_tasks': {
        'exchange': 'cpu_tasks',
        'routing_key': 'cpu_tasks',
    },
}

# Маршрутизация задач
app.conf.task_routes = {
    # GPU задачи (требуют CUDA)
    'ivanna_model_manager.tasks.load_model_to_memory_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.unload_model_from_memory_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.generate_text_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.merge_models_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.run_evaluation_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.run_fine_tuning_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.process_document_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.create_embeddings_for_collection_task': {'queue': 'gpu_tasks'},
    
    # CPU задачи (могут выполняться на любом воркере)
    'ivanna_model_manager.tasks.download_model_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.cleanup_download_jobs': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.sync_model_statuses_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.refresh_model_ttl': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.health_check_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.process_uploaded_dataset': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.build_faiss_index_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.update_faiss_index_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.regenerate_document_chunks_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.cleanup_stale_model_locks': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.cleanup_stale_models': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.auto_clean_memory': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.evaluate_fine_tuned_model_task': {'queue': 'cpu_tasks'},
}

# --------------------------------------------------------------------------
# Сигналы для инициализации/очистки GPU
# --------------------------------------------------------------------------
@worker_process_init.connect
def init_worker(**kwargs):
    import logging
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Initializing Celery worker")
    
    # Очищаем кэш моделей при старте воркера
    try:
        from ivanna_model_manager.services.redis_model_service import redis_model_manager
        redis_model_manager.clean_stale_locks()
        logger.info("Cleaned stale model locks")
    except Exception as e:
        logger.error(f"Failed to clean stale locks: {e}")
    
    logger.info("=" * 50)


@worker_process_shutdown.connect
def shutdown_worker(**kwargs):
    """Очистка при завершении воркера"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Выгружаем все модели, загруженные этим воркером
        from ivanna_model_manager.services.redis_model_service import redis_model_manager
        
        for model_id in list(redis_model_manager.local_models.keys()):
            logger.info(f"🔄 Unloading {model_id} on worker shutdown")
            redis_model_manager.unload_model(model_id, force=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("🧹 Cleared CUDA cache on shutdown")
            
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")


# --------------------------------------------------------------------------
# Периодические задачи (Celery Beat)
# --------------------------------------------------------------------------
from celery.schedules import crontab

app.conf.beat_schedule = {
    # Синхронизация статусов моделей каждые 5 минут
    'sync-model-statuses-every-5-minutes': {
        'task': 'ivanna_model_manager.tasks.sync_model_statuses_task',
        'schedule': crontab(minute='*/5'),
    },
    
    # Обновление TTL моделей в Redis каждые 5 минут
    'refresh-model-ttl-every-5-minutes': {
        'task': 'ivanna_model_manager.tasks.refresh_model_ttl',
        'schedule': 300.0,
    },
    
    # Проверка здоровья каждые 15 минут
    'health-check-every-15-minutes': {
        'task': 'ivanna_model_manager.tasks.health_check_task',
        'schedule': 900.0,
    },
    
    # Очистка завершённых задач загрузки ежедневно в 3:00
    'cleanup-download-jobs-daily': {
        'task': 'ivanna_model_manager.tasks.cleanup_download_jobs',
        'schedule': crontab(minute=0, hour=3),
    },
    
    # Обновление статистики коллекций ежечасно
    'update-collection-statistics-hourly': {
        'task': 'ivanna_model_manager.tasks.update_collection_statistics_task',
        'schedule': crontab(minute=0),
    },
    
    # Валидация документов ежедневно в 2:00
    'validate-documents-daily': {
        'task': 'ivanna_model_manager.tasks.validate_documents_task',
        'schedule': crontab(minute=0, hour=2),
    },
    
    # Очистка векторных индексов ежедневно в 2:00
    'cleanup-vector-index-daily': {
        'task': 'ivanna_model_manager.tasks.cleanup_vector_index_task',
        'schedule': crontab(minute=0, hour=2),
    },
    
    # Проверка здоровья системы ежечасно
    'check-system-health-hourly': {
        'task': 'ivanna_model_manager.tasks.check_system_health_task',
        'schedule': crontab(minute=30, hour='*/1'),
    },
    
    # Генерация еженедельного отчёта в понедельник в 9:00
    'generate-system-report-weekly': {
        'task': 'ivanna_model_manager.tasks.generate_system_report_task',
        'schedule': crontab(minute=0, hour=9, day_of_week=0),
    },
    
    # Очистка зависших блокировок моделей каждые 5 минут
    'cleanup-stale-model-locks': {
        'task': 'ivanna_model_manager.tasks.cleanup_stale_model_locks',
        'schedule': crontab(minute='*/5'),
    },
    
    # Очистка зависших моделей каждые 30 минут
    'cleanup-stale-models-hourly': {
        'task': 'ivanna_model_manager.tasks.cleanup_stale_models',
        'schedule': crontab(minute='*/30'),
    },
    
    # Автоматическая очистка памяти каждые 30 минут
    'auto-clean-memory-hourly': {
        'task': 'ivanna_model_manager.tasks.auto_clean_memory',
        'schedule': crontab(minute='*/30'),
    },
}

# --------------------------------------------------------------------------
# Дополнительные настройки Celery
# --------------------------------------------------------------------------
app.conf.task_acks_late = True
app.conf.task_reject_on_worker_lost = True
app.conf.task_track_started = True
app.conf.task_time_limit = None  # Снимаем ограничение времени
app.conf.task_soft_time_limit = None  # Снимаем ограничение времени

# Настройки для отслеживания прогресса
app.conf.result_expires = 3600  # результаты хранятся 1 час
app.conf.result_backend = f"redis://{os.getenv('REDIS_HOST', 'redis')}:6379/0"

# --------------------------------------------------------------------------
# Тестовая задача
# --------------------------------------------------------------------------
@app.task(bind=True)
def debug_task(self):
    import socket
    print(f'Request: {self.request!r}')
    return {
        'status': 'ok', 
        'worker': socket.gethostname(),
        'pid': os.getpid()
    }


# --------------------------------------------------------------------------
# Инициализация при запуске beat
# --------------------------------------------------------------------------
@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Дополнительная настройка периодических задач при запуске."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Celery beat scheduler configured with periodic tasks")