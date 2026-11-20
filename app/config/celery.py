from __future__ import absolute_import, unicode_literals

import json
import logging
import multiprocessing
import os
from pathlib import Path

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown, task_postrun
from kombu.serialization import register

# Set process start method to avoid fork issues
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch

# Custom JSON encoder for Path objects
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

# Set Django settings module env var
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

# Initialize Django
import django
django.setup()

# Create Celery app instance
app = Celery('config')

# Load config from Django settings (CELERY namespace)
app.config_from_object('django.conf:settings', namespace='CELERY')

# Autodiscover tasks in installed apps
app.autodiscover_tasks()
# v5.2: soft MoE tasks live in a separate module — autodiscover loads
# only <app>.tasks
app.conf.imports = ('ivanna_model_manager.tasks_moe',
                       'ivanna_model_manager.tasks_benchmarks',)

# v5.1: web<->Celery log correlation
from ivanna_model_manager.log_context import setup_celery_log_signals
setup_celery_log_signals(app)

# Queue configuration
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

# Task routing
app.conf.task_routes = {
    # GPU tasks (require CUDA)
    'ivanna_model_manager.tasks.load_model_to_memory_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.unload_model_from_memory_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.generate_text_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.merge_models_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.run_evaluation_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.run_fine_tuning_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.process_document_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.create_embeddings_for_collection_task': {'queue': 'gpu_tasks'},
    # v5.2: soft MoE — scan+patch on GPU, rest on CPU
    'ivanna_model_manager.tasks_moe.moe_build_expert_task': {'queue': 'gpu_tasks'},
    # v5.2: llama.cpp conversion is CPU-bound, was never routed and stalled
    # in the default 'celery' queue that no worker listens to
    'ivanna_model_manager.tasks.convert_model_to_gguf_task': {'queue': 'cpu_tasks'},
    # v5.3: benchmarks — catalog/install on CPU, generation runs on GPU
    'ivanna_model_manager.tasks_benchmarks.refresh_benchmark_catalog_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks_benchmarks.install_benchmark_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks_benchmarks.run_benchmark_task': {'queue': 'gpu_tasks'},
    
    # CPU tasks (can run on any worker)
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

@task_postrun.connect
def release_gpu_after_task(**kwargs):
    # Solo workers serve many tasks in one long-lived process. Without this
    # the next GPU task inherits cached CUDA blocks from the previous one:
    # accelerate misjudges free VRAM and offloads conv modules to CPU, where
    # fast-path kernels (causal_conv1d/fla) crash with device asserts.
    # empty_cache only releases blocks no longer referenced, so models kept
    # resident by redis_model_manager (chat) stay loaded.
    try:
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------
# GPU init/cleanup signals
# --------------------------------------------------------------------------
@worker_process_init.connect
def init_worker(**kwargs):
    import logging
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Initializing Celery worker")
    
    # Clean model cache on worker start
    try:
        from ivanna_model_manager.services.redis_model_service import redis_model_manager
        redis_model_manager.clean_stale_locks()
        logger.info("Cleaned stale model locks")
    except Exception as e:
        logger.error(f"Failed to clean stale locks: {e}")
    
    logger.info("=" * 50)


@worker_process_shutdown.connect
def shutdown_worker(**kwargs):
    """Cleanup on worker shutdown."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Unload all models loaded by this worker
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
# Periodic tasks (Celery Beat)
# --------------------------------------------------------------------------
from celery.schedules import crontab

app.conf.beat_schedule = {
    # Sync model statuses every 5 minutes
    'sync-model-statuses-every-5-minutes': {
        'task': 'ivanna_model_manager.tasks.sync_model_statuses_task',
        'schedule': crontab(minute='*/5'),
    },
    
    # Refresh model TTL in Redis every 5 minutes
    'refresh-model-ttl-every-5-minutes': {
        'task': 'ivanna_model_manager.tasks.refresh_model_ttl',
        'schedule': 300.0,
    },
    
    # Health check every 15 minutes
    'health-check-every-15-minutes': {
        'task': 'ivanna_model_manager.tasks.health_check_task',
        'schedule': 900.0,
    },
    
    # Cleanup finished download jobs daily at 3:00
    'cleanup-download-jobs-daily': {
        'task': 'ivanna_model_manager.tasks.cleanup_download_jobs',
        'schedule': crontab(minute=0, hour=3),
    },
    
    # Update collection statistics hourly
    'update-collection-statistics-hourly': {
        'task': 'ivanna_model_manager.tasks.update_collection_statistics_task',
        'schedule': crontab(minute=0),
    },
    
    # Validate documents daily at 2:00
    'validate-documents-daily': {
        'task': 'ivanna_model_manager.tasks.validate_documents_task',
        'schedule': crontab(minute=0, hour=2),
    },
    
    # Cleanup vector indexes daily at 2:00
    'cleanup-vector-index-daily': {
        'task': 'ivanna_model_manager.tasks.cleanup_vector_index_task',
        'schedule': crontab(minute=0, hour=2),
    },
    
    # System health check hourly
    'check-system-health-hourly': {
        'task': 'ivanna_model_manager.tasks.check_system_health_task',
        'schedule': crontab(minute=30, hour='*/1'),
    },
    
    # Weekly report generation Monday at 9:00
    'generate-system-report-weekly': {
        'task': 'ivanna_model_manager.tasks.generate_system_report_task',
        'schedule': crontab(minute=0, hour=9, day_of_week=0),
    },
    
    # Cleanup stale model locks every 5 minutes
    'cleanup-stale-model-locks': {
        'task': 'ivanna_model_manager.tasks.cleanup_stale_model_locks',
        'schedule': crontab(minute='*/5'),
    },
    
    # Cleanup stale models every 30 minutes
    'cleanup-stale-models-hourly': {
        'task': 'ivanna_model_manager.tasks.cleanup_stale_models',
        'schedule': crontab(minute='*/30'),
    },
    
    # Automatic memory cleanup every 30 minutes
    'auto-clean-memory-hourly': {
        'task': 'ivanna_model_manager.tasks.auto_clean_memory',
        'schedule': crontab(minute='*/30'),
    },
}

# --------------------------------------------------------------------------
# Additional Celery settings
# --------------------------------------------------------------------------
app.conf.task_acks_late = True
app.conf.task_reject_on_worker_lost = True
app.conf.task_track_started = True
app.conf.task_time_limit = None  # Remove time limit
app.conf.task_soft_time_limit = None  # Remove soft time limit

# Progress tracking settings
app.conf.result_expires = 3600  # results stored for 1 hour
app.conf.result_backend = f"redis://{os.getenv('REDIS_HOST', 'redis')}:6379/0"

# --------------------------------------------------------------------------
# Test task
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
# Initialization on beat startup
# --------------------------------------------------------------------------
@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Additional periodic task setup at startup."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Celery beat scheduler configured with periodic tasks")