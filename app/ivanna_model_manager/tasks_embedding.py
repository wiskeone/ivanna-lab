import logging
import shutil
import os
import time
import threading
from pathlib import Path
from celery import shared_task
from django.core.cache import cache
from django.conf import settings
from huggingface_hub import snapshot_download, HfFileSystem
from .models import EmbeddingModel, UserProfile
from django.contrib.auth import get_user_model

# HF endpoint: official Hub by default; a mirror only via an explicit env var.
# IMPORTANT: do not default to hf-mirror.com — this module is imported before
# download_service and its os.environ value overrode that service's endpoint
# (model downloads failed with FileMetadataError on mirror redirects).
HF_ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
os.environ['HF_ENDPOINT'] = HF_ENDPOINT  # just in case

# Enable hf_transfer for faster downloads (if installed)
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

logger = logging.getLogger(__name__)

# Reduce library noise
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

EMBEDDING_MODELS_DIR = getattr(settings, 'EMBEDDING_MODELS_DIR', '/app/models/embeddings')
os.makedirs(EMBEDDING_MODELS_DIR, exist_ok=True)

MIN_FREE_SPACE = 5 * 1024**3


def check_disk_space(path, required_bytes):
    try:
        usage = shutil.disk_usage(path)
        free_bytes = usage.free
        if free_bytes < required_bytes:
            raise OSError(
                f"Недостаточно свободного места: {free_bytes / 1024**3:.2f} ГБ, "
                f"требуется {required_bytes / 1024**3:.2f} ГБ"
            )
        logger.info(f"Enough free disk space: {free_bytes / 1024**3:.2f} GB")
    except Exception as e:
        logger.error(f"Disk space check error: {e}")
        raise


@shared_task(
    bind=True,
    queue='cpu_tasks',
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    max_retries=5,
    default_retry_delay=30,
    time_limit=7200,
    soft_time_limit=6600
)
def download_embedding_model_task(self, model_id, user_id=None):
    logger.info(f"📥 Starting embedding model download task {model_id} (user_id={user_id})")

    lock_key = f"download_embedding_model:{model_id}"
    if not cache.add(lock_key, self.request.id, timeout=7200):
        existing = cache.get(lock_key)
        logger.warning(f"Model {model_id} is already downloading (lock: {existing})")
        return {'success': False, 'error': 'Already downloading'}

    try:
        model = EmbeddingModel.objects.get(id=model_id, is_active=True)
        logger.info(f"Model: {model.name} (HF ID: {model.model_id})")
    except EmbeddingModel.DoesNotExist:
        logger.error(f"Model {model_id} not found")
        cache.delete(lock_key)
        return {'success': False, 'error': 'Model not found'}

    if model.is_downloaded:
        logger.info(f"Model {model.name} is already downloaded")
        cache.delete(lock_key)
        return {'success': True, 'message': 'Already downloaded'}

    model_dir = Path(EMBEDDING_MODELS_DIR) / model.model_id.replace('/', '_')

    # Always remove the old folder before a fresh download
    if model_dir.exists():
        logger.warning(f"Folder {model_dir} exists, removing it for a clean download.")
        shutil.rmtree(model_dir, ignore_errors=True)
        time.sleep(1)
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        check_disk_space(model_dir, MIN_FREE_SPACE)
    except OSError as e:
        model.download_status = 'failed'
        model.download_error = str(e)
        model.save(update_fields=['download_status', 'download_error'])
        cache.delete(lock_key)
        raise self.retry(exc=e)

    model.download_status = 'downloading'
    model.download_progress = 0
    model.download_error = ''
    model.save(update_fields=['download_status', 'download_progress', 'download_error'])

    # Get the user's token
    hf_token = None
    if user_id is not None:
        try:
            User = get_user_model()
            user = User.objects.get(id=user_id)
            if hasattr(user, 'profile') and user.profile.huggingface_token:
                hf_token = user.profile.huggingface_token
                logger.info(f"✅ Using user token of {user.username}")
            else:
                logger.info("User token not found, continuing without a token")
        except User.DoesNotExist:
            logger.warning(f"User with id {user_id} not found, continuing without a token")
        except Exception as e:
            logger.warning(f"Error getting the token: {e}, continuing without a token")

    # Get file info from the server with an explicit endpoint
    fs = HfFileSystem(token=hf_token, endpoint=HF_ENDPOINT)
    total_files = None
    total_size = None
    try:
        all_files = fs.ls(model.model_id, recursive=True, detail=True)
        total_files = len([f for f in all_files if f['type'] == 'file'])
        total_size = sum(f['size'] for f in all_files if f['type'] == 'file')
        logger.info(f"📊 Expected files: {total_files}, total size: {total_size / 1024**3:.2f} GB")
    except Exception as e:
        logger.warning(f"Failed to list model files: {e}")

    # Background thread for progress monitoring
    stop_event = threading.Event()
    start_time = time.time()
    monitor_active = True

    def progress_monitor():
        nonlocal monitor_active
        last_log_time = time.time()
        while not stop_event.wait(1) and monitor_active:
            now = time.time()
            if now - last_log_time >= 10:
                elapsed = int(now - start_time)
                if model_dir.exists():
                    local_files = list(model_dir.rglob('*'))
                    file_count = len([f for f in local_files if f.is_file()])
                    downloaded_bytes = sum(f.stat().st_size for f in local_files if f.is_file())

                    if total_size and total_size > 0:
                        progress = min(int((downloaded_bytes / total_size) * 100), 100)
                        log_msg = (f"⏳ Download in progress... {elapsed}s elapsed, "
                                   f"progress: {progress}% ({downloaded_bytes / 1024**3:.2f}/{total_size / 1024**3:.2f} GB)")
                    elif total_files and total_files > 0:
                        progress = min(int((file_count / total_files) * 100), 100)
                        log_msg = (f"⏳ Download in progress... {elapsed}s elapsed, "
                                   f"progress: {progress}% (files: {file_count}/{total_files})")
                    else:
                        progress = 0
                        log_msg = f"⏳ Download in progress... {elapsed}s elapsed, downloaded {downloaded_bytes / 1024**3:.2f} GB"

                    logger.info(log_msg)

                    # Update progress in the DB
                    model.download_progress = progress
                    model.save(update_fields=['download_progress'])

                    last_log_time = now
                else:
                    logger.info(f"⏳ Download starting... {elapsed}s elapsed")

    monitor = threading.Thread(target=progress_monitor)
    monitor.start()

    try:
        # Download the model with an explicit endpoint
        local_path = snapshot_download(
            repo_id=model.model_id,
            local_dir=str(model_dir),
            token=hf_token,
            endpoint=HF_ENDPOINT,
        )

        # Download finished
        monitor_active = False
        logger.info(f"✅ Model {model.name} downloaded successfully to {local_path}")

        model.download_status = 'completed'
        model.download_progress = 100
        model.is_downloaded = True
        model.save(update_fields=['download_status', 'download_progress', 'is_downloaded'])

        cache.delete(lock_key)
        return {'success': True, 'model_id': model.id}

    except Exception as e:
        monitor_active = False
        logger.error(f"❌ Error downloading model {model.name}: {e}", exc_info=True)
        model.download_status = 'failed'
        model.download_error = str(e)
        model.save(update_fields=['download_status', 'download_error'])
        cache.delete(lock_key)
        raise self.retry(exc=e)

    finally:
        stop_event.set()
        monitor.join(timeout=2)

# ── v5.2: reranker model downloads (cross-encoders) ──────────────────
@shared_task(bind=True, queue='cpu_tasks', name='ivanna_model_manager.tasks_embedding.download_reranker_model_task')
def download_reranker_model_task(self, reranker_id, user_id=None):
    """Download a RerankerModel (cross-encoder) into the rerankers directory."""
    from .models import RerankerModel
    reranker = RerankerModel.objects.get(id=reranker_id)
    if reranker.download_status == 'downloading':
        return {'success': False, 'error': 'Already downloading'}
    reranker.download_status = 'downloading'
    reranker.download_error = ''
    reranker.download_progress = 0
    reranker.save(update_fields=['download_status', 'download_error', 'download_progress'])
    try:
        token = None
        if user_id:
            profile = UserProfile.objects.filter(user_id=user_id).first()
            token = profile.huggingface_token or None
        import huggingface_hub
        if token:
            huggingface_hub.login(token=token, add_to_git_credential=False)
        dest_dir = os.path.join(EMBEDDING_MODELS_DIR, 'rerankers',
                                reranker.model_id.replace('/', '_'))
        os.makedirs(dest_dir, exist_ok=True)
        huggingface_hub.snapshot_download(
            repo_id=reranker.model_id, local_dir=dest_dir, resume_download=True)
        reranker.is_downloaded = True
        reranker.download_status = 'completed'
        reranker.download_progress = 100
        reranker.save(update_fields=['is_downloaded', 'download_status', 'download_progress'])
        return {'success': True, 'path': dest_dir}
    except Exception as e:
        reranker.download_status = 'failed'
        reranker.download_error = str(e)[:500]
        reranker.save(update_fields=['download_status', 'download_error'])
        return {'success': False, 'error': str(e)}
