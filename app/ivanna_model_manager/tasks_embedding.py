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
from .models import EmbeddingModel
from django.contrib.auth import get_user_model

# HF endpoint: официальный Hub по умолчанию; зеркало — только явно через env.
# ВАЖНО: не дефолтить на hf-mirror.com — этот модуль импортируется раньше
# download_service и через os.environ перебивал его endpoint (скачивание
# моделей падало FileMetadataError на редиректах зеркала).
HF_ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
os.environ['HF_ENDPOINT'] = HF_ENDPOINT  # на всякий случай

# Включаем hf_transfer для ускорения (если установлено)
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

logger = logging.getLogger(__name__)

# Уменьшаем шум от библиотек
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
        logger.info(f"Свободного места достаточно: {free_bytes / 1024**3:.2f} ГБ")
    except Exception as e:
        logger.error(f"Ошибка при проверке диска: {e}")
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
    logger.info(f"📥 Запуск задачи загрузки модели эмбеддингов {model_id} (user_id={user_id})")

    lock_key = f"download_embedding_model:{model_id}"
    if not cache.add(lock_key, self.request.id, timeout=7200):
        existing = cache.get(lock_key)
        logger.warning(f"Модель {model_id} уже загружается (блокировка: {existing})")
        return {'success': False, 'error': 'Already downloading'}

    try:
        model = EmbeddingModel.objects.get(id=model_id, is_active=True)
        logger.info(f"Модель: {model.name} (HF ID: {model.model_id})")
    except EmbeddingModel.DoesNotExist:
        logger.error(f"Модель {model_id} не найдена")
        cache.delete(lock_key)
        return {'success': False, 'error': 'Model not found'}

    if model.is_downloaded:
        logger.info(f"Модель {model.name} уже скачана")
        cache.delete(lock_key)
        return {'success': True, 'message': 'Already downloaded'}

    model_dir = Path(EMBEDDING_MODELS_DIR) / model.model_id.replace('/', '_')

    # Всегда удаляем старую папку перед новой загрузкой
    if model_dir.exists():
        logger.warning(f"Папка {model_dir} существует, удаляем для чистой загрузки.")
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

    # Получаем токен пользователя
    hf_token = None
    if user_id is not None:
        try:
            User = get_user_model()
            user = User.objects.get(id=user_id)
            if hasattr(user, 'profile') and user.profile.huggingface_token:
                hf_token = user.profile.huggingface_token
                logger.info(f"✅ Используется токен пользователя {user.username}")
            else:
                logger.info("Токен пользователя не найден, продолжаем без токена")
        except User.DoesNotExist:
            logger.warning(f"Пользователь с id {user_id} не найден, продолжаем без токена")
        except Exception as e:
            logger.warning(f"Ошибка при получении токена: {e}, продолжаем без токена")

    # Получаем информацию о файлах на сервере с явным указанием endpoint
    fs = HfFileSystem(token=hf_token, endpoint=HF_ENDPOINT)
    total_files = None
    total_size = None
    try:
        all_files = fs.ls(model.model_id, recursive=True, detail=True)
        total_files = len([f for f in all_files if f['type'] == 'file'])
        total_size = sum(f['size'] for f in all_files if f['type'] == 'file')
        logger.info(f"📊 Ожидается файлов: {total_files}, общий размер: {total_size / 1024**3:.2f} ГБ")
    except Exception as e:
        logger.warning(f"Не удалось получить список файлов модели: {e}")

    # Фоновый поток для мониторинга прогресса
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
                        log_msg = (f"⏳ Загрузка продолжается... прошло {elapsed} сек, "
                                   f"прогресс: {progress}% ({downloaded_bytes / 1024**3:.2f}/{total_size / 1024**3:.2f} ГБ)")
                    elif total_files and total_files > 0:
                        progress = min(int((file_count / total_files) * 100), 100)
                        log_msg = (f"⏳ Загрузка продолжается... прошло {elapsed} сек, "
                                   f"прогресс: {progress}% (файлов: {file_count}/{total_files})")
                    else:
                        progress = 0
                        log_msg = f"⏳ Загрузка продолжается... прошло {elapsed} сек, скачано {downloaded_bytes / 1024**3:.2f} ГБ"

                    logger.info(log_msg)

                    # Обновляем прогресс в БД
                    model.download_progress = progress
                    model.save(update_fields=['download_progress'])

                    last_log_time = now
                else:
                    logger.info(f"⏳ Загрузка начинается... прошло {elapsed} сек")

    monitor = threading.Thread(target=progress_monitor)
    monitor.start()

    try:
        # Скачиваем модель, явно указываем endpoint
        local_path = snapshot_download(
            repo_id=model.model_id,
            local_dir=str(model_dir),
            token=hf_token,
            endpoint=HF_ENDPOINT,
        )

        # Загрузка завершена
        monitor_active = False
        logger.info(f"✅ Модель {model.name} успешно загружена в {local_path}")

        model.download_status = 'completed'
        model.download_progress = 100
        model.is_downloaded = True
        model.save(update_fields=['download_status', 'download_progress', 'is_downloaded'])

        cache.delete(lock_key)
        return {'success': True, 'model_id': model.id}

    except Exception as e:
        monitor_active = False
        logger.error(f"❌ Ошибка загрузки модели {model.name}: {e}", exc_info=True)
        model.download_status = 'failed'
        model.download_error = str(e)
        model.save(update_fields=['download_status', 'download_error'])
        cache.delete(lock_key)
        raise self.retry(exc=e)

    finally:
        stop_event.set()
        monitor.join(timeout=2)