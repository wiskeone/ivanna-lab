import os
import logging
import time
from typing import Optional, Callable
from django.utils import timezone
from huggingface_hub import snapshot_download, login, HfApi
from huggingface_hub.utils import LocalEntryNotFoundError, RepositoryNotFoundError, GatedRepoError
import traceback

from ..models import ModelCheckpoint, ModelDownloadJob

# Устанавливаем зеркало по умолчанию, если не задано в окружении
HF_ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
os.environ['HF_ENDPOINT'] = HF_ENDPOINT  # для snapshot_download и HfApi

logger = logging.getLogger(__name__)


class DownloadService:
    """
    Сервис для загрузки моделей с Hugging Face Hub с поддержкой:
    - прогресса (опционально),
    - отмены зависших блокировок,
    - сохранения информации о задаче в БД.
    """

    def __init__(self):
        # Отключаем проблемные оптимизации hf_transfer
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
        self.hf_token = None  # больше не храним токен глобально
        self.api = None

    def _init_api(self, token: Optional[str] = None):
        """Инициализирует HfApi с переданным токеном и зеркалом."""
        effective_token = token or os.environ.get('HF_TOKEN')
        if effective_token:
            self.api = HfApi(token=effective_token, endpoint=HF_ENDPOINT)
            try:
                login(token=effective_token)
                logger.info("✅ Logged into Hugging Face Hub")
            except Exception as e:
                logger.warning(f"⚠️ Failed to login to Hugging Face: {e}")
        else:
            self.api = HfApi(endpoint=HF_ENDPOINT)
        return effective_token

    def download_model_sync(
        self,
        checkpoint_id: int,
        token: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> ModelCheckpoint:
        """
        Синхронная загрузка модели с Hugging Face.
        - Создаёт/обновляет запись ModelDownloadJob.
        - Вызывает snapshot_download.
        - При успехе обновляет статус модели на 'DOWNLOADED'.
        - При ошибке — статус 'ERROR' и сохраняет сообщение.
        """
        checkpoint = ModelCheckpoint.objects.get(id=checkpoint_id)
        if not checkpoint.huggingface_id:
            raise ValueError("Hugging Face ID не указан")

        # Инициализируем API с переданным токеном
        effective_token = self._init_api(token)

        # Создаём или получаем активную задачу загрузки
        download_job, created = ModelDownloadJob.objects.get_or_create(
            checkpoint=checkpoint,
            status__in=['PENDING', 'DOWNLOADING'],
            defaults={
                'status': 'DOWNLOADING',
                'started_at': timezone.now(),
            }
        )
        if not created:
            download_job.status = 'DOWNLOADING'
            download_job.started_at = timezone.now()
            download_job.error_message = ''
            download_job.save()

        checkpoint.status = 'DOWNLOADING'
        checkpoint.save()

        def progress_cb(progress, step):
            download_job.refresh_from_db()
            download_job.progress = progress
            download_job.save()
            if progress_callback:
                progress_callback(progress, step)

        progress_cb(5, "Подготовка к загрузке...")

        try:
            # Определяем путь для сохранения
            cache_dir = os.environ.get('MODEL_CACHE_DIR', '/app/media/model_cache')
            safe_name = checkpoint.safe_name
            local_dir = os.path.join(cache_dir, safe_name)
            os.makedirs(local_dir, exist_ok=True)

            logger.info(f"⬇️ Загрузка {checkpoint.huggingface_id} -> {local_dir} (через {HF_ENDPOINT})")

            # Получаем информацию о модели (размеры, файлы) для прогресса
            try:
                model_info = self.api.model_info(checkpoint.huggingface_id)
                total_size = sum(s.size for s in model_info.siblings if s.size) if hasattr(model_info, 'siblings') else 0
                logger.info(f"Общий размер модели: {total_size / (1024**3):.2f} GB")
            except Exception:
                total_size = 0

            progress_cb(10, "Начало загрузки...")

            # Список файлов для игнорирования (чтобы не качать лишнее).
            # GGUF/GGML режем только если в репо есть другие веса: GGUF-репо
            # (например, квантованные сборки) иначе скачивалось бы пустым.
            repo_files = [s.rfilename for s in (model_info.siblings or [])] \
                if (model_info is not None and getattr(model_info, 'siblings', None)) else []
            has_other_weights = any(
                f.endswith(('.safetensors', '.bin')) for f in repo_files)
            ignore_patterns = [
                "*.msgpack",
                "*.h5",
                "*.ot",
                "*.pdf",
                "*.onnx",
                "*.tflite",
                "pytorch_model.bin",  # если есть safetensors, игнорируем bin
            ]
            if has_other_weights:
                ignore_patterns += ["*.gguf", "*.ggml"]
            elif repo_files:
                # GGUF-only репо: не тащим лишние стандартные файлы
                ignore_patterns += ["*.md", "*.gitattributes"]

            # Используем snapshot_download с явным endpoint
            local_path = snapshot_download(
                repo_id=checkpoint.huggingface_id,
                revision=checkpoint.huggingface_revision or "main",
                cache_dir=cache_dir,
                local_dir=local_dir,
                local_dir_use_symlinks=False,  # копируем файлы
                ignore_patterns=ignore_patterns,
                token=effective_token,
                resume_download=True,
                endpoint=HF_ENDPOINT,          # явно передаём зеркало
            )

            progress_cb(90, "Файлы загружены, обновление статуса...")

            # Обновляем модель
            checkpoint.local_path = local_path
            checkpoint.relative_path = f"models/{safe_name}"
            checkpoint.status = 'DOWNLOADED'
            checkpoint.error_message = ''
            checkpoint.save()

            # Обновляем задачу
            download_job.status = 'COMPLETED'
            download_job.completed_at = timezone.now()
            download_job.local_path = local_path
            download_job.progress = 100
            download_job.save()

            logger.info(f"✅ Модель {checkpoint.huggingface_id} успешно загружена")
            return checkpoint

        except GatedRepoError as e:
            logger.error(f"❌ Модель требует токен, но он не предоставлен или недействителен: {e}")
            checkpoint.status = 'ERROR'
            checkpoint.error_message = "Модель требует аутентификации. Добавьте токен Hugging Face в профиль."
            checkpoint.save()
            download_job.status = 'FAILED'
            download_job.error_message = checkpoint.error_message
            download_job.completed_at = timezone.now()
            download_job.save()
            raise

        except (RepositoryNotFoundError, LocalEntryNotFoundError) as e:
            logger.error(f"❌ Репозиторий не найден: {e}")
            checkpoint.status = 'ERROR'
            checkpoint.error_message = f"Репозиторий не найден: {str(e)}"
            checkpoint.save()
            download_job.status = 'FAILED'
            download_job.error_message = checkpoint.error_message
            download_job.completed_at = timezone.now()
            download_job.save()
            raise

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели {checkpoint.id}: {e}")
            logger.error(traceback.format_exc())
            checkpoint.status = 'ERROR'
            checkpoint.error_message = str(e)
            checkpoint.save()
            download_job.status = 'FAILED'
            download_job.error_message = str(e)
            download_job.completed_at = timezone.now()
            download_job.save()
            raise


# Глобальный экземпляр
download_service = DownloadService()