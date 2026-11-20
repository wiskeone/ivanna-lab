import os
import logging
import time
from typing import Optional, Callable
from django.utils import timezone
from huggingface_hub import snapshot_download, login, HfApi
from huggingface_hub.utils import LocalEntryNotFoundError, RepositoryNotFoundError, GatedRepoError
import traceback

from ..models import ModelCheckpoint, ModelDownloadJob

# Set the default mirror if not configured in the environment
HF_ENDPOINT = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
os.environ['HF_ENDPOINT'] = HF_ENDPOINT  # for snapshot_download and HfApi

logger = logging.getLogger(__name__)


class DownloadService:
    """
    Service for downloading models from Hugging Face Hub with support for:
    - progress reporting (optional),
    - canceling stuck locks,
    - storing job info in the DB.
    """

    def __init__(self):
        # Disable problematic hf_transfer optimizations
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
        self.hf_token = None  # no longer stored globally
        self.api = None

    def _init_api(self, token: Optional[str] = None):
        """Initialize HfApi with the given token and mirror."""
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
        Download a model from Hugging Face synchronously.
        - Creates/updates a ModelDownloadJob record.
        - Calls snapshot_download.
        - On success updates the model status to 'DOWNLOADED'.
        - On error sets status 'ERROR' and stores the message.
        """
        checkpoint = ModelCheckpoint.objects.get(id=checkpoint_id)
        if not checkpoint.huggingface_id:
            raise ValueError("Hugging Face ID не указан")

        # Initialize the API with the given token
        effective_token = self._init_api(token)

        # Create or fetch the active download job
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
            # Determine the save path
            cache_dir = os.environ.get('MODEL_CACHE_DIR', '/app/media/model_cache')
            safe_name = checkpoint.safe_name
            local_dir = os.path.join(cache_dir, safe_name)
            os.makedirs(local_dir, exist_ok=True)

            logger.info(f"⬇️ Downloading {checkpoint.huggingface_id} -> {local_dir} (via {HF_ENDPOINT})")

            # Fetch model info (sizes, files) for progress reporting
            param_count = 0
            try:
                model_info = self.api.model_info(checkpoint.huggingface_id)
                total_size = sum(s.size for s in model_info.siblings if s.size) if hasattr(model_info, 'siblings') else 0
                logger.info(f"Total model size: {total_size / (1024**3):.2f} GB")
                # v5.2: parameter count comes from safetensors metadata (otherwise size_gb in the registry = 0)
                st = getattr(model_info, 'safetensors', None)
                if st and getattr(st, 'total_params', None):
                    param_count = int(st.total_params)
            except Exception:
                total_size = 0

            progress_cb(10, "Начало загрузки...")

            # List of files to ignore (avoid downloading unnecessary data).
            # GGUF/GGML are excluded only if the repo has other weights: a GGUF repo
            # (e.g. quantized builds) would otherwise be downloaded empty.
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
                "pytorch_model.bin",  # ignore bin when safetensors exist
            ]
            if has_other_weights:
                ignore_patterns += ["*.gguf", "*.ggml"]
            elif repo_files:
                # GGUF-only repo: do not drag in extra standard files
                ignore_patterns += ["*.md", "*.gitattributes"]

            # Use snapshot_download with an explicit endpoint
            local_path = snapshot_download(
                repo_id=checkpoint.huggingface_id,
                revision=checkpoint.huggingface_revision or "main",
                cache_dir=cache_dir,
                local_dir=local_dir,
                local_dir_use_symlinks=False,  # copy files
                ignore_patterns=ignore_patterns,
                token=effective_token,
                resume_download=True,
                endpoint=HF_ENDPOINT,          # pass the mirror explicitly
            )

            progress_cb(90, "Файлы загружены, обновление статуса...")

            # Update the model
            checkpoint.local_path = local_path
            checkpoint.relative_path = f"models/{safe_name}"
            checkpoint.status = 'DOWNLOADED'
            checkpoint.error_message = ''
            if param_count and not checkpoint.parameter_count:
                checkpoint.parameter_count = param_count
            checkpoint.save()

            # Update the job
            download_job.status = 'COMPLETED'
            download_job.completed_at = timezone.now()
            download_job.local_path = local_path
            download_job.progress = 100
            download_job.save()

            logger.info(f"✅ Model {checkpoint.huggingface_id} downloaded successfully")
            return checkpoint

        except GatedRepoError as e:
            logger.error(f"❌ Model requires a token, but it was not provided or is invalid: {e}")
            checkpoint.status = 'ERROR'
            checkpoint.error_message = "Модель требует аутентификации. Добавьте токен Hugging Face в профиль."
            checkpoint.save()
            download_job.status = 'FAILED'
            download_job.error_message = checkpoint.error_message
            download_job.completed_at = timezone.now()
            download_job.save()
            raise

        except (RepositoryNotFoundError, LocalEntryNotFoundError) as e:
            logger.error(f"❌ Repository not found: {e}")
            checkpoint.status = 'ERROR'
            checkpoint.error_message = f"Репозиторий не найден: {str(e)}"
            checkpoint.save()
            download_job.status = 'FAILED'
            download_job.error_message = checkpoint.error_message
            download_job.completed_at = timezone.now()
            download_job.save()
            raise

        except Exception as e:
            logger.error(f"❌ Model download error {checkpoint.id}: {e}")
            logger.error(traceback.format_exc())
            checkpoint.status = 'ERROR'
            checkpoint.error_message = str(e)
            checkpoint.save()
            download_job.status = 'FAILED'
            download_job.error_message = str(e)
            download_job.completed_at = timezone.now()
            download_job.save()
            raise


# Global instance
download_service = DownloadService()