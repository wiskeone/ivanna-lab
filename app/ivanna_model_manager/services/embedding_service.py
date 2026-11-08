import logging
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from django.conf import settings
from ..models import EmbeddingModel

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Сервис для работы с эмбеддингами с поддержкой моделей из БД."""

    def __init__(self):
        # Кэш моделей: ключ может быть как model_id (int), так и model_name (str)
        self._model_cache: Dict[str, SentenceTransformer] = {}
        self.default_model_name = getattr(
            settings, 'DEFAULT_EMBEDDING_MODEL',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        self.embedding_models_dir = getattr(
            settings, 'EMBEDDING_MODELS_DIR', '/app/models/embeddings'
        )

    def _get_model_by_id(self, model_id: int) -> Optional[SentenceTransformer]:
        """
        Загружает модель эмбеддингов по её ID из базы данных.
        Кэширует в self._model_cache по ключу f"id:{model_id}" и по model_name.
        """
        cache_key = f"id:{model_id}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        try:
            model_obj = EmbeddingModel.objects.get(id=model_id, is_active=True)
            model_name = model_obj.model_id  # поле model_id хранит имя модели на HF
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_path = None

            # Если модель помечена как скачанная и существует локально
            if model_obj.is_downloaded:
                local_dir = Path(self.embedding_models_dir) / model_obj.model_id.replace('/', '_')
                if local_dir.exists():
                    model_path = str(local_dir)
                    logger.info(f"Loading embedding model from local path: {model_path}")

            try:
                if model_path:
                    model = SentenceTransformer(model_path, device=device)
                else:
                    model = SentenceTransformer(model_name, device=device)
                logger.info(f"✅ Loaded embedding model '{model_name}' (ID={model_id}) on {device}")
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                logger.warning(f"Failed to load {model_name} on {device}: {e}. Falling back to CPU.")
                if model_path:
                    model = SentenceTransformer(model_path, device='cpu')
                else:
                    model = SentenceTransformer(model_name, device='cpu')
                logger.info(f"✅ Loaded embedding model '{model_name}' (ID={model_id}) on CPU (fallback)")

            self._model_cache[cache_key] = model
            self._model_cache[model_name] = model
            return model
        except EmbeddingModel.DoesNotExist:
            logger.error(f"Embedding model with ID {model_id} not found")
            return None
        except Exception as e:
            logger.error(f"Error loading embedding model ID {model_id}: {e}")
            return None

    def _get_model_by_name(self, model_name: Optional[str] = None) -> SentenceTransformer:
        """
        Загружает модель по имени (старый способ). Кэширует в self._model_cache.
        """
        model_name = model_name or self.default_model_name
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        logger.info(f"📦 Loading embedding model by name: {model_name}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            model = SentenceTransformer(model_name, device=device)
            logger.info(f"✅ Model {model_name} loaded on {device}")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.warning(f"Failed to load {model_name} on {device}: {e}. Falling back to CPU.")
            model = SentenceTransformer(model_name, device='cpu')
            logger.info(f"✅ Model {model_name} loaded on CPU (fallback)")
        self._model_cache[model_name] = model
        return model

    def get_model(self, model_id: Optional[int] = None, model_name: Optional[str] = None) -> Optional[SentenceTransformer]:
        """
        Получить модель эмбеддингов.
        Если указан model_id, загружается модель из БД.
        Иначе используется model_name или модель по умолчанию.
        """
        if model_id is not None:
            return self._get_model_by_id(model_id)
        else:
            return self._get_model_by_name(model_name)

    def create_embedding(
        self,
        text: str,
        model_id: Optional[int] = None,
        model_name: Optional[str] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Создать эмбеддинг для одного текста."""
        model = self.get_model(model_id, model_name)
        if model is None:
            raise ValueError(f"Could not load embedding model (id={model_id}, name={model_name})")
        embedding = model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        return embedding

    def create_embeddings_batch(
        self,
        texts: List[str],
        model_id: Optional[int] = None,
        model_name: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """Создать эмбеддинги для батча текстов с fallback на CPU при CUDA OOM."""
        model = self.get_model(model_id, model_name)
        if model is None:
            raise ValueError(f"Could not load embedding model (id={model_id}, name={model_name})")

        # Запоминаем исходное устройство
        original_device = next(model.parameters()).device
        embeddings = None

        try:
            # Пробуем на текущем устройстве (GPU или CPU)
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"CUDA OOM during batch encoding (batch_size={batch_size}). "
                               f"Falling back to CPU for this batch.")
                # Перемещаем модель на CPU и пробуем снова
                model.to('cpu')
                try:
                    embeddings = model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                finally:
                    # Возвращаем модель на исходное устройство, если оно было GPU
                    if original_device.type == 'cuda':
                        model.to(original_device)
            else:
                # Другая ошибка — пробрасываем дальше
                raise

        if embeddings is None:
            raise RuntimeError("Failed to compute embeddings after fallback attempts.")

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        return embeddings

    def get_embedding_dimension(self, model_id: Optional[int] = None, model_name: Optional[str] = None) -> int:
        """Размерность эмбеддингов для указанной модели."""
        model = self.get_model(model_id, model_name)
        if model is None:
            return 0
        return model.get_sentence_embedding_dimension()

    def clear_cache(self):
        """Очистить кэш моделей (освободить память)."""
        self._model_cache.clear()
        logger.info("🧹 Embedding model cache cleared")

    def download_model(self, model_id: int, progress_callback=None) -> bool:
        """
        Загружает модель эмбеддингов по её ID в базу данных.
        Возвращает True при успехе.
        """
        try:
            model_obj = EmbeddingModel.objects.get(id=model_id, is_active=True)
            model_name = model_obj.model_id
            cache_dir = settings.HF_HOME / 'hub'

            from huggingface_hub import snapshot_download

            def download_progress_callback(current, total):
                if total > 0 and progress_callback:
                    percent = int(current / total * 100)
                    progress_callback(percent, f"Загрузка {current}/{total} файлов")

            # Скачиваем модель
            local_path = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                resume_download=True,
                local_files_only=False,
                ignore_patterns=['*.safetensors?'],
            )
            logger.info(f"✅ Модель {model_name} загружена в {local_path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_id}: {e}", exc_info=True)
            return False


# Глобальный экземпляр
embedding_service = EmbeddingService()