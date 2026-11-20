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
    """Service for working with embeddings, backed by models from the DB."""

    def __init__(self):
        # Model cache: the key can be either model_id (int) or model_name (str)
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
        Load an embedding model by its DB ID.
        Caches in self._model_cache under the key f"id:{model_id}" and model_name.
        """
        cache_key = f"id:{model_id}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        try:
            model_obj = EmbeddingModel.objects.get(id=model_id, is_active=True)
            model_name = model_obj.model_id  # the model_id field stores the HF model name
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_path = None

            # If the model is marked as downloaded and exists locally
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
        Load a model by name (legacy way). Caches in self._model_cache.
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
        Get an embedding model.
        If model_id is given, the model is loaded from the DB.
        Otherwise model_name or the default model is used.
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
        """Create an embedding for a single text."""
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
        """Create embeddings for a batch of texts, falling back to CPU on CUDA OOM."""
        model = self.get_model(model_id, model_name)
        if model is None:
            raise ValueError(f"Could not load embedding model (id={model_id}, name={model_name})")

        # Remember the original device
        original_device = next(model.parameters()).device
        embeddings = None

        try:
            # Try on the current device (GPU or CPU)
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
                # Move the model to CPU and retry
                model.to('cpu')
                try:
                    embeddings = model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True
                    )
                finally:
                    # Restore the model to its original device if it was GPU
                    if original_device.type == 'cuda':
                        model.to(original_device)
            else:
                # Some other error — re-raise
                raise

        if embeddings is None:
            raise RuntimeError("Failed to compute embeddings after fallback attempts.")

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
        return embeddings

    def get_embedding_dimension(self, model_id: Optional[int] = None, model_name: Optional[str] = None) -> int:
        """Embedding dimension for the given model."""
        model = self.get_model(model_id, model_name)
        if model is None:
            return 0
        return model.get_sentence_embedding_dimension()

    def clear_cache(self):
        """Clear the model cache (free memory)."""
        self._model_cache.clear()
        logger.info("🧹 Embedding model cache cleared")

    def download_model(self, model_id: int, progress_callback=None) -> bool:
        """
        Download the embedding model for the given DB ID.
        Returns True on success.
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

            # Download the model
            local_path = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                resume_download=True,
                local_files_only=False,
                ignore_patterns=['*.safetensors?'],
            )
            logger.info(f"✅ Model {model_name} downloaded to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Model download error {model_id}: {e}", exc_info=True)
            return False


# Global instance
embedding_service = EmbeddingService()