import faiss
import numpy as np
import pickle
import os
import logging
from typing import List, Tuple, Optional, Dict, Any
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


class FAISSService:
    """
    Сервис для работы с FAISS индексами.
    - Поддержка различных типов индексов (Flat, IVF, HNSW).
    - Сохранение/загрузка индекса с метаданными.
    - Поиск с возвратом chunk_id и расстояния/сходства.
    - Статистика индекса.
    """

    def __init__(self, index_dir: Optional[str] = None):
        self.index_dir = index_dir or str(settings.FAISS_INDEX_PATH)
        os.makedirs(self.index_dir, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.chunk_ids: List[int] = []          # соответствие позиции в FAISS -> ID чанка
        self.dimension: Optional[int] = None
        self.is_trained: bool = False
        self.metric: str = 'cosine'            # 'cosine', 'l2', 'ip'

    # ----------------------------------------------------------------------
    # Внутренние методы для работы с путями
    # ----------------------------------------------------------------------
    def _get_index_path(self, collection_id: int) -> str:
        return os.path.join(self.index_dir, f'collection_{collection_id}.faiss')

    def _get_metadata_path(self, collection_id: int) -> str:
        return os.path.join(self.index_dir, f'collection_{collection_id}_meta.pkl')

    # ----------------------------------------------------------------------
    # Создание и настройка индекса
    # ----------------------------------------------------------------------
    def create_index(
        self,
        dimension: int,
        index_type: str = 'flat',
        metric: str = 'cosine'
    ) -> faiss.Index:
        """
        Создать новый FAISS индекс.
        - flat: точный, но медленный для >100k векторов.
        - ivf: инвертированный файл, быстрый, требует обучения.
        - hnsw: графовый, быстрый и точный, но больше памяти.
        """
        self.dimension = dimension
        self.metric = metric
        self.chunk_ids = []
        self.is_trained = False

        # Определяем метрику
        if metric == 'cosine' or metric == 'ip':
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        elif metric == 'l2':
            faiss_metric = faiss.METRIC_L2
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Создаём индекс в зависимости от типа
        if index_type == 'flat':
            self.index = faiss.IndexFlat(dimension, faiss_metric)
            self.is_trained = True  # плоский индекс не требует обучения
            logger.info(f"✅ Created Flat index, dim={dimension}, metric={metric}")

        elif index_type == 'ivf':
            nlist = 100  # количество кластеров
            quantizer = faiss.IndexFlatL2(dimension) if metric == 'l2' else faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss_metric)
            logger.info(f"✅ Created IVF index, dim={dimension}, nlist={nlist}, metric={metric}")

        elif index_type == 'hnsw':
            M = 16  # количество связей на узел
            self.index = faiss.IndexHNSWFlat(dimension, M, faiss_metric)
            self.is_trained = True  # HNSW не требует отдельного обучения
            logger.info(f"✅ Created HNSW index, dim={dimension}, M={M}, metric={metric}")

        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        return self.index

    def train(self, vectors: np.ndarray) -> None:
        """
        Обучить индекс (необходимо для IVF).
        Векторы должны быть float32, форма (n_vectors, dimension).
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index first.")
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if vectors.shape[0] < 100:
                logger.warning(f"Too few vectors for training ({vectors.shape[0]}), generating synthetic data")
                # Генерируем синтетические данные
                n_synthetic = max(10000, self.dimension * 2)
                synthetic = np.random.randn(n_synthetic, self.dimension).astype('float32')
                # Нормализуем для косинусной метрики
                if self.metric == 'cosine':
                    synthetic = synthetic / np.linalg.norm(synthetic, axis=1, keepdims=True)
                self.index.train(synthetic)
                logger.info(f"✅ IVF index trained on {n_synthetic} synthetic vectors")
            else:
                self.index.train(vectors)
                logger.info(f"✅ IVF index trained on {vectors.shape[0]} real vectors")
            self.is_trained = True

    # ----------------------------------------------------------------------
    # Добавление векторов
    # ----------------------------------------------------------------------
    def add_vectors(self, vectors: np.ndarray, chunk_ids: List[int]) -> int:
        """
        Добавить векторы в индекс.
        Возвращает количество успешно добавленных векторов.
        """
        if self.index is None:
            raise ValueError("Index not created")
        if len(vectors) != len(chunk_ids):
            raise ValueError(f"vectors count {len(vectors)} != chunk_ids count {len(chunk_ids)}")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} != index dimension {self.dimension}")

        # Нормализуем для косинусной метрики
        if self.metric == 'cosine':
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

        # Обучение, если требуется
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.train(vectors)

        # Добавляем
        self.index.add(vectors.astype('float32'))
        self.chunk_ids.extend(chunk_ids)
        logger.info(f"➕ Added {len(vectors)} vectors, total: {self.index.ntotal}")
        return len(vectors)

    # ----------------------------------------------------------------------
    # Поиск
    # ----------------------------------------------------------------------
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Поиск k ближайших соседей.
        Возвращает список кортежей (chunk_id, similarity_score).
        similarity_score для cosine/ip = скалярное произведение (чем выше, тем лучше).
        Для l2 — преобразуем в сходство (1/(1+distance)).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Подготовка вектора запроса
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector).astype('float32')
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1).astype('float32')
        if query_vector.shape[1] != self.dimension:
            raise ValueError(f"Query dim {query_vector.shape[1]} != index dim {self.dimension}")

        # Нормализация для cosine
        if self.metric == 'cosine':
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

        # Для IVF можно задать количество зондов
        if nprobe is not None and hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe

        # Поиск
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunk_ids):
                chunk_id = self.chunk_ids[idx]
                dist = distances[0][i]
                # Преобразование расстояния в оценку сходства
                if self.metric == 'l2':
                    score = 1.0 / (1.0 + dist)
                else:  # cosine/ip
                    score = float(dist)  # уже скалярное произведение (от -1 до 1)
                results.append((chunk_id, score))

        return results

    # ----------------------------------------------------------------------
    # Сохранение / загрузка
    # ----------------------------------------------------------------------
    def save_index(self, collection_id: int) -> Tuple[str, str]:
        """
        Сохранить индекс и метаданные на диск.
        Возвращает (index_path, metadata_path).
        """
        if self.index is None:
            raise ValueError("No index to save")

        index_path = self._get_index_path(collection_id)
        metadata_path = self._get_metadata_path(collection_id)

        faiss.write_index(self.index, index_path)

        metadata = {
            'chunk_ids': self.chunk_ids,
            'dimension': self.dimension,
            'total_vectors': self.index.ntotal,
            'is_trained': self.is_trained,
            'metric': self.metric,
            'index_type': self._detect_index_type(),
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"💾 Index saved: {index_path} ({self.index.ntotal} vectors)")
        return index_path, metadata_path

    def load_index(self, collection_id: int) -> bool:
        """
        Загрузить индекс с диска.
        Возвращает True при успехе.
        """
        index_path = self._get_index_path(collection_id)
        metadata_path = self._get_metadata_path(collection_id)

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning(f"Index files for collection {collection_id} not found: {index_path} or {metadata_path}")
            return False

        try:
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.chunk_ids = metadata.get('chunk_ids', [])
            self.dimension = metadata.get('dimension')
            self.is_trained = metadata.get('is_trained', False)
            self.metric = metadata.get('metric', 'cosine')
            logger.info(f"📂 Index loaded for collection {collection_id}: {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load index for collection {collection_id}: {e}")
            return False

    def delete_index(self, collection_id: int) -> int:
        """Удалить файлы индекса коллекции."""
        index_path = self._get_index_path(collection_id)
        metadata_path = self._get_metadata_path(collection_id)
        deleted = 0
        for path in [index_path, metadata_path]:
            if os.path.exists(path):
                os.remove(path)
                deleted += 1
        if deleted:
            logger.info(f"🗑️ Deleted index files for collection {collection_id}")
        # Сброс состояния
        self.index = None
        self.chunk_ids = []
        self.dimension = None
        self.is_trained = False
        return deleted

    # ----------------------------------------------------------------------
    # Статистика
    # ----------------------------------------------------------------------
    def get_index_stats(self, collection_id: int) -> Dict[str, Any]:
        """Получить статистику индекса (без загрузки в память)."""
        index_path = self._get_index_path(collection_id)
        metadata_path = self._get_metadata_path(collection_id)

        stats = {
            'exists': False,
            'total_vectors': 0,
            'dimension': 0,
            'index_type': 'none',
            'is_trained': False,
            'file_size_mb': 0,
            'metric': 'unknown',
        }

        if os.path.exists(index_path):
            stats['exists'] = True
            stats['file_size_mb'] = os.path.getsize(index_path) / (1024 * 1024)
            # Пытаемся прочитать заголовок индекса без полной загрузки
            try:
                index = faiss.read_index(index_path, faiss.IO_FLAG_ONDISK | faiss.IO_FLAG_READ_ONLY)
                stats['total_vectors'] = index.ntotal
                stats['dimension'] = index.d
                if hasattr(index, 'is_trained'):
                    stats['is_trained'] = index.is_trained
                stats['index_type'] = self._detect_index_type_from_index(index)
                del index
            except Exception as e:
                logger.debug(f"Cannot read index header: {e}")

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    meta = pickle.load(f)
                stats['dimension'] = meta.get('dimension', stats['dimension'])
                stats['total_vectors'] = meta.get('total_vectors', stats['total_vectors'])
                stats['index_type'] = meta.get('index_type', stats['index_type'])
                stats['is_trained'] = meta.get('is_trained', stats['is_trained'])
                stats['metric'] = meta.get('metric', 'cosine')
            except Exception as e:
                logger.debug(f"Cannot read metadata: {e}")

        return stats

    # ----------------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------------
    def _detect_index_type(self) -> str:
        """Определить тип текущего индекса."""
        if isinstance(self.index, faiss.IndexFlat):
            return 'flat'
        elif isinstance(self.index, faiss.IndexIVFFlat):
            return 'ivf'
        elif isinstance(self.index, faiss.IndexHNSWFlat):
            return 'hnsw'
        else:
            return 'unknown'

    @staticmethod
    def _detect_index_type_from_index(index: faiss.Index) -> str:
        """Определить тип индекса по объекту FAISS."""
        if isinstance(index, faiss.IndexFlat):
            return 'flat'
        elif isinstance(index, faiss.IndexIVFFlat):
            return 'ivf'
        elif isinstance(index, faiss.IndexHNSWFlat):
            return 'hnsw'
        else:
            return 'unknown'

    def get_vectors_count(self) -> int:
        """Количество векторов в загруженном индексе."""
        return self.index.ntotal if self.index else 0