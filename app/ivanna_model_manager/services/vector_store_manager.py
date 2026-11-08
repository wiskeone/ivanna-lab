import os
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from django.conf import settings
from django.utils import timezone

from .faiss_service import FAISSService
from ..models import DocumentChunk

logger = logging.getLogger(__name__)

# Глобальный экземпляр менеджера (синглтон)
_vector_store_manager = None


def get_vector_store_manager():
    """Получить глобальный экземпляр VectorStoreManager."""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager


class VectorStoreManager:
    """
    Менеджер векторных хранилищ на базе FAISS.
    - Управляет кэшем загруженных индексов (FAISSService) для коллекций.
    - Предоставляет высокоуровневые методы для индексации, поиска, удаления.
    - Сохраняет совместимость с существующим кодом (tasks.py).
    """

    def __init__(self):
        self.config = getattr(settings, 'VECTOR_STORE_CONFIG', {
            'type': 'faiss',
            'dimension': 384,
            'metric': 'cosine',
            'use_gpu': False,
            'index_path': str(settings.FAISS_INDEX_PATH)
        })
        self._index_cache: Dict[int, FAISSService] = {}  # collection_id -> FAISSService
        self._default_dimension = self.config.get('dimension', 384)
        self._default_metric = self.config.get('metric', 'cosine')

    def _get_faiss_service(self, collection_id: Optional[int] = None) -> FAISSService:
        """
        Получить FAISSService для указанной коллекции.
        Если коллекция не указана, создаётся новый сервис без индекса.
        """
        if collection_id is None:
            return FAISSService()

        if collection_id not in self._index_cache:
            svc = FAISSService()
            if svc.load_index(collection_id):
                self._index_cache[collection_id] = svc
            else:
                # Индекс не загружен, но вернём пустой сервис — позже можно будет создать
                self._index_cache[collection_id] = svc
        return self._index_cache[collection_id]

    # ----------------------------------------------------------------------
    # Индексация чанков
    # ----------------------------------------------------------------------
    def index_chunks(
        self,
        chunks: List[DocumentChunk],
        collection_id: Optional[int] = None
    ) -> int:
        """
        Индексировать список чанков в FAISS.
        - chunks: список чанков (должны иметь embedding_vector).
        - collection_id: ID коллекции (обязателен, если нужно сохранить индекс).
        Возвращает количество успешно добавленных векторов.
        """
        if not chunks:
            return 0

        # Фильтруем чанки с эмбеддингами
        valid_chunks = [c for c in chunks if c.has_embedding and c.embedding_vector is not None]
        if not valid_chunks:
            logger.warning("No chunks with embeddings to index")
            return 0

        # Собираем векторы и ID
        vectors = []
        chunk_ids = []
        dimension = None

        for chunk in valid_chunks:
            vec = chunk.embedding_vector
            if isinstance(vec, list):
                vec = np.array(vec, dtype='float32')
            if dimension is None:
                dimension = len(vec)
            elif len(vec) != dimension:
                logger.error(f"Chunk {chunk.id} has inconsistent dimension {len(vec)} != {dimension}")
                continue
            vectors.append(vec)
            chunk_ids.append(chunk.id)

        if not vectors:
            return 0

        vectors_array = np.stack(vectors).astype('float32')

        # Получаем или создаём индекс для коллекции
        if collection_id is not None:
            svc = self._get_faiss_service(collection_id)
            # Если индекс не создан, создаём его
            if svc.index is None:
                svc.create_index(
                    dimension=dimension,
                    index_type='flat',  # можно параметризовать, но для простоты flat
                    metric=self._default_metric
                )
            # Добавляем векторы
            added = svc.add_vectors(vectors_array, chunk_ids)
            # Сохраняем индекс на диск
            svc.save_index(collection_id)
            return added
        else:
            # Без коллекции не сохраняем, просто возвращаем количество
            logger.warning("Indexing without collection_id: not saved to disk")
            return len(vectors)

    # ----------------------------------------------------------------------
    # Поиск
    # ----------------------------------------------------------------------
    def search(
        self,
        query_embedding: np.ndarray,
        collection_id: Optional[int] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск по векторному индексу.
        - query_embedding: вектор запроса (1D или 2D).
        - collection_id: ID коллекции (обязателен).
        - k: количество результатов.
        - filters: игнорируются в этой реализации (оставлено для совместимости).
        Возвращает список результатов с ключами:
            - chunk_id, score, metadata (содержит chunk_id, document_id и т.д.)
        """
        if collection_id is None:
            logger.error("Search requires collection_id")
            return []

        svc = self._get_faiss_service(collection_id)
        if svc.index is None or svc.index.ntotal == 0:
            logger.warning(f"No index loaded for collection {collection_id}")
            return []

        # Выполняем поиск
        results = svc.search(query_embedding, k=k)

        # Форматируем результат
        formatted = []
        for chunk_id, score in results:
            formatted.append({
                'chunk_id': chunk_id,
                'score': score,
                'metadata': {
                    'chunk_id': chunk_id,
                    'collection_id': collection_id,
                }
            })
        return formatted

    # ----------------------------------------------------------------------
    # Удаление чанков из индекса
    # ----------------------------------------------------------------------
    def delete_chunks(
        self,
        chunk_ids: List[int],
        collection_id: Optional[int] = None
    ) -> int:
        """
        Удалить чанки из индекса.
        В текущей реализации FAISS не поддерживает удаление отдельных векторов.
        Поэтому помечаем чанки как is_active=False в БД, а индекс перестраиваем целиком.
        Возвращает количество удалённых векторов (всегда 0, если не перестраивали).
        """
        if not chunk_ids or collection_id is None:
            return 0

        # Помечаем чанки как неактивные
        DocumentChunk.objects.filter(id__in=chunk_ids).update(is_active=False)

        # Перестраиваем индекс полностью (можно запустить задачу)
        from ..tasks_rag import rebuild_collection_index_task
        rebuild_collection_index_task.delay(collection_id)

        logger.info(f"Marked {len(chunk_ids)} chunks as inactive, rebuild scheduled")
        return len(chunk_ids)  # для совместимости

    # ----------------------------------------------------------------------
    # Управление индексом коллекции
    # ----------------------------------------------------------------------
    def build_index(
        self,
        collection_id: int,
        index_type: str = 'flat',
        force: bool = False
    ) -> bool:
        """
        Построить индекс для коллекции с нуля.
        - Собирает все чанки коллекции с эмбеддингами.
        - Создаёт FAISS индекс.
        - Сохраняет на диск.
        """
        from ..models import DocumentChunk
        import numpy as np
        from django.utils import timezone

        # Получаем все чанки коллекции с эмбеддингами
        chunks = DocumentChunk.objects.filter(
            document__collection_id=collection_id,
            has_embedding=True,
            is_active=True
        ).exclude(embedding_vector__isnull=True)

        if not chunks.exists():
            logger.warning(f"No chunks with embeddings for collection {collection_id}")
            return False

        # Определяем размерность
        sample_vec = chunks.first().embedding_vector
        dimension = len(sample_vec) if isinstance(sample_vec, list) else sample_vec.shape[0]

        # Создаём новый сервис
        svc = FAISSService()
        svc.create_index(dimension=dimension, index_type=index_type, metric=self._default_metric)

        # Добавляем векторы
        vectors = []
        chunk_ids = []
        for chunk in chunks.iterator():
            vec = np.array(chunk.embedding_vector, dtype='float32')
            vectors.append(vec)
            chunk_ids.append(chunk.id)
            if len(vectors) >= 1000:  # добавляем батчами
                svc.add_vectors(np.stack(vectors), chunk_ids)
                vectors = []
                chunk_ids = []

        if vectors:
            svc.add_vectors(np.stack(vectors), chunk_ids)

        # Сохраняем
        svc.save_index(collection_id)

        # Обновляем кэш
        self._index_cache[collection_id] = svc

        # Обновляем поле коллекции
        from ..models import DocumentCollection
        DocumentCollection.objects.filter(id=collection_id).update(
            vector_index_created=True,
            total_vectors=svc.index.ntotal,
            vector_dimension=dimension,
            index_metadata={
                'index_type': index_type,
                'metric': self._default_metric,
                'built_at': timezone.now().isoformat(),  # ✅ правильно
            }
        )

        logger.info(f"✅ Index built for collection {collection_id}: {svc.index.ntotal} vectors")
        return True

    def update_index(self, collection_id: int) -> int:
        """
        Обновить индекс: добавить новые чанки, появившиеся после последней индексации.
        """
        from ..models import DocumentChunk, DocumentCollection

        collection = DocumentCollection.objects.filter(id=collection_id).first()
        if not collection or not collection.vector_index_created:
            logger.warning(f"Collection {collection_id} has no index, cannot update")
            return 0

        # Определяем дату последнего обновления из метаданных
        last_update = collection.index_metadata.get('last_update')
        if last_update:
            from datetime import datetime
            if isinstance(last_update, str):
                try:
                    last_update = datetime.fromisoformat(last_update)
                except ValueError:
                    last_update = None

        # Новые чанки с эмбеддингами
        query = DocumentChunk.objects.filter(
            document__collection_id=collection_id,
            has_embedding=True,
            is_active=True
        ).exclude(embedding_vector__isnull=True)

        if last_update:
            query = query.filter(created_at__gt=last_update)

        new_chunks = list(query)
        if not new_chunks:
            return 0

        # Добавляем в индекс
        added = self.index_chunks(new_chunks, collection_id)

        # Обновляем метаданные
        collection.total_vectors = self._get_faiss_service(collection_id).index.ntotal
        collection.index_metadata['last_update'] = timezone.now().isoformat()
        collection.index_metadata['last_update_count'] = added
        collection.save(update_fields=['total_vectors', 'index_metadata'])

        logger.info(f"Updated index for collection {collection_id}: added {added} vectors")
        return added

    # ----------------------------------------------------------------------
    # Статистика
    # ----------------------------------------------------------------------
    def get_statistics(self, collection_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Получить статистику векторного хранилища.
        """
        if collection_id is None:
            # Общая статистика по всем коллекциям
            from ..models import DocumentCollection
            collections = DocumentCollection.objects.filter(is_active=True)
            return {
                'total_collections': collections.count(),
                'collections_with_index': collections.filter(vector_index_created=True).count(),
                'total_indexed_vectors': sum(c.total_vectors for c in collections),
            }
        else:
            svc = self._get_faiss_service(collection_id)
            stats = svc.get_index_stats(collection_id)
            stats['collection_id'] = collection_id
            return stats

    # ----------------------------------------------------------------------
    # Очистка кэша
    # ----------------------------------------------------------------------
    def clear_cache(self):
        """Очистить кэш загруженных индексов."""
        self._index_cache.clear()
        logger.info("Vector store cache cleared")