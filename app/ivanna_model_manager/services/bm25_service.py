import logging
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Service:
    """
    Сервис для ранжирования документов по алгоритму BM25.
    """

    def __init__(self):
        self._bm25_cache = {}  # collection_id -> (BM25, texts)
        self._id_map_cache = {}  # collection_id -> list of chunk_ids (порядок соответствует индексам)
        logger.info("BM25Service initialized")

    def _tokenize(self, text: str) -> List[str]:
        """Простая токенизация по словам (можно заменить на более продвинутую)."""
        return text.lower().split()

    def build_index(self, collection_id: int, texts: List[str], chunk_ids: List[int]) -> None:
        """
        Строит BM25 индекс для коллекции и сохраняет в кэше.
        """
        tokenized_corpus = [self._tokenize(doc) for doc in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        self._bm25_cache[collection_id] = bm25
        self._id_map_cache[collection_id] = chunk_ids
        logger.info(f"BM25 index built for collection {collection_id} with {len(texts)} documents")

    def search(self, query: str, collection_id: int, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Выполняет поиск по коллекции с использованием BM25.
        Возвращает список результатов с полями:
        - chunk_id: идентификатор чанка
        - score: BM25 score
        - text: текст документа (опционально)
        """
        if collection_id not in self._bm25_cache:
            raise ValueError(f"BM25 index for collection {collection_id} not built. Call build_index first.")

        bm25 = self._bm25_cache[collection_id]
        chunk_ids = self._id_map_cache[collection_id]

        tokenized_query = self._tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # Сортируем по убыванию
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'chunk_id': chunk_ids[idx],
                    'score': float(scores[idx]),
                    'text': None  # текст можно не возвращать, он будет получен отдельно
                })
        return results

    def clear_cache(self, collection_id: Optional[int] = None):
        """
        Очищает кэш BM25 для указанной коллекции или всего.
        """
        if collection_id:
            self._bm25_cache.pop(collection_id, None)
            self._id_map_cache.pop(collection_id, None)
            logger.info(f"BM25 cache cleared for collection {collection_id}")
        else:
            self._bm25_cache.clear()
            self._id_map_cache.clear()
            logger.info("BM25 cache cleared")