import os
import time
import logging
import torch
import langdetect
import numpy as np
from typing import List, Dict, Any, Optional, Union
from django.db.models import Q
from django.utils import timezone
from django.conf import settings
from huggingface_hub import scan_cache_dir
from sentence_transformers import CrossEncoder

from .embedding_service import embedding_service
from .faiss_service import FAISSService
from .bm25_service import BM25Service
from ..models import (
    DocumentCollection, Document, DocumentChunk, EmbeddingModel,
    SearchQuery, SearchResult, RAGSession, RetrievedChunk,
    ModelCheckpoint
)

logger = logging.getLogger(__name__)

# HF endpoint: не перебивать окружение зеркалом жёстко — этот модуль
# импортируется первым через services/__init__ и отравлял os.environ
# для download_service (скачивание моделей падало FileMetadataError).
os.environ.setdefault('HF_ENDPOINT', 'https://huggingface.co')

# ----------------------------------------------------------------------
# Вспомогательная функция очистки кэша для указанной модели
# ----------------------------------------------------------------------
def _clean_hf_cache_for_model(repo_id: str):
    try:
        cache_info = scan_cache_dir()
        repos_to_delete = [r for r in cache_info.repos if r.repo_id == repo_id]
        if not repos_to_delete:
            logger.info(f"No cache entries found for {repo_id}")
            return
        # Собираем строковые идентификаторы ревизий
        revisions = []
        for repo in repos_to_delete:
            for rev in repo.revisions:
                revisions.append(rev.commit_hash)  # или rev.revision, смотря что есть
        strategy = cache_info.delete_revisions(*revisions)
        strategy.execute()
        logger.info(f"Cleaned global HF cache for {repo_id}")
    except Exception as e:
        logger.warning(f"Failed to clean HF cache for {repo_id}: {e}")

class EnhancedSearchService:
    """
    Улучшенный сервис поиска с поддержкой:
    - семантического поиска (FAISS) с использованием моделей эмбеддингов из БД
    - ключевого поиска (BM25)
    - гибридного поиска с весами
    - учёта свежести документа
    - реранжирования (кросс-энкодер с автоопределением языка)
    - поддержки нескольких коллекций
    """

    def __init__(self):
        self.embedding = embedding_service
        self.faiss = FAISSService()
        self._faiss_index_cache = {}
        self._rerankers = {}  # язык -> модель
        self.bm25_service = BM25Service() if getattr(settings, 'KEYWORD_SEARCH_ENGINE', 'bm25') == 'bm25' else None
        self._bm25_index_built = set()  # коллекции, для которых построен BM25 индекс

    def _clean_hf_cache_for_model(self, repo_id: str):
        """Удаляет все ревизии указанной модели из глобального кэша Hugging Face."""
        try:
            cache_info = scan_cache_dir()
            repos_to_delete = [r for r in cache_info.repos if r.repo_id == repo_id]
            if not repos_to_delete:
                logger.info(f"No cache entries found for {repo_id}")
                return
            revisions = []
            for repo in repos_to_delete:
                for rev in repo.revisions:
                    revisions.append(rev.commit_hash)
            strategy = cache_info.delete_revisions(*revisions)
            strategy.execute()
            logger.info(f"Cleaned global HF cache for {repo_id}")
        except Exception as e:
            logger.warning(f"Failed to clean HF cache for {repo_id}: {e}")

    def _load_cross_encoder_with_retry(self, model_name: str, max_retries: int = 2) -> Optional[CrossEncoder]:
        """Загружает кросс-энкодер с повторными попытками, используя CPU при OOM."""
        devices_to_try = ['cuda' if torch.cuda.is_available() else 'cpu', 'cpu']
        for attempt in range(max_retries):
            for device in devices_to_try:
                try:
                    logger.info(f"Loading cross-encoder {model_name} on {device}, attempt {attempt+1}")
                    model = CrossEncoder(model_name, max_length=512, device=device)
                    logger.info(f"✅ Cross-encoder {model_name} loaded successfully on {device}")
                    return model
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and device == 'cuda':
                        logger.warning(f"CUDA OOM on {device}, falling back to CPU for this attempt")
                        continue
                    else:
                        logger.warning(f"Failed to load {model_name} on {device}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name} on {device}: {e}")

            # Если не удалось загрузить на этом проходе, чистим кэш и повторяем
            if attempt < max_retries - 1:
                self._clean_hf_cache_for_model(model_name)
                time.sleep(2)
        logger.error(f"Giving up after {max_retries} attempts")
        return None

    def _get_reranker(self, query: str):
        """
        Определяет язык запроса и возвращает соответствующий кросс-энкодер.
        Для русского языка использует DiTy/cross-encoder-russian-msmarco.
        Для всех остальных случаев — английскую модель.
        """
        try:
            lang = langdetect.detect(query)
        except:
            lang = 'en'

        model_name_en = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

        if lang == 'ru':
            model_name_ru = 'DiTy/cross-encoder-russian-msmarco'
            if model_name_ru not in self._rerankers:
                logger.info(f"Loading Russian cross-encoder: {model_name_ru}")
                model = self._load_cross_encoder_with_retry(model_name_ru)
                if model is None:
                    logger.warning("Falling back to English cross-encoder")
                    model = self._load_cross_encoder_with_retry(model_name_en)
                    if model is None:
                        logger.error("Could not load any cross-encoder, reranking disabled")
                        return None, lang
                    lang = 'en'
                self._rerankers[model_name_ru] = model
            return self._rerankers[model_name_ru], lang
        else:
            if model_name_en not in self._rerankers:
                model = self._load_cross_encoder_with_retry(model_name_en)
                if model is None:
                    logger.error("Could not load English cross-encoder, reranking disabled")
                    return None, lang
                self._rerankers[model_name_en] = model
            return self._rerankers[model_name_en], lang

    # ----------------------------------------------------------------------
    # Семантический поиск (FAISS)
    # ----------------------------------------------------------------------
    def semantic_search(
        self,
        query: str,
        collection_id: Optional[int] = None,
        embedding_model_id: Optional[int] = None,
        k: int = 10,
        threshold: float = 0.5,
        nprobe: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        logger.info(f"semantic_search: query='{query[:50]}...', collection_id={collection_id}, k={k}, threshold={threshold}")
        try:
            query_emb = self.embedding.create_embedding(query, model_id=embedding_model_id, normalize=True)
            logger.info(f"Query embedding created, shape: {query_emb.shape}")
        except Exception as e:
            logger.error(f"Failed to create query embedding: {e}")
            return []

        if collection_id:
            if collection_id not in self._faiss_index_cache:
                faiss_svc = FAISSService()
                if faiss_svc.load_index(collection_id):
                    self._faiss_index_cache[collection_id] = faiss_svc
                    logger.info(f"FAISS index for collection {collection_id} loaded, total vectors: {faiss_svc.index.ntotal}")
                else:
                    logger.warning(f"FAISS index for collection {collection_id} not found")
                    return []
            faiss_svc = self._faiss_index_cache[collection_id]

            # Проверка размерности
            if query_emb.ndim == 1:
                query_emb = query_emb.reshape(1, -1)
            if query_emb.shape[1] != faiss_svc.dimension:
                logger.error(f"Query dim {query_emb.shape[1]} != index dim {faiss_svc.dimension}")
                return []

            results = faiss_svc.search(query_emb, k=k * 2, nprobe=nprobe)
            logger.info(f"FAISS raw results: {len(results)} chunks")
            formatted = []
            for chunk_id, score in results:
                if score < threshold:
                    continue
                try:
                    chunk = DocumentChunk.objects.select_related('document').get(id=chunk_id, is_active=True)
                    formatted.append({
                        'chunk_id': chunk.id,
                        'document_id': chunk.document_id,
                        'similarity': score,
                        'semantic_score': score,
                        'content': chunk.content,
                        'document_title': chunk.document.title,
                        'document_uploaded_at': chunk.document.uploaded_at,
                        'chunk_index': chunk.chunk_index,
                        'metadata': chunk.metadata or {},
                        'source': 'semantic'
                    })
                    if len(formatted) >= k:
                        break
                except DocumentChunk.DoesNotExist:
                    continue
            logger.info(f"After threshold filtering, returning {len(formatted)} semantic results")
            return formatted
        else:
            logger.warning("Semantic search without collection_id is not supported")
            return []

    # ----------------------------------------------------------------------
    # Ключевой поиск (BM25)
    # ----------------------------------------------------------------------
    def keyword_search(
        self,
        query: str,
        collection_id: Optional[int] = None,
        k: int = 10,
        use_bm25: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Поиск по ключевым словам с использованием BM25.
        """
        logger.info(f"keyword_search: query='{query[:50]}...', collection_id={collection_id}, k={k}, use_bm25={use_bm25}")
        if not collection_id:
            logger.warning("Keyword search without collection_id is not supported")
            return []

        if use_bm25 and self.bm25_service:
            # Получаем все чанки коллекции (активные)
            chunks_qs = DocumentChunk.objects.filter(
                document__collection_id=collection_id,
                is_active=True
            ).values_list('id', 'content')
            chunks_list = list(chunks_qs)
            logger.info(f"Loaded {len(chunks_list)} chunks for BM25 indexing")
            if not chunks_list:
                return []

            # Проверяем, есть ли уже индекс в кэше
            if collection_id not in self._bm25_index_built:
                texts = [c[1] for c in chunks_list]
                chunk_ids = [c[0] for c in chunks_list]
                self.bm25_service.build_index(collection_id, texts, chunk_ids)
                self._bm25_index_built.add(collection_id)
                logger.info(f"BM25 index built for collection {collection_id}")

            # Выполняем поиск
            bm25_results = self.bm25_service.search(query, collection_id, top_k=k)
            logger.info(f"BM25 returned {len(bm25_results)} results")

            # Преобразуем в формат, ожидаемый другими методами
            results = []
            for res in bm25_results:
                chunk_id = res['chunk_id']
                try:
                    chunk = DocumentChunk.objects.select_related('document').get(id=chunk_id)
                except DocumentChunk.DoesNotExist:
                    continue
                results.append({
                    'chunk_id': chunk.id,
                    'document_id': chunk.document_id,
                    'keyword_score': res['score'],
                    'content': chunk.content,
                    'document_title': chunk.document.title,
                    'document_uploaded_at': chunk.document.uploaded_at,
                    'chunk_index': chunk.chunk_index,
                    'metadata': chunk.metadata or {},
                    'source': 'keyword'
                })
            return results[:k]

        else:
            # Fallback на старый метод с извлечением ключевых слов
            logger.info("BM25 not available, falling back to keyword extraction")
            keywords = self._extract_keywords(query)
            if not keywords:
                return []

            filters = Q(is_active=True) & Q(document__collection_id=collection_id)
            chunks = DocumentChunk.objects.filter(filters).select_related('document')[:5000]
            logger.info(f"Checking {chunks.count()} chunks for keyword match")

            results = []
            for chunk in chunks:
                score = self._calculate_keyword_score(chunk.content, keywords, use_bm25=False)
                if score > 0.1:
                    results.append({
                        'chunk_id': chunk.id,
                        'document_id': chunk.document_id,
                        'keyword_score': score,
                        'content': chunk.content,
                        'document_title': chunk.document.title,
                        'document_uploaded_at': chunk.document.uploaded_at,
                        'chunk_index': chunk.chunk_index,
                        'metadata': chunk.metadata or {},
                        'source': 'keyword'
                    })

            results.sort(key=lambda x: x['keyword_score'], reverse=True)
            logger.info(f"Fallback keyword search returned {len(results)} results")
            return results[:k]

    # ----------------------------------------------------------------------
    # Гибридный поиск (объединение семантики и ключевых слов)
    # ----------------------------------------------------------------------
    def enhanced_hybrid_search(
        self,
        query: str,
        collection_ids: Optional[Union[int, List[int]]] = None,
        embedding_model_id: Optional[int] = None,
        k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.2,
        freshness_weight: float = 0.1,
        use_reranking: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Гибридный поиск с объединением результатов по нескольким коллекциям.
        embedding_model_id передаётся в семантический поиск.
        """
        logger.info(f"enhanced_hybrid_search: query='{query[:50]}...', collection_ids={collection_ids}, k={k}, weights: s={semantic_weight}, kw={keyword_weight}, f={freshness_weight}, reranking={use_reranking}")
        if collection_ids is None:
            collection_ids = []
        elif isinstance(collection_ids, int):
            collection_ids = [collection_ids]

        all_results = []
        for cid in collection_ids:
            # Выполняем поиск по каждой коллекции
            logger.info(f"Processing collection {cid}")
            semantic_results = self.semantic_search(
                query,
                collection_id=cid,
                embedding_model_id=embedding_model_id,
                k=k*3,
                threshold=0.3
            )
            keyword_results = self.keyword_search(query, collection_id=cid, k=k*2)

            logger.info(f"Collection {cid}: semantic={len(semantic_results)}, keyword={len(keyword_results)}")

            merged = self._merge_search_results(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )
            if freshness_weight > 0:
                merged = self._add_freshness_score(merged, weight=freshness_weight)
            all_results.extend(merged)
            logger.info(f"After merge (with freshness), collection {cid} contributed {len(merged)} results")

        # Объединяем результаты, удаляем дубликаты по chunk_id
        seen = set()
        unique_results = []
        for r in all_results:
            if r['chunk_id'] not in seen:
                seen.add(r['chunk_id'])
                unique_results.append(r)

        logger.info(f"After deduplication: {len(unique_results)} results")

        # Реранжирование по общему пулу
        if use_reranking and unique_results:
            logger.info("Applying reranking")
            unique_results = self._rerank_results(query, unique_results)
            logger.info(f"After reranking: {len(unique_results)} results")

        unique_results.sort(key=lambda x: x['final_score'], reverse=True)
        final = unique_results[:k]
        logger.info(f"Final results: {len(final)}")
        return final

    # ----------------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------------
    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        """Извлечение значимых ключевых слов из запроса (fallback)."""
        import re
        words = re.findall(r'\b[а-яА-ЯёЁa-zA-Z]{3,}\b', query.lower())
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'не', 'что',
                      'это', 'как', 'так', 'у', 'к', 'о', 'об', 'за', 'из', 'или'}
        keywords = [w for w in words if w not in stop_words]
        return list(set(keywords))

    @staticmethod
    def _calculate_keyword_score(content: str, keywords: List[str], use_bm25: bool = True) -> float:
        """Вычисление релевантности по ключевым словам (fallback)."""
        if not keywords or not content:
            return 0.0

        content_lower = content.lower()
        total_score = 0.0
        for kw in keywords:
            freq = content_lower.count(kw)
            if freq > 0:
                # Упрощённый BM25 без нормализации по длине
                k1 = 1.2
                total_score += (freq * (k1 + 1)) / (freq + k1)
        return total_score / len(keywords)

    @staticmethod
    def _merge_search_results(
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict]:
        """Объединение результатов семантического и ключевого поиска."""
        merged = {}
        for r in semantic_results:
            cid = r['chunk_id']
            merged[cid] = r.copy()
            merged[cid]['semantic_score'] = r.get('similarity', 0)
            merged[cid]['keyword_score'] = 0.0
            merged[cid]['final_score'] = merged[cid]['semantic_score'] * semantic_weight

        for r in keyword_results:
            cid = r['chunk_id']
            if cid in merged:
                merged[cid]['keyword_score'] = r['keyword_score']
                merged[cid]['final_score'] += r['keyword_score'] * keyword_weight
                merged[cid]['source'] = 'hybrid'
            else:
                new = r.copy()
                new['semantic_score'] = 0.0
                new['final_score'] = r['keyword_score'] * keyword_weight
                new['source'] = 'keyword'
                merged[cid] = new

        return list(merged.values())

    @staticmethod
    def _add_freshness_score(results: List[Dict], weight: float = 0.1) -> List[Dict]:
        """Добавление веса свежести документа."""
        if not results:
            return results

        max_date = None
        for r in results:
            d = r.get('document_uploaded_at')
            if d and (max_date is None or d > max_date):
                max_date = d

        if not max_date:
            return results

        max_ts = max_date.timestamp() if hasattr(max_date, 'timestamp') else timezone.now().timestamp()

        for r in results:
            d = r.get('document_uploaded_at')
            if d:
                doc_ts = d.timestamp() if hasattr(d, 'timestamp') else timezone.now().timestamp()
                freshness = max(0.0, 1.0 - (max_ts - doc_ts) / (365 * 24 * 3600))
                r['freshness_score'] = freshness
                r['final_score'] += freshness * weight
        return results

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Реренжирование результатов с использованием кросс-энкодера."""
        reranker, lang = self._get_reranker(query)
        if reranker is None:
            logger.warning("No reranker available, skipping reranking")
            return results

        try:
            logger.info(f"Reranking {len(results)} results, language={lang}")
            pairs = [(query, r['content'][:1000]) for r in results]
            scores = reranker.predict(pairs)
            for i, r in enumerate(results):
                r['reranker_score'] = float(scores[i])
                r['final_score'] = (r['final_score'] + r['reranker_score']) / 2
            logger.info("Reranking completed")
        except Exception as e:
            logger.warning(f"Reranking failed: {e}", exc_info=True)
        return results

    def invalidate_bm25_cache(self, collection_id: int):
        """
        Инвалидирует кэш BM25 для коллекции (например, после изменения документов).
        """
        if self.bm25_service:
            self.bm25_service.clear_cache(collection_id)
            self._bm25_index_built.discard(collection_id)
            logger.info(f"BM25 cache invalidated for collection {collection_id}")


class RAGService:
    """
    Основной сервис для RAG (Retrieval-Augmented Generation).
    - Обработка запроса: поиск + генерация ответа.
    - Управление сессиями RAG.
    - Fallback-механизмы при отсутствии индекса.
    """

    def __init__(self):
        self.engine = EnhancedSearchService()
        self.embedding = embedding_service

    def process_query(
        self,
        query: str,
        user_id: str,
        collection_id: Optional[int] = None,
        document_ids: List[int] = None,
        system_prompt: str = "",
        llm_model_id: Optional[int] = None,
        embedding_model_id: Optional[int] = None,
        search_config: Dict[str, Any] = None,
        generation_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Полный цикл RAG: поиск + генерация ответа.
        Если embedding_model_id не указан, используется модель из коллекции (если есть) или глобальная по умолчанию.
        """
        from datetime import datetime
        rag_session = RAGSession.objects.create(
            session_id=f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}",
            user_id=user_id,
            original_query=query,
            system_prompt=system_prompt,
            status='SEARCHING'
        )

        try:
            # 1. Выбор модели эмбеддингов
            if embedding_model_id:
                embedding_model = EmbeddingModel.objects.get(id=embedding_model_id)
            else:
                embedding_model = EmbeddingModel.objects.filter(is_default=True).first()
            rag_session.embedding_model = embedding_model

            # 2. Выбор LLM
            if llm_model_id:
                llm_model = ModelCheckpoint.objects.get(id=llm_model_id, is_active=True)
            else:
                llm_model = ModelCheckpoint.objects.filter(
                    parameter_count__gte=6_000_000_000,
                    parameter_count__lte=8_000_000_000,
                    status='LOADED',
                    is_active=True
                ).first()
            rag_session.llm_model = llm_model
            rag_session.save()

            # 3. Поиск релевантных чанков
            search_start = timezone.now()
            collection = None
            if collection_id:
                collection = DocumentCollection.objects.filter(id=collection_id).first()
                if not collection:
                    return self._error_response(rag_session, f"Collection {collection_id} not found")

            # Определяем, какую модель эмбеддингов использовать для поиска
            effective_embedding_id = None
            if collection and collection.embedding_model:
                effective_embedding_id = collection.embedding_model.id
            elif embedding_model:
                effective_embedding_id = embedding_model.id

            if collection_id and collection:
                if not collection.vector_index_created:
                    logger.info(f"Collection {collection_id} has no vector index, using text fallback")
                    search_results = self._text_search_fallback(
                        query, collection_id, document_ids,
                        k=search_config.get('top_k', 5) if search_config else 5
                    )
                else:
                    config = search_config or {}
                    search_results = self.engine.enhanced_hybrid_search(
                        query=query,
                        collection_ids=collection_id,
                        embedding_model_id=effective_embedding_id,
                        k=config.get('top_k', 5),
                        semantic_weight=config.get('semantic_weight', 0.7),
                        keyword_weight=config.get('keyword_weight', 0.2),
                        freshness_weight=config.get('freshness_weight', 0.1),
                        use_reranking=config.get('use_reranking', True)
                    )
            else:
                search_results = self._text_search_fallback(
                    query, None, document_ids,
                    k=search_config.get('top_k', 5) if search_config else 5
                )

            search_time = (timezone.now() - search_start).total_seconds() * 1000

            if not search_results:
                rag_session.status = 'FAILED'
                rag_session.save()
                return {
                    'success': False,
                    'error': 'No relevant documents found',
                    'session_id': rag_session.session_id
                }

            # 4. Сохраняем поисковый запрос
            search_query = SearchQuery.objects.create(
                query_text=query,
                search_type='HYBRID_ENHANCED' if collection_id and collection and collection.vector_index_created else 'TEXT_FALLBACK',
                collection_filter_id=collection_id,
                user_id=user_id,
                processing_time_ms=search_time,
                search_config=search_config or {}
            )
            rag_session.search_query = search_query

            # 5. Сохраняем результаты в SearchResult и RetrievedChunk
            retrieved_chunks = []
            for i, result in enumerate(search_results[:search_config.get('top_k', 5)]):
                chunk_id = result['chunk_id']
                try:
                    chunk = DocumentChunk.objects.get(id=chunk_id, is_active=True)
                except DocumentChunk.DoesNotExist:
                    continue

                SearchResult.objects.create(
                    query=search_query,
                    chunk=chunk,
                    relevance_score=result.get('final_score', result.get('similarity', 0)),
                    rank_position=i + 1,
                    scoring_method='hybrid_enhanced',
                    metadata={
                        'semantic_score': result.get('semantic_score', 0),
                        'keyword_score': result.get('keyword_score', 0),
                        'freshness_score': result.get('freshness_score', 0),
                        'reranker_score': result.get('reranker_score', 0),
                        'final_score': result.get('final_score', 0),
                        'source': result.get('source', 'unknown')
                    }
                )

                retrieved_chunk = RetrievedChunk.objects.create(
                    session=rag_session,
                    chunk=chunk,
                    retrieval_rank=i + 1,
                    relevance_score=result.get('final_score', 0),
                    used_in_context=True,
                    context_position=i
                )
                retrieved_chunks.append(retrieved_chunk)

            rag_session.mark_search_completed(search_time, search_query)

            # 6. Формируем контекст и промпт
            context = self._build_context(search_results[:5])
            augmented_prompt = self._augment_prompt(query, context, system_prompt)
            rag_session.augmented_prompt = augmented_prompt
            rag_session.save()

            # 7. Генерация ответа через LLM
            if llm_model and llm_model.status == 'LOADED':
                from .redis_model_service import redis_model_manager

                gen_config = generation_config or {
                    'max_tokens': 1000,
                    'temperature': 0.7,
                    'top_p': 0.9
                }

                generation_start = timezone.now()
                result = redis_model_manager.generate_text(
                    model_id=llm_model.huggingface_id or llm_model.name,
                    prompt=augmented_prompt,
                    **gen_config
                )
                generation_time = (timezone.now() - generation_start).total_seconds() * 1000

                if result.get('success'):
                    answer = result['response']
                    tokens = {
                        'input': result.get('input_tokens', 0),
                        'output': result.get('output_tokens', 0),
                    }

                    rag_session.mark_generation_completed(
                        answer=answer,
                        metadata=result.get('metadata', {}),
                        generation_time_ms=generation_time,
                        tokens=tokens
                    )

                    for rc in retrieved_chunks:
                        rc.chunk.mark_used()

                    return {
                        'success': True,
                        'session_id': rag_session.session_id,
                        'answer': answer,
                        'sources': [
                            {
                                'document_id': rc.chunk.document.id,
                                'document_title': rc.chunk.document.title,
                                'chunk_id': rc.chunk.id,
                                'content': rc.chunk.content[:200] + '...',
                                'relevance_score': rc.relevance_score,
                                'search_scores': rc.metadata or {}
                            }
                            for rc in retrieved_chunks[:3]
                        ],
                        'metrics': {
                            'search_time_ms': search_time,
                            'generation_time_ms': generation_time,
                            'total_time_ms': search_time + generation_time,
                            'retrieved_chunks': len(retrieved_chunks),
                            'input_tokens': tokens['input'],
                            'output_tokens': tokens['output']
                        }
                    }
                else:
                    raise Exception(f"LLM generation failed: {result.get('error')}")
            else:
                raise Exception("LLM model is not loaded")

        except Exception as e:
            logger.error(f"RAG processing error: {e}", exc_info=True)
            rag_session.status = 'FAILED'
            rag_session.save()
            return self._error_response(rag_session, str(e))

    # ----------------------------------------------------------------------
    # Поиск только (без генерации)
    # ----------------------------------------------------------------------
    def search(
        self,
        query: str,
        user_id: str,
        collection_ids: Optional[Union[int, List[int]]] = None,
        document_ids: List[int] = None,
        top_k: int = 5,
        search_type: str = 'hybrid',
        embedding_model_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Только поиск документов, без генерации ответа.
        Если embedding_model_id не указан, используется глобальная модель по умолчанию.
        """
        logger.info(f"=== RAGService.search called ===")
        logger.info(f"  query: {query[:100]}")
        logger.info(f"  collection_ids: {collection_ids}")
        logger.info(f"  top_k: {top_k}")
        logger.info(f"  search_type: {search_type}")
        logger.info(f"  embedding_model_id: {embedding_model_id}")

        # Определяем модель эмбеддингов
        effective_embedding_id = embedding_model_id
        if not effective_embedding_id:
            default_model = EmbeddingModel.objects.filter(is_default=True).first()
            if not default_model:
                logger.error("No default embedding model available")
                return {'success': False, 'error': 'No embedding model available'}
            effective_embedding_id = default_model.id
            logger.info(f"Using default embedding model: {default_model.name} (ID={effective_embedding_id})")
        else:
            # Получаем модель по ID для логирования
            try:
                model_obj = EmbeddingModel.objects.get(id=effective_embedding_id)
                logger.info(f"Using specified embedding model: {model_obj.name} (ID={effective_embedding_id})")
            except EmbeddingModel.DoesNotExist:
                logger.warning(f"Embedding model with ID {effective_embedding_id} not found, will fail later")

        if collection_ids:
            if isinstance(collection_ids, int):
                collection_ids = [collection_ids]
            logger.info(f"Searching in collections: {collection_ids}")

            if search_type == 'semantic':
                if len(collection_ids) != 1:
                    return {'success': False, 'error': 'Semantic search requires exactly one collection'}
                logger.info("Performing semantic search")
                results = self.engine.semantic_search(
                    query,
                    collection_id=collection_ids[0],
                    embedding_model_id=effective_embedding_id,
                    k=top_k,
                    **kwargs
                )
            elif search_type == 'keyword':
                if len(collection_ids) != 1:
                    return {'success': False, 'error': 'Keyword search requires exactly one collection'}
                logger.info("Performing keyword search")
                results = self.engine.keyword_search(query, collection_id=collection_ids[0], k=top_k)
            else:  # hybrid
                logger.info("Performing enhanced hybrid search")
                results = self.engine.enhanced_hybrid_search(
                    query=query,
                    collection_ids=collection_ids,
                    embedding_model_id=effective_embedding_id,
                    k=top_k,
                    semantic_weight=kwargs.get('semantic_weight', 0.7),
                    keyword_weight=kwargs.get('keyword_weight', 0.2),
                    freshness_weight=kwargs.get('freshness_weight', 0.1),
                    use_reranking=kwargs.get('use_reranking', True)
                )
        else:
            logger.info("No collection_ids, using text fallback search")
            results = self._text_search_fallback(query, None, document_ids, k=top_k)

        logger.info(f"Search returned {len(results)} raw results")

        formatted = []
        for r in results:
            formatted.append({
                'chunk_id': r['chunk_id'],
                'document_id': r['document_id'],
                'document_title': r.get('document_title', ''),
                'content': r.get('content', ''),
                'relevance_score': r.get('final_score', r.get('similarity', r.get('keyword_score', 0))),
                'scores': {
                    'semantic': r.get('semantic_score', 0),
                    'keyword': r.get('keyword_score', 0),
                    'freshness': r.get('freshness_score', 0),
                    'reranker': r.get('reranker_score', 0)
                },
                'metadata': r.get('metadata', {})
            })

        logger.info(f"Formatted {len(formatted)} results")
        return {
            'success': True,
            'query': query,
            'total_results': len(formatted),
            'search_type': search_type,
            'results': formatted
        }

    # ----------------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------------
    @staticmethod
    def _build_context(search_results: List[Dict]) -> str:
        parts = []
        for i, r in enumerate(search_results, 1):
            score = r.get('final_score', r.get('similarity', 0))
            parts.append(
                f"[Документ {i}: {r.get('document_title', 'Без названия')}, "
                f"релевантность: {score:.3f}]\n{r.get('content', '')}"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _augment_prompt(query: str, context: str, system_prompt: str = "") -> str:
        base = system_prompt or "Ты — полезный ассистент с доступом к документам пользователя. Отвечай на основе предоставленных документов. Если информации недостаточно, скажи об этом."
        return f"""{base}

Контекст из документов:
{context}

Вопрос пользователя: {query}

Ответ:"""

    def _text_search_fallback(
        self,
        query: str,
        collection_id: Optional[int] = None,
        document_ids: List[int] = None,
        k: int = 5
    ) -> List[Dict]:
        from django.db.models import Q

        filters = Q(document__is_active=True)
        if collection_id:
            filters &= Q(document__collection_id=collection_id)
        if document_ids:
            filters &= Q(document_id__in=document_ids)

        keywords = EnhancedSearchService._extract_keywords(query)
        if not keywords:
            return []

        chunks = DocumentChunk.objects.filter(filters).select_related('document')[:2000]

        results = []
        for chunk in chunks:
            score = EnhancedSearchService._calculate_keyword_score(chunk.content, keywords, use_bm25=False)
            if score > 0.2:
                results.append({
                    'chunk_id': chunk.id,
                    'document_id': chunk.document_id,
                    'document_title': chunk.document.title,
                    'content': chunk.content,
                    'similarity': score,
                    'final_score': score,
                    'source': 'text_fallback'
                })
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:k]

    @staticmethod
    def _error_response(session: RAGSession, error: str) -> Dict[str, Any]:
        return {
            'success': False,
            'error': error,
            'session_id': session.session_id
        }