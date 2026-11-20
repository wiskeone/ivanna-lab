import os
import logging
from datetime import datetime, timedelta

import requests
import torch
import time

from django.db.models import Q, Count, Sum
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.utils.translation import gettext as _
from django.http import JsonResponse, HttpResponse
import json
import csv
import re

from django.contrib.auth.decorators import login_required, permission_required
from django.views.decorators.http import require_http_methods
from django.contrib.auth.mixins import LoginRequiredMixin

from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, BasePermission
from rest_framework.response import Response
from rest_framework.views import APIView

from ..models import (
    RerankerModel,
    Experiment, ModelCheckpoint, ModelMergeJob, InferenceSession,
    ModelEvaluation, ChatSession, ModelComparison, FineTuningJob,
    ModelDownloadJob, EvaluationDataset, ModelEvaluationTask, EvaluationResult,
    DocumentCollection, Document, DocumentChunk, EmbeddingModel,
    SearchQuery, SearchResult, RAGSession, RetrievedChunk,
    SearchQualityFeedback, ModelConversionJob, DistillationQuestionSet, 
    DistillationChunkGroup, DistillationQuestion, DistillationAnswerJob, 
    DistillationAnswer, DistillationDataset, ManualInstructionSet,
    ManualInstruction, HuggingFaceDataset, 
)
from ..serializers import (
    RerankerModelSerializer,
    ExperimentSerializer, ModelCheckpointSerializer, ModelMergeJobSerializer,
    InferenceSessionSerializer, ModelEvaluationSerializer,
    ChatSessionSerializer, ModelComparisonSerializer, FineTuningJobSerializer,
    ModelDownloadJobSerializer, EvaluationDatasetSerializer,
    ModelEvaluationTaskSerializer, EvaluationResultSerializer,
    DocumentCollectionSerializer, DocumentSerializer, DocumentChunkSerializer,
    EmbeddingModelSerializer, SearchQuerySerializer, SearchResultSerializer,
    RAGSessionSerializer, RetrievedChunkSerializer,
    SearchQualityFeedbackSerializer,
    TextGenerationSerializer, ModelLoadSerializer, ModelConversionJobSerializer,
    DistillationQuestionSetSerializer, DistillationChunkGroupSerializer,
    DistillationQuestionSerializer, DistillationAnswerJobSerializer, 
    DistillationDatasetSerializer, ManualInstructionSetSerializer, 
    ManualInstructionSerializer, HuggingFaceDatasetSerializer, 
)
from ..services.redis_model_service import redis_model_manager
from ..services.chat_service import chat_service
from ..services.rag_service import RAGService, EnhancedSearchService
from ..tasks import (
    download_model_task, load_model_to_memory_task, unload_model_from_memory_task,
    merge_models_task, process_document_task, create_embeddings_for_collection_task,
    build_faiss_index_task, update_faiss_index_task,
    run_evaluation_task, process_uploaded_dataset, 
    run_fine_tuning_task, evaluate_fine_tuned_model_task,
    generate_questions_for_set_task, generate_answers_for_questions_task, 
    regenerate_answer_task, 
)

from celery.result import AsyncResult

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

from .core import IsCollectionOwnerOrPublic



# ============================================================================
# ПОИСК
# ============================================================================
@api_view(['POST'])
@login_required
def rag_search(request):
    import time
    from ..models import DocumentCollection
    from ..services.rag_service import EnhancedSearchService

    data = request.data
    query = data.get('query', '').strip()
    collection_id = data.get('collection_id')
    search_type = data.get('search_type', 'hybrid')
    top_k = data.get('top_k', 10)
    threshold = data.get('threshold', 0.5)
    semantic_weight = data.get('semantic_weight', 0.7)
    keyword_weight = data.get('keyword_weight', 0.2)
    freshness_weight = data.get('freshness_weight', 0.1)
    # Audit fix: a JSON "false" string is truthy — coerce properly
    use_reranking = str(data.get('use_reranking', True)).lower() not in ('false', '0', 'no')

    if not query:
        return JsonResponse({'success': False, 'error': 'Empty query'}, status=400)
    if not collection_id:
        return JsonResponse({'success': False, 'error': 'No collection selected'}, status=400)

    try:
        collection = DocumentCollection.objects.get(
            Q(id=collection_id) &
            (Q(owner_id=str(request.user.id)) | Q(is_public=True)) &
            Q(is_active=True)
        )
    except DocumentCollection.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Collection not found'}, status=404)

    # Определяем модель эмбеддингов, которая использовалась для индексации
    embedding_model = collection.get_effective_embedding_model()
    embedding_model_id = embedding_model.id if embedding_model else None

    search_service = EnhancedSearchService()
    start = time.time()

    if search_type == 'semantic':
        results = search_service.semantic_search(
            query, collection_id,
            embedding_model_id=embedding_model_id,
            k=top_k, threshold=threshold
        )
    elif search_type == 'keyword':
        results = search_service.keyword_search(query, collection_id, k=top_k)
    else:  # hybrid
        results = search_service.enhanced_hybrid_search(
            query=query,
            collection_ids=collection_id,
            embedding_model_id=embedding_model_id,  # <-- важно!
            k=top_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            freshness_weight=freshness_weight,
            use_reranking=use_reranking
        )

    processing_time = (time.time() - start) * 1000
    logger.info(f"Search returned {len(results)} results in {processing_time:.2f} ms")

    # Форматируем результат
    formatted_results = []
    for r in results:
        formatted_results.append({
            'chunk_id': r['chunk_id'],
            'document_id': r['document_id'],
            'document_title': r.get('document_title', ''),
            'content': r['content'][:500] + ('...' if len(r['content']) > 500 else ''),
            'relevance_score': r.get('final_score', r.get('similarity', r.get('keyword_score', 0))),
            'chunk_index': r.get('chunk_index', 0),
            'document_uploaded_at': r.get('document_uploaded_at'),
            'scores': {
                'semantic': r.get('semantic_score', 0),
                'keyword': r.get('keyword_score', 0),
                'freshness': r.get('freshness_score', 0),
                'reranker': r.get('reranker_score', 0)
            }
        })

    return JsonResponse({
        'success': True,
        'query': query,
        'collection': {'id': collection.id, 'name': collection.name},
        'total_results': len(formatted_results),
        'results': formatted_results,
        'processing_time_ms': processing_time
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_rag_search_suggestions(request):
    """Возвращает подсказки для поиска по коллекции."""
    query = request.GET.get('q', '').strip()
    collection_id = request.GET.get('collection_id')
    limit = int(request.GET.get('limit', 8))
    
    if not query or len(query) < 2 or not collection_id:
        return Response({'success': False, 'suggestions': []})
    
    try:
        collection = DocumentCollection.objects.get(
            Q(id=collection_id) & (Q(owner_id=str(request.user.id)) | Q(is_public=True)),
            is_active=True
        )
    except DocumentCollection.DoesNotExist:
        return Response({'success': False, 'error': 'Collection not found'}, status=404)
    
    suggestions = []
    
    # Поиск по заголовкам документов
    doc_titles = list(collection.documents.filter(
        Q(title__icontains=query) | Q(original_filename__icontains=query),
        is_active=True
    ).values_list('title', flat=True).distinct()[:limit])
    
    for title in doc_titles:
        if title:
            suggestions.append({'text': title, 'type': 'document_title'})
    
    # Если осталось место, ищем по содержимому чанков
    if len(suggestions) < limit:
        chunks = DocumentChunk.objects.filter(
            document__collection=collection,
            content__icontains=query,
            is_active=True
        ).select_related('document').distinct()[:limit * 2]
        
        seen_titles = set(doc_titles)
        for chunk in chunks:
            if len(suggestions) >= limit:
                break
            preview = chunk.content[:50] + ('...' if len(chunk.content) > 50 else '')
            if preview not in seen_titles:
                suggestions.append({'text': preview, 'type': 'chunk'})
                seen_titles.add(preview)
    
    return Response({'success': True, 'suggestions': suggestions})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_rag_collection_stats(request, collection_id):
    """Возвращает статистику коллекции."""
    try:
        collection = DocumentCollection.objects.get(
            Q(id=collection_id) & (Q(owner_id=str(request.user.id)) | Q(is_public=True)),
            is_active=True
        )
    except DocumentCollection.DoesNotExist:
        return Response({'success': False, 'error': 'Collection not found'}, status=404)
    
    total_docs = collection.documents.filter(is_active=True).count()
    total_chunks = DocumentChunk.objects.filter(document__collection=collection, is_active=True).count()
    chunks_with_emb = DocumentChunk.objects.filter(
        document__collection=collection, has_embedding=True, is_active=True
    ).count()
    
    stats = {
        'documents': {'total': total_docs},
        'chunks': {'total': total_chunks, 'with_embeddings': chunks_with_emb},
        'index': {
            'vector_index_created': collection.vector_index_created,
            'total_vectors': collection.total_vectors,
            'dimension': collection.vector_dimension,
        }
    }
    return Response({'success': True, 'stats': stats})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_rag_search_history(request):
    """Возвращает историю поисков пользователя."""
    collection_id = request.GET.get('collection_id')
    limit = int(request.GET.get('limit', 10))
    
    queries = SearchQuery.objects.filter(user_id=str(request.user.id))
    if collection_id:
        queries = queries.filter(collection_filter_id=collection_id)
    
    queries = queries.order_by('-created_at')[:limit]
    
    history = []
    for q in queries:
        history.append({
            'id': q.id,
            'query_text': q.query_text,
            'created_at': q.created_at.isoformat(),
            'results_count': q.results.count(),
            'collection_name': q.collection_filter.name if q.collection_filter else None,
        })
    
    return Response({'success': True, 'history': history})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_rag_export_search_results(request):
    """Экспорт результатов поиска в JSON или CSV."""
    data = request.data
    results = data.get('results', [])
    format = data.get('format', 'json')
    collection_name = data.get('collection_name', '')
    
    if not results:
        return Response({'success': False, 'error': 'No results to export'}, status=400)
    
    if format == 'json':
        response = HttpResponse(
            json.dumps(results, ensure_ascii=False, indent=2),
            content_type='application/json; charset=utf-8'
        )
        response['Content-Disposition'] = f'attachment; filename="search_results_{collection_name}.json"'
        return response
    
    elif format == 'csv':
        response = HttpResponse(content_type='text/csv; charset=utf-8')
        response['Content-Disposition'] = f'attachment; filename="search_results_{collection_name}.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Rank', 'Document Title', 'Chunk ID', 'Content Preview', 'Relevance Score', 'Semantic Score', 'Keyword Score', 'Freshness Score'])
        
        for i, r in enumerate(results, 1):
            writer.writerow([
                i,
                r.get('document_title', ''),
                r.get('chunk_id', ''),
                (r.get('content', '')[:200] + '...') if len(r.get('content', '')) > 200 else r.get('content', ''),
                r.get('relevance_score', 0),
                r.get('scores', {}).get('semantic', 0),
                r.get('scores', {}).get('keyword', 0),
                r.get('scores', {}).get('freshness', 0)
            ])
        return response
    
    return Response({'success': False, 'error': 'Unsupported format'}, status=400)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_document_preview(request, document_id):
    """Возвращает содержимое документа или конкретного чанка для предпросмотра."""
    try:
        doc = Document.objects.get(id=document_id, is_active=True)
        if doc.collection:
            if doc.collection.owner_id != str(request.user.id) and not doc.collection.is_public:
                return Response({'success': False, 'error': 'Access denied'}, status=403)
    except Document.DoesNotExist:
        return Response({'success': False, 'error': 'Document not found'}, status=404)
    
    chunk_id = request.GET.get('chunk_id')
    if chunk_id:
        try:
            chunk = DocumentChunk.objects.get(id=chunk_id, document=doc, is_active=True)
            content = chunk.content
        except DocumentChunk.DoesNotExist:
            return Response({'success': False, 'error': 'Chunk not found'}, status=404)
    else:
        content = (doc.processed_content or doc.raw_content or '')[:5000]
        if len(content) >= 5000:
            content += '... (обрезано)'
    
    return Response({
        'success': True,
        'content': content,
        'document_id': doc.id,
        'title': doc.title,
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_document_chunk_detail(request, chunk_id):
    """
    Возвращает детальную информацию о чанке документа.
    """
    try:
        chunk = DocumentChunk.objects.select_related('document').get(id=chunk_id, is_active=True)
        # Проверка доступа
        if chunk.document.collection:
            if (chunk.document.collection.owner_id != str(request.user.id) and 
                not chunk.document.collection.is_public):
                return Response({'error': 'Access denied'}, status=403)
        serializer = DocumentChunkSerializer(chunk)
        return Response(serializer.data)
    except DocumentChunk.DoesNotExist:
        return Response({'error': 'Chunk not found'}, status=404)


# ============================================================================
# RAG API
# ============================================================================
class DocumentCollectionViewSet(viewsets.ModelViewSet):
    """API для коллекций документов."""
    queryset = DocumentCollection.objects.filter(is_active=True)
    serializer_class = DocumentCollectionSerializer
    permission_classes = [IsAuthenticated, IsCollectionOwnerOrPublic]

    def get_queryset(self):
        user = self.request.user
        return self.queryset.filter(
            Q(owner_id=str(user.id)) | Q(is_public=True)
        )

    def perform_create(self, serializer):
        serializer.save(owner_id=str(self.request.user.id))

    @action(detail=True, methods=['get'])
    def documents(self, request, pk=None):
        collection = self.get_object()
        docs = collection.documents.filter(is_active=True)
        serializer = DocumentSerializer(docs, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def statistics(self, request, pk=None):
        collection = self.get_object()
        stats = {
            'total_documents': collection.total_documents,
            'total_chunks': collection.total_chunks,
            'total_size_mb': collection.total_size_mb,
            'vector_index_created': collection.vector_index_created,
            'total_vectors': collection.total_vectors,
            'vector_dimension': collection.vector_dimension,
            'embedding_progress': collection.get_embedding_progress(),
        }
        return Response(stats)

    @action(detail=True, methods=['post'])
    def build_index(self, request, pk=None):
        collection = self.get_object()
        if not collection.owner_id == str(request.user.id):
            return Response({'error': 'Permission denied'}, status=403)
        index_type = request.data.get('index_type', 'flat')
        task = build_faiss_index_task.delay(collection.id, index_type)
        return Response({'success': True, 'task_id': task.id})

    @action(detail=True, methods=['post'])
    def create_embeddings(self, request, pk=None):
        collection = self.get_object()
        if not collection.owner_id == str(request.user.id):
            return Response({'error': 'Permission denied'}, status=403)
        task = create_embeddings_for_collection_task.delay(collection.id)
        return Response({'success': True, 'task_id': task.id})

    @action(detail=True, methods=['get'])
    def documents(self, request, pk=None):
        collection = self.get_object()
        docs = collection.documents.filter(is_active=True)
        serializer = DocumentSerializer(docs, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def chunks(self, request, pk=None):
        collection = self.get_object()
        chunks = DocumentChunk.objects.filter(
            document__collection=collection,
            is_active=True
        ).select_related('document').order_by('document_id', 'chunk_index')
        results = []
        for chunk in chunks:
            results.append({
                'id': chunk.id,
                'document_id': chunk.document.id,
                'document_title': chunk.document.title,
                'content': chunk.content,
                'content_preview': chunk.content[:200] + ('...' if len(chunk.content) > 200 else ''),
                'chunk_index': chunk.chunk_index,
                'token_count': chunk.token_count,
                'has_embedding': chunk.has_embedding,
                'created_at': chunk.created_at.isoformat() if chunk.created_at else None,
            })
        return Response({'chunks': results})


class DocumentViewSet(viewsets.ModelViewSet):
    """API для документов."""
    queryset = Document.objects.filter(is_active=True)
    serializer_class = DocumentSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        # Доступные коллекции
        coll_ids = DocumentCollection.objects.filter(
            Q(owner_id=str(user.id)) | Q(is_public=True), is_active=True
        ).values_list('id', flat=True)
        return self.queryset.filter(collection_id__in=coll_ids)

    @action(detail=True, methods=['get'])
    def chunks(self, request, pk=None):
        doc = self.get_object()
        chunks = doc.chunks.filter(is_active=True)
        serializer = DocumentChunkSerializer(chunks, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def process(self, request, pk=None):
        doc = self.get_object()
        # Audit fix: unlike the UI views, the API path never reset the status
        # before dispatch — a document stuck in PROCESSING (lost worker) could
        # never be reprocessed via the API
        if doc.status == 'PROCESSING':
            doc.status = 'UPLOADED'
            doc.processing_progress = 0
            doc.save(update_fields=['status', 'processing_progress'])
        task = process_document_task.delay(doc.id)
        return Response({'success': True, 'task_id': task.id})

    @action(detail=True, methods=['post'])
    def reindex(self, request, pk=None):
        doc = self.get_object()
        doc.chunks.update(is_active=False)
        doc.status = 'UPLOADED'
        doc.save()
        task = process_document_task.delay(doc.id)
        return Response({'success': True, 'task_id': task.id})


class RAGSessionViewSet(viewsets.ModelViewSet):
    """API для RAG-сессий."""
    queryset = RAGSession.objects.all()
    serializer_class = RAGSessionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return self.queryset.filter(user_id=str(self.request.user.id))

    @action(detail=False, methods=['post'])
    def chat(self, request):
        """RAG-чат: поиск + генерация."""
        rag = RAGService()
        result = rag.process_query(
            query=request.data.get('query'),
            user_id=str(request.user.id),
            collection_id=request.data.get('collection_id'),
            system_prompt=request.data.get('system_prompt', ''),
            llm_model_id=request.data.get('llm_model_id'),
            embedding_model_id=request.data.get('embedding_model_id'),
            search_config=request.data.get('search_config', {}),
            generation_config=request.data.get('generation_config', {})
        )
        return Response(result)

    @action(detail=True, methods=['post'])
    def feedback(self, request, pk=None):
        session = self.get_object()
        rating = request.data.get('rating')
        comment = request.data.get('comment', '')
        if not rating or not 1 <= int(rating) <= 5:
            return Response({'error': 'Rating must be 1-5'}, status=400)
        session.add_feedback(int(rating), comment)
        SearchQualityFeedback.objects.create(
            rag_session=session,
            user_id=str(request.user.id),
            rating=int(rating),
            comment=comment
        )
        return Response({'status': 'Feedback saved'})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_rag_statistics(request):
    """
    Возвращает статистику RAG для текущего пользователя: количество коллекций и документов.
    """
    from ..models import DocumentCollection, Document

    # Коллекции, доступные пользователю (свои + публичные)
    collections = DocumentCollection.objects.filter(
        Q(owner_id=str(request.user.id)) | Q(is_public=True),
        is_active=True
    )
    total_collections = collections.count()

    # Документы в этих коллекциях
    documents = Document.objects.filter(
        collection__in=collections,
        is_active=True
    )
    total_documents = documents.count()

    return Response({
        'total_collections': total_collections,
        'total_documents': total_documents,
    })


# ============================================================================
# Embedding
# ============================================================================
class EmbeddingModelViewSet(viewsets.ModelViewSet):
    queryset = EmbeddingModel.objects.all()
    serializer_class = EmbeddingModelSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=True, methods=['post'])
    def set_default(self, request, pk=None):
        model = self.get_object()
        # Снимаем флаг is_default со всех других моделей
        EmbeddingModel.objects.filter(is_default=True).update(is_default=False)
        model.is_default = True
        model.save()
        return Response({'status': 'default set'})

    @action(detail=False, methods=['get'])
    def default(self, request):
        model = EmbeddingModel.objects.filter(is_default=True).first()
        if model:
            serializer = self.get_serializer(model)
            return Response(serializer.data)
        return Response({'detail': 'No default model'}, status=404)

    @action(detail=True, methods=['post'])
    def load(self, request, pk=None):
        # Загрузить модель в память (если нужно)
        # Может быть заглушкой, так как модели эмбеддингов обычно лёгкие
        model = self.get_object()
        from ..services.embedding_service import embedding_service
        try:
            embedding_service.get_model(model.model_id)  # загрузит в кэш
            return Response({'status': 'loaded'})
        except Exception as e:
            return Response({'error': str(e)}, status=500)

    @action(detail=True, methods=['post'])
    def unload(self, request, pk=None):
        # Выгрузить из памяти
        model = self.get_object()
        from ..services.embedding_service import embedding_service
        embedding_service._model_cache.pop(model.model_id, None)
        return Response({'status': 'unloaded'})

    @action(detail=False, methods=['get'])
    def search_hf(self, request):
        # Поиск моделей эмбеддингов на Hugging Face
        query = request.GET.get('q', '')
        # Используем HuggingFace API для поиска моделей с тегом sentence-transformers или feature-extraction
        from huggingface_hub import HfApi
        api = HfApi()
        models = api.list_models(
            search=query,
            filter='sentence-transformers,feature-extraction',
            sort='downloads',
            limit=20
        )
        results = []
        for m in models:
            results.append({
                'id': m.id,
                'name': m.id.split('/')[-1],
                'author': m.id.split('/')[0],
                'downloads': m.downloads,
                'likes': m.likes,
                'tags': m.tags[:5],
                'pipeline_tag': m.pipeline_tag,
            })
        return Response(results)


# ── v5.2: реестр реранкеров ───────────────────────────────────────────────
class RerankerModelViewSet(viewsets.ModelViewSet):
    """CRUD моделей реранкинга (cross-encoders) для RAG-поиска."""
    queryset = RerankerModel.objects.all()
    serializer_class = RerankerModelSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        instance = serializer.save()
        if instance.is_default:
            RerankerModel.objects.exclude(pk=instance.pk).update(is_default=False)
