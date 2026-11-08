import os
import logging
from datetime import datetime, timedelta

import torch
import time

from django.db.models import Q, Count, Sum
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.http import JsonResponse, HttpResponse
import json
import csv
import re

from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth.mixins import LoginRequiredMixin

from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, BasePermission
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import (
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
from .serializers import (
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
from .services.redis_model_service import redis_model_manager
from .services.chat_service import chat_service
from .services.rag_service import RAGService, EnhancedSearchService
from .tasks import (
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

# ============================================================================
# ПЕРМИШЕНЫ
# ============================================================================
class IsCollectionOwnerOrPublic(BasePermission):
    """Доступ к коллекции: владелец или публичная."""
    def has_object_permission(self, request, view, obj):
        if obj.owner_id == str(request.user.id):
            return True
        if obj.is_public:
            return True
        return request.user.is_staff

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def task_result(request, task_id):
    """Возвращает статус задачи Celery с безопасной сериализацией."""
    from celery.result import AsyncResult
    import traceback
    
    try:
        task = AsyncResult(task_id)
        
        # Базовый ответ
        response = {
            'task_id': task_id,
            'state': task.state,
            'status': task.status,
        }
        
        if task.ready():
            try:
                # Пытаемся получить результат
                result = task.result
                
                # Если результат - исключение, преобразуем в словарь
                if isinstance(result, Exception):
                    response['result'] = {
                        'success': False,
                        'error': str(result),
                        'error_type': result.__class__.__name__
                    }
                elif result is not None:
                    # Проверяем, что результат можно сериализовать
                    try:
                        json.dumps(result)
                        response['result'] = result
                    except (TypeError, ValueError):
                        # Если не сериализуется, преобразуем в безопасный формат
                        response['result'] = {
                            'success': True if result.get('success') else False,
                            'message': str(result)
                        }
                else:
                    response['result'] = {}
                    
            except Exception as e:
                response['result'] = {
                    'success': False,
                    'error': f'Error getting task result: {str(e)}'
                }
        else:
            # Для задач в процессе
            try:
                if hasattr(task, 'info') and task.info:
                    if isinstance(task.info, dict):
                        # Убеждаемся, что все значения сериализуемы
                        safe_info = {}
                        for k, v in task.info.items():
                            try:
                                json.dumps({k: v})
                                safe_info[k] = v
                            except (TypeError, ValueError):
                                safe_info[k] = str(v)
                        response['info'] = safe_info
                    else:
                        response['info'] = {'message': str(task.info)}
                else:
                    response['info'] = {}
            except Exception as e:
                response['info'] = {'error': str(e)}
        
        return Response(response)
        
    except Exception as e:
        # Возвращаем понятную ошибку
        return Response({
            'task_id': task_id,
            'state': 'FAILURE',
            'error': str(e),
            'traceback': traceback.format_exc() if settings.DEBUG else None
        }, status=500)

# ============================================================================
# API ViewSets
# ============================================================================
class ExperimentViewSet(viewsets.ModelViewSet):
    """API для экспериментов."""
    queryset = Experiment.objects.filter(is_active=True)
    serializer_class = ExperimentSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=True, methods=['post'])
    def deactivate(self, request, pk=None):
        experiment = self.get_object()
        experiment.deactivate()
        return Response({'message': f'Эксперимент "{experiment.name}" деактивирован'})

    @action(detail=True, methods=['post'])
    def create_checkpoint(self, request, pk=None):
        experiment = self.get_object()
        if not experiment.base_checkpoint:
            return Response({'error': 'Нет базового чекпоинта'}, status=400)
        base = experiment.base_checkpoint
        new = ModelCheckpoint.objects.create(
            name=f"{experiment.name}_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            display_name=f"Чекпоинт {experiment.name}",
            model_type=base.model_type,
            huggingface_id=base.huggingface_id,
            huggingface_revision=base.huggingface_revision,
            loader_config=base.loader_config.copy(),
            local_path=base.local_path,
            relative_path=base.relative_path,
            status='DOWNLOADING',
            parameter_count=base.parameter_count,
            quantization=base.quantization,
            context_length=base.context_length,
            vocab_size=base.vocab_size,
            license_type=base.license_type,
            is_safe=base.is_safe,
            safety_score=base.safety_score,
            experiment=experiment,
            parent=base,
            is_active=True
        )
        return Response(ModelCheckpointSerializer(new).data, status=201)


class ModelCheckpointViewSet(viewsets.ModelViewSet):
    """API для моделей."""
    queryset = ModelCheckpoint.objects.all()
    serializer_class = ModelCheckpointSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=True, methods=['post'])
    def download(self, request, pk=None):
        """Запуск загрузки модели (API)."""
        model = self.get_object()
        if not model.huggingface_id:
            return Response({'error': 'No Hugging Face ID'}, status=400)
        task = download_model_task.delay(model.id, user_id=request.user.id)
        return Response({'message': 'Download started', 'task_id': task.id})

    @action(detail=True, methods=['post'])
    def load(self, request, pk=None):
        """Загрузка модели в память с выбором квантизации."""
        model = self.get_object()
        serializer = ModelLoadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        if not model.local_path or not os.path.exists(model.local_path):
            return Response({'error': 'Model not downloaded locally'}, status=400)

        # Обновляем квантование, если указано и не равно 'none'
        quantization = serializer.validated_data.get('quantization')
        if quantization and quantization != 'none':
            model.quantization = quantization
            model.save()

        # Запускаем задачу с переданной квантизацией
        task = load_model_to_memory_task.delay(
            model.id,
            quantization=quantization if quantization != 'none' else None
        )
        return Response({'message': 'Loading started', 'task_id': task.id})

    @action(detail=True, methods=['post'])
    def unload(self, request, pk=None):
        """Выгрузка модели из памяти с опцией force."""
        model = self.get_object()
        model_id = model.huggingface_id or model.name
        
        force = request.data.get('force', False)
        
        # Запускаем задачу выгрузки
        task = unload_model_from_memory_task.delay(model.id, force=force)
        
        return Response({
            'message': 'Unload task started',
            'task_id': task.id,
            'model_id': model_id,
            'force': force
        })

    @action(detail=True, methods=['post'])
    def generate(self, request, pk=None):
        """Генерация текста."""
        model = self.get_object()
        serializer = TextGenerationSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        model_id = model.huggingface_id or model.name
        info = redis_model_manager.get_model_info(model_id)
        if not info or not info.get('loaded'):
            return Response({'error': 'Model not loaded in memory'}, status=400)

        result = redis_model_manager.generate_text(
            model_id=model_id,
            prompt=serializer.validated_data['prompt'],
            max_tokens=serializer.validated_data.get('max_tokens', 512),
            temperature=serializer.validated_data.get('temperature', 0.7),
            top_p=serializer.validated_data.get('top_p', 0.9),
            top_k=serializer.validated_data.get('top_k', 50),
            repetition_penalty=serializer.validated_data.get('repetition_penalty', 1.0),
            system_prompt=serializer.validated_data.get('system_prompt', ''),
        )

        if result.get('success'):
            # Создаём сессию инференса для статистики
            session = InferenceSession.objects.create(
                checkpoint=model,
                session_id=f"api_{request.user.id}_{model.id}_{int(timezone.now().timestamp())}",
                user_ip=request.META.get('REMOTE_ADDR', ''),
                user_agent=request.META.get('HTTP_USER_AGENT', ''),
                api_key=request.user.username,
            )
            session.add_request(
                input_tokens=result.get('input_tokens', 0),
                output_tokens=result.get('output_tokens', 0),
                duration=result.get('generation_time', 0),
            )
            return Response({
                'success': True,
                'response': result['response'],
                'tokens': {
                    'input': result.get('input_tokens', 0),
                    'output': result.get('output_tokens', 0),
                    'total': result.get('total_tokens', 0),
                },
                'generation_time': result.get('generation_time', 0),
                'tokens_per_second': result.get('tokens_per_second', 0),
            })
        return Response({'success': False, 'error': result.get('error')}, status=500)

    @action(detail=False, methods=['get'])
    def memory_status(self, request):
        """Статус памяти GPU/CPU."""
        memory = redis_model_manager.get_available_memory()
        loaded = redis_model_manager.get_all_model_infos()
        return Response({
            'device': memory.get('device', 'unknown'),
            'memory': memory,
            'loaded_models': loaded,
            'total_loaded': len(loaded),
        })

    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        model = self.get_object()
        model_id = model.huggingface_id or model.name
        info = redis_model_manager.get_model_info(model_id)
        loading_status = redis_model_manager.get_loading_status(model_id)
        is_loaded = bool(info and info.get('loaded'))

        actual_quantization = None
        device = None
        if is_loaded:
            actual_quantization = info.get('quantization')
            device = info.get('device')
        if not actual_quantization and model.loader_config:
            actual_quantization = model.loader_config.get('actual_quantization')
        if not device and model.loader_config:
            device = model.loader_config.get('target_device')

        # Синхронизация статуса
        if is_loaded and model.status != 'LOADED':
            model.status = 'LOADED'
            model.save(update_fields=['status'])

        memory_info = redis_model_manager.get_available_memory()

        download_jobs = ModelDownloadJobSerializer(model.download_jobs.order_by('-created_at'), many=True).data

        return Response({
            'id': model.id,
            'name': model.display_name,
            'status': model.status,
            'is_loaded_in_memory': is_loaded,
            'loaded_quantization': actual_quantization,
            'quantization': model.quantization,
            'device': device,
            'memory_status': memory_info,
            'redis_info': info,
            'loading_status': loading_status,
            'download_jobs': download_jobs,
            'local_path': model.local_path,
            'has_local_copy': bool(model.local_path and os.path.exists(model.local_path)),
            'huggingface_id': model.huggingface_id,
            'created_at': model.created_at,
            'updated_at': model.updated_at,
            'progress_info': loading_status if loading_status.get('status') in ['LOADING', 'DOWNLOADING'] else None,
        })


class ModelMergeJobViewSet(viewsets.ModelViewSet):
    """API для слияния моделей."""
    queryset = ModelMergeJob.objects.all()
    serializer_class = ModelMergeJobSerializer
    permission_classes = [IsAuthenticated]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        merge_job = serializer.save()
        task = merge_models_task.delay(merge_job.id)
        return Response({
            'message': 'Merge job created',
            'task_id': task.id,
            'merge_job_id': merge_job.id,
        }, status=201)

    @action(detail=True, methods=['post'])
    def start(self, request, pk=None):
        merge_job = self.get_object()
        if merge_job.status not in ['PENDING', 'FAILED']:
            return Response({'error': 'Job already running or completed'}, status=400)
        task = merge_models_task.delay(merge_job.id)
        merge_job.status = 'VALIDATING'
        merge_job.save()
        return Response({'message': 'Merge started', 'task_id': task.id})

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        merge_job = self.get_object()
        if merge_job.status in ['PENDING', 'VALIDATING', 'RUNNING']:
            merge_job.status = 'CANCELLED'
            merge_job.save()
            return Response({'message': 'Merge cancelled'})
        return Response({'error': 'Cannot cancel'}, status=400)

    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        job = self.get_object()
        return Response({
            'id': job.id,
            'status': job.status,
            'progress': job.progress,
            'current_step': job.current_step,
            'logs': job.logs,
            'error_details': job.error_details,
            'output_model_id': job.output_model.id if job.output_model else None,
        })


class InferenceSessionViewSet(viewsets.ReadOnlyModelViewSet):
    """API для сессий инференса (только чтение)."""
    queryset = InferenceSession.objects.all()
    serializer_class = InferenceSessionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        qs = super().get_queryset()
        if self.request.query_params.get('checkpoint_id'):
            qs = qs.filter(checkpoint_id=self.request.query_params['checkpoint_id'])
        if self.request.query_params.get('active_only') == 'true':
            qs = qs.filter(is_active=True)
        return qs.order_by('-started_at')


class ModelEvaluationViewSet(viewsets.ModelViewSet):
    """API для оценок моделей."""
    queryset = ModelEvaluation.objects.all()
    serializer_class = ModelEvaluationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        qs = super().get_queryset()
        if self.request.query_params.get('checkpoint_id'):
            qs = qs.filter(checkpoint_id=self.request.query_params['checkpoint_id'])
        if self.request.query_params.get('dataset_name'):
            qs = qs.filter(dataset_name=self.request.query_params['dataset_name'])
        return qs.order_by('-evaluated_at')


class ModelDownloadJobViewSet(viewsets.ReadOnlyModelViewSet):
    """API для задач загрузки."""
    queryset = ModelDownloadJob.objects.all()
    serializer_class = ModelDownloadJobSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['get'])
    def active(self, request):
        jobs = ModelDownloadJob.objects.filter(
            status__in=['PENDING', 'DOWNLOADING']
        ).order_by('-created_at')
        serializer = self.get_serializer(jobs, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        job = self.get_object()
        if job.status in ['PENDING', 'DOWNLOADING']:
            job.status = 'CANCELLED'
            job.completed_at = timezone.now()
            job.save()
            if job.checkpoint:
                job.checkpoint.status = 'PENDING'
                job.checkpoint.save()
            return Response({'message': 'Download cancelled'})
        return Response({'error': 'Cannot cancel'}, status=400)

# ============================================================================
# ПОИСК
# ============================================================================
@api_view(['POST'])
@login_required
def rag_search(request):
    import time
    from .models import DocumentCollection
    from .services.rag_service import EnhancedSearchService

    data = request.data
    query = data.get('query', '').strip()
    collection_id = data.get('collection_id')
    search_type = data.get('search_type', 'hybrid')
    top_k = data.get('top_k', 10)
    threshold = data.get('threshold', 0.5)
    semantic_weight = data.get('semantic_weight', 0.7)
    keyword_weight = data.get('keyword_weight', 0.2)
    freshness_weight = data.get('freshness_weight', 0.1)
    use_reranking = data.get('use_reranking', True)

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

# ============================================================================
# НОВЫЙ ЭНДПОИНТ ДЛЯ ПОЛУЧЕНИЯ ДАННЫХ ЧАНКА
# ============================================================================

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
# ЧАТ API
# ============================================================================

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_chat_session_preview(request, session_id):
    """
    Возвращает последние сообщения сессии для предпросмотра.
    """
    from .models import ChatSession
    from .services.chat_service import chat_service

    try:
        session = ChatSession.objects.get(session_id=session_id, user_id=str(request.user.id))
    except ChatSession.DoesNotExist:
        return Response({'success': False, 'error': 'Session not found'}, status=404)

    # Пытаемся получить из Redis, иначе из БД
    redis_session = chat_service.get_session(session_id)
    if redis_session and redis_session.get('conversation_history'):
        messages = redis_session['conversation_history'][-5:]  # последние 5
    else:
        messages = session.conversation_history[-5:] if session.conversation_history else []

    # Убираем лишние поля, если нужно
    simplified = [{'role': m['role'], 'content': m['content'][:200]} for m in messages]

    return Response({'success': True, 'messages': simplified})

class ChatSessionViewSet(viewsets.ModelViewSet):
    """API для чат-сессий."""
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'session_id'  # теперь URL используют session_id вместо pk

    def get_queryset(self):
        return self.queryset.filter(user_id=str(self.request.user.id), is_active=True)

    def create(self, request, *args, **kwargs):
        """Создание сессии с моделью."""
        model_id = request.data.get('checkpoint_id') or request.data.get('model_id')
        if not model_id:
            return Response({'error': 'model_id required'}, status=400)
        try:
            result = chat_service.create_session(str(request.user.id), model_id)
            session = ChatSession.objects.get(session_id=result['session_id'])
            serializer = self.get_serializer(session)
            return Response(serializer.data, status=201)
        except Exception as e:
            logger.error(f"Chat session create error: {e}")
            return Response({'error': str(e)}, status=500)

    @action(detail=True, methods=['post'])
    def send_message(self, request, session_id=None):
        session = self.get_object()
        serializer = TextGenerationSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        result = chat_service.generate_response(
            session_id=session.session_id,
            message=serializer.validated_data['prompt'],
            generation_config_overrides={
                'max_tokens': serializer.validated_data.get('max_tokens', 512),
                'temperature': serializer.validated_data.get('temperature', 0.7),
                'top_p': serializer.validated_data.get('top_p', 0.9),
                'top_k': serializer.validated_data.get('top_k', 50),
                'repetition_penalty': serializer.validated_data.get('repetition_penalty', 1.0),
            }
        )
        if result.get('async'):
            return Response({'success': True, 'task_id': result['task_id'], 'async': True})
        return Response(result)

    @action(detail=True, methods=['post'])
    def clear(self, request, session_id=None):
        """Очистить историю."""
        session = self.get_object()
        success = chat_service.clear_history(session.session_id)
        return Response({'success': success})

    @action(detail=True, methods=['post'])
    def change_model(self, request, session_id=None):
        """Сменить модель."""
        session = self.get_object()
        model_id = request.data.get('model_id')
        if not model_id:
            return Response({'error': 'model_id required'}, status=400)
        try:
            new_model = ModelCheckpoint.objects.get(id=model_id, is_active=True)
            session.checkpoint = new_model
            session.save()
            # Обновить в Redis
            redis_data = chat_service.get_session(session.session_id)
            if redis_data:
                redis_data['model_id'] = model_id
                chat_service.redis_manager.redis.setex(
                    f"chat:session:{session.session_id}",
                    86400,
                    json.dumps(redis_data, default=str)
                )
            return Response({'success': True, 'model_name': new_model.display_name})
        except Exception as e:
            return Response({'error': str(e)}, status=400)

    @action(detail=False, methods=['get'])
    def my_sessions(self, request):
        """Список сессий пользователя."""
        sessions = chat_service.list_user_sessions(str(request.user.id))
        return Response({'success': True, 'sessions': sessions})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def force_local_load_model(request, model_id):
    """Принудительная локальная загрузка модели с выбором квантизации."""
    try:
        from .models import ModelCheckpoint
        from .tasks import load_model_to_memory_task
        from .services.redis_model_service import redis_model_manager

        model = get_object_or_404(ModelCheckpoint, id=model_id, is_active=True)

        # Проверяем, не загружена ли уже модель
        model_id_str = model.huggingface_id or model.name
        if redis_model_manager.get_model_info(model_id_str):
            return Response({
                'success': False,
                'error': 'Модель уже загружена в память',
                'model_id': model_id,
            }, status=400)

        if not model.local_path or not os.path.exists(model.local_path):
            return Response({
                'success': False,
                'error': 'Модель не скачана локально'
            }, status=400)

        quantization = request.data.get('quantization')
        # Если квантизация не указана, берём из модели (или None)
        if quantization is None:
            quantization = model.quantization or None

        # Очищаем возможные старые блокировки
        redis_model_manager.clean_stale_locks()

        task = load_model_to_memory_task.delay(model.id, quantization=quantization)
        return Response({
            'success': True,
            'message': f'Задача загрузки модели {model.display_name} запущена',
            'task_id': task.id,
            'model_id': model_id,
            'quantization': quantization,
        })
    except Exception as e:
        logger.error(f"❌ Error force loading model: {str(e)}", exc_info=True)
        return Response({'success': False, 'error': str(e)}, status=500)
        
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
def api_user_rag_collections(request):
    """Возвращает список доступных пользователю RAG коллекций."""
    from .models import DocumentCollection
    from django.db.models import Q, Count
    
    collections = DocumentCollection.objects.filter(
        Q(owner_id=str(request.user.id)) | Q(is_public=True),
        is_active=True
    ).annotate(
        doc_count=Count('documents', filter=Q(documents__is_active=True)),
        chunk_count=Count('documents__chunks', filter=Q(documents__chunks__is_active=True)),
        has_embeddings=Count('documents__chunks', filter=Q(documents__chunks__has_embedding=True))
    ).order_by('name')
    
    result = []
    for c in collections:
        result.append({
            'id': c.id,
            'name': c.name,
            'description': c.description,
            'doc_count': c.doc_count,
            'chunk_count': c.chunk_count,
            'has_embeddings': c.has_embeddings > 0,
            'vector_index_created': c.vector_index_created,
            'is_public': c.is_public,
            'is_owner': c.owner_id == str(request.user.id),
            'embedding_progress': round(
                (c.has_embeddings / c.chunk_count * 100) if c.chunk_count > 0 else 0, 1
            ),
        })
    
    return Response({'success': True, 'collections': result})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_task_status(request, task_id):
    """Возвращает статус задачи Celery."""
    from celery.result import AsyncResult
    
    try:
        task = AsyncResult(task_id)
        
        response = {
            'task_id': task_id,
            'state': task.state,
            'status': task.status,
        }
        
        if task.ready():
            response['result'] = task.result
        else:
            # Для задач в процессе, возвращаем meta информацию
            if hasattr(task, 'info') and task.info:
                response['info'] = task.info
            else:
                response['info'] = {}
        
        return Response(response)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_chat_update_rag_config(request, session_id):
    """Обновление RAG конфигурации для сессии."""
    from .services.chat_service import chat_service
    
    try:
        rag_config = request.data
        if not isinstance(rag_config, dict):
            return Response({'error': 'Invalid config format'}, status=400)
        
        success = chat_service.update_rag_config(session_id, rag_config)
        if success:
            return Response({'success': True})
        else:
            return Response({'error': 'Session not found'}, status=404)
    except Exception as e:
        logger.error(f"Error updating RAG config: {e}", exc_info=True)
        return Response({'error': str(e)}, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_chat_send_message(request, session_id):
    """Отправка сообщения в чат (асинхронная)."""
    from .services.chat_service import chat_service
    from .serializers import TextGenerationSerializer

    try:
        # Проверяем существование сессии
        session = chat_service.get_session(session_id)
        if not session:
            return Response({'error': 'Session not found'}, status=404)

        serializer = TextGenerationSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        # Извлекаем RAG-параметры
        use_rag = request.data.get('use_rag', False)
        rag_config = request.data.get('rag_config', None)

        result = chat_service.generate_response(
            session_id=session_id,
            message=serializer.validated_data['prompt'],
            generation_config_overrides={
                'max_tokens': serializer.validated_data.get('max_tokens', 512),
                'temperature': serializer.validated_data.get('temperature', 0.7),
                'top_p': serializer.validated_data.get('top_p', 0.9),
                'top_k': serializer.validated_data.get('top_k', 50),
                'repetition_penalty': serializer.validated_data.get('repetition_penalty', 1.0),
            },
            use_rag=use_rag,
            rag_config=rag_config
        )

        return Response(result)
    except Exception as e:
        logger.error(f"Error sending message: {e}", exc_info=True)
        return Response({'error': str(e)}, status=500)

# ============================================================================
# ОЦЕНКА (EVALUATION)
# ============================================================================
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def evaluation_datasets_api(request):
    datasets = EvaluationDataset.objects.all().order_by('-created_at')
    serializer = EvaluationDatasetSerializer(datasets, many=True)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_dataset_api(request):
    """Загрузка датасета из файла."""
    from .serializers import QuestionUploadSerializer
    serializer = QuestionUploadSerializer(data=request.data)
    if serializer.is_valid():
        try:
            uploaded = request.FILES['file']
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'evaluation_datasets')
            os.makedirs(upload_dir, exist_ok=True)
            file_name = f"dataset_{int(timezone.now().timestamp())}_{uploaded.name}"
            file_path = os.path.join(upload_dir, file_name)
            with open(file_path, 'wb+') as f:
                for chunk in uploaded.chunks():
                    f.write(chunk)

            dataset = EvaluationDataset.objects.create(
                name=serializer.validated_data['name'],
                description=serializer.validated_data.get('description', ''),
                file_path=file_path,
                tags=serializer.validated_data.get('tags', []),
                source_type='FILE',
                created_by=request.user.username
            )
            task = process_uploaded_dataset.delay(
                file_path,
                dataset.id,
                serializer.validated_data['file_format']
            )
            return Response({
                'success': True,
                'dataset_id': dataset.id,
                'task_id': task.id,
            })
        except Exception as e:
            return Response({'success': False, 'error': str(e)}, status=500)
    return Response(serializer.errors, status=400)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_evaluation_api(request):
    """Запуск задачи оценки."""
    from .serializers import StartEvaluationSerializer
    serializer = StartEvaluationSerializer(data=request.data)
    if serializer.is_valid():
        try:
            task = ModelEvaluationTask.objects.create(
                name=serializer.validated_data['name'],
                description=serializer.validated_data.get('description', ''),
                dataset_id=serializer.validated_data['dataset_id'],
                generation_config=serializer.validated_data.get('generation_config', {}),
                created_by=request.user.username
            )
            task.evaluation_models.set(serializer.validated_data['model_ids'])
            task.save()
            # RAG параметры
            use_rag = request.data.get('use_rag', False)
            collection_ids = request.data.get('collection_ids', [])
            celery_task = run_evaluation_task.delay(
                task.id,
                use_rag=use_rag,
                collection_ids=collection_ids
            )
            task.celery_task_id = celery_task.id
            task.save()
            return Response({
                'success': True,
                'task_id': task.id,
                'celery_task_id': celery_task.id,
            })
        except Exception as e:
            return Response({'success': False, 'error': str(e)}, status=500)
    return Response(serializer.errors, status=400)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def evaluation_task_status_api(request, task_id):
    try:
        task = ModelEvaluationTask.objects.get(id=task_id)
        serializer = ModelEvaluationTaskSerializer(task)
        data = serializer.data

        # Добавляем информацию о чекпоинте
        if task.metadata and 'checkpoint' in task.metadata:
            data['checkpoint'] = task.metadata['checkpoint']
            data['has_checkpoint'] = True
        else:
            data['has_checkpoint'] = False

        # Если задача выполняется, добавляем прогресс из Celery
        if task.status == 'RUNNING' and task.celery_task_id:
            from celery.result import AsyncResult
            res = AsyncResult(task.celery_task_id)
            if res.info and isinstance(res.info, dict):
                if 'progress' in res.info:
                    data['progress'] = res.info['progress']
                if 'current_step' in res.info:
                    data['current_step'] = res.info['current_step']
        
        return Response(data)
    except ModelEvaluationTask.DoesNotExist:
        return Response({'error': 'Task not found'}, status=404)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def evaluation_results_api(request, task_id):
    try:
        task = ModelEvaluationTask.objects.get(id=task_id)
        results = EvaluationResult.objects.filter(task=task).select_related('model')
        grouped = {}
        for r in results:
            mid = r.model.id
            if mid not in grouped:
                grouped[mid] = {
                    'model': {
                        'id': r.model.id,
                        'name': r.model.display_name,
                    },
                    'results': []
                }
            grouped[mid]['results'].append({
                'question_id': r.question_id,
                'question_text': r.question_text[:200],
                'response': r.model_response[:500],
                'response_time': r.response_time,
                'tokens_used': r.tokens_used,
                'manual_rating': r.manual_rating,
            })
        return Response({
            'success': True,
            'task_id': task_id,
            'task_name': task.name,
            'task_status': task.status,
            'results_by_model': grouped,
            'summary_metrics': task.summary_metrics or {},
        })
    except ModelEvaluationTask.DoesNotExist:
        return Response({'error': 'Task not found'}, status=404)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def rate_response_api(request):
    """Оценка ответа модели."""
    from .serializers import RatingSerializer
    serializer = RatingSerializer(data=request.data)
    if serializer.is_valid():
        try:
            task = ModelEvaluationTask.objects.get(id=serializer.validated_data['task_id'])
            task.add_rating(
                model_id=serializer.validated_data['model_id'],
                question_id=serializer.validated_data['question_id'],
                rating=serializer.validated_data['rating'],
                user_id=request.user.username
            )
            # Обновить детальный результат
            EvaluationResult.objects.filter(
                task=task,
                model_id=serializer.validated_data['model_id'],
                question_id=serializer.validated_data['question_id']
            ).update(
                manual_rating=serializer.validated_data['rating'],
                rating_comment=serializer.validated_data.get('comment', ''),
                rated_by=request.user.username,
                rated_at=timezone.now()
            )
            return Response({'success': True, 'summary_metrics': task.summary_metrics})
        except Exception as e:
            return Response({'success': False, 'error': str(e)}, status=500)
    return Response(serializer.errors, status=400)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def compare_models_api(request, task_id):
    """Сравнение моделей по вопросам."""
    try:
        task = ModelEvaluationTask.objects.get(id=task_id)
        if task.status != 'COMPLETED':
            return Response({'error': 'Task not completed'}, status=400)

        questions = task.dataset.questions if task.dataset else []
        models = list(task.evaluation_models.all())
        results = task.results or {}

        comparison = []
        for q in questions[:10]:  # первые 10 вопросов
            qid = str(q.get('id', ''))
            qtext = q.get('text', '')
            if not qid or not qtext:
                continue
            resp_list = []
            for m in models:
                mid = str(m.id)
                if mid in results and qid in results[mid]:
                    resp = results[mid][qid]
                    resp_list.append({
                        'model_id': m.id,
                        'model_name': m.display_name,
                        'response': resp.get('response', ''),
                        'tokens_used': resp.get('tokens_used', 0),
                        'response_time': resp.get('response_time', 0),
                    })
            if resp_list:
                comparison.append({
                    'question_id': qid,
                    'question_text': qtext[:200],
                    'model_responses': resp_list
                })
        return Response({'success': True, 'comparison': comparison})
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def compare_tasks_api(request, dataset_id):
    """
    Возвращает данные для сравнения нескольких задач по одному датасету.
    Ожидает JSON: {"task_ids": [1,2,3]}
    """
    task_ids = request.data.get('task_ids', [])
    if not task_ids:
        return Response({'success': False, 'error': 'No task ids provided'}, status=400)

    tasks = ModelEvaluationTask.objects.filter(
        id__in=task_ids,
        dataset_id=dataset_id,
        status='COMPLETED'
    ).prefetch_related('evaluation_models')

    if not tasks.exists():
        return Response({'success': False, 'error': 'No completed tasks found'}, status=404)

    dataset = tasks.first().dataset
    questions = dataset.questions if dataset else []

    tasks_data = []
    for task in tasks:
        models = list(task.evaluation_models.all())
        # Получаем детальные результаты
        detailed_results = {}
        for result in EvaluationResult.objects.filter(task=task).select_related('model'):
            qid = str(result.question_id)
            if qid not in detailed_results:
                detailed_results[qid] = {}
            detailed_results[qid][result.model.id] = {
                'response': result.model_response,
                'rating': result.manual_rating,
                'rag': False,  # можно добавить позже
                'response_time': result.response_time,
                'tokens': result.tokens_used,
            }

        # Дополняем из task.results (если чего-то не хватает)
        for model_id_str, model_results in (task.results or {}).items():
            model_id = int(model_id_str)
            for qid, ans in model_results.items():
                if qid not in detailed_results:
                    detailed_results[qid] = {}
                if model_id not in detailed_results[qid]:
                    detailed_results[qid][model_id] = {
                        'response': ans.get('response', ''),
                        'rating': None,
                        'rag': False,
                        'response_time': ans.get('response_time', 0),
                        'tokens': ans.get('tokens_used', 0),
                    }

        models_info = []
        for model in models:
            models_info.append({
                'model_id': model.id,
                'model_name': model.display_name,
                'answers': {
                    str(q['id']): detailed_results.get(str(q['id']), {}).get(model.id, {
                        'response': '',
                        'rating': None,
                        'rag': False,
                        'response_time': 0,
                        'tokens': 0
                    })
                    for q in questions if q.get('id')
                }
            })

        tasks_data.append({
            'task_id': task.id,
            'task_name': task.name,
            'models': models_info
        })

    return Response({
        'success': True,
        'questions': questions,
        'tasks': tasks_data
    })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def evaluation_dataset_questions(request, dataset_id):
    dataset = get_object_or_404(EvaluationDataset, id=dataset_id)
    if not dataset.file_path or not os.path.exists(dataset.file_path):
        return Response({'success': False, 'error': 'Файл датасета не найден'}, status=404)

    page = int(request.GET.get('page', 1))
    per_page = int(request.GET.get('per_page', 1000))
    start = (page - 1) * per_page
    end = start + per_page

    questions = []
    with open(dataset.file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if idx >= end:
                break
            try:
                item = json.loads(line)
                questions.append({
                    'id': idx + 1,
                    'text': item.get('instruction', item.get('text', '')),
                    'answer': item.get('output', item.get('answer', '')),
                    'category': item.get('category', ''),
                    'difficulty': item.get('difficulty', 'medium')
                })
            except json.JSONDecodeError:
                continue

    total = dataset.total_questions
    total_pages = (total + per_page - 1) // per_page
    remaining = total - end

    return Response({
        'success': True,
        'questions': questions,
        'page': page,
        'per_page': per_page,
        'total': total,
        'total_pages': total_pages,
        'remaining': max(remaining, 0)
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def evaluation_dataset_import(request, pk):
    dataset = get_object_or_404(EvaluationDataset, pk=pk)
    uploaded_file = request.FILES.get('file')
    if not uploaded_file:
        return Response({'success': False, 'error': 'Файл не загружен'}, status=400)
    if not uploaded_file.name.endswith('.jsonl'):
        return Response({'success': False, 'error': 'Поддерживаются только файлы .jsonl'}, status=400)

    # Сохраняем файл
    file_path = dataset.file_path
    if not file_path:
        file_name = f"dataset_{dataset.id}.jsonl"
        file_path = os.path.join(settings.MEDIA_ROOT, 'evaluation_datasets', file_name)
        dataset.file_path = file_path

    with open(file_path, 'wb+') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    # Подсчитываем количество записей
    total = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
    dataset.total_questions = total
    dataset.save()

    return Response({'success': True, 'total': total})

# ============================================================================
# HUGGING FACE API (ПОИСК МОДЕЛЕЙ)
# ============================================================================
class HuggingFaceAPI(APIView):
    """API для взаимодействия с Hugging Face Hub: поиск моделей, эмбеддингов, датасетов."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        action = request.GET.get('action', '')
        if action == 'search':
            return self.search_models(request)
        elif action == 'top_models':
            return self.get_top_models(request)
        elif action == 'model_info':
            return self.get_model_info(request)
        elif action == 'validate':
            return self.validate_model(request)
        elif action == 'search_embedding':
            return self.search_embedding_models(request)
        elif action == 'top_embedding_models':
            return self.get_top_embedding_models(request)
        elif action == 'validate_embedding':
            return self.validate_embedding_model(request)
        elif action == 'search_datasets':
            return self.search_datasets(request)
        elif action == 'top_datasets':
            return self.top_datasets(request)
        elif action == 'validate_dataset':
            return self.validate_dataset(request)
        else:
            return Response({'error': 'Неизвестное действие'}, status=400)

    # =========================================================================
    # Методы для обычных (LLM) моделей
    # =========================================================================
    def search_models(self, request):
        from huggingface_hub import HfApi
        query = request.GET.get('q', '')
        model_type = request.GET.get('type', '')
        limit = int(request.GET.get('limit', 10))

        filter_tags = None
        if model_type == 'QWEN':
            filter_tags = 'qwen,text-generation'
        elif model_type == 'LLAMA':
            filter_tags = 'llama,text-generation'
        elif model_type == 'MISTRAL':
            filter_tags = 'mistral,text-generation'
        elif model_type == 'GEMMA':
            filter_tags = 'gemma,text-generation'
        elif model_type == 'DEEPSEEK':
            filter_tags = 'text-generation'
        elif model_type == 'PHI':
            filter_tags = 'text-generation'
        elif model_type == 'YI':
            filter_tags = 'text-generation'

        api = HfApi()
        try:
            models = api.list_models(
                search=query or None,
                filter=filter_tags,
                sort="downloads",
                limit=limit,
                full=True
            )
            results = []
            for m in models:
                results.append({
                    'id': m.id,
                    'name': m.id.split('/')[-1],
                    'author': m.id.split('/')[0],
                    'downloads': m.downloads,
                    'likes': m.likes,
                    'tags': m.tags[:5] if m.tags else [],
                    'pipeline_tag': m.pipeline_tag,
                    'created_at': m.created_at.isoformat() if m.created_at else None,
                })
            return Response({'results': results, 'total': len(results)})
        except Exception as e:
            return Response({'error': str(e), 'results': []}, status=500)

    def get_top_models(self, request):
        from huggingface_hub import HfApi
        model_type = request.GET.get('type', 'QWEN')
        limit = int(request.GET.get('limit', 5))

        type_config = {
            'QWEN': {'search': 'qwen', 'filter': 'qwen,text-generation'},
            'LLAMA': {'search': 'llama', 'filter': 'llama,text-generation'},
            'MISTRAL': {'search': 'mistral', 'filter': 'mistral,text-generation'},
            'GEMMA': {'search': 'gemma', 'filter': 'gemma,text-generation'},
            'DEEPSEEK': {'search': 'deepseek', 'filter': 'text-generation'},
            'PHI': {'search': 'phi-3', 'filter': 'text-generation'},
            'YI': {'search': 'yi-1.5', 'filter': 'text-generation'},
            'ALL': {'search': '', 'filter': 'text-generation'},
        }
        cfg = type_config.get(model_type, type_config['ALL'])

        api = HfApi()
        try:
            models = api.list_models(
                search=cfg['search'],
                filter=cfg['filter'],
                sort="downloads",
                limit=limit
            )
            results = []
            for m in models:
                results.append({
                    'id': m.id,
                    'name': m.id.split('/')[-1],
                    'full_name': m.id,
                    'author': m.id.split('/')[0],
                    'downloads': m.downloads,
                    'likes': m.likes,
                    'created_at': m.created_at.isoformat()[:10] if m.created_at else 'N/A',
                })
            return Response({'type': model_type, 'models': results})
        except Exception as e:
            logger.error(f"Error in get_top_models: {e}", exc_info=True)
            return Response({'error': str(e), 'models': []}, status=500)

    def get_model_info(self, request):
        from huggingface_hub import HfApi
        model_id = request.GET.get('model_id', '')
        if not model_id:
            return Response({'error': 'model_id required'}, status=400)
        try:
            api = HfApi()
            info = api.model_info(model_id)
            return Response({
                'id': info.id,
                'author': info.author,
                'downloads': info.downloads,
                'likes': info.likes,
                'tags': info.tags[:10] if info.tags else [],
                'pipeline_tag': info.pipeline_tag,
                'private': info.private,
                'gated': info.gated,
            })
        except Exception as e:
            return Response({'error': str(e), 'model_id': model_id}, status=404)

    def validate_model(self, request):
        model_id = request.GET.get('model_id', '')
        if not model_id:
            return Response({'error': 'Не указан ID модели'}, status=400)

        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.model_info(model_id, timeout=10)
            
            # Значения по умолчанию
            context_length = 4096
            vocab_size = 32000
            parameter_count = 0
            
            # Пытаемся получить конфиг через прямой запрос к raw config.json
            try:
                config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
                headers = {"User-Agent": "IvannaModelManager/1.0"}
                resp = requests.get(config_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    config = resp.json()
                    context_length = config.get('max_position_embeddings', 4096)
                    vocab_size = config.get('vocab_size', 32000)
                    if 'num_parameters' in config:
                        parameter_count = config['num_parameters']
            except Exception as e:
                logger.warning(f"Could not fetch config for {model_id}: {e}")
            
            # Если parameter_count всё ещё 0, попробуем извлечь из тегов или названия
            if parameter_count == 0 and info.tags:
                for tag in info.tags:
                    tag_lower = tag.lower()
                    if tag_lower.endswith('b') and tag_lower[:-1].isdigit():
                        parameter_count = int(tag_lower[:-1]) * 1_000_000_000
                        break
                    elif tag_lower.endswith('m') and tag_lower[:-1].isdigit():
                        parameter_count = int(tag_lower[:-1]) * 1_000_000
                        break
            
            # Если не нашли, пробуем из названия модели (например, "7b")
            if parameter_count == 0:
                match = re.search(r'(\d+\.?\d*)[bB]', model_id)
                if match:
                    parameter_count = int(float(match.group(1)) * 1_000_000_000)
            
            return Response({
                'valid': True,
                'exists': True,
                'model_id': info.id,
                'author': info.author,
                'downloads': info.downloads,
                'private': info.private,
                'gated': info.gated,
                'context_length': context_length,
                'vocab_size': vocab_size,
                'parameter_count': parameter_count,
            })
        except Exception as e:
            logger.error(f"Validate model error for {model_id}: {e}", exc_info=True)
            return Response({
                'valid': False,
                'exists': False,
                'model_id': model_id,
                'error': str(e)
            }, status=404)

    # =========================================================================
    # Методы для моделей эмбеддингов
    # =========================================================================
    def search_embedding_models(self, request):
        from huggingface_hub import HfApi
        import logging
        logger = logging.getLogger(__name__)

        query = request.GET.get('q', '')
        page = int(request.GET.get('page', 1))
        limit = int(request.GET.get('limit', 10))

        api = HfApi()
        try:
            # Получаем до 100 моделей (чтобы не грузить API слишком сильно)
            all_models = list(api.list_models(
                search=query or None,
                filter="sentence-transformers,feature-extraction",
                sort="downloads",
                limit=100
            ))
            total = len(all_models)
            start = (page - 1) * limit
            end = start + limit
            page_models = all_models[start:end]

            results = []
            for m in page_models:
                dim = None
                if m.tags:
                    for tag in m.tags:
                        if tag.startswith('dim'):
                            dim = tag.replace('dim', '')
                            break
                results.append({
                    'id': m.id,
                    'name': m.id.split('/')[-1],
                    'author': m.id.split('/')[0],
                    'downloads': m.downloads,
                    'likes': m.likes,
                    'tags': m.tags[:10] if m.tags else [],
                    'dimension': dim,
                })
            return Response({'results': results, 'total': total})
        except Exception as e:
            logger.error(f"search_embedding_models error: {e}", exc_info=True)
            return Response({'error': str(e), 'results': []}, status=500)

    def get_top_embedding_models(self, request):
        from huggingface_hub import HfApi
        import logging
        logger = logging.getLogger(__name__)

        limit = int(request.GET.get('limit', 5))
        api = HfApi()
        try:
            models = list(api.list_models(
                filter="sentence-transformers,feature-extraction",
                sort="downloads",
                limit=limit
            ))
            results = []
            for m in models:
                dim = None
                if m.tags:
                    for tag in m.tags:
                        if tag.startswith('dim'):
                            dim = tag.replace('dim', '')
                            break
                results.append({
                    'id': m.id,
                    'name': m.id.split('/')[-1],
                    'author': m.id.split('/')[0],
                    'downloads': m.downloads,
                    'likes': m.likes,
                    'tags': m.tags[:10] if m.tags else [],
                    'dimension': dim,
                })
            return Response({'models': results})
        except Exception as e:
            logger.error(f"get_top_embedding_models error: {e}", exc_info=True)
            return Response({'error': str(e), 'models': []}, status=500)

    def validate_embedding_model(self, request):
        model_id = request.GET.get('model_id', '')
        if not model_id:
            return Response({'error': 'model_id required'}, status=400)
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            info = api.model_info(model_id, timeout=10)
            if info.pipeline_tag not in ['feature-extraction', 'sentence-similarity']:
                return Response({'valid': False, 'exists': True, 'error': 'Not an embedding model'})
            dim = None
            if info.tags:
                for tag in info.tags:
                    if tag.startswith('dim'):
                        dim = tag.replace('dim', '')
                        break
            return Response({
                'valid': True,
                'exists': True,
                'id': info.id,
                'author': info.author,
                'downloads': info.downloads,
                'likes': info.likes,
                'tags': info.tags[:10],
                'dimension': dim,
                'license': info.cardData.get('license', 'unknown'),
            })
        except Exception as e:
            return Response({'valid': False, 'exists': False, 'error': str(e)}, status=404)

    # =========================================================================
    # Методы для датасетов
    # =========================================================================
    def search_datasets(self, request):
        """
        Поиск датасетов на Hugging Face Hub с ручной сортировкой и пагинацией.
        Ограничивает количество получаемых датасетов для производительности.
        """
        from huggingface_hub import HfApi
        import logging
        from datetime import datetime

        logger = logging.getLogger(__name__)

        query = request.GET.get('q', '').strip()
        limit = int(request.GET.get('limit', 10))
        page = int(request.GET.get('page', 1))
        sort_by = request.GET.get('sort', 'downloads')

        api = HfApi()
        try:
            # Получаем ограниченное количество датасетов (500) – достаточно для топ-20 и первых страниц поиска
            all_datasets = list(api.list_datasets(search=query or None, limit=500))

            # Сортировка на клиенте (так как параметр sort не всегда поддерживается)
            if sort_by == 'downloads':
                all_datasets.sort(key=lambda x: getattr(x, 'downloads', 0), reverse=True)
            elif sort_by == 'likes':
                all_datasets.sort(key=lambda x: getattr(x, 'likes', 0), reverse=True)
            elif sort_by == 'created':
                all_datasets.sort(key=lambda x: x.created_at if x.created_at else datetime.min, reverse=True)

            total = len(all_datasets)
            start = (page - 1) * limit
            end = start + limit
            paginated = all_datasets[start:end]

            results = []
            for ds in paginated:
                description = ''
                if hasattr(ds, 'cardData') and ds.cardData:
                    description = ds.cardData.get('description', '')
                elif hasattr(ds, 'description'):
                    description = ds.description or ''

                results.append({
                    'id': ds.id,
                    'name': ds.id.split('/')[-1],
                    'author': ds.id.split('/')[0],
                    'downloads': getattr(ds, 'downloads', 0) or 0,
                    'likes': getattr(ds, 'likes', 0) or 0,
                    'tags': ds.tags[:5] if ds.tags else [],
                    'description': description[:200],
                    'created_at': ds.created_at.isoformat() if ds.created_at else None,
                })

            return Response({'results': results, 'total': total})
        except Exception as e:
            logger.error(f"Error in search_datasets: {e}", exc_info=True)
            return Response({'error': str(e), 'results': []}, status=500)

    def top_datasets(self, request):
        """Возвращает топ-датасеты по загрузкам (пустой поиск)."""
        # Перенаправляем на search_datasets с пустым q
        new_request = request
        # Создаём копию GET параметров и устанавливаем q=''
        if hasattr(request, 'GET') and isinstance(request.GET, dict):
            # Просто вызываем search_datasets с теми же параметрами, но q заменён на ''
            get_params = request.GET.copy()
            get_params['q'] = ''
            request.GET = get_params
        else:
            # Fallback
            request.GET = request.GET.copy()
            request.GET['q'] = ''
        return self.search_datasets(request)

    def validate_dataset(self, request):
        repo_id = request.GET.get('repo_id', '')
        if not repo_id:
            return Response({'error': 'repo_id required'}, status=400)
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            info = api.dataset_info(repo_id, timeout=10)
            return Response({
                'valid': True,
                'exists': True,
                'id': info.id,
                'author': info.id.split('/')[0],
                'downloads': info.downloads,
                'likes': info.likes,
                'tags': info.tags[:10] if info.tags else [],
                'description': info.cardData.get('description', '')[:200] if info.cardData else '',
            })
        except Exception as e:
            return Response({'valid': False, 'exists': False, 'error': str(e)}, status=404)

# ============================================================================
# СИСТЕМНЫЕ API (ЕДИНЫЙ HEALTH CHECK)
# ============================================================================
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_health_check(request):
    """Единый эндпоинт для проверки здоровья."""
    from django.db import connection
    from celery.result import AsyncResult

    health = {
        'status': 'healthy',
        'timestamp': timezone.now().isoformat(),
        'services': {},
    }

    # База данных
    try:
        with connection.cursor() as c:
            c.execute("SELECT 1")
        health['services']['database'] = 'connected'
    except:
        health['services']['database'] = 'disconnected'
        health['status'] = 'degraded'

    # Redis
    try:
        redis_model_manager.redis.ping()
        health['services']['redis'] = 'connected'
    except:
        health['services']['redis'] = 'disconnected'
        health['status'] = 'degraded'

    # Модели
    health['models'] = {
        'total': ModelCheckpoint.objects.filter(is_active=True).count(),
        'loaded': ModelCheckpoint.objects.filter(status='LOADED', is_active=True).count(),
        'downloading': ModelCheckpoint.objects.filter(status='DOWNLOADING', is_active=True).count(),
    }

    return Response(health)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def gpu_status(request):
    """Статус GPU."""
    mem = redis_model_manager.get_available_memory()
    loaded = redis_model_manager.get_all_model_infos()
    return Response({
        'cuda_available': mem.get('device') == 'cuda',
        'memory': mem,
        'loaded_models': loaded,
        'timestamp': timezone.now().isoformat(),
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def system_metrics(request):
    """Системные метрики (CPU, память, диск)."""
    import psutil
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    net = psutil.net_io_counters()
    return Response({
        'cpu': {'percent': cpu, 'cores': psutil.cpu_count()},
        'memory': {
            'total_gb': mem.total / 1024**3,
            'available_gb': mem.available / 1024**3,
            'percent': mem.percent,
        },
        'disk': {
            'total_gb': disk.total / 1024**3,
            'free_gb': disk.free / 1024**3,
            'percent': disk.percent,
        },
        'network': {
            'bytes_sent_gb': net.bytes_sent / 1024**3,
            'bytes_recv_gb': net.bytes_recv / 1024**3,
        },
        'timestamp': timezone.now().isoformat(),
    })


# ============================================================================
# СТАТИСТИКА (УПРОЩЁННАЯ)
# ============================================================================
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_statistics_view(request):
    stats = {
        'total_models': ModelCheckpoint.objects.filter(is_active=True).count(),
        'loaded_models': ModelCheckpoint.objects.filter(status='LOADED', is_active=True).count(),
        'experiments': Experiment.objects.filter(is_active=True).count(),
        'active_chat_sessions': ChatSession.objects.filter(is_active=True).count(),
        'total_evaluation_tasks': ModelEvaluationTask.objects.count(),
        'pending_evaluation_tasks': ModelEvaluationTask.objects.filter(status__in=['PENDING', 'RUNNING']).count(),
    }
    return Response(stats)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_rag_statistics(request):
    """
    Возвращает статистику RAG для текущего пользователя: количество коллекций и документов.
    """
    from .models import DocumentCollection, Document

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
# FineTuning
# ============================================================================

class FineTuningJobViewSet(viewsets.ModelViewSet):
    """API для задач fine-tuning"""
    queryset = FineTuningJob.objects.all()
    serializer_class = FineTuningJobSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return self.queryset.filter(created_by=self.request.user.username)
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user.username)
    
    @action(detail=True, methods=['post'])
    def start(self, request, pk=None):
        """Запуск задачи fine-tuning"""
        job = self.get_object()
        # Если нужно принимать флаг restart, он может быть в request.data
        restart = request.data.get('restart', False)
        
        if job.status not in ['PENDING', 'FAILED', 'CANCELLED']:
            return Response(
                {'error': f'Job already in status {job.status}'},
                status=400
            )
        
        task = run_fine_tuning_task.delay(job.id, restart=restart)
        job.celery_task_id = task.id
        job.status = 'PENDING'
        job.save()
        
        return Response({'success': True, 'task_id': task.id, 'job_id': job.id})
    
    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """Отмена задачи"""
        job = self.get_object()
        
        if job.status in ['TRAINING', 'PREPARING'] and job.celery_task_id:
            from celery.result import AsyncResult
            AsyncResult(job.celery_task_id).revoke(terminate=True)
            
            job.status = 'CANCELLED'
            job.completed_at = timezone.now()
            job.save()
            
            return Response({'success': True, 'message': 'Job cancelled'})
        
        return Response({'error': 'Cannot cancel job'}, status=400)
    
    @action(detail=True, methods=['get'])
    def logs(self, request, pk=None):
        """Получение логов задачи"""
        job = self.get_object()
        return Response({'logs': job.logs})
    
    @action(detail=True, methods=['post'])
    def evaluate(self, request, pk=None):
        """Запуск оценки на датасете"""
        job = self.get_object()
        dataset_id = request.data.get('dataset_id')
        
        if job.status != 'COMPLETED':
            return Response({'error': 'Job not completed'}, status=400)
        
        task = evaluate_fine_tuned_model_task.delay(job.id, dataset_id)
        
        return Response({
            'success': True,
            'task_id': task.id
        })

# ============================================================================
# Conversion
# ============================================================================

class ModelConversionJobViewSet(viewsets.ModelViewSet):
    queryset = ModelConversionJob.objects.all()
    serializer_class = ModelConversionJobSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return self.queryset.filter(created_by=self.request.user.username)

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user.username)

    @action(detail=True, methods=['post'])
    def start(self, request, pk=None):
        job = self.get_object()
        if job.status not in ['PENDING', 'FAILED', 'CANCELLED']:
            return Response({'error': f'Job already in status {job.status}'}, status=400)
        task = convert_model_to_gguf_task.delay(job.id)
        job.celery_task_id = task.id
        job.status = 'PENDING'
        job.save()
        return Response({'success': True, 'task_id': task.id, 'job_id': job.id})

    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        job = self.get_object()
        if job.status in ['PREPARING', 'CONVERTING'] and job.celery_task_id:
            from celery.result import AsyncResult
            AsyncResult(job.celery_task_id).revoke(terminate=True)
            job.status = 'CANCELLED'
            job.completed_at = timezone.now()
            job.save()
            return Response({'success': True, 'message': 'Job cancelled'})
        return Response({'error': 'Cannot cancel job'}, status=400)

    @action(detail=True, methods=['get'])
    def logs(self, request, pk=None):
        job = self.get_object()
        return Response({'logs': job.logs})


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

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_evaluation_stats(request):
    """
    Заглушка для статистики по задачам оценки.
    Возвращает нулевые значения, чтобы не ломать фронтенд.
    """
    return Response({
        'total_tasks': 0,
        'active_tasks': 0,
        'completed_tasks': 0,
        'total_datasets': 0,
    })


# ============================================================================
# Distillation
# ============================================================================

class DistillationQuestionSetViewSet(viewsets.ModelViewSet):
    """Управление наборами для дистилляции."""
    queryset = DistillationQuestionSet.objects.all()
    serializer_class = DistillationQuestionSetSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        qs = super().get_queryset()
        if self.request.user.is_authenticated:
            return qs.filter(Q(created_by=self.request.user.username) | Q(created_by=''))
        return qs.none()

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user.username)

    @action(detail=True, methods=['post'])
    def auto_create_groups(self, request, pk=None):
        """
        Автоматическое создание групп: разбивает список чанков на окна.
        Ожидает JSON: {
            "chunk_ids": [1,2,3,...],
            "window_size": 3,
            "num_questions": 3,
            "include_partial": false
        }
        """
        qset = self.get_object()
        chunk_ids = request.data.get('chunk_ids', [])
        window_size = request.data.get('window_size', 3)
        num_questions = request.data.get('num_questions', 3)
        include_partial = request.data.get('include_partial', False)

        if not chunk_ids:
            return Response({'error': 'chunk_ids required'}, status=400)

        # Удаляем старые группы, если нужно (по желанию)
        # qset.groups.all().delete()

        created = 0
        for i in range(0, len(chunk_ids), window_size):
            group_chunks = chunk_ids[i:i+window_size]
            if len(group_chunks) == window_size:
                DistillationChunkGroup.objects.create(
                    question_set=qset,
                    chunk_ids=group_chunks,
                    num_questions=num_questions,
                    order=i // window_size
                )
                created += 1
            elif include_partial and group_chunks:
                DistillationChunkGroup.objects.create(
                    question_set=qset,
                    chunk_ids=group_chunks,
                    num_questions=num_questions,
                    order=i // window_size
                )
                created += 1

        return Response({'success': True, 'groups_created': created})

    @action(detail=True, methods=['post'])
    def start_generation(self, request, pk=None):
        """
        Запуск генерации вопросов для всех групп набора.
        Ожидает: teacher_model_id, опционально quantization и generation_config.
        """
        qset = self.get_object()
        if qset.status not in ['PENDING', 'FAILED']:
            return Response({'error': 'Cannot start: invalid status'}, status=400)

        teacher_model_id = request.data.get('teacher_model_id')
        if not teacher_model_id:
            return Response({'error': 'teacher_model_id required'}, status=400)

        quantization = request.data.get('quantization', 'int4')

        # Обновляем конфигурацию набора, если передана
        if 'generation_config' in request.data:
            qset.default_generation_config.update(request.data['generation_config'])
            qset.save(update_fields=['default_generation_config'])

        
        task = generate_questions_for_set_task.delay(qset.id, teacher_model_id, quantization)
        qset.celery_task_id = task.id
        qset.save(update_fields=['celery_task_id'])

        return Response({'success': True, 'task_id': task.id})

    @action(detail=True, methods=['post'])
    def resume(self, request, pk=None):
        qset = self.get_object()
        if qset.status not in ['FAILED', 'PAUSED', 'PENDING']:
            return Response({'error': 'Cannot resume: invalid status'}, status=400)

        teacher_model_id = request.data.get('teacher_model_id')
        if not teacher_model_id:
            return Response({'error': 'teacher_model_id required'}, status=400)

        quantization = request.data.get('quantization', 'int4')

        # Сохраняем celery_task_id, чтобы можно было отслеживать
        task = generate_questions_for_set_task.delay(qset.id, teacher_model_id, quantization)
        qset.celery_task_id = task.id
        qset.status = 'PENDING'  # или RUNNING? лучше PENDING, а задача сама переведёт в RUNNING
        qset.save(update_fields=['celery_task_id', 'status'])

        return Response({'success': True, 'task_id': task.id})

    @action(detail=True, methods=['get'])
    def questions(self, request, pk=None):
        """Возвращает список вопросов набора."""
        qset = self.get_object()
        questions = DistillationQuestion.objects.filter(group__question_set=qset).order_by('-created_at')
        serializer = DistillationQuestionSerializer(questions, many=True)
        return Response(serializer.data)


class DistillationChunkGroupViewSet(viewsets.ModelViewSet):
    """Управление группами чанков."""
    queryset = DistillationChunkGroup.objects.all()
    serializer_class = DistillationChunkGroupSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return super().get_queryset().filter(question_set__created_by=self.request.user.username)

    @action(detail=True, methods=['post'])
    def update_chunks(self, request, pk=None):
        """Обновить список чанков группы."""
        group = self.get_object()
        chunk_ids = request.data.get('chunk_ids', [])
        if not chunk_ids:
            return Response({'error': 'chunk_ids required'}, status=400)
        group.chunk_ids = chunk_ids
        group.save()
        return Response({'success': True})

    @action(detail=False, methods=['post'])
    def bulk_update(self, request):
        """Массовое обновление групп."""
        question_set_id = request.data.get('question_set_id')
        groups_data = request.data.get('groups', [])
        if not question_set_id:
            return Response({'error': 'question_set_id required'}, status=400)
        question_set = get_object_or_404(DistillationQuestionSet, id=question_set_id, created_by=request.user.username)
        for g in groups_data:
            group_id = g.get('id')
            if group_id:
                group = get_object_or_404(DistillationChunkGroup, id=group_id, question_set=question_set)
                group.name = g.get('name', group.name)
                group.chunk_ids = g.get('chunk_ids', group.chunk_ids)
                group.num_questions = g.get('num_questions', group.num_questions)
                group.order = g.get('order', group.order)
                group.save()
            else:
                # Создаём новую группу
                DistillationChunkGroup.objects.create(
                    question_set=question_set,
                    name=g.get('name', ''),
                    chunk_ids=g.get('chunk_ids', []),
                    num_questions=g.get('num_questions', 3),
                    order=g.get('order', question_set.groups.count())
                )
        return Response({'success': True})


class DistillationQuestionViewSet(viewsets.ModelViewSet):
    """Управление вопросами дистилляции."""
    queryset = DistillationQuestion.objects.all()
    serializer_class = DistillationQuestionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return super().get_queryset().filter(group__question_set__created_by=self.request.user.username)

    @action(detail=True, methods=['post'])
    def update_answer(self, request, pk=None):
        """Вручную обновить ответ на вопрос."""
        question = self.get_object()
        answer = request.data.get('answer', '')
        if not answer:
            return Response({'error': 'answer required'}, status=400)
        question.answer_text = answer
        question.status = 'reviewed'
        question.save()
        return Response({'success': True})

    @action(detail=False, methods=['post'])
    def batch_update_answers(self, request):
        """Массовое обновление ответов."""
        updates = request.data.get('updates', [])  # список {id, answer}
        for upd in updates:
            try:
                q = DistillationQuestion.objects.get(id=upd['id'])
                q.answer_text = upd['answer']
                q.status = 'reviewed'
                q.save()
            except Exception:
                pass
        return Response({'success': True})


class DistillationAnswerJobViewSet(viewsets.ModelViewSet):
    """Управление задачами генерации ответов."""
    queryset = DistillationAnswerJob.objects.all()
    serializer_class = DistillationAnswerJobSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return super().get_queryset().filter(created_by=self.request.user.username)

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user.username)

    @action(detail=True, methods=['post'])
    def start(self, request, pk=None):
        job = self.get_object()
        if job.status not in ['PENDING', 'FAILED']:
            return Response({'error': 'Cannot start: invalid status'}, status=400)

        teacher_model_id = request.data.get('teacher_model_id')
        if not teacher_model_id:
            return Response({'error': 'teacher_model_id required'}, status=400)

        quantization = request.data.get('quantization', 'int4')

        if 'generation_config' in request.data:
            job.generation_config.update(request.data['generation_config'])
            job.save(update_fields=['generation_config'])

        task = generate_answers_for_questions_task.delay(job.id, teacher_model_id, quantization)
        return Response({'success': True, 'task_id': task.id})

    @action(detail=True, methods=['post'])
    def resume(self, request, pk=None):
        job = self.get_object()
        if job.status not in ['FAILED', 'PAUSED', 'PENDING']:
            return Response({'error': 'Cannot resume: invalid status'}, status=400)

        teacher_model_id = request.data.get('teacher_model_id')
        if not teacher_model_id:
            return Response({'error': 'teacher_model_id required'}, status=400)

        quantization = request.data.get('quantization', 'int4')

        task = generate_answers_for_questions_task.delay(job.id, teacher_model_id, quantization)
        job.celery_task_id = task.id
        job.status = 'PENDING'
        job.save(update_fields=['celery_task_id', 'status'])

        return Response({'success': True, 'task_id': task.id})

    @action(detail=False, methods=['post'])
    def regenerate_answer(self, request):
        question_id = request.data.get('question_id')
        answer_job_id = request.data.get('answer_job_id')
        teacher_model_id = request.data.get('teacher_model_id')
        quantization = request.data.get('quantization', 'int4')
        generation_config = request.data.get('generation_config', {})

        if not all([question_id, answer_job_id, teacher_model_id]):
            return Response({'error': 'Необходимы question_id, answer_job_id, teacher_model_id'}, status=400)

        try:
            job = DistillationAnswerJob.objects.get(id=answer_job_id, created_by=request.user.username)
            question = DistillationQuestion.objects.get(id=question_id)
        except (DistillationAnswerJob.DoesNotExist, DistillationQuestion.DoesNotExist):
            return Response({'error': 'Объект не найден'}, status=404)

        from .tasks import regenerate_answer_task
        task = regenerate_answer_task.delay(question_id, answer_job_id, teacher_model_id, quantization, generation_config)
        return Response({'success': True, 'task_id': task.id})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def distillation_task_status(request, question_set_id):
    """Возвращает статус задачи генерации вопросов."""
    qset = get_object_or_404(DistillationQuestionSet, id=question_set_id, created_by=request.user.username)
    # Получаем последние 15 вопросов для динамического отображения
    recent_questions = DistillationQuestion.objects.filter(
        group__question_set=qset
    ).order_by('-created_at')[:15]
    return Response({
        'status': qset.status,
        'status_display': qset.get_status_display(),
        'progress': qset.progress,
        'current_step': qset.current_step,
        'logs': qset.logs,
        'checkpoint_data': qset.checkpoint_data,
        'recent_questions': [{
            'id': q.id,
            'text': q.question_text[:150],
            'status': q.status,
            'status_display': q.get_status_display(),
            'created_at': q.created_at.isoformat()
        } for q in recent_questions]
    })

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def distillation_answer_job_status(request, job_id):
    """Возвращает статус задачи генерации ответов."""
    job = get_object_or_404(DistillationAnswerJob, id=job_id, created_by=request.user.username)
    recent_answers = DistillationAnswer.objects.filter(answer_job=job).select_related('question').order_by('-created_at')[:15]
    return Response({
        'status': job.status,
        'status_display': job.get_status_display(),
        'progress': job.progress,
        'current_step': job.current_step,
        'logs': job.logs,
        'checkpoint_data': job.checkpoint_data,
        'recent_answers': [{
            'id': ans.id,
            'question_id': ans.question.id,
            'question_text': ans.question.question_text[:150],
            'answer_text': ans.answer_text[:200] if ans.answer_text else '',
            'status': ans.question.status,
            'status_display': ans.question.get_status_display(),
            'created_at': ans.created_at.isoformat()
        } for ans in recent_answers]
    })

@action(detail=True, methods=['post'])
def resume(self, request, pk=None):
    """Возобновление задачи из чекпоинта."""
    qset = self.get_object()
    if not qset.checkpoint_data:
        return Response({'error': 'Нет чекпоинта для возобновления'}, status=400)
    if qset.status == 'RUNNING':
        return Response({'error': 'Задача уже выполняется'}, status=400)
    
    # Получаем последние сохранённые параметры из checkpoint_data
    checkpoint = qset.checkpoint_data
    teacher_model_id = checkpoint.get('teacher_model_id')
    quantization = checkpoint.get('quantization', 'int4')
    if not teacher_model_id:
        return Response({'error': 'Не указана модель-учитель в чекпоинте'}, status=400)
    
    # Запускаем задачу с параметрами из чекпоинта
    task = generate_questions_for_set_task.delay(qset.id, teacher_model_id, quantization)
    qset.celery_task_id = task.id
    qset.status = 'PENDING'
    qset.save(update_fields=['celery_task_id', 'status'])
    
    return Response({'success': True, 'task_id': task.id})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def distillation_dataset_builder_data(request):
    """Возвращает данные для сборщика датасета: вопросы и ответы по выбранным задачам."""
    question_set_id = request.GET.get('question_set')
    answer_job_ids = request.GET.getlist('answer_jobs')
    
    if not question_set_id:
        return Response({'error': 'question_set required'}, status=400)
    
    try:
        question_set = DistillationQuestionSet.objects.get(
            id=question_set_id, 
            created_by=request.user.username
        )
    except DistillationQuestionSet.DoesNotExist:
        return Response({'error': 'Набор вопросов не найден'}, status=404)
    
    # Вопросы через group__question_set
    questions = DistillationQuestion.objects.filter(
        group__question_set=question_set
    ).order_by('id')
    
    # ✅ Получаем все ответы для выбранных задач, сортируем по -created_at
    answers_by_question = {}
    
    for job_id in answer_job_ids:
        try:
            job = DistillationAnswerJob.objects.get(
                id=job_id, 
                created_by=request.user.username
            )
        except DistillationAnswerJob.DoesNotExist:
            continue
        
        # ✅ Сортируем по -created_at, чтобы последний ответ был первым
        answers = DistillationAnswer.objects.filter(
            answer_job=job
        ).select_related('question').order_by('-created_at')
        
        for ans in answers:
            # ✅ Если для этого вопроса ещё нет ответа — сохраняем (это будет последний)
            if ans.question_id not in answers_by_question:
                answers_by_question[ans.question_id] = []
            
            answers_by_question[ans.question_id].append({
                'answer_id': ans.id,
                'answer_text': ans.answer_text,
                'job_name': job.name,
                'job_id': job.id,
                'created_at': ans.created_at.isoformat() if ans.created_at else None
            })
    
    # Формируем данные для каждого вопроса
    result = []
    for q in questions:
        chunks = []
        if q.group and q.group.chunk_ids:
            chunk_objects = DocumentChunk.objects.filter(
                id__in=q.group.chunk_ids, 
                is_active=True
            )
            chunks = [chunk.content for chunk in chunk_objects]
        
        question_answers = answers_by_question.get(q.id, [])
        
        result.append({
            'id': q.id,
            'question_text': q.question_text,
            'context_chunks': chunks,
            'answers': question_answers,  # Список всех версий ответов
            'latest_answer': question_answers[0] if question_answers else None,  # Последний ответ
            'has_multiple_answers': len(question_answers) > 1
        })
    
    return Response({
        'questions': result,
        'question_set_name': question_set.name,
        'question_set_id': question_set.id,
    })

class DistillationDatasetViewSet(viewsets.ModelViewSet):
    queryset = DistillationDataset.objects.all()
    serializer_class = DistillationDatasetSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return self.queryset.filter(created_by=self.request.user.username)

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user.username)

    @action(detail=True, methods=['get'])
    def export_jsonl(self, request, pk=None):
        """Экспорт датасета в JSONL для fine-tuning."""
        dataset = self.get_object()
        items = dataset.items
        response = HttpResponse(content_type='application/jsonl')
        response['Content-Disposition'] = f'attachment; filename="{dataset.name}.jsonl"'
        for item in items:
            line = json.dumps({
                'instruction': item['question_text'],
                'output': item['answer_text']
            }, ensure_ascii=False)
            response.write(line + '\n')
        return response

    @action(detail=False, methods=['post'])
    def create_dataset(self, request):
        """Создание датасета из выбранных вопросов и ответов"""
        name = request.data.get('name')
        description = request.data.get('description', '')
        items = request.data.get('items', [])
        
        if not name or not items:
            return Response({'error': 'name и items обязательны'}, status=400)
        
        dataset = DistillationDataset.objects.create(
            name=name,
            description=description,
            items=items,
            created_by=request.user.username
        )
        
        return Response({
            'id': dataset.id,
            'name': dataset.name,
            'item_count': len(items)
        }, status=201)


# ============================================================================
# Instruction
# ============================================================================
class ManualInstructionSetViewSet(viewsets.ModelViewSet):
    queryset = ManualInstructionSet.objects.all()
    serializer_class = ManualInstructionSetSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return self.queryset.filter(created_by=self.request.user.username)

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user.username)

    @action(detail=True, methods=['post'])
    def export_jsonl(self, request, pk=None):
        """Экспорт набора в JSONL для fine-tuning."""
        instruction_set = self.get_object()
        instructions = instruction_set.instructions.all().order_by('order')
        response = HttpResponse(content_type='application/jsonl')
        response['Content-Disposition'] = f'attachment; filename="{instruction_set.name}.jsonl"'
        for instr in instructions:
            line = json.dumps({
                'instruction': instr.question,
                'output': instr.answer,
                'context_chunks': instr.context_chunks
            }, ensure_ascii=False)
            response.write(line + '\n')
        return response

    @action(detail=True, methods=['post'])
    def upload_to_huggingface(self, request, pk=None):
        """Загрузка датасета на Hugging Face Hub."""
        instruction_set = self.get_object()
        from huggingface_hub import HfApi, Repository
        import tempfile
        import os

        api = HfApi()
        repo_id = request.data.get('repo_id')
        token = request.data.get('token')
        if not repo_id or not token:
            return Response({'error': 'Не указаны repo_id и token'}, status=400)

        # Создаём временную папку и сохраняем JSONL
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, f"{instruction_set.name}.jsonl")
            with open(file_path, 'w', encoding='utf-8') as f:
                for instr in instruction_set.instructions.all().order_by('order'):
                    line = json.dumps({
                        'instruction': instr.question,
                        'output': instr.answer
                    }, ensure_ascii=False)
                    f.write(line + '\n')
            # Загружаем на HF
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=f"{instruction_set.name}.jsonl",
                repo_id=repo_id,
                token=token
            )
        return Response({'success': True, 'message': f'Датасет загружен в {repo_id}'})

    @action(detail=True, methods=['post'])
    def sync_instructions(self, request, pk=None):
        """
        Синхронизирует инструкции набора: удаляет старые, создаёт новые.
        Всё в одной транзакции — без дублирования.
        """
        from django.db import transaction
        
        instruction_set = self.get_object()
        
        # Проверка прав
        if instruction_set.created_by != request.user.username and not request.user.is_staff:
            return Response({'error': 'Нет прав'}, status=403)
        
        instructions_data = request.data.get('instructions', [])
        
        if not isinstance(instructions_data, list):
            return Response({'error': 'instructions должен быть списком'}, status=400)
        
        try:
            with transaction.atomic():  # ✅ Атомарная транзакция
                # 1. Удаляем ВСЕ существующие инструкции этого набора
                deleted_count, _ = ManualInstruction.objects.filter(
                    instruction_set=instruction_set
                ).delete()
                
                # 2. Создаём новые инструкции
                created_instructions = []
                for idx, instr_data in enumerate(instructions_data):
                    instruction = ManualInstruction.objects.create(
                        instruction_set=instruction_set,
                        question=instr_data.get('question', ''),
                        answer=instr_data.get('answer', ''),
                        context_chunks=instr_data.get('context_chunks', []),
                        order=instr_data.get('order', idx)
                    )
                    created_instructions.append(instruction)
                
                return Response({
                    'success': True,
                    'deleted': deleted_count,
                    'created': len(created_instructions),
                    'instructions': ManualInstructionSerializer(created_instructions, many=True).data
                }, status=200)
                
        except Exception as e:
            logger.error(f"Error syncing instructions: {e}", exc_info=True)
            return Response({'error': str(e)}, status=500)


class ManualInstructionViewSet(viewsets.ModelViewSet):
    queryset = ManualInstruction.objects.all()
    serializer_class = ManualInstructionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        qs = super().get_queryset()
        instruction_set = self.request.query_params.get('instruction_set')
        if instruction_set:
            qs = qs.filter(instruction_set_id=instruction_set)
        return qs


# ============================================================================
# Instruction
# ============================================================================
class HuggingFaceDatasetViewSet(viewsets.ModelViewSet):
    queryset = HuggingFaceDataset.objects.all()
    serializer_class = HuggingFaceDatasetSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return self.queryset.filter(created_by=self.request.user.username)

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user.username)

    @action(detail=True, methods=['post'])
    def import_dataset(self, request, pk=None):
        """Запуск импорта датасета из HF."""
        dataset = self.get_object()
        if dataset.status not in ['PENDING', 'FAILED']:
            return Response({'error': 'Датасет уже обрабатывается или завершён'}, status=400)

        from ..tasks import import_huggingface_dataset_task
        task = import_huggingface_dataset_task.delay(dataset.id)
        dataset.celery_task_id = task.id
        dataset.status = 'PENDING'
        dataset.save()
        return Response({'success': True, 'task_id': task.id})