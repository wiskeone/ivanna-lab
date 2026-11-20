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



# ============================================================================
# PERMISSIONS
# ============================================================================
class IsCollectionOwnerOrPublic(BasePermission):
    """Collection access: owner or public."""
    def has_object_permission(self, request, view, obj):
        if obj.owner_id == str(request.user.id):
            return True
        if obj.is_public:
            return True
        return request.user.is_staff


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def task_result(request, task_id):
    """Returns Celery task status with safe serialization."""
    from celery.result import AsyncResult
    import traceback
    
    try:
        task = AsyncResult(task_id)
        
        # Base response
        response = {
            'task_id': task_id,
            'state': task.state,
            'status': task.status,
        }
        
        if task.ready():
            try:
                # Try to get the result
                result = task.result
                
                # If the result is an exception, convert it to a dict
                if isinstance(result, Exception):
                    response['result'] = {
                        'success': False,
                        'error': str(result),
                        'error_type': result.__class__.__name__
                    }
                elif result is not None:
                    # Check that the result is serializable
                    try:
                        json.dumps(result)
                        response['result'] = result
                    except (TypeError, ValueError):
                        # If not serializable, convert to a safe format
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
            # For tasks still in progress
            try:
                if hasattr(task, 'info') and task.info:
                    if isinstance(task.info, dict):
                        # Ensure all values are serializable
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
        # Return a friendly error
        return Response({
            'task_id': task_id,
            'state': 'FAILURE',
            'error': str(e),
            'traceback': traceback.format_exc() if settings.DEBUG else None
        }, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def force_local_load_model(request, model_id):
    """Force local model load with a quantization choice."""
    try:
        from ..models import ModelCheckpoint
        from ..tasks import load_model_to_memory_task
        from ..services.redis_model_service import redis_model_manager

        model = get_object_or_404(ModelCheckpoint, id=model_id, is_active=True)

        # Check if the model is already loaded
        model_id_str = model.huggingface_id or model.name
        if redis_model_manager.get_model_info(model_id_str):
            return Response({
                'success': False,
                'error': _('Модель уже загружена в память'),
                'model_id': model_id,
            }, status=400)

        if not model.local_path or not os.path.exists(model.local_path):
            return Response({
                'success': False,
                'error': _('Модель не скачана локально')
            }, status=400)

        quantization = request.data.get('quantization')
        # If quantization is not specified, fall back to the model's value (or None)
        if quantization is None:
            quantization = model.quantization or None

        # Clean up any stale locks
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


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_user_rag_collections(request):
    """Returns RAG collections available to the user."""
    from ..models import DocumentCollection
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
    """Returns Celery task status."""
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
            # For tasks still in progress, return meta info
            if hasattr(task, 'info') and task.info:
                response['info'] = task.info
            else:
                response['info'] = {}
        
        return Response(response)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


# ============================================================================
# SYSTEM API (UNIFIED HEALTH CHECK)
# ============================================================================
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_health_check(request):
    """Unified health check endpoint."""
    from django.db import connection
    from celery.result import AsyncResult

    health = {
        'status': 'healthy',
        'timestamp': timezone.now().isoformat(),
        'services': {},
    }

    # Database
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

    # Models
    health['models'] = {
        'total': ModelCheckpoint.objects.filter(is_active=True).count(),
        'loaded': ModelCheckpoint.objects.filter(status='LOADED', is_active=True).count(),
        'downloading': ModelCheckpoint.objects.filter(status='DOWNLOADING', is_active=True).count(),
    }

    return Response(health)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def gpu_status(request):
    """GPU status + models resident in memory (from Redis)."""
    mem = redis_model_manager.get_available_memory()
    loaded = redis_model_manager.get_all_model_infos()
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                reserved = torch.cuda.memory_reserved(i)
                total = props.total_memory
                gpus.append({
                    'index': i,
                    'name': props.name,
                    'total_memory_gb': round(total / 1024**3, 2),
                    'reserved_gb': round(reserved / 1024**3, 2),
                    'used_percent': round(reserved / total * 100, 1) if total else 0,
                })
    except Exception:
        pass
    return Response({
        'cuda_available': mem.get('device') == 'cuda',
        'gpu_count': len(gpus),
        'gpus': gpus,
        'memory': mem,        # legacy consumers (gpu_status.html)
        'device_memory': mem,
        'loaded_models': loaded,
        'timestamp': timezone.now().isoformat(),
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def system_metrics(request):
    """System metrics (CPU, memory, disk, network)."""
    import psutil
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    net = psutil.net_io_counters()
    used_gb = (mem.total - mem.available) / 1024**3
    return Response({
        'cpu': {
            'percent': cpu,
            'cores': psutil.cpu_count(),
            'physical_cores': psutil.cpu_count(logical=False) or psutil.cpu_count(),
            'logical_cores': psutil.cpu_count(logical=True),
        },
        'memory': {
            'total_gb': mem.total / 1024**3,
            'available_gb': mem.available / 1024**3,
            # explicit used/free aliases: the page used to read free_gb which
            # never existed here (undefinedGB in the UI)
            'used_gb': used_gb,
            'free_gb': mem.available / 1024**3,
            'percent': mem.percent,
        },
        'disk': {
            'total_gb': disk.total / 1024**3,
            'free_gb': disk.free / 1024**3,
            'used_gb': disk.used / 1024**3,
            'percent': disk.percent,
        },
        'network': {
            'bytes_sent_gb': net.bytes_sent / 1024**3,
            'bytes_recv_gb': net.bytes_recv / 1024**3,
            'bytes_sent': net.bytes_sent,
            'bytes_recv': net.bytes_recv,
            'packets_sent': net.packets_sent,
            'packets_recv': net.packets_recv,
        },
        'timestamp': timezone.now().isoformat(),
    })


# ============================================================================
# STATISTICS (SIMPLIFIED)
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
