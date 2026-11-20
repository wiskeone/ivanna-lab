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
# API ViewSets
# ============================================================================
class ExperimentViewSet(viewsets.ModelViewSet):
    """API for experiments."""
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
            return Response({'error': _('Нет базового чекпоинта')}, status=400)
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
    """API for models."""
    queryset = ModelCheckpoint.objects.all()
    serializer_class = ModelCheckpointSerializer
    permission_classes = [IsAuthenticated]

    @action(detail=True, methods=['post'])
    def download(self, request, pk=None):
        """Start model download (API)."""
        model = self.get_object()
        if not model.huggingface_id:
            return Response({'error': 'No Hugging Face ID'}, status=400)
        task = download_model_task.delay(model.id, user_id=request.user.id)
        return Response({'message': 'Download started', 'task_id': task.id})

    @action(detail=True, methods=['post'])
    def load(self, request, pk=None):
        """Load model into memory with quantization choice."""
        model = self.get_object()
        serializer = ModelLoadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        if not model.local_path or not os.path.exists(model.local_path):
            return Response({'error': 'Model not downloaded locally'}, status=400)

        # Update quantization if specified and not 'none'
        quantization = serializer.validated_data.get('quantization')
        if quantization and quantization != 'none':
            model.quantization = quantization
            model.save()

        # Start the task with the given quantization
        task = load_model_to_memory_task.delay(
            model.id,
            quantization=quantization if quantization != 'none' else None
        )
        return Response({'message': 'Loading started', 'task_id': task.id})

    @action(detail=True, methods=['post'])
    def unload(self, request, pk=None):
        """Unload model from memory with force option."""
        model = self.get_object()
        model_id = model.huggingface_id or model.name
        
        force = request.data.get('force', False)
        
        # Start the unload task
        task = unload_model_from_memory_task.delay(model.id, force=force)
        
        return Response({
            'message': 'Unload task started',
            'task_id': task.id,
            'model_id': model_id,
            'force': force
        })

    @action(detail=True, methods=['post'])
    def generate(self, request, pk=None):
        """Text generation."""
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
            # Create an inference session for statistics
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
        """GPU/CPU memory status."""
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

        # Sync model status
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
    """API for model merge jobs."""
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
    """API for inference sessions (read-only)."""
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
    """API for model evaluations."""
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
    """API for download jobs."""
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
