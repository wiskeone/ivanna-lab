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



class FineTuningJobViewSet(viewsets.ModelViewSet):
    """API for fine-tuning jobs"""
    queryset = FineTuningJob.objects.all()
    serializer_class = FineTuningJobSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return self.queryset.filter(created_by=self.request.user.username)
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user.username)
    
    @action(detail=True, methods=['post'])
    def start(self, request, pk=None):
        """Start a fine-tuning job"""
        job = self.get_object()
        # The restart flag, if accepted, comes in request.data
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
        """Cancel a job"""
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
        """Get job logs"""
        job = self.get_object()
        return Response({'logs': job.logs})
    
    @action(detail=True, methods=['post'])
    def evaluate(self, request, pk=None):
        """Start evaluation on a dataset"""
        job = self.get_object()
        dataset_id = request.data.get('dataset_id')
        
        if job.status != 'COMPLETED':
            return Response({'error': 'Job not completed'}, status=400)
        
        task = evaluate_fine_tuned_model_task.delay(job.id, dataset_id)
        
        return Response({
            'success': True,
            'task_id': task.id
        })


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


@login_required
@require_http_methods(["POST"])
def api_fine_tuning_pause(request, job_id):
    """Request a pause: the trainer stops at the nearest step, the checkpoint
    is saved and the status becomes PAUSED."""
    from ..models import FineTuningJob
    job = FineTuningJob.objects.filter(id=job_id).first()
    if not job:
        return JsonResponse({'success': False, 'error': 'Job not found'}, status=404)
    if job.status != 'RUNNING':
        return JsonResponse({'success': False,
                             'error': f'Пауза доступна только для RUNNING (сейчас {job.status})'},
                            status=400)
    job.request_pause()
    return JsonResponse({'success': True, 'status': job.status,
                         'message': 'Пауза запрошена — тренер остановится на ближайшем шаге'})


@login_required
@require_http_methods(["POST"])
def api_fine_tuning_resume(request, job_id):
    """Resume: clear the pause and re-enqueue — the trainer
    picks up from the last checkpoint."""
    from ..models import FineTuningJob
    job = FineTuningJob.objects.filter(id=job_id).first()
    if not job:
        return JsonResponse({'success': False, 'error': 'Job not found'}, status=404)
    if job.status != 'PAUSED':
        return JsonResponse({'success': False,
                             'error': f'Резюме доступно только для PAUSED (сейчас {job.status})'},
                            status=400)
    job.resume()
    from ..tasks import resume_fine_tuning_task
    task = resume_fine_tuning_task.delay(job.id)
    return JsonResponse({'success': True, 'task_id': task.id,
                         'status': job.status})
