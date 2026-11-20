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



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_chat_session_preview(request, session_id):
    """
    Returns the latest session messages for preview.
    """
    from ..models import ChatSession
    from ..services.chat_service import chat_service

    try:
        session = ChatSession.objects.get(session_id=session_id, user_id=str(request.user.id))
    except ChatSession.DoesNotExist:
        return Response({'success': False, 'error': 'Session not found'}, status=404)

    # Try Redis first, fall back to the DB
    redis_session = chat_service.get_session(session_id)
    if redis_session and redis_session.get('conversation_history'):
        messages = redis_session['conversation_history'][-5:]  # last 5
    else:
        messages = session.conversation_history[-5:] if session.conversation_history else []

    # Trim extra fields if needed
    simplified = [{'role': m['role'], 'content': m['content'][:200]} for m in messages]

    return Response({'success': True, 'messages': simplified})


class ChatSessionViewSet(viewsets.ModelViewSet):
    """API for chat sessions."""
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'session_id'  # URLs now use session_id instead of pk

    def get_queryset(self):
        return self.queryset.filter(user_id=str(self.request.user.id), is_active=True)

    def create(self, request, *args, **kwargs):
        """Create a session with a model."""
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
        """Clear history."""
        session = self.get_object()
        success = chat_service.clear_history(session.session_id)
        return Response({'success': success})

    @action(detail=True, methods=['post'])
    def change_model(self, request, session_id=None):
        """Change the model."""
        session = self.get_object()
        model_id = request.data.get('model_id')
        if not model_id:
            return Response({'error': 'model_id required'}, status=400)
        try:
            new_model = ModelCheckpoint.objects.get(id=model_id, is_active=True)
            session.checkpoint = new_model
            session.save()
            # Update in Redis
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
        """List the user's sessions."""
        sessions = chat_service.list_user_sessions(str(request.user.id))
        return Response({'success': True, 'sessions': sessions})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def api_chat_update_rag_config(request, session_id):
    """Update the RAG config for a session."""
    from ..services.chat_service import chat_service
    
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
    """Send a chat message (async)."""
    from ..services.chat_service import chat_service
    from ..serializers import TextGenerationSerializer

    try:
        # Check that the session exists
        session = chat_service.get_session(session_id)
        if not session:
            return Response({'error': 'Session not found'}, status=404)

        serializer = TextGenerationSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        # Extract RAG parameters
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


@login_required
@require_http_methods(["GET"])
def api_generation_presets_list(request):
    """List generation presets (for the chat/comparison dropdown)."""
    from ..models import GenerationPreset
    presets = GenerationPreset.objects.all().values(
        'id', 'name', 'description', 'config', 'created_at'
    )
    return JsonResponse({'success': True, 'presets': list(presets)})


@login_required
@require_http_methods(["POST"])
def api_generation_preset_create(request):
    """Create a preset: {name, description?, config} — the config is validated
    and normalized (key whitelist, range clamping)."""
    import json as _json
    from ..models import GenerationPreset
    from ..services.generation_presets import normalize_preset_config

    try:
        data = _json.loads(request.body or '{}')
    except _json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)

    name = (data.get('name') or '').strip()
    if not name:
        return JsonResponse({'success': False, 'error': 'Name is required'}, status=400)

    config, warnings = normalize_preset_config(data.get('config', {}))
    preset = GenerationPreset.objects.create(
        name=name,
        description=(data.get('description') or '')[:500],
        config=config,
        created_by=request.user if request.user.is_authenticated else None,
    )
    return JsonResponse({'success': True, 'id': preset.id, 'config': config,
                         'warnings': warnings})


@login_required
@require_http_methods(["DELETE"])
def api_generation_preset_delete(request, preset_id):
    from ..models import GenerationPreset
    preset = GenerationPreset.objects.filter(id=preset_id).first()
    if not preset:
        return JsonResponse({'success': False, 'error': 'Not found'}, status=404)
    # only the author or staff can delete
    if not (request.user.is_staff or preset.created_by_id == request.user.id):
        return JsonResponse({'success': False, 'error': 'No permissions'}, status=403)
    preset.delete()
    return JsonResponse({'success': True})


@login_required
@require_http_methods(["POST"])
def api_chat_apply_preset(request, session_id):
    """Apply a preset to a session: merge into the session generation_config (Redis)
    + return the config to the client for input synchronization."""
    import json as _json
    from ..models import GenerationPreset
    from ..services.chat_service import chat_service
    from ..services.generation_presets import apply_preset_to_config

    try:
        data = _json.loads(request.body or '{}')
    except _json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    preset = GenerationPreset.objects.filter(id=data.get('preset_id')).first()
    if not preset:
        return JsonResponse({'success': False, 'error': 'Preset not found'}, status=404)

    session = chat_service.get_session(session_id)
    if not session:
        return JsonResponse({'success': False, 'error': 'Session not found'}, status=404)

    session = apply_preset_to_config(session, {'generation_config': preset.config})
    # Persist to Redis the same way update_rag_config does
    redis_key = f"chat:session:{session_id}"
    chat_service.redis_manager.redis.setex(redis_key, 86400,
                                           _json.dumps(session, default=str))
    return JsonResponse({'success': True, 'generation_config': session['generation_config'],
                         'preset': preset.name})
