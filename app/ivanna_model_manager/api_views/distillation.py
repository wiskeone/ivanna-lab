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



class DistillationQuestionSetViewSet(viewsets.ModelViewSet):
    """Manage distillation question sets."""
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
        Auto-create groups: splits the chunk list into windows.
        Expects JSON: {
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

        # Remove old groups if needed (optional)
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
        Start question generation for all groups of the set.
        Expects: teacher_model_id, optionally quantization and generation_config.
        """
        qset = self.get_object()
        if qset.status not in ['PENDING', 'FAILED']:
            return Response({'error': 'Cannot start: invalid status'}, status=400)

        teacher_model_id = request.data.get('teacher_model_id')
        if not teacher_model_id:
            return Response({'error': 'teacher_model_id required'}, status=400)

        quantization = request.data.get('quantization', 'int4')

        # Update the set config if provided
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

        # Save celery_task_id for tracking
        task = generate_questions_for_set_task.delay(qset.id, teacher_model_id, quantization)
        qset.celery_task_id = task.id
        qset.status = 'PENDING'  # PENDING is safer; the task itself switches to RUNNING
        qset.save(update_fields=['celery_task_id', 'status'])

        return Response({'success': True, 'task_id': task.id})

    @action(detail=True, methods=['get'])
    def questions(self, request, pk=None):
        """Returns the set's questions."""
        qset = self.get_object()
        questions = DistillationQuestion.objects.filter(group__question_set=qset).order_by('-created_at')
        serializer = DistillationQuestionSerializer(questions, many=True)
        return Response(serializer.data)


class DistillationChunkGroupViewSet(viewsets.ModelViewSet):
    """Manage chunk groups."""
    queryset = DistillationChunkGroup.objects.all()
    serializer_class = DistillationChunkGroupSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return super().get_queryset().filter(question_set__created_by=self.request.user.username)

    @action(detail=True, methods=['post'])
    def update_chunks(self, request, pk=None):
        """Update the group's chunk list."""
        group = self.get_object()
        chunk_ids = request.data.get('chunk_ids', [])
        if not chunk_ids:
            return Response({'error': 'chunk_ids required'}, status=400)
        group.chunk_ids = chunk_ids
        group.save()
        return Response({'success': True})

    @action(detail=False, methods=['post'])
    def bulk_update(self, request):
        """Bulk update groups."""
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
                # Create a new group
                DistillationChunkGroup.objects.create(
                    question_set=question_set,
                    name=g.get('name', ''),
                    chunk_ids=g.get('chunk_ids', []),
                    num_questions=g.get('num_questions', 3),
                    order=g.get('order', question_set.groups.count())
                )
        return Response({'success': True})


class DistillationQuestionViewSet(viewsets.ModelViewSet):
    """Manage distillation questions."""
    queryset = DistillationQuestion.objects.all()
    serializer_class = DistillationQuestionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return super().get_queryset().filter(group__question_set__created_by=self.request.user.username)

    @action(detail=True, methods=['post'])
    def update_answer(self, request, pk=None):
        """Manually update a question's answer."""
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
        """Bulk update answers."""
        updates = request.data.get('updates', [])  # list of {id, answer}
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
    """Manage answer generation jobs."""
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
            return Response({'error': _('Необходимы question_id, answer_job_id, teacher_model_id')}, status=400)

        try:
            job = DistillationAnswerJob.objects.get(id=answer_job_id, created_by=request.user.username)
            question = DistillationQuestion.objects.get(id=question_id)
        except (DistillationAnswerJob.DoesNotExist, DistillationQuestion.DoesNotExist):
            return Response({'error': _('Объект не найден')}, status=404)

        from ..tasks import regenerate_answer_task
        task = regenerate_answer_task.delay(question_id, answer_job_id, teacher_model_id, quantization, generation_config)
        return Response({'success': True, 'task_id': task.id})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def distillation_task_status(request, question_set_id):
    """Returns question generation task status."""
    qset = get_object_or_404(DistillationQuestionSet, id=question_set_id, created_by=request.user.username)
    # Get the latest 15 questions for dynamic display
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
    """Returns answer generation job status."""
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
    """Resume the task from a checkpoint."""
    qset = self.get_object()
    if not qset.checkpoint_data:
        return Response({'error': _('Нет чекпоинта для возобновления')}, status=400)
    if qset.status == 'RUNNING':
        return Response({'error': _('Задача уже выполняется')}, status=400)
    
    # Get the last saved parameters from checkpoint_data
    checkpoint = qset.checkpoint_data
    teacher_model_id = checkpoint.get('teacher_model_id')
    quantization = checkpoint.get('quantization', 'int4')
    if not teacher_model_id:
        return Response({'error': _('Не указана модель-учитель в чекпоинте')}, status=400)
    
    # Start the task with the checkpoint parameters
    task = generate_questions_for_set_task.delay(qset.id, teacher_model_id, quantization)
    qset.celery_task_id = task.id
    qset.status = 'PENDING'
    qset.save(update_fields=['celery_task_id', 'status'])
    
    return Response({'success': True, 'task_id': task.id})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def distillation_dataset_builder_data(request):
    """Returns dataset builder data: questions and answers for the selected jobs."""
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
        return Response({'error': _('Набор вопросов не найден')}, status=404)
    
    # Questions via group__question_set
    questions = DistillationQuestion.objects.filter(
        group__question_set=question_set
    ).order_by('id')
    
    # Fetch all answers for the selected jobs, sorted by -created_at
    answers_by_question = {}
    
    for job_id in answer_job_ids:
        try:
            job = DistillationAnswerJob.objects.get(
                id=job_id, 
                created_by=request.user.username
            )
        except DistillationAnswerJob.DoesNotExist:
            continue
        
        # Sort by -created_at so the latest answer comes first
        answers = DistillationAnswer.objects.filter(
            answer_job=job
        ).select_related('question').order_by('-created_at')
        
        for ans in answers:
            # Keep only the first seen answer per question (the latest one)
            if ans.question_id not in answers_by_question:
                answers_by_question[ans.question_id] = []
            
            answers_by_question[ans.question_id].append({
                'answer_id': ans.id,
                'answer_text': ans.answer_text,
                'job_name': job.name,
                'job_id': job.id,
                'created_at': ans.created_at.isoformat() if ans.created_at else None
            })
    
    # Build data for each question
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
            'answers': question_answers,  # all answer versions
            'latest_answer': question_answers[0] if question_answers else None,  # latest answer
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
        """Export the dataset to JSONL for fine-tuning."""
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
        """Create a dataset from the selected questions and answers"""
        name = request.data.get('name')
        description = request.data.get('description', '')
        items = request.data.get('items', [])
        
        if not name or not items:
            return Response({'error': _('name и items обязательны')}, status=400)
        
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
        """Export the set to JSONL for fine-tuning."""
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
        """Upload the dataset to the Hugging Face Hub."""
        instruction_set = self.get_object()
        from huggingface_hub import HfApi, Repository
        import tempfile
        import os

        api = HfApi()
        repo_id = request.data.get('repo_id')
        token = request.data.get('token')
        if not repo_id or not token:
            return Response({'error': _('Не указаны repo_id и token')}, status=400)

        # Create a temp dir and save the JSONL
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, f"{instruction_set.name}.jsonl")
            with open(file_path, 'w', encoding='utf-8') as f:
                for instr in instruction_set.instructions.all().order_by('order'):
                    line = json.dumps({
                        'instruction': instr.question,
                        'output': instr.answer
                    }, ensure_ascii=False)
                    f.write(line + '\n')
            # Upload to HF
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
        Sync the set's instructions: deletes old ones, creates new ones.
        All in a single transaction — no duplicates.
        """
        from django.db import transaction
        
        instruction_set = self.get_object()
        
        # Permission check
        if instruction_set.created_by != request.user.username and not request.user.is_staff:
            return Response({'error': _('Нет прав')}, status=403)
        
        instructions_data = request.data.get('instructions', [])
        
        if not isinstance(instructions_data, list):
            return Response({'error': _('instructions должен быть списком')}, status=400)
        
        try:
            with transaction.atomic():  # atomic transaction
                # 1. Delete ALL existing instructions of this set
                deleted_count, _ = ManualInstruction.objects.filter(
                    instruction_set=instruction_set
                ).delete()
                
                # 2. Create the new instructions
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
        """Start dataset import from HF."""
        dataset = self.get_object()
        if dataset.status not in ['PENDING', 'FAILED']:
            return Response({'error': _('Датасет уже обрабатывается или завершён')}, status=400)

        from ..tasks import import_huggingface_dataset_task
        task = import_huggingface_dataset_task.delay(dataset.id)
        dataset.celery_task_id = task.id
        dataset.status = 'PENDING'
        dataset.save()
        return Response({'success': True, 'task_id': task.id})
