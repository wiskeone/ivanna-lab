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
    from ..serializers import QuestionUploadSerializer
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
    from ..serializers import StartEvaluationSerializer
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
    from ..serializers import RatingSerializer
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
        return Response({'success': False, 'error': _('Файл датасета не найден')}, status=404)

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


@login_required
@require_http_methods(["PUT"])
def evaluation_dataset_question_update(request, dataset_id, question_id):
    """v5.0: обновление одного вопроса в JSONL-датасете (inline-редактирование)."""
    dataset = get_object_or_404(EvaluationDataset, id=dataset_id)
    if not dataset.file_path or not os.path.exists(dataset.file_path):
        return Response({'success': False, 'error': _('Файл датасета не найден')}, status=404)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return Response({'success': False, 'error': _('Неверный JSON')}, status=400)

    text = body.get('text', '').strip()
    if not text:
        return Response({'success': False, 'error': _('Текст вопроса обязателен')}, status=400)

    lines = []
    with open(dataset.file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                lines.append(item)
            except json.JSONDecodeError:
                continue

    qid = int(question_id)
    if qid < 1 or qid > len(lines):
        return Response({'success': False, 'error': _('Вопрос не найден')}, status=404)

    item = lines[qid - 1]
    item['instruction'] = text
    item['output'] = body.get('answer', item.get('output', item.get('answer', '')))
    item['category'] = body.get('category', item.get('category', ''))
    item['difficulty'] = body.get('difficulty', item.get('difficulty', 'medium'))
    lines[qid - 1] = item

    with open(dataset.file_path, 'w', encoding='utf-8') as f:
        for item in lines:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return Response({'success': True, 'id': qid, 'message': _('Вопрос обновлён')})



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
        return Response({'success': False, 'error': _('Файл не загружен')}, status=400)
    if not uploaded_file.name.endswith('.jsonl'):
        return Response({'success': False, 'error': _('Поддерживаются только файлы .jsonl')}, status=400)

    # Сохраняем файл
    file_path = dataset.file_path
    if not file_path:
        file_name = f"dataset_{dataset.id}.jsonl"
        file_path = os.path.join(settings.MEDIA_ROOT, 'evaluation_datasets', file_name)
        dataset.file_path = file_path

    with open(file_path, 'wb+') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)

    # Audit fix: only line counting happened here — dataset.questions stayed
    # empty (trainers/evaluator read questions), invalid lines were counted
    questions = []
    bad_lines = 0
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if not isinstance(item, dict):
                bad_lines += 1
                continue
            text = (item.get('instruction') or item.get('text')
                    or item.get('question') or '')
            answer = (item.get('output') or item.get('answer')
                      or item.get('response') or '')
            text, answer = str(text).strip(), str(answer).strip()
            if not text:
                bad_lines += 1
                continue
            questions.append({'id': idx, 'text': text, 'answer': answer})

    dataset.questions = questions
    dataset.total_questions = len(questions)
    dataset.save(update_fields=['questions', 'total_questions', 'updated_at'])

    return Response({'success': True, 'total': len(questions), 'skipped_lines': bad_lines})


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
