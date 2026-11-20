import io
import os
import json
import re
import time
import uuid
import hashlib
import logging
from datetime import datetime
from django.contrib.auth.models import User
from huggingface_hub import HfApi

from django.db.models import Q, Count, Sum, OuterRef, Subquery
from django.shortcuts import render, redirect, get_object_or_404
from django.utils.translation import gettext as _
from django.http import JsonResponse, HttpResponse, Http404
from django.contrib import messages
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, DetailView, CreateView, UpdateView, TemplateView
from django.urls import reverse, reverse_lazy
from django.views.decorators.http import require_POST, require_GET, require_http_methods
from django.core.paginator import Paginator
from django.core.exceptions import PermissionDenied
from django.conf import settings
from django.utils import timezone
from django.db import transaction

from ..models import (
    Experiment, ModelCheckpoint, ModelVersion, ModelMergeJob,
    InferenceSession, ModelEvaluation, ChatSession, ModelComparison,
    FineTuningJob, ModelDownloadJob, EvaluationDataset,
    ModelEvaluationTask, EvaluationResult,
    DocumentCollection, Document, DocumentChunk,
    BenchmarkRun,
    MoEExpertJob,
    EmbeddingModel, RAGSession, SearchQuery, FineTuningMethod,
    FineTuningJob, FineTuningCheckpoint, FineTuningEvaluation,
    FineTuningTemplate, UserProfile, TrainingMetric,
    ModelConversionJob, DistillationQuestionSet,
    DistillationChunkGroup, DistillationQuestion,
    DistillationAnswerJob, ModelCheckpoint, DocumentCollection,
    DistillationAnswer, DistillationDataset, ManualInstructionSet,
    ManualInstruction, 
)
from ..forms import (
    ExperimentForm, ModelCheckpointForm, ModelConfigForm,
    ModelMergeForm, InferenceTestForm, EvaluationForm,
    ModelUploadForm, EvaluationDatasetForm, QuestionUploadForm,
    CreateEvaluationTaskForm, RatingForm,
    MultipleDocumentUploadForm, DocumentUploadForm,
    DocumentEditForm, DocumentChunkEditForm, DocumentBulkActionForm,
    DocumentCollectionForm, FineTuningJobForm, FineTuningTemplateForm,
    FineTuningTemplateSelectForm, UserProfileForm, ModelConversionForm,
    EmbeddingModelForm, DistillationAnswerJobForm, LocalModelForm, 

)
from ..services.redis_model_service import redis_model_manager
from ..services.chat_service import chat_service
from ..services.download_service import download_service
from ..services.rag_service import RAGService
from ..tasks import (
    download_model_task, load_model_to_memory_task, unload_model_from_memory_task,
    merge_models_task, process_document_task, create_embeddings_for_collection_task,
    build_faiss_index_task, update_faiss_index_task, regenerate_document_chunks_task,
    run_evaluation_task, process_uploaded_dataset, sync_model_statuses_task,
    health_check_task, generate_questions_for_set_task,
    generate_answers_for_questions_task, HuggingFaceDataset, 
)

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)



# ============================================================================
# ОЦЕНКА МОДЕЛЕЙ (EVALUATION)
# ============================================================================
@login_required
def evaluation_dashboard(request):
    """Панель управления оценкой."""
    context = {
        'title': 'Оценка моделей',
        'total_datasets': EvaluationDataset.objects.count(),
        'total_tasks': ModelEvaluationTask.objects.count(),
        'active_tasks': ModelEvaluationTask.objects.filter(status__in=['PENDING', 'RUNNING']).count(),
        'completed_tasks': ModelEvaluationTask.objects.filter(status='COMPLETED').count(),
        'recent_tasks': ModelEvaluationTask.objects.select_related('dataset').order_by('-created_at')[:5],
        'recent_datasets': EvaluationDataset.objects.order_by('-created_at')[:5],
        'available_models': ModelCheckpoint.objects.filter(status='LOADED', is_active=True).order_by('name'),
    }
    return render(request, 'ivanna_model_manager/evaluation_dashboard.html', context)


class EvaluationDatasetListView(LoginRequiredMixin, ListView):
    model = EvaluationDataset
    template_name = 'ivanna_model_manager/evaluation_dataset_list.html'
    context_object_name = 'datasets'
    paginate_by = 20

    def get_queryset(self):
        qs = super().get_queryset()
        if self.request.GET.get('q'):
            qs = qs.filter(name__icontains=self.request.GET['q'])
        return qs.order_by('-created_at')


class EvaluationDatasetDetailView(LoginRequiredMixin, DetailView):
    model = EvaluationDataset
    template_name = 'ivanna_model_manager/evaluation_dataset_detail.html'
    context_object_name = 'dataset'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['related_tasks'] = self.object.evaluation_tasks.order_by('-created_at')[:10]
        context['sample_questions'] = self.object.questions[:10] if self.object.questions else []
        return context


@login_required
def upload_dataset_view(request):
    """Загрузка датасета из файла."""
    if request.method == 'POST':
        form = QuestionUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                uploaded = request.FILES['file']
                upload_dir = os.path.join(settings.MEDIA_ROOT, 'evaluation_datasets')
                os.makedirs(upload_dir, exist_ok=True)
                file_name = f"dataset_{int(time.time())}_{uploaded.name}"
                file_path = os.path.join(upload_dir, file_name)
                with open(file_path, 'wb+') as f:
                    for chunk in uploaded.chunks():
                        f.write(chunk)

                dataset = EvaluationDataset.objects.create(
                    name=form.cleaned_data['name'],
                    description=form.cleaned_data.get('description', ''),
                    file_path=file_path,
                    tags=form.cleaned_data.get('tags', []),
                    source_type='FILE',
                    created_by=request.user.username
                )
                process_uploaded_dataset.delay(file_path, dataset.id, form.cleaned_data['file_format'])
                messages.success(request, f'Датасет "{dataset.name}" создан, обработка запущена')
                return redirect('evaluation_dataset_detail', pk=dataset.id)
            except Exception as e:
                logger.error(f"Upload dataset error: {e}")
                messages.error(request, f'Ошибка: {e}')
    else:
        form = QuestionUploadForm()
    return render(request, 'ivanna_model_manager/evaluation_dataset_upload.html', {'form': form})


@login_required
def create_dataset_manual_view(request):
    """Создание датасета вручную."""
    if request.method == 'POST':
        form = EvaluationDatasetForm(request.POST)
        if form.is_valid():
            try:
                dataset = form.save(commit=False)
                dataset.created_by = request.user.username
                questions_json = request.POST.get('questions_json')
                if questions_json:
                    dataset.questions = json.loads(questions_json)
                    dataset.total_questions = len(dataset.questions)
                dataset.save()
                messages.success(request, f'Датасет "{dataset.name}" создан')
                return redirect('evaluation_dataset_detail', pk=dataset.id)
            except Exception as e:
                messages.error(request, f'Ошибка: {e}')
    else:
        form = EvaluationDatasetForm()
    return render(request, 'ivanna_model_manager/evaluation_dataset_create.html', {'form': form})


LARGE_DATASET_THRESHOLD = 10000  # датасеты больше этого считаем большими


@login_required
def edit_dataset_questions_view(request, pk):
    dataset = get_object_or_404(EvaluationDataset, pk=pk)

    # Убедимся, что файл существует
    if not dataset.file_path:
        file_name = f"dataset_{dataset.id}.jsonl"
        dataset.file_path = os.path.join(settings.MEDIA_ROOT, 'evaluation_datasets', file_name)
        dataset.save()
    file_path = dataset.file_path
    file_exists = os.path.exists(file_path)

    total_questions = dataset.total_questions

    # Определяем, большой ли датасет
    has_large_dataset = total_questions > LARGE_DATASET_THRESHOLD

    # Загружаем вопросы только для некрупных датасетов
    questions = []
    if not has_large_dataset and file_exists:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
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
        except Exception as e:
            logger.error(f"Error reading dataset file {file_path}: {e}")
            messages.warning(request, f'Ошибка чтения файла: {e}')
    elif not file_exists:
        logger.warning(f"Dataset file not found: {file_path}")
        messages.warning(request, _('Файл датасета не найден. Возможно, импорт не завершился.'))

    # Для GET-запроса
    if request.method == 'GET':
        # Для больших датасетов не передаём JSON, только предупреждение
        questions_json = json.dumps(questions, ensure_ascii=False, indent=2) if not has_large_dataset else ''
        context = {
            'dataset': dataset,
            'questions_json': questions_json,
            'total_questions': total_questions,
            'has_large_dataset': has_large_dataset,
            'file_path': file_path,
            'file_exists': file_exists,
        }
        return render(request, 'ivanna_model_manager/evaluation_dataset_edit.html', context)

    # Обработка POST (только для некрупных датасетов)
    if has_large_dataset:
        messages.error(request, _('Редактирование большого датасета через веб-интерфейс невозможно. Используйте экспорт/импорт.'))
        return redirect('evaluation_dataset_detail', pk=dataset.id)

    questions_json = request.POST.get('questions_json', '[]')
    try:
        new_questions = json.loads(questions_json)
        if not isinstance(new_questions, list):
            raise ValueError('Данные должны быть массивом')

        # Проверка структуры
        for q in new_questions:
            if 'text' not in q and 'instruction' not in q:
                raise ValueError('Каждый вопрос должен иметь поле "text" или "instruction"')
            if 'answer' not in q and 'output' not in q:
                raise ValueError('Каждый вопрос должен иметь поле "answer" или "output"')

        # Сохраняем в файл
        normalized = []
        with open(file_path, 'w', encoding='utf-8') as f:
            for i, q in enumerate(new_questions, 1):
                item = {
                    'instruction': q.get('text', q.get('instruction', '')),
                    'output': q.get('answer', q.get('output', '')),
                    'category': q.get('category', ''),
                    'difficulty': q.get('difficulty', 'medium')
                }
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
                # Audit fix: trainers/evaluator read dataset.questions, but the
                # edit view only rewrote the file — the JSONField diverged from
                # the file and training used stale data
                normalized.append({
                    'id': i,
                    'text': item['instruction'],
                    'answer': item['output'],
                    'category': item['category'],
                    'difficulty': item['difficulty'],
                })
        dataset.questions = normalized
        dataset.total_questions = len(normalized)
        dataset.save()

        messages.success(request, f'Датасет успешно обновлён. Сохранено {len(new_questions)} вопросов.')
        return redirect('evaluation_dataset_detail', pk=dataset.id)

    except Exception as e:
        messages.error(request, f'Ошибка при сохранении: {e}')
        logger.error(f"Error saving dataset {dataset.id}: {e}")
        return redirect('evaluation_dataset_edit', pk=dataset.id)


def export_evaluation_dataset(request, pk):
    dataset = get_object_or_404(EvaluationDataset, pk=pk)
    if not dataset.file_path or not os.path.exists(dataset.file_path):
        messages.error(request, _('Файл датасета не найден'))
        return redirect('evaluation_dataset_detail', pk=pk)
    response = HttpResponse(content_type='application/jsonl')
    response['Content-Disposition'] = f'attachment; filename="{dataset.name}.jsonl"'
    with open(dataset.file_path, 'r', encoding='utf-8') as f:
        response.write(f.read())
    return response


@login_required
def evaluation_task_list_view(request):
    """Список задач оценки."""
    tasks = ModelEvaluationTask.objects.select_related('dataset').prefetch_related('evaluation_models')
    if request.GET.get('status'):
        tasks = tasks.filter(status=request.GET['status'])
    if request.GET.get('q'):
        tasks = tasks.filter(Q(name__icontains=request.GET['q']) | Q(description__icontains=request.GET['q']))
    paginator = Paginator(tasks.order_by('-created_at'), 20)
    page = paginator.get_page(request.GET.get('page'))
    return render(request, 'ivanna_model_manager/evaluation_task_list.html', {
        'page_obj': page,
        'status_choices': ModelEvaluationTask.STATUS_CHOICES,
    })


@login_required
def evaluation_task_detail_view(request, pk):
    """Детали задачи оценки."""
    task = get_object_or_404(ModelEvaluationTask.objects.select_related('dataset'), pk=pk)
    results = EvaluationResult.objects.filter(task=task).select_related('model')
    results_by_model = {}
    for r in results:
        results_by_model.setdefault(r.model.id, {'model': r.model, 'results': []})['results'].append(r)
    
    # Проверяем, активна ли Celery задача
    task_active = False
    if task.celery_task_id and task.status in ['RUNNING', 'PENDING']:
        from celery.result import AsyncResult
        res = AsyncResult(task.celery_task_id)
        task_active = res.state in ['PENDING', 'STARTED', 'RETRY']
    
    context = {
        'task': task,
        'results_by_model': results_by_model,
        'rating_form': RatingForm(),
        'task_results': task.results or {},
        'task_ratings': task.ratings or {},
        'summary_metrics': task.summary_metrics or {},
        'models': task.evaluation_models.all(),
        'task_active': task_active,  # Добавляем флаг
    }
    return render(request, 'ivanna_model_manager/evaluation_task_detail.html', context)


@login_required
@require_POST
def force_reset_stuck_task(request, task_id):
    """Принудительный сброс зависшей задачи (без перезапуска)"""
    task = get_object_or_404(ModelEvaluationTask, id=task_id)
    
    # Проверяем права
    if task.created_by != request.user.username and not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    # Сбрасываем статус
    old_status = task.status
    task.status = 'FAILED'
    task.error_message = f"Task was stuck in {old_status} state and manually reset"
    task.completed_at = timezone.now()
    task.save()
    
    # Отменяем Celery задачу если есть
    if task.celery_task_id:
        from celery.result import AsyncResult
        AsyncResult(task.celery_task_id).revoke(terminate=True)
    
    logger.info(f"🔄 Force reset stuck task {task_id} from {old_status} to FAILED")
    
    return JsonResponse({
        'success': True, 
        'message': 'Task has been reset',
        'old_status': old_status
    })


@login_required
def create_evaluation_task_view(request):
    """
    Создание задачи оценки.
    Теперь с опцией RAG: можно выбрать коллекции и использовать RAG-поиск.
    Использует transaction.on_commit для гарантии, что задача Celery
    запускается только после полного сохранения объекта в БД.
    """
    logger = logging.getLogger(__name__)

    if request.method == 'POST':
        form = CreateEvaluationTaskForm(request.POST)
        
        # Извлекаем RAG-параметры из POST
        use_rag = request.POST.get('use_rag') == 'on'
        collection_ids = request.POST.getlist('rag_collections')
        # Преобразуем в int, отбрасываем нечисловые
        collection_ids = [int(cid) for cid in collection_ids if cid.isdigit()]
        
        # ЛОГИРОВАНИЕ
        logger.info(f"Creating evaluation task: use_rag={use_rag}, collection_ids={collection_ids}")
        
        if form.is_valid():
            try:
                # Получаем значение квантизации (может быть пустым)
                quantization = form.cleaned_data.get('quantization') or None

                task = ModelEvaluationTask.objects.create(
                    name=form.cleaned_data['name'],
                    description=form.cleaned_data.get('description', ''),
                    dataset=form.cleaned_data['dataset'],
                    generation_config={
                        'max_tokens': form.cleaned_data['max_tokens'],
                        'temperature': form.cleaned_data['temperature'],
                        'top_p': form.cleaned_data['top_p'],
                        'top_k': form.cleaned_data['top_k'],
                        'repetition_penalty': form.cleaned_data['repetition_penalty'],
                        'quantization': quantization,
                        'use_rag': use_rag,
                        'rag_collections': collection_ids,
                    },
                    created_by=request.user.username,
                )
                task.evaluation_models.set(form.cleaned_data['models'])
                task.save()

                # Запускаем задачу после коммита транзакции
                transaction.on_commit(lambda: run_evaluation_task.delay(
                    task.id,
                    use_rag=use_rag,
                    collection_ids=collection_ids
                ))

                messages.success(request, f'Задача "{task.name}" создана и запущена в фоновом режиме')
                return redirect('evaluation_task_detail', pk=task.id)

            except Exception as e:
                logger.error(f"Create evaluation task error: {e}", exc_info=True)
                messages.error(request, f'Ошибка при создании задачи: {e}')
    else:
        form = CreateEvaluationTaskForm()

    # Подготовка данных для шаблона
    collections = DocumentCollection.objects.filter(
        Q(owner_id=str(request.user.id)) | Q(is_public=True),
        is_active=True
    ).order_by('name')

    context = {
        'title': 'Создание задачи оценки',
        'form': form,
        'available_datasets': EvaluationDataset.objects.filter(total_questions__gt=0),
        'available_models': ModelCheckpoint.objects.filter(
            status='LOADED', is_active=True
        ).order_by('name'),
        'rag_collections': collections,
    }
    return render(request, 'ivanna_model_manager/evaluation_task_create.html', context)


@login_required
def evaluation_task_progress_view(request, task_id):
    """Прогресс выполнения задачи (AJAX)."""
    try:
        task = ModelEvaluationTask.objects.get(id=task_id)
        from celery.result import AsyncResult
        celery_status = {}
        if task.celery_task_id:
            res = AsyncResult(task.celery_task_id)
            celery_status = {
                'state': res.state,
                'status': res.status,
                'info': res.info if hasattr(res, 'info') else None,
            }
        return JsonResponse({
            'success': True,
            'task_id': task_id,
            'status': task.status,
            'progress': task.progress,
            'current_step': task.current_step,
            'celery': celery_status,
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
@require_POST
def rate_response_view(request):
    """Оценка ответа модели (AJAX)."""
    try:
        task_id = request.POST.get('task_id')
        model_id = request.POST.get('model_id')
        question_id = request.POST.get('question_id')
        rating = request.POST.get('rating')
        comment = request.POST.get('comment', '')
        if not all([task_id, model_id, question_id, rating]):
            return JsonResponse({'success': False, 'error': 'Missing fields'})

        task = ModelEvaluationTask.objects.get(id=task_id)
        task.add_rating(model_id, question_id, int(rating), request.user.username)

        result = EvaluationResult.objects.filter(
            task=task,
            model_id=model_id,
            question_id=question_id
        ).first()
        if result:
            result.manual_rating = int(rating)
            result.rating_comment = comment
            result.rated_by = request.user.username
            result.rated_at = timezone.now()
            result.save()

        summary = task.calculate_summary_metrics()
        return JsonResponse({'success': True, 'summary_metrics': summary})
    except Exception as e:
        logger.error(f"Rate response error: {e}")
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def compare_models_view(request, task_id):
    """Сравнение ответов моделей (страница)."""
    from ..models import ModelEvaluationTask, EvaluationResult
    import logging
    logger = logging.getLogger(__name__)

    task = get_object_or_404(ModelEvaluationTask, pk=task_id)

    if task.status != 'COMPLETED':
        messages.warning(request, _('Задача ещё не завершена'))
        return redirect('evaluation_task_detail', pk=task_id)

    questions = task.dataset.questions if task.dataset else []
    models = list(task.evaluation_models.all())
    results = task.results or {}
    ratings = task.ratings or {}

    if not models or not questions:
        return render(request, 'ivanna_model_manager/evaluation_compare.html', {
            'task': task,
            'questions_with_answers': [],
            'models': models,
            'question_ratings': {},
            'summary_stats': {},
            'no_data': True
        })

    # Загружаем все EvaluationResult для этой задачи
    eval_results = EvaluationResult.objects.filter(task=task).select_related('model')
    er_dict = {}
    for er in eval_results:
        mid = er.model_id
        qid = er.question_id
        er_dict[(mid, qid)] = er

    questions_with_answers = []
    for q in questions:
        q_id = str(q.get('id', ''))
        q_text = q.get('text', '')
        if not q_id or not q_text:
            continue

        model_responses = []
        for m in models:
            m_id = str(m.id)
            # Данные из task.results
            resp = results.get(m_id, {}).get(q_id, {})
            response_text = resp.get('response', '')
            rag_used = resp.get('rag_used', False)
            rag_sources = resp.get('rag_sources_count', 0)
            tokens_used = resp.get('tokens_used', 0)
            response_time = resp.get('response_time', 0)

            # Если есть EvaluationResult, используем его данные (приоритет)
            er = er_dict.get((m.id, int(q_id)))
            if er:
                if not response_text and er.model_response:
                    response_text = er.model_response
                if er.metadata:
                    rag_used = er.metadata.get('rag_used', rag_used)
                    rag_sources = er.metadata.get('rag_sources_count', rag_sources)
                tokens_used = er.tokens_used
                response_time = er.response_time

            if response_text is None:
                response_text = ''

            model_responses.append({
                'model': m,
                'response': response_text,
                'tokens_used': tokens_used,
                'response_time': response_time,
                'rag_used': rag_used,
                'rag_sources': rag_sources,
            })

        questions_with_answers.append({
            'id': q_id,
            'text': q_text,
            'model_responses': model_responses,
        })

    # Формируем словарь рейтингов по вопросам
    question_ratings = {}
    for q in questions:
        q_id = str(q.get('id'))
        ratings_for_question = []
        for model in models:
            rating_info = ratings.get(str(model.id), {}).get(q_id)
            if rating_info and isinstance(rating_info, dict):
                rating = rating_info.get('rating')
                if rating is not None:
                    ratings_for_question.append(rating)
            elif rating_info and isinstance(rating_info, (int, float)):
                ratings_for_question.append(rating_info)
        if ratings_for_question:
            avg_rating = sum(ratings_for_question) / len(ratings_for_question)
            question_ratings[q_id] = {
                'avg': avg_rating,
                'count': len(ratings_for_question)
            }
        else:
            question_ratings[q_id] = None

    # Сводная статистика по моделям
    summary_stats = {}
    if models and questions_with_answers:
        avg_times = []
        avg_tokens = []
        for model in models:
            times = []
            tokens = []
            for qa in questions_with_answers:
                for resp in qa['model_responses']:
                    if resp['model'].id == model.id:
                        if resp['response_time']:
                            times.append(resp['response_time'])
                        if resp['tokens_used']:
                            tokens.append(resp['tokens_used'])
            avg_times.append(sum(times) / len(times) if times else 0)
            avg_tokens.append(sum(tokens) / len(tokens) if tokens else 0)

        summary_stats = {
            'avg_time': avg_times,
            'avg_tokens': avg_tokens,
        }

    context = {
        'task': task,
        'questions_with_answers': questions_with_answers,
        'models': models,
        'question_ratings': question_ratings,
        'summary_stats': summary_stats,
        'no_data': False,
    }
    return render(request, 'ivanna_model_manager/evaluation_compare.html', context)


@login_required
@require_POST
def restart_evaluation_task(request, task_id):
    task = get_object_or_404(ModelEvaluationTask, id=task_id)
    
    # Проверяем права
    if task.created_by != request.user.username and not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)
    
    # Определяем режим рестарта
    resume = request.POST.get('resume') == 'true'
    
    if resume and task.metadata.get('checkpoint'):
        # Возобновляем с чекпоинта
        task.status = 'PENDING'  # Celery начнет с чекпоинта
        task.error_message = ''
        logger.info(f"🔄 Resuming task {task_id} from checkpoint")
    else:
        # Полный рестарт
        task.status = 'PENDING'
        task.progress = 0
        task.current_step = ''
        task.results = {}
        task.ratings = {}
        task.summary_metrics = {}
        task.error_message = ''
        task.started_at = None
        task.completed_at = None
        task.metadata = {}  # очищаем метаданные
        task.save()
        
        # Удаляем старые результаты
        from ..models import EvaluationResult
        deleted, _ = EvaluationResult.objects.filter(task=task).delete()
        logger.info(f"Deleted {deleted} old evaluation results for task {task_id}")
    
    # Отменяем старую задачу
    if task.celery_task_id:
        from celery.result import AsyncResult
        AsyncResult(task.celery_task_id).revoke(terminate=True)
    
    # Запускаем новую
    celery_task = run_evaluation_task.delay(task.id)
    task.celery_task_id = celery_task.id
    task.save()
    
    return JsonResponse({
        'success': True, 
        'message': 'Task resumed from checkpoint' if resume else 'Task restarted',
        'resumed': resume
    })


@login_required
@require_POST
def delete_evaluation_task(request, task_id):
    task = get_object_or_404(ModelEvaluationTask, id=task_id)
    
    # Отменяем Celery задачу, если она ещё выполняется
    if task.celery_task_id and task.status == 'RUNNING':
        from celery.result import AsyncResult
        AsyncResult(task.celery_task_id).revoke(terminate=True)
    
    task_name = task.name
    task.delete()
    return JsonResponse({'success': True, 'message': f'Задача "{task_name}" удалена'})


@login_required
def get_model_results(request, task_id, model_id):
    """Детальные результаты конкретной модели (AJAX)."""
    task = get_object_or_404(ModelEvaluationTask, id=task_id)
    model = get_object_or_404(ModelCheckpoint, id=model_id)
    results = EvaluationResult.objects.filter(task=task, model=model).order_by('question_id')
    data = [{
        'question_id': r.question_id,
        'question_text': r.question_text,
        'model_response': r.model_response,
        'response_time': r.response_time,
        'tokens_used': r.tokens_used,
        'manual_rating': r.manual_rating,
    } for r in results]
    return JsonResponse({'success': True, 'results': data})
