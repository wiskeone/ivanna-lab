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



class DistillationQuestionSetListView(LoginRequiredMixin, ListView):
    model = DistillationQuestionSet
    template_name = 'ivanna_model_manager/distillation/question_set_list.html'
    context_object_name = 'question_sets'
    paginate_by = 20

    def get_queryset(self):
        return DistillationQuestionSet.objects.filter(
            created_by=self.request.user.username
        ).order_by('-created_at')


class DistillationQuestionSetCreateView(LoginRequiredMixin, CreateView):
    model = DistillationQuestionSet
    template_name = 'ivanna_model_manager/distillation/question_set_form.html'
    fields = ['name', 'description', 'collection', 'default_generation_config']
    success_url = reverse_lazy('distillation_question_set_list')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['collections'] = DocumentCollection.objects.filter(
            Q(owner_id=str(self.request.user.id)) | Q(is_public=True),
            is_active=True
        )
        context['available_models'] = ModelCheckpoint.objects.filter(is_active=True)
        return context

    def form_valid(self, form):
        form.instance.created_by = self.request.user.username
        return super().form_valid(form)


class DistillationQuestionSetDetailView(LoginRequiredMixin, DetailView):
    model = DistillationQuestionSet
    template_name = 'ivanna_model_manager/distillation/question_set_detail.html'
    context_object_name = 'question_set'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['groups'] = self.object.groups.all().order_by('order')
        context['available_models'] = ModelCheckpoint.objects.filter(is_active=True)
        context['available_quantizations'] = ModelCheckpoint.QUANTIZATION_CHOICES
        # Add questions with pagination
        all_questions = DistillationQuestion.objects.filter(group__question_set=self.object).order_by('-created_at')
        paginator = Paginator(all_questions, 20)  # 20 per page
        page_number = self.request.GET.get('page', 1)
        page_obj = paginator.get_page(page_number)
        context['page_obj'] = page_obj
        return context


class DistillationQuestionSetUpdateView(LoginRequiredMixin, UpdateView):
    model = DistillationQuestionSet
    template_name = 'ivanna_model_manager/distillation/question_set_form.html'
    fields = ['name', 'description', 'collection', 'default_generation_config']
    # Remove success_url from class attributes
    # success_url = reverse_lazy('distillation_question_set_detail')  # ❌ Remove this line

    def get_queryset(self):
        return DistillationQuestionSet.objects.filter(created_by=self.request.user.username)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['collections'] = DocumentCollection.objects.filter(
            Q(owner_id=str(self.request.user.id)) | Q(is_public=True),
            is_active=True
        )
        context['available_models'] = ModelCheckpoint.objects.filter(is_active=True)
        return context

    def get_success_url(self):
        return reverse('distillation_question_set_detail', args=[self.object.id])


@login_required
def distillation_question_set_edit_groups(request, pk):
    """Редактирование групп чанков набора."""
    question_set = get_object_or_404(DistillationQuestionSet, pk=pk, created_by=request.user.username)

    if request.method == 'POST':
        # Обработка действий: добавить группу, удалить, обновить чанки
        action = request.POST.get('action')
        if action == 'add_group':
            group = DistillationChunkGroup.objects.create(
                question_set=question_set,
                name=request.POST.get('group_name', ''),
                chunk_ids=request.POST.getlist('chunk_ids'),
                num_questions=int(request.POST.get('num_questions', 3)),
                order=question_set.groups.count()
            )
        elif action == 'delete_group':
            group_id = request.POST.get('group_id')
            if group_id:
                DistillationChunkGroup.objects.filter(id=group_id, question_set=question_set).delete()
        elif action == 'update_group':
            group_id = request.POST.get('group_id')
            if group_id:
                group = get_object_or_404(DistillationChunkGroup, id=group_id, question_set=question_set)
                group.name = request.POST.get('group_name', group.name)
                group.chunk_ids = request.POST.getlist('chunk_ids')
                group.num_questions = int(request.POST.get('num_questions', group.num_questions))
                group.save()
        return redirect('distillation_question_set_detail', pk=question_set.id)

    # GET: показываем форму
    all_chunks = DocumentChunk.objects.filter(
        document__collection=question_set.collection,
        is_active=True
    ).order_by('document_id', 'chunk_index')
    context = {
        'question_set': question_set,
        'groups': question_set.groups.all().order_by('order'),
        'all_chunks': all_chunks,
        'available_quantizations': ModelCheckpoint.QUANTIZATION_CHOICES,
    }
    return render(request, 'ivanna_model_manager/distillation/question_set_edit_groups.html', context)


@login_required
@require_POST
def distillation_start_question_generation(request, pk):
    """Запуск генерации вопросов для набора (через Celery)."""
    question_set = get_object_or_404(DistillationQuestionSet, pk=pk, created_by=request.user.username)

    teacher_model_id = request.POST.get('teacher_model_id')
    quantization = request.POST.get('quantization', 'int4')
    if not teacher_model_id:
        messages.error(request, _('Не выбрана модель-учитель'))
        return redirect('distillation_question_set_detail', pk=question_set.id)

    # Обновляем конфигурацию, если передана
    gen_config = {}
    if request.POST.get('prompt_template'):
        gen_config['prompt_template'] = request.POST['prompt_template']
    if request.POST.get('max_tokens'):
        gen_config['max_tokens'] = int(request.POST['max_tokens'])
    if request.POST.get('temperature'):
        gen_config['temperature'] = float(request.POST['temperature'])
    if gen_config:
        question_set.default_generation_config.update(gen_config)
        question_set.save(update_fields=['default_generation_config'])

    task = generate_questions_for_set_task.delay(question_set.id, teacher_model_id, quantization)
    messages.success(request, f'Задача на генерацию вопросов запущена (ID: {task.id})')
    return redirect('distillation_question_set_detail', pk=question_set.id)


class DistillationQuestionListView(LoginRequiredMixin, ListView):
    """Список вопросов (можно фильтровать по набору)."""
    model = DistillationQuestion
    template_name = 'ivanna_model_manager/distillation/question_list.html'
    context_object_name = 'questions'
    paginate_by = 50

    def get_queryset(self):
        qs = DistillationQuestion.objects.filter(group__question_set__created_by=self.request.user.username)
        question_set_id = self.request.GET.get('question_set')
        if question_set_id:
            qs = qs.filter(group__question_set_id=question_set_id)
        status = self.request.GET.get('status')
        if status:
            qs = qs.filter(status=status)
        return qs.order_by('-created_at')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['question_sets'] = DistillationQuestionSet.objects.filter(created_by=self.request.user.username)
        return context


@login_required
def distillation_question_edit(request, pk):
    """Редактирование вопроса (вручную изменить текст вопроса и ответа)."""
    question = get_object_or_404(DistillationQuestion, pk=pk, group__question_set__created_by=request.user.username)

    if request.method == 'POST':
        question.question_text = request.POST.get('question_text', question.question_text)
        question.answer_text = request.POST.get('answer_text', question.answer_text)
        if 'status' in request.POST:
            question.status = request.POST['status']
        question.save()
        messages.success(request, _('Вопрос сохранён'))
        return redirect('distillation_question_list')

    return render(request, 'ivanna_model_manager/distillation/question_edit.html', {'question': question})


class DistillationAnswerJobListView(LoginRequiredMixin, ListView):
    model = DistillationAnswerJob
    template_name = 'ivanna_model_manager/distillation/answer_job_list.html'
    context_object_name = 'jobs'
    paginate_by = 20

    def get_queryset(self):
        return DistillationAnswerJob.objects.filter(created_by=self.request.user.username).order_by('-created_at')


class DistillationAnswerJobCreateView(LoginRequiredMixin, CreateView):
    model = DistillationAnswerJob
    form_class = DistillationAnswerJobForm
    template_name = 'ivanna_model_manager/distillation/answer_job_form.html'
    success_url = reverse_lazy('distillation_answer_job_list')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['question_sets'] = DistillationQuestionSet.objects.filter(created_by=self.request.user.username)
        context['available_models'] = ModelCheckpoint.objects.filter(is_active=True)
        context['default_generation_config'] = {
            'max_tokens': 1024,
            'temperature': 0.6,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.0,
            'prompt_template': ''
        }
        return context

    def form_valid(self, form):
        # Устанавливаем created_by
        form.instance.created_by = self.request.user.username

        # Собираем generation_config из POST-параметров
        generation_config = {
            'max_tokens': int(self.request.POST.get('max_tokens', 1024)),
            'temperature': float(self.request.POST.get('temperature', 0.6)),
            'top_p': float(self.request.POST.get('top_p', 0.9)),
            'top_k': int(self.request.POST.get('top_k', 50)),
            'repetition_penalty': float(self.request.POST.get('repetition_penalty', 1.0)),
        }
        prompt_template = self.request.POST.get('prompt_template', '').strip()
        if prompt_template:
            generation_config['prompt_template'] = prompt_template
        form.instance.generation_config = generation_config

        # Обработка выбора вопросов
        question_set = form.cleaned_data.get('question_set')
        question_ids_str = self.request.POST.get('question_ids_str', '')   # ключ должен совпадать с именем поля в шаблоне

        if question_set:
            questions = DistillationQuestion.objects.filter(group__question_set=question_set)   # исправлено
            form.instance.question_ids = list(questions.values_list('id', flat=True))
            form.instance.planned_count = questions.count()
            form.instance.question_set = question_set   # убедимся, что question_set установлен
        elif question_ids_str:
            question_ids = [int(id.strip()) for id in question_ids_str.split(',') if id.strip().isdigit()]
            form.instance.question_ids = question_ids
            form.instance.planned_count = len(question_ids)
            form.instance.question_set = None
        else:
            form.add_error(None, 'Необходимо выбрать набор вопросов или указать ID конкретных вопросов.')
            return self.form_invalid(form)

        # Сохраняем объект
        response = super().form_valid(form)
        messages.success(self.request, f'Задача "{self.object.name}" успешно создана. Генерация ответов начнётся после запуска.')
        return response

    def form_invalid(self, form):
        logger.error("Form errors: %s", form.errors)
        messages.error(self.request, 'Ошибка при создании задачи. Проверьте введённые данные.')
        return super().form_invalid(form)


class DistillationAnswerJobDetailView(LoginRequiredMixin, DetailView):
    model = DistillationAnswerJob
    template_name = 'ivanna_model_manager/distillation/answer_job_detail.html'
    context_object_name = 'job'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Получаем вопросы
        if self.object.question_ids:
            questions = DistillationQuestion.objects.filter(id__in=self.object.question_ids)
        elif self.object.question_set:
            questions = DistillationQuestion.objects.filter(group__question_set=self.object.question_set)
        else:
            questions = DistillationQuestion.objects.none()
        
        # Получаем все ответы для этой задачи, отсортированные по убыванию даты
        answers = DistillationAnswer.objects.filter(
            answer_job=self.object
        ).select_related('question').order_by('-created_at')
        
        # Группируем ответы по вопросам
        answers_by_question = {}
        for ans in answers:
            if ans.question_id not in answers_by_question:
                answers_by_question[ans.question_id] = []
            answers_by_question[ans.question_id].append(ans)
        
        qa_pairs = []
        for q in questions:
            question_answers = answers_by_question.get(q.id, [])
            latest_answer = question_answers[0] if question_answers else None
            qa_pairs.append({
                'question': q,
                'answer': latest_answer,
                'all_answers': question_answers,  # для истории версий
                'has_multiple_answers': len(question_answers) > 1
            })
        
        context['qa_pairs'] = qa_pairs
        context['total_questions'] = questions.count()
        context['answered_count'] = len(answers_by_question)
        context['available_models'] = ModelCheckpoint.objects.filter(is_active=True).order_by('display_name')
        
        return context


@login_required
@require_POST
def distillation_start_answer_generation(request, pk):
    """Запуск генерации ответов для задачи."""
    job = get_object_or_404(DistillationAnswerJob, pk=pk, created_by=request.user.username)

    teacher_model_id = request.POST.get('teacher_model_id')
    quantization = request.POST.get('quantization', 'int4')
    if not teacher_model_id:
        messages.error(request, _('Не выбрана модель-учитель'))
        return redirect('distillation_answer_job_detail', pk=job.id)

    # Обновляем конфигурацию
    gen_config = {}
    if request.POST.get('prompt_template'):
        gen_config['prompt_template'] = request.POST['prompt_template']
    if request.POST.get('max_tokens'):
        gen_config['max_tokens'] = int(request.POST['max_tokens'])
    if request.POST.get('temperature'):
        gen_config['temperature'] = float(request.POST['temperature'])
    if gen_config:
        job.generation_config.update(gen_config)
        job.save(update_fields=['generation_config'])

    task = generate_answers_for_questions_task.delay(job.id, teacher_model_id, quantization)
    messages.success(request, f'Задача на генерацию ответов запущена (ID: {task.id})')
    return redirect('distillation_answer_job_detail', pk=job.id)


@login_required
@require_POST
def distillation_resume_question_generation(request, pk):
    question_set = get_object_or_404(DistillationQuestionSet, pk=pk, created_by=request.user.username)
    if not question_set.checkpoint_data:
        messages.error(request, _('Нет чекпоинта для возобновления'))
        return redirect('distillation_question_set_detail', pk=question_set.id)

    teacher_model_id = question_set.checkpoint_data.get('teacher_model_id')
    quantization = question_set.checkpoint_data.get('quantization', 'int4')
    if not teacher_model_id:
        messages.error(request, _('В чекпоинте не указана модель-учитель'))
        return redirect('distillation_question_set_detail', pk=question_set.id)

    task = generate_questions_for_set_task.delay(question_set.id, teacher_model_id, quantization)
    question_set.celery_task_id = task.id
    question_set.status = 'PENDING'
    question_set.save(update_fields=['celery_task_id', 'status'])
    messages.success(request, f'Задача возобновлена (ID: {task.id})')
    return redirect('distillation_question_set_detail', pk=question_set.id)


@login_required
@require_POST
def distillation_resume_answer_generation(request, pk):
    """Возобновление задачи генерации ответов из чекпоинта."""
    job = get_object_or_404(DistillationAnswerJob, pk=pk, created_by=request.user.username)

    if not job.checkpoint_data:
        messages.error(request, _('Нет чекпоинта для возобновления'))
        return redirect('distillation_answer_job_detail', pk=job.id)

    if job.status == 'RUNNING':
        messages.error(request, _('Задача уже выполняется'))
        return redirect('distillation_answer_job_detail', pk=job.id)

    # Берём параметры из чекпоинта (модель-учитель, квантизация)
    teacher_model_id = job.checkpoint_data.get('teacher_model_id')
    quantization = job.checkpoint_data.get('quantization', 'int4')
    if not teacher_model_id:
        messages.error(request, _('В чекпоинте не указана модель-учитель'))
        return redirect('distillation_answer_job_detail', pk=job.id)

    # Запускаем задачу (она сама продолжит с сохранённого чекпоинта)
    task = generate_answers_for_questions_task.delay(job.id, teacher_model_id, quantization)
    job.celery_task_id = task.id
    job.status = 'PENDING'  # задача сама переведёт в RUNNING
    job.save(update_fields=['celery_task_id', 'status'])

    messages.success(request, f'Задача на генерацию ответов возобновлена (ID: {task.id})')
    return redirect('distillation_answer_job_detail', pk=job.id)


@login_required
@require_POST
def distillation_delete_answer_job(request, pk):
    """Удаление задачи генерации ответов."""
    job = get_object_or_404(DistillationAnswerJob, pk=pk, created_by=request.user.username)
    DistillationAnswer.objects.filter(answer_job=job).delete()
    job.delete()
    return JsonResponse({'success': True, 'message': 'Job deleted'})


@login_required
@require_POST
def distillation_restart_answer_job(request, pk):
    """Перезапуск задачи: удаление всех ответов и сброс статуса."""
    job = get_object_or_404(DistillationAnswerJob, pk=pk, created_by=request.user.username)
    deleted_count = DistillationAnswer.objects.filter(answer_job=job).count()
    DistillationAnswer.objects.filter(answer_job=job).delete()
    job.status = 'PENDING'
    job.progress = 0.0
    job.current_step = ''
    job.processed_count = 0
    job.checkpoint_data = {}
    job.logs += f"[{timezone.now()}] Задача перезапущена пользователем {request.user.username}. Удалено ответов: {deleted_count}\n"
    job.save()
    return JsonResponse({'success': True, 'message': 'Job restarted'})


class DistillationDatasetListView(LoginRequiredMixin, ListView):
    template_name = 'ivanna_model_manager/distillation/dataset_list.html'
    context_object_name = 'datasets'
    paginate_by = 20

    def get_queryset(self):
        # Объединяем DistillationDataset и ManualInstructionSet
        distillation_datasets = DistillationDataset.objects.filter(created_by=self.request.user.username)
        manual_sets = ManualInstructionSet.objects.filter(created_by=self.request.user.username)

        # Преобразуем в список словарей с общими полями
        combined = []
        for ds in distillation_datasets:
            combined.append({
                'id': ds.id,
                'name': ds.name,
                'description': ds.description,
                'created_at': ds.created_at,
                'type': 'distillation',
                'item_count': len(ds.items) if ds.items else 0,
                'url_detail': reverse('distillation_dataset_detail', args=[ds.id]),
                'url_export': reverse('distillation_dataset_export', args=[ds.id]),
            })
        for ms in manual_sets:
            combined.append({
                'id': ms.id,
                'name': ms.name,
                'description': ms.description,
                'created_at': ms.created_at,
                'type': 'manual',
                'item_count': ms.instructions.count(),
                'url_detail': reverse('manual_instruction_builder') + f'?set_id={ms.id}',
                'url_export': reverse('manual_instruction_set_export', args=[ms.id]),
            })

        # Сортировка
        sort = self.request.GET.get('sort', '-created_at')
        reverse_sort = sort.startswith('-')
        key_field = sort.lstrip('-')
        combined.sort(key=lambda x: x[key_field], reverse=reverse_sort)

        # Поиск по имени
        q = self.request.GET.get('q', '')
        if q:
            combined = [d for d in combined if q.lower() in d['name'].lower()]

        return combined

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['total_datasets'] = len(self.get_queryset())
        return context


class DistillationDatasetDetailView(LoginRequiredMixin, DetailView):
    model = DistillationDataset
    template_name = 'ivanna_model_manager/distillation/dataset_detail.html'
    context_object_name = 'dataset'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Добавим возможность фильтрации? Пока ничего не добавляем
        return context


class DistillationDatasetBuilderView(LoginRequiredMixin, TemplateView):
    template_name = 'ivanna_model_manager/distillation/dataset_builder.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['question_sets'] = DistillationQuestionSet.objects.filter(created_by=self.request.user.username).order_by('-created_at')
        context['answer_jobs'] = DistillationAnswerJob.objects.filter(created_by=self.request.user.username).order_by('-created_at')
        return context


@login_required
def distillation_dataset_export(request, pk):
    dataset = get_object_or_404(DistillationDataset, pk=pk, created_by=request.user.username)
    response = HttpResponse(content_type='application/jsonl')
    response['Content-Disposition'] = f'attachment; filename="{dataset.name}.jsonl"'
    for item in dataset.items:
        line = json.dumps({
            'instruction': item['question_text'],
            'output': item['answer_text']
        }, ensure_ascii=False)
        response.write(line + '\n')
    return response


class ManualInstructionBuilderView(LoginRequiredMixin, TemplateView):
    template_name = 'ivanna_model_manager/manual_instruction_builder.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['collections'] = DocumentCollection.objects.filter(
            Q(owner_id=str(self.request.user.id)) | Q(is_public=True),
            is_active=True
        )
        context['available_models'] = ModelCheckpoint.objects.filter(is_active=True)
        
        set_id = self.request.GET.get('set_id')
        if set_id:
            instruction_set = get_object_or_404(ManualInstructionSet, id=set_id, created_by=self.request.user.username)
            context['instruction_set'] = instruction_set
            instructions = instruction_set.instructions.all().order_by('order')
            instructions_data = []
            for instr in instructions:
                instructions_data.append({
                    'id': instr.id,
                    'question': instr.question,
                    'answer': instr.answer,
                    'context_chunks': instr.context_chunks,
                    'order': instr.order
                })
            context['instructions_json'] = json.dumps(instructions_data, ensure_ascii=False)
        else:
            context['instructions_json'] = '[]'
        return context


def manual_instruction_set_detail(request, pk):
    instruction_set = get_object_or_404(ManualInstructionSet, pk=pk, created_by=request.user.username)
    return render(request, 'ivanna_model_manager/manual_instruction_set_detail.html', {'instruction_set': instruction_set})


def manual_instruction_set_export(request, pk):
    instruction_set = get_object_or_404(ManualInstructionSet, pk=pk, created_by=request.user.username)
    response = HttpResponse(content_type='application/jsonl')
    response['Content-Disposition'] = f'attachment; filename="{instruction_set.name}.jsonl"'
    for instr in instruction_set.instructions.all().order_by('order'):
        line = json.dumps({
            'instruction': instr.question,
            'output': instr.answer
        }, ensure_ascii=False)
        response.write(line + '\n')
    return response


# ============================================================================
# ХФ датасеты
# ============================================================================
class HuggingFaceDatasetListView(LoginRequiredMixin, ListView):
    model = HuggingFaceDataset
    template_name = 'ivanna_model_manager/hf_dataset_list.html'
    context_object_name = 'datasets'
    paginate_by = 20

    def get_queryset(self):
        return HuggingFaceDataset.objects.filter(created_by=self.request.user.username).order_by('-created_at')


class HuggingFaceDatasetCreateView(LoginRequiredMixin, CreateView):
    model = HuggingFaceDataset
    fields = ['repo_id', 'config', 'split', 'name', 'description']
    template_name = 'ivanna_model_manager/hf_dataset_form.html'
    success_url = reverse_lazy('hf_dataset_list')

    def form_valid(self, form):
        form.instance.created_by = self.request.user.username
        response = super().form_valid(form)
        from ..tasks import import_huggingface_dataset_task
        import_huggingface_dataset_task.delay(self.object.id)
        messages.success(self.request, f'Датасет "{self.object.name}" создан, начат импорт.')
        return response


class HuggingFaceDatasetDetailView(LoginRequiredMixin, DetailView):
    model = HuggingFaceDataset
    template_name = 'ivanna_model_manager/hf_dataset_detail.html'
    context_object_name = 'dataset'


def hf_dataset_start_import(request, pk):
    dataset = get_object_or_404(HuggingFaceDataset, pk=pk, created_by=request.user.username)
    if dataset.status not in ['PENDING', 'FAILED']:
        messages.warning(request, _('Датасет уже обрабатывается'))
        return redirect('hf_dataset_detail', pk=pk)

    from ..tasks import import_huggingface_dataset_task
    task = import_huggingface_dataset_task.delay(dataset.id)
    dataset.celery_task_id = task.id
    dataset.status = 'PENDING'
    dataset.save()
    messages.success(request, f'Импорт датасета запущен (ID: {task.id})')
    return redirect('hf_dataset_detail', pk=pk)


@login_required
@require_POST
def hf_dataset_delete(request, pk):
    dataset = get_object_or_404(HuggingFaceDataset, pk=pk, created_by=request.user.username)
    dataset.delete()
    return JsonResponse({'success': True, 'message': _('Датасет удалён')})
