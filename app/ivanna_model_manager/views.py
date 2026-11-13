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

from django.db.models import Q, Count, Sum
from django.shortcuts import render, redirect, get_object_or_404
from django.utils.translation import gettext as _
from django.http import JsonResponse, HttpResponse
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

from .models import (
    Experiment, ModelCheckpoint, ModelVersion, ModelMergeJob,
    InferenceSession, ModelEvaluation, ChatSession, ModelComparison,
    FineTuningJob, ModelDownloadJob, EvaluationDataset,
    ModelEvaluationTask, EvaluationResult,
    DocumentCollection, Document, DocumentChunk,
    EmbeddingModel, RAGSession, SearchQuery, FineTuningMethod,
    FineTuningJob, FineTuningCheckpoint, FineTuningEvaluation,
    FineTuningTemplate, UserProfile, TrainingMetric,
    ModelConversionJob, DistillationQuestionSet,
    DistillationChunkGroup, DistillationQuestion,
    DistillationAnswerJob, ModelCheckpoint, DocumentCollection,
    DistillationAnswer, DistillationDataset, ManualInstructionSet,
    ManualInstruction, 
)
from .forms import (
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
from .services.redis_model_service import redis_model_manager
from .services.chat_service import chat_service
from .services.download_service import download_service
from .services.rag_service import RAGService
from .tasks import (
    download_model_task, load_model_to_memory_task, unload_model_from_memory_task,
    merge_models_task, process_document_task, create_embeddings_for_collection_task,
    build_faiss_index_task, update_faiss_index_task, regenerate_document_chunks_task,
    run_evaluation_task, process_uploaded_dataset, sync_model_statuses_task,
    health_check_task, generate_questions_for_set_task,
    generate_answers_for_questions_task, HuggingFaceDataset, 
)

logger = logging.getLogger(__name__)

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================
def get_document_type(filename):
    ext = os.path.splitext(filename)[1].lower()
    mapping = {
        '.pdf': 'PDF', '.docx': 'DOCX', '.doc': 'DOCX',
        '.txt': 'TEXT', '.html': 'HTML', '.htm': 'HTML',
        '.md': 'MARKDOWN', '.csv': 'CSV', '.xlsx': 'EXCEL',
        '.xls': 'EXCEL',
    }
    return mapping.get(ext, 'UNKNOWN')


def sanitize_filename(filename):
    import re
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
    filename = filename.replace(' ', '_')
    return re.sub(r'_+', '_', filename)

# ============================================================================
# ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ
# ============================================================================

@login_required
def profile_view(request):
    # Создаём профиль, если его ещё нет
    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)

    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, _('Профиль успешно обновлён.'))
            return redirect('profile')
    else:
        form = UserProfileForm(instance=profile)

    return render(request, 'ivanna_model_manager/profile.html', {
        'form': form,
        'active_tab': 'profile',
        'user': request.user,
    })

# ============================================================================
# ГЛАВНАЯ ПАНЕЛЬ
# ============================================================================
class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'ivanna_model_manager/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        user_id_str = str(user.id)

        # Глобальная статистика (модели общие)
        context['total_models'] = ModelCheckpoint.objects.filter(is_active=True).count()
        context['loaded_models'] = ModelCheckpoint.objects.filter(status='LOADED', is_active=True).count()
        context['downloading_models'] = ModelCheckpoint.objects.filter(status='DOWNLOADING', is_active=True).count()
        context['error_models'] = ModelCheckpoint.objects.filter(status='ERROR', is_active=True).count()
        context['active_experiments'] = Experiment.objects.filter(is_active=True).count()

        # Статистика пользователя
        # Мои коллекции и документы (RAG)
        my_collections = DocumentCollection.objects.filter(owner_id=user_id_str, is_active=True)
        context['my_collections_count'] = my_collections.count()
        context['my_documents_count'] = Document.objects.filter(collection__in=my_collections, is_active=True).count()

        # Мои чат-сессии
        context['my_chat_sessions'] = ChatSession.objects.filter(user_id=user_id_str, is_active=True).count()

        # Мои задачи оценки
        my_evaluation_tasks = ModelEvaluationTask.objects.filter(created_by=user.username)
        context['my_evaluation_tasks_total'] = my_evaluation_tasks.count()
        context['my_evaluation_tasks_active'] = my_evaluation_tasks.filter(status__in=['PENDING', 'RUNNING']).count()
        context['my_evaluation_tasks_completed'] = my_evaluation_tasks.filter(status='COMPLETED').count()

        # Мои задачи fine-tuning
        my_ft_jobs = FineTuningJob.objects.filter(created_by=user.username)
        context['my_ft_jobs_total'] = my_ft_jobs.count()
        context['my_ft_jobs_active'] = my_ft_jobs.filter(status__in=['PREPARING', 'TRAINING']).count()
        context['my_ft_jobs_completed'] = my_ft_jobs.filter(status='COMPLETED').count()

        # Мои задачи конвертации (GGUF)
        my_conv_jobs = ModelConversionJob.objects.filter(created_by=user.username)
        context['my_conv_jobs_total'] = my_conv_jobs.count()
        context['my_conv_jobs_active'] = my_conv_jobs.filter(status__in=['PENDING', 'PREPARING', 'CONVERTING']).count()
        context['my_conv_jobs_completed'] = my_conv_jobs.filter(status='COMPLETED').count()

        # Мои задачи слияния (глобально, пока без фильтрации по пользователю)
        context['merge_jobs_active'] = ModelMergeJob.objects.filter(status__in=['PENDING', 'VALIDATING', 'RUNNING']).count()

        # Статистика эмбеддингов
        embedding_models = EmbeddingModel.objects.filter(is_active=True)
        context['embedding_total'] = embedding_models.count()
        context['embedding_downloaded'] = embedding_models.filter(is_downloaded=True).count()
        context['embedding_downloading'] = embedding_models.filter(download_status='downloading').count()
        context['embedding_failed'] = embedding_models.filter(download_status='failed').count()

        # Недавние загрузки эмбеддингов (для списка)
        context['recent_embeddings'] = embedding_models.exclude(download_status='not_started').order_by('-updated_at')[:5]

        # ========== НОВЫЕ БЛОКИ ДЛЯ ДИСТИЛЛЯЦИИ ==========
        # Мои наборы для дистилляции (вопросы)
        my_question_sets = DistillationQuestionSet.objects.filter(created_by=user.username)
        context['my_question_sets_total'] = my_question_sets.count()
        context['my_question_sets_active'] = my_question_sets.filter(status__in=['PENDING', 'RUNNING']).count()
        context['my_question_sets_completed'] = my_question_sets.filter(status='COMPLETED').count()

        # Мои задачи генерации ответов
        my_answer_jobs = DistillationAnswerJob.objects.filter(created_by=user.username)
        context['my_answer_jobs_total'] = my_answer_jobs.count()
        context['my_answer_jobs_active'] = my_answer_jobs.filter(status__in=['PENDING', 'RUNNING']).count()
        context['my_answer_jobs_completed'] = my_answer_jobs.filter(status='COMPLETED').count()

        # Объединённые активные задачи пользователя (для списка последних)
        recent_tasks = []

        # Задачи оценки
        for task in my_evaluation_tasks.order_by('-created_at')[:3]:
            recent_tasks.append({
                'id': task.id,
                'type': 'evaluation',
                'name': task.name,
                'status': task.status,
                'progress': task.progress,
                'created_at': task.created_at,
                'url': reverse('evaluation_task_detail', args=[task.id])
            })

        # Задачи fine-tuning
        for job in my_ft_jobs.order_by('-created_at')[:3]:
            recent_tasks.append({
                'id': job.id,
                'type': 'fine_tuning',
                'name': job.name,
                'status': job.status,
                'progress': job.progress,
                'created_at': job.created_at,
                'url': reverse('fine_tuning_detail', args=[job.id])
            })

        # Задачи конвертации
        for job in my_conv_jobs.order_by('-created_at')[:3]:
            recent_tasks.append({
                'id': job.id,
                'type': 'conversion',
                'name': job.name,
                'status': job.status,
                'progress': job.progress,
                'created_at': job.created_at,
                'url': reverse('conversion_detail', args=[job.id])
            })

        # Задачи дистилляции (генерация вопросов)
        for qset in my_question_sets.order_by('-created_at')[:3]:
            recent_tasks.append({
                'id': qset.id,
                'type': 'distillation_questions',
                'name': qset.name,
                'status': qset.status,
                'progress': qset.progress,
                'created_at': qset.created_at,
                'url': reverse('distillation_question_set_detail', args=[qset.id])
            })

        # Задачи дистилляции (генерация ответов)
        for job in my_answer_jobs.order_by('-created_at')[:3]:
            recent_tasks.append({
                'id': job.id,
                'type': 'distillation_answers',
                'name': job.name,
                'status': job.status,
                'progress': job.progress,
                'created_at': job.created_at,
                'url': reverse('distillation_answer_job_detail', args=[job.id])
            })

        # Сортировка по дате создания
        recent_tasks.sort(key=lambda x: x['created_at'], reverse=True)
        context['recent_tasks'] = recent_tasks[:5]

        # Последние модели (глобальные)
        context['recent_models'] = ModelCheckpoint.objects.filter(is_active=True).order_by('-created_at')[:10]

        # Активные задачи слияния (глобальные)
        context['recent_jobs'] = ModelMergeJob.objects.filter(status__in=['PENDING', 'VALIDATING', 'RUNNING']).order_by('-created_at')[:5]

        # Метрики пользователя (общее количество сгенерированных токенов)
        user_sessions = InferenceSession.objects.filter(api_key=user.username)
        total_tokens = user_sessions.aggregate(total=Sum('input_tokens') + Sum('output_tokens'))['total'] or 0
        context['user_total_tokens'] = total_tokens

        return context

@login_required
def fine_tuning_logs(request, pk):
    job = get_object_or_404(FineTuningJob, pk=pk, created_by=request.user.username)
    return JsonResponse({'logs': job.logs})

@login_required
def fine_tuning_progress(request, pk):
    job = get_object_or_404(FineTuningJob, pk=pk, created_by=request.user.username)
    return JsonResponse({
        'progress': job.progress,
        'current_step': job.current_step,
        'status': job.status
    })

# ============================================================================
# МОДЕЛИ
# ============================================================================
class ModelListView(LoginRequiredMixin, ListView):
    template_name = 'ivanna_model_manager/model_list.html'
    context_object_name = 'models'
    paginate_by = 20

    def get_queryset(self):
        qs = ModelCheckpoint.objects.filter(is_active=True)
        # Фильтры
        if self.request.GET.get('model_type'):
            qs = qs.filter(model_type=self.request.GET['model_type'])
        if self.request.GET.get('status'):
            qs = qs.filter(status=self.request.GET['status'])
        if self.request.GET.get('experiment_id'):
            qs = qs.filter(experiment_id=self.request.GET['experiment_id'])
        if self.request.GET.get('q'):
            qs = qs.filter(Q(name__icontains=self.request.GET['q']) | Q(display_name__icontains=self.request.GET['q']))
        # Сортировка
        sort = self.request.GET.get('sort_by', '-created_at')
        if sort.lstrip('-') in ['name', 'display_name', 'created_at', 'parameter_count']:
            qs = qs.order_by(sort)
        else:
            qs = qs.order_by('-created_at')
        return qs.select_related('experiment', 'parent')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        models = context['models']
        total_size_gb = 0
        loaded_models = 0
        quantized_models = 0

        for model in models:
            total_size_gb += model.size_gb
            if model.status == 'LOADED':
                loaded_models += 1
            if model.is_quantized:
                quantized_models += 1

            # Проверяем, загружена ли модель в память (Redis)
            model_id = model.huggingface_id or model.name
            info = redis_model_manager.get_model_info(model_id)
            model.is_loaded_in_memory = bool(info and info.get('loaded'))

        # Получаем текущее значение сортировки из запроса (для сохранения в фильтрах)
        current_sort = self.request.GET.get('sort_by', '-created_at')

        context.update({
            'total_models': models.count(),
            'total_size_gb': round(total_size_gb, 2),
            'loaded_models': loaded_models,
            'quantized_models': quantized_models,
            'model_types': ModelCheckpoint.MODEL_TYPE_CHOICES,
            'status_choices': ModelCheckpoint.STATUS_CHOICES,
            'experiments': Experiment.objects.filter(is_active=True),
            'current_filters': {
                'model_type': self.request.GET.get('model_type', ''),
                'status': self.request.GET.get('status', ''),
                'experiment_id': self.request.GET.get('experiment_id', ''),
                'q': self.request.GET.get('q', ''),
                'sort_by': current_sort,
            }
        })
        return context


class ModelDetailView(LoginRequiredMixin, DetailView):
    """Детальная информация о модели с отображением конфигурации."""
    model = ModelCheckpoint
    template_name = 'ivanna_model_manager/model_detail.html'
    context_object_name = 'model'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        obj = self.object
        context['has_local_copy'] = bool(obj.local_path and os.path.exists(obj.local_path))

        # Загружаем конфигурацию loader_config в JSON для отображения
        import json
        config_json = json.dumps(obj.loader_config, indent=2, ensure_ascii=False)
        context['config_json'] = config_json

        # Версии модели
        context['versions'] = obj.versions.order_by('-created_at')[:10]

        # Оценки
        context['evaluations'] = obj.evaluations.filter(is_valid=True).order_by('-evaluated_at')[:5]

        # Активные сессии
        context['active_sessions'] = obj.inference_sessions.filter(is_active=True).order_by('-last_activity')[:5]

        # Подходящие модели для слияния
        context['merge_candidates'] = ModelCheckpoint.objects.filter(
            status='LOADED',
            is_active=True,
            model_type=obj.model_type,
            parameter_count=obj.parameter_count
        ).exclude(id=obj.id)[:10]

        # Последние задачи слияния
        context['merge_jobs'] = ModelMergeJob.objects.filter(
            Q(base_model=obj) | Q(merge_model=obj)
        ).order_by('-created_at')[:5]

        # Размер и квантизация
        context['size_gb'] = obj.size_gb
        context['is_quantized'] = obj.is_quantized

        return context


class LocalModelCreateView(LoginRequiredMixin, CreateView):
    """Создание модели с локальным путём (без Hugging Face)."""
    template_name = 'ivanna_model_manager/local_model_form.html'
    form_class = LocalModelForm  # которую нужно создать
    success_url = reverse_lazy('model_list')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['experiments'] = Experiment.objects.filter(is_active=True)
        return context

    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, f'Модель "{form.instance.display_name}" добавлена')
        return response


class UploadModelView(LoginRequiredMixin, CreateView):
    template_name = 'ivanna_model_manager/upload_model.html'
    form_class = ModelUploadForm
    success_url = reverse_lazy('model_list')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Загрузить новую модель'
        context['experiments'] = Experiment.objects.filter(is_active=True)
        context['model_choices'] = [
            {'value': 'QWEN', 'label': 'Qwen', 'icon': 'bi-robot', 'tags': ['многоязычный', 'код']},
            {'value': 'LLAMA', 'label': 'Llama', 'icon': 'bi-vector-pen', 'tags': ['meta', 'инструкции']},
            {'value': 'MISTRAL', 'label': 'Mistral', 'icon': 'bi-wind', 'tags': ['MOE', 'французский']},
            {'value': 'GEMMA', 'label': 'Gemma', 'icon': 'bi-gem', 'tags': ['google', 'лёгкий']},
            {'value': 'DEEPSEEK', 'label': 'DeepSeek', 'icon': 'bi-search', 'tags': ['китайский', 'код']},
            {'value': 'PHI', 'label': 'Phi', 'icon': 'bi-lightning', 'tags': ['малый', 'быстрый']},
            {'value': 'YI', 'label': 'Yi', 'icon': 'bi-translate', 'tags': ['китайский', 'многоязычный']},
            {'value': 'CUSTOM', 'label': 'Кастомная', 'icon': 'bi-cpu', 'tags': ['локальная']},
        ]
        return context

    def form_valid(self, form):
        model = form.save()
        model.fill_missing_loader_config()
        model.save()
        model.create_version('v1.0', 'Создание модели')
        if form.cleaned_data.get('start_download') and model.huggingface_id:
            download_model_task.delay(model.id)
            messages.success(self.request, f'Модель "{model.display_name}" создана и поставлена в очередь на загрузку')
        else:
            messages.success(self.request, f'Модель "{model.display_name}" создана')
        return super().form_valid(form)


class ModelUpdateView(LoginRequiredMixin, UpdateView):
    model = ModelCheckpoint
    form_class = ModelCheckpointForm
    template_name = 'ivanna_model_manager/model_form.html'

    def get_success_url(self):
        return reverse_lazy('model_detail', kwargs={'pk': self.object.id})

    def form_valid(self, form):
        response = super().form_valid(form)
        if form.changed_data:
            form.instance.create_version(
                f"edit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                f"Изменены поля: {', '.join(form.changed_data)}"
            )
        messages.success(self.request, f'Модель "{form.instance.name}" обновлена')
        return response

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = f'Редактировать модель: {self.object.name}'
        context['experiments'] = Experiment.objects.filter(is_active=True)
        return context


@login_required
def download_model_view(request, pk):
    """Запуск загрузки модели (синхронный запуск задачи)."""
    model = get_object_or_404(ModelCheckpoint, pk=pk, is_active=True)
    if model.status == 'LOADED':
        messages.warning(request, _('Модель уже загружена'))
        return redirect('model_detail', pk=model.id)
    if not model.huggingface_id:
        messages.error(request, _('У модели не указан Hugging Face ID'))
        return redirect('model_detail', pk=model.id)

    download_model_task.delay(model.id, user_id=request.user.id)
    messages.info(request, f'Загрузка модели "{model.name}" запущена в фоновом режиме')
    return redirect('model_detail', pk=model.id)


@login_required
@permission_required('ivanna_model_manager.can_delete_protected', raise_exception=True)
def model_delete_view(request, pk):
    """Мягкое удаление модели (деактивация)."""
    model = get_object_or_404(ModelCheckpoint, pk=pk)
    if request.method == 'POST':
        # Отменяем активные загрузки
        ModelDownloadJob.objects.filter(
            checkpoint=model,
            status__in=['PENDING', 'DOWNLOADING']
        ).update(status='CANCELLED', completed_at=timezone.now())
        model.is_active = False
        model.status = 'ARCHIVED'
        model.save()
        messages.success(request, f'Модель "{model.display_name}" удалена (деактивирована)')
        return redirect('model_list')

    context = {
        'model': model,
        'related_count': {
            'sessions': model.inference_sessions.filter(is_active=True).count(),
            'versions': model.versions.count(),
            'evaluations': model.evaluations.count(),
            'chat_sessions': model.chat_sessions.count(),
        },
        'active_downloads': ModelDownloadJob.objects.filter(
            checkpoint=model,
            status__in=['PENDING', 'DOWNLOADING']
        ).count(),
        'has_local_files': bool(model.local_path and os.path.exists(model.local_path)),
    }
    return render(request, 'ivanna_model_manager/model_confirm_delete.html', context)


@login_required
def model_config_view(request, pk):
    """Просмотр конфигурации модели."""
    import json
    model = get_object_or_404(ModelCheckpoint, pk=pk, is_active=True)
    if not model.loader_config:
        model.loader_config = model.get_default_config()
        model.save()
    return render(request, 'ivanna_model_manager/model_config.html', {
        'model': model,
        'config_json': json.dumps(model.loader_config, indent=2, ensure_ascii=False)
    })


@login_required
def create_model_version_view(request, pk):
    """Создание новой версии модели."""
    model = get_object_or_404(ModelCheckpoint, pk=pk, is_active=True)
    version_name = request.POST.get('version_name', f'v{model.versions.count() + 1}')
    notes = request.POST.get('notes', '')
    model.create_version(version_name, notes)
    messages.success(request, f'Версия {version_name} создана')
    return redirect('model_detail', pk=model.id)


@login_required
def restore_model_version_view(request, model_pk, version_pk):
    """Восстановление версии модели."""
    model = get_object_or_404(ModelCheckpoint, pk=model_pk, is_active=True)
    version = get_object_or_404(ModelVersion, pk=version_pk, checkpoint=model)
    version.restore()
    messages.success(request, f'Версия {version.version} восстановлена')
    return redirect('model_detail', pk=model.id)


# ============================================================================
# ТЕСТИРОВАНИЕ МОДЕЛИ (ИНФЕРЕНС)
# ============================================================================
class InferenceTestView(LoginRequiredMixin, DetailView):
    model = ModelCheckpoint
    template_name = 'ivanna_model_manager/inference_test.html'
    context_object_name = 'model'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = InferenceTestForm()
        return context

    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        form = InferenceTestForm(request.POST)
        if not form.is_valid():
            return render(request, self.template_name, {'form': form, 'model': self.object})

        model_id = self.object.huggingface_id or self.object.name
        # Проверяем, загружена ли модель
        info = redis_model_manager.get_model_info(model_id)
        if not info or not info.get('loaded'):
            messages.error(request, _('Модель не загружена в память'))
            return redirect('model_detail', pk=self.object.id)

        session = InferenceSession.objects.create(
            checkpoint=self.object,
            session_id=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_ip=request.META.get('REMOTE_ADDR', ''),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            api_key=request.user.username,
        )

        try:
            start = time.time()
            result = redis_model_manager.generate_text(
                model_id=model_id,
                prompt=form.cleaned_data['prompt'],
                max_tokens=form.cleaned_data['max_tokens'],
                temperature=form.cleaned_data['temperature'],
                top_p=form.cleaned_data['top_p'],
                top_k=form.cleaned_data['top_k'],
                repetition_penalty=form.cleaned_data['repetition_penalty'],
            )
            gen_time = time.time() - start

            if result.get('success'):
                response_text = result['response']
                output_tokens = result.get('output_tokens', 0)
                session.add_request(
                    input_tokens=result.get('input_tokens', 0),
                    output_tokens=output_tokens,
                    duration=gen_time
                )
                context = self.get_context_data()
                context.update({
                    'response': response_text,
                    'prompt': form.cleaned_data['prompt'],
                    'session': session,
                    'response_metadata': {
                        'generation_time': round(gen_time, 2),
                        'tokens_generated': output_tokens,
                        'tokens_per_second': round(output_tokens / gen_time, 1) if gen_time > 0 else 0,
                        'input_tokens': result.get('input_tokens', 0),
                        'total_tokens': result.get('total_tokens', 0),
                    }
                })
                messages.success(request, f'Сгенерировано {output_tokens} токенов за {gen_time:.2f} сек')
                return render(request, self.template_name, context)
            else:
                session.error_count += 1
                session.last_error = result.get('error', 'Unknown error')
                session.save()
                messages.error(request, f'Ошибка генерации: {result.get("error")}')
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            messages.error(request, f'Ошибка: {str(e)}')

        return redirect('model_detail', pk=self.object.id)


# ============================================================================
# ЭКСПЕРИМЕНТЫ
# ============================================================================
class ExperimentListView(LoginRequiredMixin, ListView):
    model = Experiment
    template_name = 'ivanna_model_manager/experiment_list.html'
    context_object_name = 'experiments'

    def get_queryset(self):
        qs = Experiment.objects.all()
        if self.request.GET.get('active') == 'true':
            qs = qs.filter(is_active=True)
        elif self.request.GET.get('active') == 'false':
            qs = qs.filter(is_active=False)
        if self.request.GET.get('q'):
            qs = qs.filter(name__icontains=self.request.GET['q'])
        return qs.order_by('-created_at')


class ExperimentDetailView(LoginRequiredMixin, DetailView):
    model = Experiment
    template_name = 'ivanna_model_manager/experiment_detail.html'
    context_object_name = 'experiment'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        checkpoints = self.object.checkpoints.filter(is_active=True)
        context.update({
            'checkpoints': checkpoints,
            'model_count': checkpoints.count(),
            'loaded_count': checkpoints.filter(status='LOADED').count(),
        })
        return context


class ExperimentCreateView(LoginRequiredMixin, CreateView):
    model = Experiment
    form_class = ExperimentForm
    template_name = 'ivanna_model_manager/experiment_form.html'
    success_url = reverse_lazy('experiment_list')

    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, f'Эксперимент "{form.instance.name}" создан')
        return response


class ExperimentUpdateView(LoginRequiredMixin, UpdateView):
    model = Experiment
    form_class = ExperimentForm
    template_name = 'ivanna_model_manager/experiment_form.html'

    def get_success_url(self):
        return reverse_lazy('experiment_detail', kwargs={'pk': self.object.id})

    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, f'Эксперимент "{form.instance.name}" обновлён')
        return response


@login_required
def experiment_deactivate_view(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk, is_active=True)
    if request.method == 'POST':
        experiment.deactivate()
        messages.success(request, f'Эксперимент "{experiment.name}" деактивирован')
        return redirect('experiment_list')
    return render(request, 'ivanna_model_manager/experiment_confirm_deactivate.html', {'experiment': experiment})


@login_required
def experiment_activate_view(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk)
    if request.method == 'POST':
        experiment.is_active = True
        experiment.save()
        messages.success(request, f'Эксперимент "{experiment.name}" активирован')
        return redirect('experiment_detail', pk=experiment.id)
    return render(request, 'ivanna_model_manager/experiment_confirm_activate.html', {'experiment': experiment})


@login_required
def create_checkpoint_from_experiment(request, pk):
    experiment = get_object_or_404(Experiment, pk=pk, is_active=True)
    if not experiment.base_checkpoint:
        messages.error(request, _('У эксперимента нет базового чекпоинта'))
        return redirect('experiment_detail', pk=experiment.id)
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
    messages.success(request, f'Чекпоинт "{new.name}" создан')
    return redirect('model_detail', pk=new.id)


# ============================================================================
# СЛИЯНИЕ МОДЕЛЕЙ
# ============================================================================
class ModelMergeView(LoginRequiredMixin, CreateView):
    model = ModelMergeJob
    form_class = ModelMergeForm
    template_name = 'ivanna_model_manager/merge_detail.html'
    success_url = reverse_lazy('model_list')

    def get_initial(self):
        initial = super().get_initial()
        if self.request.GET.get('base_model'):
            try:
                initial['base_model'] = ModelCheckpoint.objects.get(id=self.request.GET['base_model'])
            except:
                pass
        if self.request.GET.get('merge_model'):
            try:
                initial['merge_model'] = ModelCheckpoint.objects.get(id=self.request.GET['merge_model'])
            except:
                pass
        return initial

    def form_valid(self, form):
        # Проверка совместимости моделей
        base = form.cleaned_data['base_model']
        merge = form.cleaned_data['merge_model']
        if base.model_type != merge.model_type:
            form.add_error(None, 'Модели должны быть одного типа')
            return self.form_invalid(form)
        if base.parameter_count != merge.parameter_count:
            form.add_error(None, 'Модели должны иметь одинаковое количество параметров')
            return self.form_invalid(form)
        response = super().form_valid(form)
        # Запускаем задачу слияния
        merge_models_task.delay(self.object.id)
        messages.success(self.request, f'Задача слияния "{self.object.name}" запущена')
        return response

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Модели, доступные для слияния (должны быть скачаны локально)
        context['base_models'] = ModelCheckpoint.objects.filter(
            is_active=True,
            status__in=['DOWNLOADED', 'LOADED']
        ).order_by('display_name')
        context['merge_models'] = ModelCheckpoint.objects.filter(
            is_active=True,
            status__in=['DOWNLOADED', 'LOADED']
        ).order_by('display_name')
        context['merge_methods'] = ModelMergeJob.MERGE_METHOD_CHOICES
        return context


@login_required
def start_merge_view(request, pk):
    """Быстрый старт слияния из карточки модели."""
    base = get_object_or_404(ModelCheckpoint, pk=pk, is_active=True)
    merge_id = request.POST.get('merge_model_id')
    if not merge_id:
        messages.error(request, _('Не выбрана модель для слияния'))
        return redirect('model_detail', pk=base.id)
    merge = get_object_or_404(ModelCheckpoint, pk=merge_id, is_active=True)
    if base.model_type != merge.model_type or base.parameter_count != merge.parameter_count:
        messages.error(request, _('Модели несовместимы'))
        return redirect('model_detail', pk=base.id)

    job = ModelMergeJob.objects.create(
        name=f"merge_{base.name}_{merge.name}_{int(time.time())}",
        base_model=base,
        merge_model=merge,
        merge_method=request.POST.get('method', 'TIES'),
        merge_ratio=float(request.POST.get('ratio', 0.5))
    )
    merge_models_task.delay(job.id)
    messages.success(request, f'Слияние запущено')
    return redirect('model_detail', pk=base.id)


# ============================================================================
# ЧАТ
# ============================================================================
@login_required
def chat_view(request, session_id=None):
    """Основное представление чата с поддержкой создания новой сессии и проверки статуса модели."""
    from .models import ChatSession, ModelCheckpoint
    from .services.chat_service import chat_service
    from .services.redis_model_service import redis_model_manager
    from .tasks import sync_model_statuses_task
    from django.db import IntegrityError
    import os

    logger = logging.getLogger(__name__)
    
    logger.info(f"🚀 chat_view called with session_id={session_id}")

    # Запускаем синхронизацию статусов моделей в фоне (не блокируем)
    try:
        sync_model_statuses_task.delay()
    except Exception as e:
        logger.error(f"Ошибка запуска синхронизации моделей: {e}")

    # Получаем все активные модели с информацией о загрузке
    available_models = ModelCheckpoint.objects.filter(is_active=True).order_by('-status', 'name')
    models_with_info = []
    loaded_models_count = 0

    for model in available_models:
        model_identifier = model.huggingface_id or model.name
        redis_info = redis_model_manager.get_model_info(model_identifier)
        is_loaded = bool(redis_info and redis_info.get('loaded'))
        is_local = model_identifier in redis_model_manager.local_models
        if is_loaded:
            loaded_models_count += 1

        # Форматируем количество параметров
        params = model.parameter_count
        if params >= 1_000_000_000:
            params_h = f"{params / 1_000_000_000:.1f}B"
        elif params >= 1_000_000:
            params_h = f"{params / 1_000_000:.1f}M"
        else:
            params_h = str(params)

        models_with_info.append({
            'model': model,
            'model_id': model.id,
            'name': model.display_name or model.name,
            'model_identifier': model_identifier,
            'is_loaded': is_loaded,
            'is_loaded_in_this_process': is_local,
            'device': redis_info.get('device', 'Не загружена') if redis_info else 'Не загружена',
            'status': model.status,
            'parameters_human': params_h,
            'size_gb': model.size_gb,
        })

    # Определяем, нужно ли создать новую сессию
    create_new = request.GET.get('new') == 'true'
    force_new = request.GET.get('force_new') == 'true'

    chat_session = None
    session_data = None

    # Если явно запрошено создание новой сессии
    if create_new or force_new:
        # Пытаемся создать сессию с первой загруженной моделью (если есть)
        default_model_id = None
        for minfo in models_with_info:
            if minfo['is_loaded']:
                default_model_id = minfo['model'].id
                break
        # Если нет загруженных, создаём сессию без модели
        creation_result = chat_service.create_session(str(request.user.id), default_model_id)
        session_id = creation_result['session_id']
        session_data = chat_service.get_session(session_id)
        # Создаём запись в БД, если её ещё нет
        try:
            chat_session = ChatSession.objects.get(session_id=session_id)
        except ChatSession.DoesNotExist:
            model = None
            if default_model_id:
                try:
                    model = ModelCheckpoint.objects.get(id=default_model_id, is_active=True)
                except ModelCheckpoint.DoesNotExist:
                    pass
            chat_session = ChatSession.objects.create(
                session_id=session_id,
                user_id=str(request.user.id),
                checkpoint=model,
                is_active=True,
                created_at=timezone.now(),
                last_activity=timezone.now(),
                generation_config={}
            )
        logger.info(f"✅ Создана новая сессия по запросу: {session_id}")
        return redirect('chat_session', session_id=session_id)

    # Если передан session_id, пытаемся получить сессию
    elif session_id:
        try:
            # Сначала пробуем получить сессию из БД
            try:
                chat_session = ChatSession.objects.get(
                    session_id=session_id,
                    user_id=str(request.user.id),
                    is_active=True
                )
            except ChatSession.DoesNotExist:
                logger.warning(f"Session {session_id} not found in DB, redirecting to management")
                return redirect('chat_sessions_management')
            
            # Затем получаем данные из Redis (без force_refresh, чтобы не потерять сообщения)
            session_data = chat_service.get_session(session_id)
            
            # Если в Redis нет данных, восстанавливаем из БД
            if not session_data:
                logger.info(f"Session {session_id} not in Redis, restoring from DB")
                session_data = {
                    'session_id': chat_session.session_id,
                    'user_id': chat_session.user_id,
                    'model_id': chat_session.checkpoint.id if chat_session.checkpoint else None,
                    'conversation_history': chat_session.conversation_history or [],
                    'session_config': {
                        'loader_overrides': {},
                        'generation_config': chat_session.generation_config or {},
                        'system_prompt': chat_session.system_prompt or 'Ты полезный ассистент. Отвечай на русском языке.',
                    },
                    'rag_config': {},
                }
                # Сохраняем в Redis для будущих запросов
                redis_key = f"chat:session:{session_id}"
                chat_service.redis_manager.redis.setex(
                    redis_key, 86400, 
                    json.dumps(session_data, default=str)
                )
            
            # Проверяем, есть ли сообщения в Redis, но не было ли их в БД
            # Это может произойти при проблемах с записью в БД
            if session_data.get('conversation_history') and not chat_session.conversation_history:
                logger.info(f"Session {session_id} has history in Redis but not in DB, updating DB")
                for msg in session_data['conversation_history']:
                    if msg['role'] == 'user':
                        chat_session.add_message('user', msg['content'])
                    else:
                        chat_session.add_message('assistant', msg['content'], 
                                                msg.get('metadata', {}))
            
        except Exception as e:
            logger.error(f"Ошибка получения сессии {session_id}: {e}")
            return redirect('chat_sessions_management')

    # Если сессия не указана и не создаём новую, пытаемся взять последнюю активную
    if not session_id:
        return redirect('chat_sessions_management')

    # Получаем историю сообщений - приоритет у Redis
    messages_history = []
    if session_data:
        messages_history = session_data.get('conversation_history', [])
    elif chat_session:
        messages_history = chat_session.conversation_history or []
    
    logger.info(f"📊 Final messages_history length: {len(messages_history)}")
    user_msgs = [m for m in messages_history if m.get('role') == 'user']
    logger.info(f"👤 User messages count: {len(user_msgs)}")

    # Контекст для шаблона
    context = {
        'chat_session': chat_session,
        'session_data': session_data,
        'models_with_info': models_with_info,
        'messages': messages_history,
        'user_sessions': chat_service.list_user_sessions(str(request.user.id)),
        'loaded_models_count': loaded_models_count,
        'has_loaded_models': loaded_models_count > 0,
        'active_tab': 'chat',
    }

    # Добавляем отладочную информацию при DEBUG
    if settings.DEBUG:
        context['debug'] = {
            'session_id': session_id,
            'pid': os.getpid(),
            'local_models': list(redis_model_manager.local_models.keys()),
        }

    return render(request, 'ivanna_model_manager/chat.html', context)


@login_required
def create_chat_session(request):
    """Создание новой чат-сессии."""
    model_id = request.POST.get('model_id')
    if not model_id:
        messages.error(request, _('Не выбрана модель'))
        return redirect('chat_view')
    try:
        result = chat_service.create_session(str(request.user.id), model_id)
        return redirect('chat_view', session_id=result['session_id'])
    except Exception as e:
        logger.error(f"Create session error: {e}")
        messages.error(request, f'Ошибка: {e}')
        return redirect('chat_view')


@login_required
@require_POST
def send_message(request, session_id):
    """Отправка сообщения в чат (AJAX)."""
    try:
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        if not message:
            return JsonResponse({'error': 'Message is empty'}, status=400)
        result = chat_service.generate_response(session_id, message)
        return JsonResponse(result)
    except Exception as e:
        logger.error(f"Send message error: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required
def chat_sessions_management(request):
    """Страница управления сессиями."""
    sessions = chat_service.list_user_sessions(str(request.user.id))
    available_models = ModelCheckpoint.objects.filter(is_active=True).order_by('name')
    for m in available_models:
        m.is_loaded_in_memory_cache = bool(redis_model_manager.get_model_info(m.huggingface_id or m.name))

    # Статистика
    total_sessions = len(sessions)
    active_sessions = total_sessions  # поскольку мы уже фильтруем только активные
    loaded_models_count = sum(1 for s in sessions if s.get('model_loaded', False))

    context = {
        'sessions': sessions,
        'available_models': available_models,
        'page_title': 'Мои чат-сессии',
        'total_sessions': total_sessions,
        'active_sessions': active_sessions,
        'loaded_models_count': loaded_models_count,
    }
    return render(request, 'ivanna_model_manager/chat_sessions_management.html', context)


@login_required
@require_POST
def rename_chat_session(request, session_id):
    new_name = request.POST.get('name', '').strip()
    if not new_name:
        return JsonResponse({'success': False, 'error': _('Имя не может быть пустым')})
    try:
        session = ChatSession.objects.get(session_id=session_id, user_id=str(request.user.id))
        session.notes = new_name
        session.save()
        # Обновляем в Redis
        redis_data = chat_service.get_session(session_id)
        if redis_data:
            redis_data['display_name'] = new_name
            chat_service.redis_manager.redis.setex(
                f"chat:session:{session_id}",
                86400,
                json.dumps(redis_data, default=str)
            )
        return JsonResponse({'success': True})
    except Exception as e:
        logger.error(f"Rename error: {e}")
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
@require_POST
def clear_chat_session(request, session_id):
    success = chat_service.clear_history(session_id)
    return JsonResponse({'success': success})


@login_required
@require_POST
def delete_chat_session(request, session_id):
    try:
        chat_service.close_session(session_id)
        return JsonResponse({'success': True})
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
@require_POST
def change_chat_session_model(request, session_id):
    model_id = request.POST.get('model_id')
    if not model_id:
        return JsonResponse({'success': False, 'error': 'Model ID required'})
    try:
        session = ChatSession.objects.get(session_id=session_id, user_id=str(request.user.id))
        new_model = ModelCheckpoint.objects.get(id=model_id, is_active=True)
        session.checkpoint = new_model
        session.save()
        # Обновляем Redis
        redis_data = chat_service.get_session(session_id)
        if redis_data:
            redis_data['model_id'] = model_id
            chat_service.redis_manager.redis.setex(
                f"chat:session:{session_id}",
                86400,
                json.dumps(redis_data, default=str)
            )
        return JsonResponse({'success': True, 'new_model_name': new_model.display_name})
    except Exception as e:
        logger.error(f"Change model error: {e}")
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
@require_POST
def export_chat_session(request, session_id):
    """Экспорт сессии в JSON."""
    session = get_object_or_404(ChatSession, session_id=session_id, user_id=str(request.user.id))
    redis_data = chat_service.get_session(session_id) or {}
    export = {
        'session_id': session.session_id,
        'user_id': session.user_id,
        'model_name': session.checkpoint.display_name if session.checkpoint else None,
        'model_id': session.checkpoint.id if session.checkpoint else None,
        'created_at': session.created_at.isoformat(),
        'last_activity': session.last_activity.isoformat(),
        'total_messages': session.total_messages,
        'total_tokens': session.total_input_tokens + session.total_output_tokens,
        'conversation_history': redis_data.get('conversation_history', []),
        'notes': session.notes,
        'export_date': timezone.now().isoformat(),
    }
    if request.POST.get('download') == 'true':
        response = HttpResponse(
            json.dumps(export, ensure_ascii=False, indent=2),
            content_type='application/json; charset=utf-8'
        )
        response['Content-Disposition'] = f'attachment; filename="chat_{session_id}.json"'
        return response
    return JsonResponse({'success': True, 'export_data': export})


# ============================================================================
# СРАВНЕНИЕ МОДЕЛЕЙ (заглушка, для совместимости)
# ============================================================================
class ComparisonView(LoginRequiredMixin, ListView):
    model = ModelComparison
    template_name = 'ivanna_model_manager/comparison_list.html'
    context_object_name = 'comparisons'


class ComparisonCreateView(LoginRequiredMixin, CreateView):
    model = ModelComparison
    fields = ['name', 'description']
    template_name = 'ivanna_model_manager/comparison_create.html'
    success_url = reverse_lazy('comparison_list')


class ComparisonDetailView(LoginRequiredMixin, DetailView):
    model = ModelComparison
    template_name = 'ivanna_model_manager/comparison_detail.html'
    context_object_name = 'comparison'


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
        with open(file_path, 'w', encoding='utf-8') as f:
            for q in new_questions:
                item = {
                    'instruction': q.get('text', q.get('instruction', '')),
                    'output': q.get('answer', q.get('output', '')),
                    'category': q.get('category', ''),
                    'difficulty': q.get('difficulty', 'medium')
                }
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        dataset.total_questions = len(new_questions)
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
    from .models import ModelEvaluationTask, EvaluationResult
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
        from .models import EvaluationResult
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


# ============================================================================
# RAG: КОЛЛЕКЦИИ, ДОКУМЕНТЫ, ПОИСК, ЧАТ
# ============================================================================
@login_required
def rag_collection_list(request):
    collections = DocumentCollection.objects.filter(
        Q(owner_id=str(request.user.id)) | Q(is_public=True),
        is_active=True
    ).order_by('-created_at')
    paginator = Paginator(collections, 20)
    page = paginator.get_page(request.GET.get('page'))
    total_docs = sum(c.documents.filter(is_active=True).count() for c in page)
    return render(request, 'ivanna_model_manager/rag_collection_list.html', {
        'page_obj': page,
        'total_collections': collections.count(),
        'total_documents': total_docs,
        'active_tab': 'rag',
    })


@login_required
def rag_collection_detail(request, pk):
    collection = get_object_or_404(
        DocumentCollection,
        Q(owner_id=str(request.user.id)) | Q(is_public=True),
        pk=pk,
        is_active=True
    )
    documents = collection.documents.filter(is_active=True).order_by('-uploaded_at')
    # Аннотируем количество чанков и эмбеддингов
    documents = documents.annotate(
        ann_chunk_count=Count('chunks', filter=Q(chunks__is_active=True)),
        ann_embedding_count=Count('chunks', filter=Q(chunks__has_embedding=True))
    )
    stats = {
        'total_documents': documents.count(),
        'by_status': list(documents.values('status').annotate(count=Count('id'))),
        'by_type': list(documents.values('document_type').annotate(count=Count('id'))),
        'total_size_mb': documents.aggregate(total=Sum('file_size_mb'))['total'] or 0,
        'total_chunks': documents.aggregate(total=Sum('ann_chunk_count'))['total'] or 0,
        'chunks_with_embeddings': DocumentChunk.objects.filter(
            document__collection=collection,
            has_embedding=True,
            is_active=True
        ).count(),
    }
    paginator = Paginator(documents, 10)
    page = paginator.get_page(request.GET.get('page'))
    return render(request, 'ivanna_model_manager/rag_collection_detail.html', {
        'collection': collection,
        'documents': documents,
        'page_obj': page,
        'stats': stats,
        'active_tab': 'rag',
    })


@login_required
def rag_collection_create(request):
    if request.method == 'POST':
        form = DocumentCollectionForm(request.POST, user=request.user)
        if form.is_valid():
            collection = form.save(commit=False)
            collection.owner_id = str(request.user.id)
            if not request.user.is_staff:
                collection.is_public = False
            collection.save()
            messages.success(request, f'Коллекция "{collection.name}" создана')
            if request.POST.get('action') == 'create_and_add':
                return redirect('rag_upload_documents')
            elif request.POST.get('action') == 'create_and_chat':
                return redirect('rag_chat')
            return redirect('rag_collection_detail', pk=collection.id)
    else:
        form = DocumentCollectionForm(user=request.user)
    return render(request, 'ivanna_model_manager/rag_collection_create.html', {
        'form': form,
        'active_tab': 'rag',
    })


@login_required
@require_POST
def rag_collection_delete(request, pk):
    collection = get_object_or_404(
        DocumentCollection,
        pk=pk,
        owner_id=str(request.user.id)
    )
    collection.is_active = False
    collection.save()
    return JsonResponse({'success': True, 'message': _('Коллекция удалена')})


@login_required
def document_list(request, collection_id=None):
    if collection_id:
        collection = get_object_or_404(
            DocumentCollection,
            Q(owner_id=str(request.user.id)) | Q(is_public=True),
            id=collection_id,
            is_active=True
        )
        docs = collection.documents.filter(is_active=True)
    else:
        # Все документы из доступных коллекций
        collections = DocumentCollection.objects.filter(
            Q(owner_id=str(request.user.id)) | Q(is_public=True),
            is_active=True
        ).values_list('id', flat=True)
        docs = Document.objects.filter(collection_id__in=list(collections), is_active=True)

    # Фильтры
    if request.GET.get('status'):
        docs = docs.filter(status=request.GET['status'])
    if request.GET.get('type'):
        docs = docs.filter(document_type=request.GET['type'])
    if request.GET.get('q'):
        docs = docs.filter(
            Q(title__icontains=request.GET['q']) |
            Q(original_filename__icontains=request.GET['q']) |
            Q(description__icontains=request.GET['q'])
        )

    sort = request.GET.get('sort', '-uploaded_at')
    if sort.lstrip('-') in ['title', 'document_type', 'status', 'uploaded_at', 'file_size_mb']:
        docs = docs.order_by(sort)
    else:
        docs = docs.order_by('-uploaded_at')

    paginator = Paginator(docs, 25)
    page = paginator.get_page(request.GET.get('page'))

    stats = {
        'total': docs.count(),
        'by_status': list(docs.values('status').annotate(count=Count('id'))),
        'by_type': list(docs.values('document_type').annotate(count=Count('id'))),
        'total_size_mb': docs.aggregate(total=Sum('file_size_mb'))['total'] or 0,
    }
    return render(request, 'ivanna_model_manager/document_list.html', {
        'collection': collection if collection_id else None,
        'page_obj': page,
        'stats': stats,
        'active_tab': 'rag',
        'filters': {
            'status': request.GET.get('status', ''),
            'type': request.GET.get('type', ''),
            'q': request.GET.get('q', ''),
            'sort': sort,
        }
    })


@login_required
def document_detail(request, pk):
    doc = get_object_or_404(Document, pk=pk, is_active=True)
    # Проверка доступа
    if doc.collection:
        if doc.collection.owner_id != str(request.user.id) and not doc.collection.is_public:
            raise PermissionDenied
    elif hasattr(doc, 'owner_id') and doc.owner_id != str(request.user.id):
        raise PermissionDenied

    chunks = doc.chunks.filter(is_active=True).order_by('chunk_index')
    from .models import DocumentProcessor
    history = DocumentProcessor.objects.filter(document=doc).order_by('-started_at')[:10]
    similar = Document.objects.none()
    if doc.collection:
        similar = Document.objects.filter(
            collection=doc.collection,
            document_type=doc.document_type,
            is_active=True
        ).exclude(id=doc.id).order_by('-uploaded_at')[:5]

    if request.method == 'POST' and 'quick_edit' in request.POST:
        form = DocumentEditForm(request.POST, instance=doc, user=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, _('Документ обновлён'))
            return redirect('document_detail', pk=doc.id)
    else:
        form = DocumentEditForm(instance=doc, user=request.user)

    return render(request, 'ivanna_model_manager/document_detail.html', {
        'document': doc,
        'chunks': chunks,
        'processing_history': history,
        'similar_docs': similar,
        'form': form,
        'active_tab': 'rag',
    })


@login_required
def document_edit(request, pk):
    """
    Редактирование метаданных документа.
    Поддерживает AJAX и обычные POST-запросы.
    При установке флага regenerate_chunks запускает задачу пересоздания чанков.
    """
    document = get_object_or_404(Document, pk=pk, is_active=True)

    # Проверка прав: владелец коллекции или администратор
    if not request.user.is_staff:
        if document.collection and document.collection.owner_id != str(request.user.id):
            raise PermissionDenied("You do not have permission to edit this document.")

    if request.method == 'POST':
        form = DocumentEditForm(request.POST, instance=document, user=request.user)
        if form.is_valid():
            doc = form.save(commit=False)
            # Обновляем метаданные из полей формы (chunk_size, chunk_overlap)
            metadata = doc.metadata or {}
            chunk_size = form.cleaned_data.get('chunk_size')
            chunk_overlap = form.cleaned_data.get('chunk_overlap')
            if chunk_size is not None:
                metadata['chunk_size'] = chunk_size
            if chunk_overlap is not None:
                metadata['chunk_overlap'] = chunk_overlap
            doc.metadata = metadata
            doc.save()

            # Если отмечена опция пересоздания чанков – запускаем задачу
            task_id = None
            if form.cleaned_data.get('regenerate_chunks'):
                task = regenerate_document_chunks_task.delay(doc.id)
                task_id = task.id
                msg = "Документ обновлён, запущено пересоздание чанков."
            else:
                msg = "Документ успешно обновлён."

            # Ответ для AJAX
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': msg,
                    'redirect_url': reverse('document_detail', args=[doc.id]),
                    'task_id': task_id
                })
            # Обычный POST – редирект с сообщением
            messages.success(request, msg)
            return redirect('document_detail', pk=doc.id)
        else:
            # Ошибки валидации
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                errors = {field: [str(e) for e in errs] for field, errs in form.errors.items()}
                return JsonResponse({'success': False, 'errors': errors}, status=400)
            # При обычном POST покажем форму с ошибками (рендер ниже)
    else:
        form = DocumentEditForm(instance=document, user=request.user)

    # Для GET или не AJAX POST с ошибками – рендерим шаблон
    collections = DocumentCollection.objects.filter(
        Q(owner_id=str(request.user.id)) | Q(is_public=True),
        is_active=True
    ).order_by('name')

    context = {
        'form': form,
        'document': document,
        'collections': collections,
    }
    return render(request, 'ivanna_model_manager/document_edit.html', context)


# --------------------------------------------------------------------------
# УДАЛЕНИЕ ДОКУМЕНТА (мягкое удаление)
# --------------------------------------------------------------------------
@login_required
@require_POST
def document_delete(request, pk):
    try:
        document = get_object_or_404(Document, pk=pk, is_active=True)

        if not request.user.is_staff:
            if document.collection and document.collection.owner_id != str(request.user.id):
                return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)

        # Логируем удаление
        DeletedDocumentLog.objects.create(
            original_name=document.original_filename,
            title=document.title,
            document_type=document.document_type,
            file_size_mb=document.file_size_mb,
            collection_name=document.collection.name if document.collection else '',
            total_chunks=document.chunks.filter(is_active=True).count(),
        )

        document.is_active = False
        document.status = 'ARCHIVED'
        document.save()

        if document.collection:
            document.collection.update_statistics()

        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'message': _('Документ удалён'),
                'redirect_url': reverse('document_list')
            })

        messages.success(request, _('Документ удалён'))
        return redirect('document_list')

    except Exception as e:
        logger.error(f"Error deleting document {pk}: {e}", exc_info=True)
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
        raise


# --------------------------------------------------------------------------
# ПЕРЕЗАГРУЗКА (ЗАМЕНА) ФАЙЛА ДОКУМЕНТА
# --------------------------------------------------------------------------
@login_required
@require_POST
def document_reload(request, pk):
    """
    Замена файла документа. Принимает новый файл, сохраняет его,
    обновляет метаданные и запускает полную переобработку (process_document_task).
    """
    document = get_object_or_404(Document, pk=pk, is_active=True)

    # Проверка прав
    if not request.user.is_staff:
        if document.collection and document.collection.owner_id != str(request.user.id):
            return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)

    if 'file' not in request.FILES:
        return JsonResponse({'success': False, 'error': 'No file uploaded'}, status=400)

    uploaded_file = request.FILES['file']

    # Валидация расширения (можно вынести в отдельную функцию)
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    allowed_extensions = getattr(settings, 'ALLOWED_DOCUMENT_EXTENSIONS', {
        '.pdf': 'PDF', '.docx': 'DOCX', '.doc': 'DOCX',
        '.txt': 'TXT', '.html': 'HTML', '.htm': 'HTML',
        '.md': 'MARKDOWN', '.csv': 'CSV', '.xlsx': 'EXCEL', '.xls': 'EXCEL'
    })
    if ext not in allowed_extensions:
        return JsonResponse({'success': False, 'error': f'File type {ext} not allowed'}, status=400)

    # Генерируем уникальное имя для сохранения
    safe_name = f"{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'documents')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, safe_name)

    # Сохраняем файл
    with open(file_path, 'wb+') as dest:
        for chunk in uploaded_file.chunks():
            dest.write(chunk)

    # Удаляем старый файл (если он существует и не используется другими документами)
    if document.file_path and os.path.exists(document.full_file_path):
        try:
            os.remove(document.full_file_path)
        except OSError as e:
            logger.warning(f"Could not delete old file {document.full_file_path}: {e}")

    # Обновляем поля документа
    document.original_filename = uploaded_file.name
    document.file_path = os.path.join('documents', safe_name)
    document.file_size_mb = uploaded_file.size / (1024 * 1024)
    document.document_type = allowed_extensions.get(ext, 'UNKNOWN')
    document.content_hash = ''  # будет вычислено при обработке
    document.raw_content = ''
    document.processed_content = ''
    # Сбрасываем статус
    document.status = 'UPLOADED'
    document.processing_progress = 0
    document.error_message = ''
    document.processed_at = None
    document.indexed_at = None
    # Деактивируем старые чанки (они будут удалены задачей или позже)
    document.chunks.update(is_active=False)
    document.save()

    # Запускаем обработку
    task = process_document_task.delay(document.id)

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({
            'success': True,
            'message': _('Файл заменён, запущена обработка'),
            'task_id': task.id,
            'redirect_url': reverse('document_detail', args=[document.id])
        })

    messages.success(request, _('Файл заменён, запущена обработка'))
    return redirect('document_detail', pk=document.id)

@login_required
def rag_search_page(request, collection_id=None):
    """
    Страница расширенного поиска по документам RAG.
    Поддерживает выбор коллекции, тип поиска, параметры гибридного поиска,
    историю запросов и сохранённые поиски.
    """
    from .models import DocumentCollection, SearchQuery, DocumentChunk
    from django.db.models import Q, Count, Avg
    from django.core.paginator import Paginator

    # Доступные коллекции
    collections = DocumentCollection.objects.filter(
        Q(owner_id=str(request.user.id)) | Q(is_public=True),
        is_active=True
    ).annotate(
        doc_count=Count('documents', filter=Q(documents__is_active=True)),
        chunk_count=Count('documents__chunks', filter=Q(documents__chunks__is_active=True)),
        has_embeddings=Count('documents__chunks', filter=Q(documents__chunks__has_embedding=True))
    ).order_by('-created_at')

    # Выбранная коллекция
    selected_collection = None
    collection_stats = None
    if collection_id:
        try:
            selected_collection = collections.get(id=collection_id)
            total_docs = selected_collection.documents.filter(is_active=True).count()
            total_chunks = DocumentChunk.objects.filter(
                document__collection=selected_collection,
                is_active=True
            ).count()
            chunks_with_emb = DocumentChunk.objects.filter(
                document__collection=selected_collection,
                has_embedding=True,
                is_active=True
            ).count()
            collection_stats = {
                'total_documents': total_docs,
                'total_chunks': total_chunks,
                'chunks_with_embeddings': chunks_with_emb,
                'embedding_progress': round((chunks_with_emb / total_chunks * 100) if total_chunks > 0 else 0, 1),
                'vector_index_created': selected_collection.vector_index_created,
                'total_vectors': selected_collection.total_vectors,
                'embedding_model': selected_collection.embedding_model.name if selected_collection.embedding_model else 'Не задана',
                'chunk_size': selected_collection.chunk_size,
                'chunk_overlap': selected_collection.chunk_overlap,
                'created_at': selected_collection.created_at,
                'is_owner': selected_collection.owner_id == str(request.user.id),
            }
        except DocumentCollection.DoesNotExist:
            messages.warning(request, _('Коллекция не найдена или нет доступа'))

    # История поиска
    search_history = SearchQuery.objects.filter(
        user_id=str(request.user.id)
    ).select_related('collection_filter').order_by('-created_at')[:20]

    # Популярные запросы
    popular_queries = SearchQuery.objects.filter(
        user_id=str(request.user.id)
    ).values('query_text').annotate(
        count=Count('id'),
        avg_time=Avg('processing_time_ms')
    ).order_by('-count')[:10]

    # Сохранённые поиски (если модель существует)
    saved_searches = []
    try:
        from .models import SavedSearch
        saved_searches = SavedSearch.objects.filter(
            user_id=str(request.user.id)
        ).order_by('-is_favorite', '-last_used_at')[:10]
    except ImportError:
        pass

    # Статистика
    total_searches = SearchQuery.objects.filter(user_id=str(request.user.id)).count()
    searches_today = SearchQuery.objects.filter(
        user_id=str(request.user.id),
        created_at__date=timezone.now().date()
    ).count()
    avg_response_time = SearchQuery.objects.filter(
        user_id=str(request.user.id)
    ).aggregate(avg=Avg('processing_time_ms'))['avg'] or 0

    context = {
        'active_tab': 'rag',
        'collections': collections,
        'selected_collection': selected_collection,
        'collection_stats': collection_stats,
        'search_history': search_history,
        'popular_queries': popular_queries,
        'saved_searches': saved_searches,
        'stats': {
            'total_searches': total_searches,
            'searches_today': searches_today,
            'avg_response_time': round(avg_response_time, 2),
        },
        # Параметры по умолчанию (будут использоваться в JavaScript)
        'default_search_type': 'hybrid',
        'default_top_k': 10,
        'default_semantic_weight': 0.7,
        'default_keyword_weight': 0.2,
        'default_freshness_weight': 0.1,
        'default_use_reranking': True,
        'default_threshold': 0.5,
    }
    return render(request, 'ivanna_model_manager/rag_search_page.html', context)


@login_required
@require_POST
def ajax_rag_advanced_search(request):
    """
    AJAX-эндпоинт для расширенного поиска.
    Использует EnhancedSearchService.
    """
    import json
    from .services.rag_service import EnhancedSearchService
    from .models import DocumentCollection, DocumentChunk, SearchQuery, SearchResult
    from django.db.models import Q
    from django.utils import timezone

    try:
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            data = request.POST

        query = data.get('query', '').strip()
        collection_id = data.get('collection_id')
        search_type = data.get('search_type', 'hybrid')
        k = int(data.get('k', 10))
        threshold = float(data.get('threshold', 0.5))
        semantic_weight = float(data.get('semantic_weight', 0.7))
        keyword_weight = float(data.get('keyword_weight', 0.2))
        freshness_weight = float(data.get('freshness_weight', 0.1))
        use_reranking = data.get('use_reranking', 'true') in ('true', 'True', True)
        save_query = data.get('save_query', 'true') in ('true', 'True', True)
        search_name = data.get('search_name', '')

        if not query:
            return JsonResponse({'success': False, 'error': _('Запрос не может быть пустым')}, status=400)
        if not collection_id:
            return JsonResponse({'success': False, 'error': _('Не выбрана коллекция')}, status=400)

        try:
            collection = DocumentCollection.objects.get(
                Q(id=collection_id) & 
                (Q(owner_id=str(request.user.id)) | Q(is_public=True)) & 
                Q(is_active=True)
            )
        except DocumentCollection.DoesNotExist:
            return JsonResponse({'success': False, 'error': _('Коллекция не найдена')}, status=404)

        # Выполняем поиск
        search_service = EnhancedSearchService()
        start_time = timezone.now()

        if search_type == 'semantic':
            results = search_service.semantic_search(query, collection.id, k=k, threshold=threshold)
        elif search_type == 'keyword':
            results = search_service.keyword_search(query, collection.id, k=k)
        else:  # hybrid
            results = search_service.enhanced_hybrid_search(
                query=query,
                collection_id=collection.id,
                k=k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                freshness_weight=freshness_weight,
                use_reranking=use_reranking
            )

        processing_time = (timezone.now() - start_time).total_seconds() * 1000

        # Сохраняем запрос, если требуется
        search_query = None
        if save_query:
            search_query = SearchQuery.objects.create(
                query_text=query,
                search_type=search_type.upper(),
                collection_filter=collection,
                user_id=str(request.user.id),
                processing_time_ms=processing_time,
                search_config={
                    'search_type': search_type,
                    'k': k,
                    'threshold': threshold,
                    'semantic_weight': semantic_weight,
                    'keyword_weight': keyword_weight,
                    'freshness_weight': freshness_weight,
                    'use_reranking': use_reranking,
                    'search_name': search_name,
                }
            )
            # Сохраняем результаты
            for i, res in enumerate(results[:k]):
                try:
                    chunk = DocumentChunk.objects.get(id=res['chunk_id'])
                    # Подготовим metadata, сериализуемый в JSON
                    metadata = res.copy()
                    # Преобразуем datetime в строку, если есть
                    if 'document_uploaded_at' in metadata and metadata['document_uploaded_at']:
                        metadata['document_uploaded_at'] = metadata['document_uploaded_at'].isoformat()
                    # Также проверяем другие возможные datetime поля (например, created_at, если есть)
                    for key in list(metadata.keys()):
                        if hasattr(metadata[key], 'isoformat'):
                            metadata[key] = metadata[key].isoformat()
                    SearchResult.objects.create(
                        query=search_query,
                        chunk=chunk,
                        relevance_score=res.get('final_score', res.get('similarity', 0)),
                        rank_position=i + 1,
                        scoring_method=search_type,
                        metadata=metadata
                    )
                except DocumentChunk.DoesNotExist:
                    continue

        # Форматируем для фронтенда
        formatted_results = []
        for res in results[:k]:
            try:
                chunk = DocumentChunk.objects.select_related('document').get(id=res['chunk_id'])
                formatted_results.append({
                    'chunk_id': chunk.id,
                    'document_id': chunk.document.id,
                    'document_title': chunk.document.title,
                    'content': chunk.content,
                    'preview': chunk.content[:300] + ('...' if len(chunk.content) > 300 else ''),
                    'relevance_score': res.get('final_score', res.get('similarity', 0)),
                    'scores': {
                        'semantic': res.get('semantic_score', 0),
                        'keyword': res.get('keyword_score', 0),
                        'freshness': res.get('freshness_score', 0),
                        'reranker': res.get('reranker_score', 0),
                        'final': res.get('final_score', 0),
                    },
                    'source': res.get('source', search_type),
                    'document_uploaded_at': chunk.document.uploaded_at.isoformat() if chunk.document.uploaded_at else None,
                    'chunk_index': chunk.chunk_index,
                })
            except DocumentChunk.DoesNotExist:
                continue

        return JsonResponse({
            'success': True,
            'query': query,
            'collection_id': collection.id,
            'collection_name': collection.name,
            'search_type': search_type,
            'total_results': len(formatted_results),
            'results': formatted_results,
            'processing_time_ms': round(processing_time, 2),
            'search_query_id': search_query.id if search_query else None,
        })
    except Exception as e:
        logger.error(f"Error in ajax_rag_advanced_search: {e}", exc_info=True)
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

@login_required
@require_POST
def document_process(request, pk):
    doc = get_object_or_404(Document, pk=pk, is_active=True)
    if not request.user.is_staff and (not doc.collection or doc.collection.owner_id != str(request.user.id)):
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)

    doc.status = 'UPLOADED'
    doc.save()
    task = process_document_task.delay(doc.id)
    return JsonResponse({'success': True, 'task_id': task.id})


@login_required
@require_POST
def document_reindex(request, pk):
    doc = get_object_or_404(Document, pk=pk, is_active=True)
    if not request.user.is_staff and (not doc.collection or doc.collection.owner_id != str(request.user.id)):
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)

    doc.chunks.update(is_active=False)
    doc.status = 'UPLOADED'
    doc.requires_reindexing = False
    doc.save()
    task = process_document_task.delay(doc.id)
    return JsonResponse({'success': True, 'task_id': task.id})


@login_required
def document_download(request, pk):
    doc = get_object_or_404(Document, pk=pk, is_active=True)
    if doc.collection:
        if doc.collection.owner_id != str(request.user.id) and not doc.collection.is_public:
            raise PermissionDenied
    response = doc.get_file_response()
    if response:
        return response
    messages.error(request, _('Файл не найден'))
    return redirect('document_detail', pk=doc.id)


@login_required
def rag_upload_documents(request):
    """
    Загрузка одного или нескольких документов в коллекцию.
    """
    from django.conf import settings

    if request.method == 'POST':
        use_multiple = 'files' in request.FILES
        if use_multiple:
            form = MultipleDocumentUploadForm(request.POST, request.FILES, user=request.user)
        else:
            form = DocumentUploadForm(request.POST, request.FILES, user=request.user)

        if form.is_valid():
            collection = form.cleaned_data['collection']
            auto_process = form.cleaned_data.get('auto_process', True)
            files = form.cleaned_data['files'] if use_multiple else [form.cleaned_data['file']]
            if not isinstance(files, (list, tuple)):
                files = [files]

            created = []
            for f in files:
                try:
                    safe_name = sanitize_filename(f.name)
                    unique = str(uuid.uuid4())[:8]
                    final = f"{unique}_{safe_name}"
                    upload_dir = os.path.join(settings.MEDIA_ROOT, 'documents')
                    os.makedirs(upload_dir, exist_ok=True)
                    full_path = os.path.join(upload_dir, final)
                    with open(full_path, 'wb+') as dest:
                        for chunk in f.chunks():
                            dest.write(chunk)

                    # Извлекаем текст для raw_content (упрощённо)
                    raw_content = ""
                    ext = os.path.splitext(f.name)[1].lower()
                    if ext == '.pdf':
                        try:
                            import pdfplumber
                            with pdfplumber.open(full_path) as pdf:
                                raw_content = "\n".join(p.extract_text() or "" for p in pdf.pages)
                        except:
                            pass
                    elif ext in ['.txt', '.md', '.csv']:
                        try:
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as rf:
                                raw_content = rf.read()
                        except:
                            pass

                    doc = Document.objects.create(
                        collection=collection,
                        original_filename=f.name,
                        title=form.cleaned_data.get('title') or os.path.splitext(f.name)[0],
                        description=form.cleaned_data.get('description', ''),
                        category=form.cleaned_data.get('category', ''),
                        tags=form.cleaned_data.get('tags', []),
                        document_type=get_document_type(f.name),
                        file_path=os.path.join('documents', final),
                        file_size_mb=f.size / (1024 * 1024),
                        raw_content=raw_content,
                        content_hash=hashlib.sha256(raw_content.encode()).hexdigest() if raw_content else '',
                        status='UPLOADED',
                        metadata={
                            'uploaded_by': str(request.user.id),
                            'uploaded_at': timezone.now().isoformat(),
                            'chunk_size': form.cleaned_data.get('chunk_size', 1000),
                            'chunk_overlap': form.cleaned_data.get('chunk_overlap', 200),
                        }
                    )
                    created.append(doc)
                    if auto_process:
                        process_document_task.delay(doc.id)
                        doc.status = 'PROCESSING'
                        doc.save()
                except Exception as e:
                    logger.error(f"Upload error: {e}")
                    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                        return JsonResponse({'success': False, 'error': str(e)}, status=500)
                    messages.error(request, f'Ошибка загрузки {f.name}: {e}')

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'success': True,
                    'message': f'Загружено {len(created)} файлов',
                    'document_ids': [d.id for d in created],
                })
            messages.success(request, f'Загружено {len(created)} файлов')
            return redirect('document_list')
        else:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'errors': form.errors}, status=400)
    else:
        show_multiple = request.GET.get('multiple') == '1'
        if show_multiple:
            form = MultipleDocumentUploadForm(user=request.user)
        else:
            form = DocumentUploadForm(user=request.user)

    collections = DocumentCollection.objects.filter(
        Q(owner_id=str(request.user.id)) | Q(is_public=True),
        is_active=True
    ).order_by('name')

    # Статистика для отображения в шаблоне
    upload_stats = {
        'max_file_size': f"{settings.MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
        'max_files': 10,  # можно вынести в settings
        'allowed_formats': ', '.join(settings.ALLOWED_DOCUMENT_EXTENSIONS.keys()),
    }

    return render(request, 'ivanna_model_manager/rag_upload_document.html', {
        'form': form,
        'show_multiple': request.GET.get('multiple') == '1',
        'collections': json.dumps([{'id': c.id, 'name': c.name, 'description': c.description or ''} for c in collections]),
        'collections_list': collections,
        'upload_stats': upload_stats,
        'active_tab': 'rag',
    })


@login_required
def document_chunk_edit(request, pk):
    chunk = get_object_or_404(DocumentChunk, pk=pk, is_active=True)
    if chunk.document.collection and chunk.document.collection.owner_id != str(request.user.id):
        raise PermissionDenied
    if request.method == 'POST':
        form = DocumentChunkEditForm(request.POST, instance=chunk)
        if form.is_valid():
            form.save()
            messages.success(request, _('Чанк обновлён'))
            return redirect('document_detail', pk=chunk.document.id)
    else:
        form = DocumentChunkEditForm(instance=chunk)
    return render(request, 'ivanna_model_manager/document_chunk_edit.html', {
        'form': form,
        'chunk': chunk,
        'document': chunk.document,
        'active_tab': 'rag',
    })


@login_required
def document_bulk_action(request):
    if request.method == 'POST':
        form = DocumentBulkActionForm(request.POST)
        if form.is_valid():
            action = form.cleaned_data['action']
            ids_str = form.cleaned_data.get('document_ids', '')
            ids = [int(i) for i in ids_str.split(',') if i.strip()]
            docs = Document.objects.filter(
                id__in=ids,
                collection__owner_id=str(request.user.id),
                is_active=True
            )
            if not docs.exists():
                messages.error(request, _('Документы не найдены или нет прав'))
                return redirect('document_list')

            if action == 'delete':
                count = 0
                for d in docs:
                    d.is_active = False
                    d.save()
                    count += 1
                messages.success(request, f'Удалено {count} документов')
            elif action == 'reprocess':
                count = 0
                for d in docs:
                    d.status = 'UPLOADED'
                    d.save()
                    process_document_task.delay(d.id)
                    count += 1
                messages.success(request, f'{count} документов отправлено на обработку')
            elif action == 'archive':
                count = 0
                for d in docs:
                    d.status = 'ARCHIVED'
                    d.save()
                    count += 1
                messages.success(request, f'Архивировано {count} документов')
            elif action == 'export':
                export = [{
                    'id': d.id,
                    'title': d.title,
                    'original_filename': d.original_filename,
                    'document_type': d.document_type,
                    'status': d.status,
                    'uploaded_at': d.uploaded_at.isoformat(),
                    'metadata': d.metadata,
                    'tags': d.tags,
                    'collection': d.collection.name if d.collection else None,
                } for d in docs]
                response = HttpResponse(
                    json.dumps(export, ensure_ascii=False, indent=2),
                    content_type='application/json'
                )
                response['Content-Disposition'] = 'attachment; filename="documents_export.json"'
                return response
            return redirect('document_list')
    else:
        form = DocumentBulkActionForm()
    return render(request, 'ivanna_model_manager/document_bulk_action.html', {'form': form, 'active_tab': 'rag'})


@login_required
def rag_chat_view(request):
    """Веб-интерфейс RAG-чата."""
    collections = DocumentCollection.objects.filter(
        Q(owner_id=str(request.user.id)) | Q(is_public=True),
        is_active=True
    ).order_by('-created_at')
    llm_models = ModelCheckpoint.objects.filter(status='LOADED', is_active=True).order_by('-parameter_count')
    embedding_models = EmbeddingModel.objects.filter(is_active=True).order_by('-is_default', 'name')
    rag_sessions = RAGSession.objects.filter(user_id=str(request.user.id)).order_by('-created_at')[:10]

    user_stats = {
        'total_sessions': RAGSession.objects.filter(user_id=str(request.user.id)).count(),
        'total_documents': Document.objects.filter(
            collection_id__in=list(collections.values_list('id', flat=True)),
            is_active=True
        ).count(),
        'recent_queries': list(rag_sessions.values_list('original_query', flat=True)[:5]),
    }
    return render(request, 'ivanna_model_manager/rag_chat.html', {
        'document_collections': collections,
        'llm_models': llm_models,
        'embedding_models': embedding_models,
        'rag_sessions': rag_sessions,
        'user_stats': user_stats,
        'active_tab': 'rag',
    })


@login_required
@require_POST
def ajax_rag_chat(request):
    """AJAX-эндпоинт для RAG-чата."""
    import json
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        collection_id = data.get('collection_id')
        if not query or not collection_id:
            return JsonResponse({'success': False, 'error': 'Missing query or collection'})
        rag = RAGService()
        result = rag.process_query(
            query=query,
            user_id=str(request.user.id),
            collection_id=collection_id,
            system_prompt=data.get('system_prompt', ''),
            llm_model_id=data.get('llm_model_id'),
            embedding_model_id=data.get('embedding_model_id'),
            search_config=data.get('search_config', {}),
            generation_config=data.get('generation_config', {})
        )
        return JsonResponse(result)
    except Exception as e:
        logger.error(f"AJAX RAG error: {e}", exc_info=True)
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def ajax_document_status(request, document_id):
    """Статус обработки документа (AJAX)."""
    doc = get_object_or_404(Document, id=document_id, is_active=True)
    # Проверка доступа
    if doc.collection:
        if doc.collection.owner_id != str(request.user.id) and not doc.collection.is_public:
            return JsonResponse({'error': 'Access denied'}, status=403)
    from .models import DocumentProcessor
    processor = DocumentProcessor.objects.filter(document=doc).order_by('-started_at').first()
    chunks = doc.chunks.filter(is_active=True)
    return JsonResponse({
        'id': doc.id,
        'title': doc.title,
        'status': doc.status,
        'progress': doc.processing_progress,
        'error_message': doc.error_message,
        'chunk_stats': {
            'total': chunks.count(),
            'with_embeddings': chunks.filter(has_embedding=True).count(),
            'percentage': round(chunks.filter(has_embedding=True).count() / max(chunks.count(), 1) * 100, 1),
        },
        'has_embeddings': doc.has_embeddings,
        'requires_reindexing': doc.requires_reindexing,
        'uploaded_at': doc.uploaded_at.isoformat() if doc.uploaded_at else None,
        'processed_at': doc.processed_at.isoformat() if doc.processed_at else None,
        'indexed_at': doc.indexed_at.isoformat() if doc.indexed_at else None,
        'processing_log': {
            'status': processor.status if processor else None,
            'progress': processor.progress if processor else 0,
            'error_message': processor.error_message if processor else None,
            'chunks_created': processor.chunks_created if processor else 0,
            'embeddings_created': processor.embeddings_created if processor else 0,
        } if processor else None,
        'collection': {
            'id': doc.collection.id,
            'name': doc.collection.name,
            'vector_index_created': doc.collection.vector_index_created,
            'total_vectors': doc.collection.total_vectors,
        } if doc.collection else None,
    })


@login_required
@require_POST
def create_collection_embeddings(request, pk):
    """Создание эмбеддингов для коллекции (AJAX)."""
    collection = get_object_or_404(
        DocumentCollection,
        pk=pk,
        owner_id=str(request.user.id)
    )
    if not collection.embedding_model:
        emb = EmbeddingModel.objects.filter(is_default=True).first()
        if not emb:
            return JsonResponse({'success': False, 'error': 'No embedding model'})
        collection.embedding_model = emb
        collection.save()
    task = create_embeddings_for_collection_task.delay(collection.id)
    return JsonResponse({'success': True, 'task_id': task.id})


@login_required
@require_POST
def build_collection_index(request, pk):
    """Построить FAISS индекс для коллекции."""
    collection = get_object_or_404(
        DocumentCollection,
        pk=pk,
        owner_id=str(request.user.id)
    )
    index_type = request.POST.get('index_type', 'flat')
    task = build_faiss_index_task.delay(collection.id, index_type)
    return JsonResponse({'success': True, 'task_id': task.id})


@login_required
@require_POST
def update_collection_index(request, pk):
    """Обновить FAISS индекс (добавить новые векторы)."""
    collection = get_object_or_404(
        DocumentCollection,
        pk=pk,
        owner_id=str(request.user.id)
    )
    task = update_faiss_index_task.delay(collection.id)
    return JsonResponse({'success': True, 'task_id': task.id})


@login_required
@require_GET
def get_index_stats(request, pk):
    """Статистика FAISS индекса коллекции."""
    collection = get_object_or_404(
        DocumentCollection,
        Q(owner_id=str(request.user.id)) | Q(is_public=True),
        pk=pk,
        is_active=True
    )
    from .services.faiss_service import FAISSService
    faiss = FAISSService()
    stats = faiss.get_index_stats(collection.id)
    stats['collection_id'] = collection.id
    stats['collection_name'] = collection.name
    return JsonResponse({'success': True, 'stats': stats})


@login_required
@require_GET
def get_collection_vector_stats(request, pk):
    """Статистика векторного индекса коллекции (альтернативный метод)."""
    return get_index_stats(request, pk)


@login_required
@require_GET
def check_index_files(request, collection_id):
    """Проверить наличие файлов индекса на диске."""
    from .services.faiss_service import FAISSService
    faiss = FAISSService()
    index_path = faiss._get_index_path(collection_id)
    meta_path = faiss._get_metadata_path(collection_id)
    return JsonResponse({
        'index_exists': os.path.exists(index_path),
        'metadata_exists': os.path.exists(meta_path),
        'index_size_mb': os.path.getsize(index_path) / (1024 * 1024) if os.path.exists(index_path) else 0,
    })


# ============================================================================
# СИСТЕМНЫЕ МЕТРИКИ И СТАТУСЫ
# ============================================================================
class SystemMetricsView(LoginRequiredMixin, TemplateView):
    template_name = 'ivanna_model_manager/metrics.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update({
            'total_models': ModelCheckpoint.objects.filter(is_active=True).count(),
            'loaded_models': ModelCheckpoint.objects.filter(status='LOADED', is_active=True).count(),
            'downloading_models': ModelCheckpoint.objects.filter(status='DOWNLOADING', is_active=True).count(),
            'active_downloads': ModelDownloadJob.objects.filter(
                status__in=['PENDING', 'DOWNLOADING']
            ).select_related('checkpoint').order_by('-started_at')[:10],
        })
        return context


@login_required
def gpu_status_view(request):
    """Страница статуса GPU."""
    import torch
    gpu_info = {'available': False, 'count': 0}
    if torch.cuda.is_available():
        gpu_info = {
            'available': True,
            'count': torch.cuda.device_count(),
            'cuda_version': torch.version.cuda,
            'device_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    context = {
        'title': 'Статус GPU',
        'gpu_info': gpu_info,
        'total_models': ModelCheckpoint.objects.filter(is_active=True).count(),
        'loaded_models': ModelCheckpoint.objects.filter(status='LOADED', is_active=True).count(),
        'active_downloads': ModelDownloadJob.objects.filter(status__in=['PENDING', 'DOWNLOADING'])[:5],
    }
    return render(request, 'ivanna_model_manager/gpu_status.html', context)


@login_required
def health_check_view(request):
    """Единый health-check эндпоинт (JSON)."""
    try:
        from django.db import connection
        with connection.cursor() as c:
            c.execute("SELECT 1")
        db_ok = True
    except:
        db_ok = False
    try:
        redis_model_manager.redis.ping()
        redis_ok = True
    except:
        redis_ok = False

    return JsonResponse({
        'status': 'healthy' if db_ok and redis_ok else 'degraded',
        'timestamp': timezone.now().isoformat(),
        'database': 'connected' if db_ok else 'disconnected',
        'redis': 'connected' if redis_ok else 'disconnected',
        'celery': 'running',  # предположим
        'models': {
            'total': ModelCheckpoint.objects.filter(is_active=True).count(),
            'loaded': ModelCheckpoint.objects.filter(status='LOADED', is_active=True).count(),
        },
    })


@login_required
def task_status(request, task_id):
    """Статус задачи Celery (AJAX)."""
    from celery.result import AsyncResult
    try:
        res = AsyncResult(task_id)
        return JsonResponse({
            'task_id': task_id,
            'state': res.state,
            'status': res.status,
            'result': res.result if res.ready() else None,
            'info': res.info if hasattr(res, 'info') else None,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ============================================================================
# СТАТИЧЕСКИЕ СТРАНИЦЫ
# ============================================================================
class AboutView(LoginRequiredMixin, TemplateView):
    template_name = 'ivanna_model_manager/about.html'


class DocumentationView(LoginRequiredMixin, TemplateView):
    template_name = 'ivanna_model_manager/docs.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['model_types'] = ModelCheckpoint.MODEL_TYPE_CHOICES
        context['merge_methods'] = ModelMergeJob.MERGE_METHOD_CHOICES
        return context


# ============================================================================
# API-ПОДОБНЫЕ ВЬЮХИ ДЛЯ ШАБЛОНОВ (НЕ DRF)
# ============================================================================
class SupportedModelTypesView(LoginRequiredMixin, TemplateView):
    """Для шаблонов, возвращает JSON."""
    def get(self, request):
        data = [{'value': c[0], 'display': c[1]} for c in ModelCheckpoint.MODEL_TYPE_CHOICES]
        return JsonResponse(data, safe=False)


class ModelStatisticsView(LoginRequiredMixin, TemplateView):
    def get(self, request):
        stats = {
            'total_models': ModelCheckpoint.objects.filter(is_active=True).count(),
            'by_status': dict(ModelCheckpoint.objects.filter(is_active=True).values_list('status').annotate(Count('id'))),
            'by_type': dict(ModelCheckpoint.objects.filter(is_active=True).values_list('model_type').annotate(Count('id'))),
            'loaded_models': ModelCheckpoint.objects.filter(status='LOADED', is_active=True).count(),
        }
        return JsonResponse(stats)

# ============================================================================
# FineTuning
# ============================================================================
@login_required
def fine_tuning_dashboard(request):
    """Дашборд fine-tuning"""
    jobs = FineTuningJob.objects.filter(
        created_by=request.user.username
    ).order_by('-created_at')
    
    context = {
        'active_tab': 'fine_tuning',
        'jobs': jobs,
        'total_jobs': jobs.count(),
        'running_jobs': jobs.filter(status__in=['PREPARING', 'TRAINING']).count(),
        'completed_jobs': jobs.filter(status='COMPLETED').count(),
    }
    return render(request, 'ivanna_model_manager/fine_tuning/dashboard.html', context)

@login_required
def fine_tuning_metrics(request, pk):
    """
    Возвращает метрики обучения в формате JSON для построения графиков.
    """
    job = get_object_or_404(FineTuningJob, pk=pk, created_by=request.user.username)
    metrics = TrainingMetric.objects.filter(job=job).order_by('step')
    data = []
    for m in metrics:
        data.append({
            'step': m.step,
            'epoch': m.epoch,
            'time': m.time_elapsed,
            'train_loss': m.train_loss,
            'eval_loss': m.eval_loss,
            'grad_norm': m.grad_norm,
            'learning_rate': m.learning_rate,
        })
    return JsonResponse({'metrics': data})

@login_required
def fine_tuning_create(request):
    if request.method == 'POST':
        form = FineTuningJobForm(request.POST)
        if form.is_valid():
            base_model = form.cleaned_data['base_model']
            if not base_model.local_path or not os.path.exists(base_model.local_path):
                form.add_error('base_model', 'Базовая модель не загружена локально. Сначала загрузите модель.')
                return render(request, 'ivanna_model_manager/fine_tuning/create.html', _get_create_context(request, form))

            method = form.cleaned_data['method']

            # Проверка наличия данных для instruction/raft
            if method in ('instruction', 'raft'):
                distillation_ids = request.POST.getlist('distillation_datasets')
                manual_ids = request.POST.getlist('manual_datasets')
                hf_ids = request.POST.getlist('hf_datasets')
                if not distillation_ids and not manual_ids and not hf_ids:
                    form.add_error(None, 'Для методов instruction и RAFT необходимы датасеты инструкций.')
                    return render(request, 'ivanna_model_manager/fine_tuning/create.html', _get_create_context(request, form))

            # Создание выходной модели
            quantization = form.cleaned_data.get('quantization', 'int4')
            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
            new_model_name = f"tuned_{method}_{base_model.name}_{timestamp}"
            new_model_display = f"{base_model.display_name} (fine-tuned, {method})"

            base_path = base_model.local_path
            new_path = None
            if base_path and os.path.exists(base_path):
                new_path = os.path.join(os.path.dirname(base_path), new_model_name)
                os.makedirs(new_path, exist_ok=True)

            loader_config = base_model.loader_config.copy() if base_model.loader_config else {}

            output_model = ModelCheckpoint.objects.create(
                name=new_model_name,
                display_name=new_model_display,
                model_type=base_model.model_type,
                huggingface_id=base_model.huggingface_id,
                huggingface_revision=base_model.huggingface_revision,
                loader_config=loader_config,
                local_path=new_path,
                relative_path=f"models/{new_model_name}",
                status='PENDING',
                parameter_count=base_model.parameter_count,
                quantization=quantization,
                context_length=base_model.context_length,
                vocab_size=base_model.vocab_size,
                license_type=base_model.license_type,
                is_safe=base_model.is_safe,
                safety_score=base_model.safety_score,
                parent=base_model,
                is_active=True,
                tags=base_model.tags + ['fine-tuned']
            )

            job = form.save(commit=False)
            job.created_by = request.user.username
            job.output_model = output_model
            job.quantization = quantization
            job.save()

            # Сохраняем выбранные коллекции и датасеты в training_config
            training_config = job.training_config or {}
            # Коллекции
            collection_ids = request.POST.getlist('document_collections')
            if collection_ids:
                training_config['document_collections'] = [int(cid) for cid in collection_ids]
            # Датасеты дистилляции
            distillation_ids = request.POST.getlist('distillation_datasets')
            if distillation_ids:
                training_config['distillation_datasets'] = [int(did) for did in distillation_ids]
            # Ручные датасеты
            manual_ids = request.POST.getlist('manual_datasets')
            if manual_ids:
                training_config['manual_datasets'] = [int(mid) for mid in manual_ids]
            # HF датасеты
            hf_ids = request.POST.getlist('hf_datasets')
            if hf_ids:
                training_config['hf_datasets'] = [int(hid) for hid in hf_ids]

            if training_config:
                job.training_config = training_config
                job.save(update_fields=['training_config'])

            messages.success(request, f'Задача "{job.name}" создана. Модель-копия "{output_model.display_name}" ожидает обучения.')
            return redirect('fine_tuning_detail', pk=job.id)

    else:
        form = FineTuningJobForm()

    return render(request, 'ivanna_model_manager/fine_tuning/create.html', _get_create_context(request, form))

@login_required
def _get_create_context(request, form):
    from .models import ModelCheckpoint, DocumentCollection, EvaluationDataset, DistillationDataset, ManualInstructionSet, HuggingFaceDataset

    return {
        'active_tab': 'fine_tuning',
        'form': form,
        'available_models': ModelCheckpoint.objects.filter(is_active=True).order_by('name'),
        'available_collections': DocumentCollection.objects.filter(is_active=True).order_by('name'),
        'available_datasets': EvaluationDataset.objects.filter(total_questions__gt=0).order_by('name'),
        'distillation_datasets': DistillationDataset.objects.filter(created_by=request.user.username),
        'manual_datasets': ManualInstructionSet.objects.filter(created_by=request.user.username),
        'hf_datasets': HuggingFaceDataset.objects.filter(created_by=request.user.username, status='COMPLETED').order_by('-created_at'),
    }


@login_required
def fine_tuning_jobs(request):
    """Список всех задач fine-tuning"""
    from .models import FineTuningJob
    
    # Фильтрация
    status = request.GET.get('status', '')
    method = request.GET.get('method', '')
    
    jobs = FineTuningJob.objects.filter(created_by=request.user.username)
    
    if status:
        jobs = jobs.filter(status=status)
    if method:
        jobs = jobs.filter(method=method)
    
    # Сортировка
    sort = request.GET.get('sort', '-created_at')
    if sort.lstrip('-') in ['name', 'status', 'created_at', 'progress']:
        jobs = jobs.order_by(sort)
    else:
        jobs = jobs.order_by('-created_at')
    
    # Пагинация
    paginator = Paginator(jobs, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'active_tab': 'fine_tuning',
        'page_obj': page_obj,
        'status_choices': [
            ('PENDING', 'В ожидании'),
            ('PREPARING', 'Подготовка'),
            ('TRAINING', 'Обучение'),
            ('COMPLETED', 'Завершено'),
            ('FAILED', 'Ошибка'),
            ('CANCELLED', 'Отменено'),
        ],
        'method_choices': [
            ('document', 'Обучение на документах'),
            ('instruction', 'Обучение на инструкциях'),
            ('raft', 'RAFT'),
        ],
        'current_filters': {
            'status': status,
            'method': method,
            'sort': sort,
        }
    }
    return render(request, 'ivanna_model_manager/fine_tuning/jobs_list.html', context)


@login_required
def fine_tuning_detail(request, pk):
    """Детали задачи fine-tuning"""
    job = get_object_or_404(
        FineTuningJob,
        id=pk,
        created_by=request.user.username
    )
    
    context = {
        'active_tab': 'fine_tuning',
        'job': job,
        'checkpoints': job.checkpoints.all().order_by('step'),
        'evaluations': job.evaluations.all().order_by('-created_at'),
    }
    return render(request, 'ivanna_model_manager/fine_tuning/detail.html', context)


@login_required
def fine_tuning_logs(request, pk):
    """Просмотр логов задачи (AJAX)"""
    job = get_object_or_404(
        FineTuningJob,
        id=pk,
        created_by=request.user.username
    )
    return JsonResponse({'logs': job.logs})


# ============================================================================
# FINE-TUNING TEMPLATES
# ============================================================================

@login_required
def fine_tuning_templates(request):
    """Список шаблонов fine-tuning"""
    from .models import FineTuningTemplate
    
    # Фильтрация
    category = request.GET.get('category', '')
    method = request.GET.get('method', '')
    
    templates = FineTuningTemplate.objects.filter(is_public=True)
    
    if category:
        templates = templates.filter(category=category)
    if method:
        templates = templates.filter(method=method)
    
    # Статистика по категориям
    categories = templates.values('category').annotate(
        count=Count('id')
    ).order_by('category')
    
    context = {
        'active_tab': 'fine_tuning',
        'templates': templates,
        'categories': categories,
        'current_filters': {
            'category': category,
            'method': method,
        }
    }
    return render(request, 'ivanna_model_manager/fine_tuning/templates.html', context)


@login_required
def fine_tuning_template_detail(request, pk):
    """Детали шаблона"""
    from .models import FineTuningTemplate
    
    template = get_object_or_404(FineTuningTemplate, pk=pk, is_public=True)
    
    context = {
        'active_tab': 'fine_tuning',
        'template': template,
    }
    return render(request, 'ivanna_model_manager/fine_tuning/template_detail.html', context)


@login_required
def fine_tuning_template_create(request):
    """Создание нового шаблона"""
    from .forms import FineTuningTemplateForm
    
    if request.method == 'POST':
        form = FineTuningTemplateForm(request.POST)
        if form.is_valid():
            template = form.save(commit=False)
            template.created_by = request.user.username
            template.save()
            messages.success(request, f'Шаблон "{template.name}" создан')
            return redirect('fine_tuning_templates')
    else:
        form = FineTuningTemplateForm()
    
    context = {
        'active_tab': 'fine_tuning',
        'form': form,
    }
    return render(request, 'ivanna_model_manager/fine_tuning/template_create.html', context)


@login_required
def fine_tuning_template_apply(request, pk):
    """Применить шаблон к созданию задачи"""
    from .models import FineTuningTemplate
    from .forms import FineTuningJobForm
    
    template = get_object_or_404(FineTuningTemplate, pk=pk, is_public=True)
    template.increment_usage()
    
    # Предзаполняем форму данными из шаблона
    initial_data = {
        'method': template.method,
        'lora_r': template.lora_r,
        'lora_alpha': template.lora_alpha,
        'lora_dropout': template.lora_dropout,
        'learning_rate': template.learning_rate,
        'batch_size': template.batch_size,
        'gradient_accumulation_steps': template.gradient_accumulation_steps,
        'num_epochs': template.num_epochs,
        'max_seq_length': template.max_seq_length,
        'distractors_per_example': template.distractors_per_example,
        'oracle_ratio': template.oracle_ratio,
        'distractor_ratio': template.distractor_ratio,
        'no_context_ratio': template.no_context_ratio,
        'use_kl_regularization': template.use_kl_regularization,
        'kl_lambda': template.kl_lambda,
        'general_instructions_ratio': template.general_instructions_ratio,
    }
    
    # Перенаправляем на создание задачи с предзаполненной формой
    return redirect('fine_tuning_create_with_template', template_id=template.id)


# ============================================================================
# CONVERSION
# ============================================================================

@login_required
def conversion_dashboard(request):
    """Дашборд конвертации"""
    jobs = ModelConversionJob.objects.filter(created_by=request.user.username).order_by('-created_at')
    context = {
        'active_tab': 'conversion',
        'jobs': jobs,
        'total_jobs': jobs.count(),
        'running_jobs': jobs.filter(status__in=['PREPARING', 'CONVERTING']).count(),
        'completed_jobs': jobs.filter(status='COMPLETED').count(),
    }
    return render(request, 'ivanna_model_manager/conversion/dashboard.html', context)

@login_required
def conversion_create(request):
    if request.method == 'POST':
        form = ModelConversionForm(request.POST)
        if form.is_valid():
            job = form.save(commit=False)
            job.created_by = request.user.username
            job.save()
            messages.success(request, f'Задача конвертации "{job.name}" создана')
            return redirect('conversion_detail', pk=job.id)
    else:
        form = ModelConversionForm()
    context = {
        'active_tab': 'conversion',
        'form': form,
        'available_models': ModelCheckpoint.objects.filter(
            is_active=True, local_path__isnull=False
        ).exclude(local_path='').order_by('name'),
    }
    return render(request, 'ivanna_model_manager/conversion/create.html', context)

@login_required
def conversion_detail(request, pk):
    job = get_object_or_404(ModelConversionJob, pk=pk, created_by=request.user.username)
    context = {
        'active_tab': 'conversion',
        'job': job,
    }
    return render(request, 'ivanna_model_manager/conversion/detail.html', context)

@login_required
def conversion_logs(request, pk):
    job = get_object_or_404(ModelConversionJob, pk=pk, created_by=request.user.username)
    return JsonResponse({'logs': job.logs})

@login_required
def conversion_progress(request, pk):
    job = get_object_or_404(ModelConversionJob, pk=pk, created_by=request.user.username)
    return JsonResponse({
        'progress': job.progress,
        'current_step': job.current_step,
        'status': job.status
    })



# ============================================================================
# Embedding
# ============================================================================
class EmbeddingModelListView(LoginRequiredMixin, ListView):
    model = EmbeddingModel
    template_name = 'ivanna_model_manager/embedding_models/embedding_model_list.html'
    context_object_name = 'models'
    paginate_by = 20

    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.order_by('-is_default', '-is_active', 'name')
        return qs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Добавляем статистику
        context['total_models'] = EmbeddingModel.objects.count()
        context['downloaded_models'] = EmbeddingModel.objects.filter(is_downloaded=True).count()
        return context


class EmbeddingModelCreateView(LoginRequiredMixin, CreateView):
    model = EmbeddingModel
    form_class = EmbeddingModelForm
    template_name = 'ivanna_model_manager/embedding_models/embedding_model_form.html'
    success_url = reverse_lazy('embedding_model_list')

    def get_initial(self):
        initial = super().get_initial()
        initial['is_active'] = True
        return initial

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Добавить модель эмбеддингов'
        return context


class EmbeddingModelUpdateView(LoginRequiredMixin, UpdateView):
    model = EmbeddingModel
    form_class = EmbeddingModelForm
    template_name = 'ivanna_model_manager/embedding_models/embedding_model_form.html'
    success_url = reverse_lazy('embedding_model_list')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = f'Редактирование модели: {self.object.display_name or self.object.name}'
        return context


@login_required
@require_POST
def embedding_model_delete(request, pk):
    model = get_object_or_404(EmbeddingModel, pk=pk)
    # Не даём удалить модель по умолчанию, если есть другие
    if model.is_default and EmbeddingModel.objects.filter(is_default=True).count() <= 1:
        return JsonResponse({'success': False, 'error': 'Cannot delete the default model'}, status=400)

    model.delete()
    return JsonResponse({'success': True})


@login_required
@require_POST
def embedding_model_download(request, pk):
    """Запуск загрузки модели эмбеддингов."""
    model = get_object_or_404(EmbeddingModel, pk=pk, is_active=True)

    if model.is_downloaded:
        return JsonResponse({'success': False, 'error': 'Model already downloaded'})

    if model.download_status == 'downloading':
        return JsonResponse({'success': False, 'error': 'Model is already downloading'})

    # Сбрасываем статус
    model.download_status = 'not_started'
    model.download_progress = 0
    model.download_error = ''
    model.save()

    # Запускаем задачу с user_id
    from .tasks_embedding import download_embedding_model_task
    task = download_embedding_model_task.delay(model.id, user_id=request.user.id)
    return JsonResponse({'success': True, 'task_id': task.id})


@login_required
def embedding_model_download_status(request, pk):
    """Возвращает статус загрузки модели."""
    model = get_object_or_404(EmbeddingModel, pk=pk)
    return JsonResponse({
        'is_downloaded': model.is_downloaded,
        'download_status': model.download_status,
        'download_progress': model.download_progress,
        'download_error': model.download_error,
    })

@login_required
@require_POST
def embedding_model_reset_download(request, pk):
    """
    Сбрасывает статус загрузки модели и удаляет Redis-блокировку.
    """
    model = get_object_or_404(EmbeddingModel, pk=pk)
    # Сбрасываем статус
    model.download_status = 'not_started'
    model.download_progress = 0
    model.download_error = ''
    model.save(update_fields=['download_status', 'download_progress', 'download_error'])
    # Удаляем Redis-блокировку
    from django.core.cache import cache
    cache.delete(f"download_embedding_model:{model.id}")
    logger.info(f"Сброшен статус загрузки для модели {model.name} (ID={model.id})")
    return JsonResponse({'success': True})

class EmbeddingModelDetailView(LoginRequiredMixin, DetailView):
    model = EmbeddingModel
    template_name = 'ivanna_model_manager/embedding_models/embedding_model_detail.html'
    context_object_name = 'model'

@login_required
def embedding_model_delete(request, pk):
    model = get_object_or_404(EmbeddingModel, pk=pk)
    if request.method == 'POST':
        model.delete()
        return redirect('embedding_model_list')
    return render(request, 'ivanna_model_manager/embedding_models/embedding_model_confirm_delete.html', {'model': model})

@login_required
@require_POST
def rebuild_embeddings_index(request, pk):
    collection = get_object_or_404(DocumentCollection, pk=pk, owner_id=str(request.user.id))
    task = rebuild_embeddings_and_index_task.delay(collection.id)
    return JsonResponse({'success': True, 'task_id': task.id})

@login_required
@require_POST
def rebuild_collection_embeddings(request, pk):
    """Полная перестройка эмбеддингов и FAISS индекса для коллекции."""
    logger.info(f"rebuild_collection_embeddings called by {request.user}, headers: {request.headers}")
    collection = get_object_or_404(DocumentCollection, pk=pk, is_active=True)
    # Проверка прав
    if not request.user.is_staff and collection.owner_id != str(request.user.id):
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)

    from .tasks_rag import rebuild_embeddings_and_index_task
    task = rebuild_embeddings_and_index_task.delay(collection.id)
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'success': True, 'task_id': task.id})
    else:
        messages.info(request, _('Задача перестройки эмбеддингов запущена'))
        return redirect('rag_collection_detail', pk=collection.id)

@login_required
@require_POST
def change_collection_embedding_model(request, pk):
    logger.info(f"Changing embedding model for collection {pk}, data: {request.POST}")  
    """
    Изменяет модель эмбеддингов для коллекции.
    """
    collection = get_object_or_404(
        DocumentCollection,
        Q(owner_id=str(request.user.id)) | Q(is_public=True),
        pk=pk,
        is_active=True
    )
    model_id = request.POST.get('embedding_model_id')
    if not model_id:
        return JsonResponse({'success': False, 'error': 'No model ID provided'})
    try:
        # Проверяем, что модель существует и активна
        model = EmbeddingModel.objects.get(id=model_id, is_active=True)
        collection.embedding_model = model
        collection.save()
        return JsonResponse({
            'success': True,
            'model_name': model.display_name or model.name,
            'model_id': model.id
        })
    except EmbeddingModel.DoesNotExist:
        logger.error(f"Embedding model with id {model_id} not found or inactive")
        return JsonResponse({'success': False, 'error': 'Model not found'})

# ============================================================================
# ПРЕДСТАВЛЕНИЯ ДЛЯ ДИСТИЛЛЯЦИИ
# ============================================================================

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

# ============================================================================
# Instruction
# ============================================================================

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
        from .tasks import import_huggingface_dataset_task
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

    from .tasks import import_huggingface_dataset_task
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

# ═══════════════════════════ МЯГКИЙ MoE (v5.2) ═══════════════════════════
# Тематические эксперты без дообучения: скан активаций → steer-патч → гейты.
# Порт weight_lab Phase A; рецепт — docs/soft_moe.md.

def _moe_stats(jobs):
    return {
        'total': jobs.count(),
        'running': jobs.filter(status__in=['PENDING', 'VALIDATING', 'SCANNING',
                                           'PATCHING', 'EVALUATING', 'ROUTING']).count(),
        'completed': jobs.filter(status='COMPLETED').count(),
        'failed': jobs.filter(status='FAILED').count(),
    }


@login_required
def moe_dashboard(request):
    from .models import MoETheme, MoEExpertJob

    themes = MoETheme.objects.all()
    jobs = MoEExpertJob.objects.select_related('theme', 'base_checkpoint', 'output_model')
    context = {
        'active_tab': 'soft_moe',
        'themes': themes,
        'jobs': jobs[:50],
        **_moe_stats(jobs),
    }
    return render(request, 'ivanna_model_manager/moe/dashboard.html', context)


def _theme_prompt_stats(theme):
    """Статистика наборов промптов темы для карточек/деталки."""
    from .services.soft_moe import prompts as moe_prompts
    out = {}
    for attr, fn in (('domain', theme.domain_prompts_path),
                     ('contrast', theme.contrast_prompts_path),
                     ('eval_domain', theme.eval_domain_path),
                     ('eval_contrast', theme.eval_contrast_path)):
        texts = moe_prompts.load_texts(fn())
        out[attr] = {'n': len(texts), **moe_prompts.length_stats(texts)}
    return out


@login_required
def moe_theme_create(request):
    from .forms import MoEThemeForm

    if request.method == 'POST':
        form = MoEThemeForm(request.POST)
        if form.is_valid():
            theme = form.save()
            messages.success(request, f'Тема «{theme.name}» создана. '
                                      f'Проверьте наборы валидацией (гейт bge).')
            return redirect('moe_theme_detail', pk=theme.pk)
    else:
        form = MoEThemeForm()
    return render(request, 'ivanna_model_manager/moe/theme_form.html',
                  {'form': form, 'active_tab': 'soft_moe', 'is_create': True})


@login_required
def moe_theme_edit(request, pk):
    from .forms import MoEThemeForm
    from .models import MoETheme

    theme = get_object_or_404(MoETheme, pk=pk)
    if request.method == 'POST':
        form = MoEThemeForm(request.POST, instance=theme)
        if form.is_valid():
            form.save()
            messages.success(request, f'Тема «{theme.name}» обновлена '
                                      f'(заполненные наборы перезаписаны).')
            return redirect('moe_theme_detail', pk=theme.pk)
    else:
        form = MoEThemeForm(instance=theme)
    return render(request, 'ivanna_model_manager/moe/theme_form.html',
                  {'form': form, 'theme': theme, 'active_tab': 'soft_moe',
                   'is_create': False})


@login_required
def moe_theme_detail(request, pk):
    from .models import MoETheme

    theme = get_object_or_404(MoETheme, pk=pk)
    jobs = theme.jobs.select_related('base_checkpoint', 'output_model')[:20]
    context = {
        'active_tab': 'soft_moe',
        'theme': theme,
        'jobs': jobs,
        'prompt_stats': _theme_prompt_stats(theme),
        'validation_report_json': json.dumps(theme.validation_report, ensure_ascii=False,
                                             indent=2) if theme.validation_report else '',
        'router_report_json': json.dumps(theme.router_report, ensure_ascii=False,
                                         indent=2) if theme.router_report else '',
    }
    return render(request, 'ivanna_model_manager/moe/theme_detail.html', context)


@login_required
@require_POST
def moe_theme_validate(request, pk):
    """Запуск гейта валидации наборов (CPU-задача)."""
    from .models import MoETheme
    from .tasks_moe import moe_validate_theme_task

    theme = get_object_or_404(MoETheme, pk=pk)
    if not theme.domain_prompts_file or not theme.contrast_prompts_file:
        messages.error(request, 'Нужны оба набора: доменные и контрастные промпты.')
        return redirect('moe_theme_detail', pk=pk)
    task = moe_validate_theme_task.delay(theme.pk)
    messages.info(request, f'Валидация наборов запущена (задача {task.id[:8]}…). '
                           f'Обновите страницу через минуту.')
    return redirect('moe_theme_detail', pk=pk)


@login_required
@require_POST
def moe_theme_router_build(request, pk):
    """Центроиды роутера + тест точности (CPU-задача)."""
    from .models import MoETheme
    from .tasks_moe import moe_router_build_task

    theme = get_object_or_404(MoETheme, pk=pk)
    if not theme.domain_prompts_file or not theme.contrast_prompts_file:
        messages.error(request, 'Нужны оба набора: доменные и контрастные промпты.')
        return redirect('moe_theme_detail', pk=pk)
    task = moe_router_build_task.delay(theme.pk)
    messages.info(request, f'Построение роутера запущено (задача {task.id[:8]}…).')
    return redirect('moe_theme_detail', pk=pk)


@login_required
@require_POST
def moe_theme_route_test(request, pk):
    """Маршрут одного запроса через центроиды темы (синхронно, для проверки)."""
    from .models import MoETheme
    from .services.soft_moe import router as moe_router

    theme = get_object_or_404(MoETheme, pk=pk)
    text = (request.POST.get('text') or '').strip()
    if not text:
        messages.error(request, 'Введите текст запроса.')
        return redirect('moe_theme_detail', pk=pk)
    try:
        C_d, C_c = moe_router.load_centroids(theme.theme_dir())
        res = moe_router.route_text(text, C_d, C_c, theme.router_threshold,
                                    embedding_model_id=theme.get_embedding_model_id())
        verdict = 'ЭКСПЕРТ' if res['route'] == 'expert' else 'GENERAL'
        messages.success(request, f'Маршрут: {verdict} (margin {res["margin"]:+.3f}, '
                                  f'cos домен {res["cos_domain"]:.3f} / контраст {res["cos_contrast"]:.3f})')
    except FileNotFoundError:
        messages.error(request, 'Центроиды не построены — сначала запустите роутер.')
    except Exception as e:
        messages.error(request, f'Ошибка маршрутизации: {e}')
    return redirect('moe_theme_detail', pk=pk)


@login_required
def moe_job_create(request):
    """Создание задачи сборки эксперта: валидации → джоба → GPU-очередь."""
    from .forms import MoEExpertJobForm
    from .models import MoEExpertJob, MoETheme
    from .tasks_moe import moe_build_expert_task

    if request.method == 'POST':
        form = MoEExpertJobForm(request.POST)
        if form.is_valid():
            theme = form.cleaned_data['theme']
            base = form.cleaned_data['base_checkpoint']

            if not base.local_path or not os.path.exists(base.local_path):
                form.add_error('base_checkpoint', 'Базовая модель не найдена локально — сначала загрузите её.')
            elif not theme.domain_prompts_file or not theme.contrast_prompts_file:
                form.add_error('theme', 'У темы нет наборов промптов.')
            elif theme.validation_status == 'FAILED':
                form.add_error('theme', 'Валидация наборов не пройдена — исправьте наборы '
                                        '(cross cos ≥ 0.5 говорит о пересечении домена и контраста).')
            else:
                job = form.save(commit=False)
                job.created_by = request.user.username
                job.name = job.name or f"moe_{theme.slug}_{base.name[:60]}"
                job.save()
                transaction.on_commit(lambda: moe_build_expert_task.delay(job.id))
                messages.success(request, f'Задача «{job.name}» поставлена в GPU-очередь: '
                                          f'скан → гейт AUC → патч → eval → роутер.')
                return redirect('moe_job_detail', pk=job.pk)
    else:
        form = MoEExpertJobForm()
        theme_id = request.GET.get('theme')
        if theme_id and MoETheme.objects.filter(pk=theme_id).exists():
            form.fields['theme'].initial = int(theme_id)

    return render(request, 'ivanna_model_manager/moe/job_form.html',
                  {'form': form, 'active_tab': 'soft_moe'})


def _moe_charts(job):
    """Данные для Plotly-визуализации страницы джобы (v5.2)."""
    import numpy as np

    charts = {}
    scan = job.scan_report or {}
    auc = scan.get('auc') or {}
    if auc.get('last'):
        charts['auc'] = {
            'layers': list(range(scan.get('n_layers', 0) + 1)),
            'train': auc['last'].get('train', []),
            'test': auc['last'].get('test', []),
            'gate': scan.get('gate_auc'),
        }
    writers = scan.get('writers') or {}
    if writers:
        per_layer = {}
        for w in writers.values():
            per_layer[w['layer']] = per_layer.get(w['layer'], 0.0) + abs(w['proj'])
        ls = sorted(per_layer)
        charts['writers'] = {'layers': ls, 'proj': [round(per_layer[l], 2) for l in ls]}

    # направления из npz: нормы по слоям + теплокарта |d̂| (слои × бакеты)
    try:
        npz_path = os.path.join(job.job_dir(), 'scan_dirs.npz')
        if os.path.exists(npz_path):
            Z = np.load(npz_path)
            D = np.abs(Z['d_last'])
            charts['dir_norms'] = [round(float(np.linalg.norm(D[l])), 2)
                                   for l in range(D.shape[0])]
            buckets = np.array_split(np.arange(D.shape[1]), 32)
            charts['dir_heat'] = [[round(float(D[l, b].mean()), 4) for b in buckets]
                                  for l in range(D.shape[0])]
    except Exception:
        pass

    patch = job.patch_report or {}
    ts = [t for t in (patch.get('tensors') or []) if 'delta_l1' in t]
    if ts:
        charts['deltas'] = [{'name': t['name'], 'delta': round(t['delta_l1'], 2)} for t in ts]
    heatmaps = [t for t in (patch.get('tensors') or []) if t.get('heatmap')]
    if heatmaps:
        charts['heatmaps'] = [{'name': t['name'], 'z': t['heatmap']} for t in heatmaps]

    sets = (job.eval_report or {}).get('sets') or {}
    if sets:
        agg = lambda k: sets.get(k, {}).get('agg', {}).get('mean_len', 0)
        charts['eval'] = {
            'labels': ['domain/base', 'domain/expert', 'general/base', 'general/expert'],
            'values': [agg('base/eval_domain'), agg('expert/eval_domain'),
                       agg('base/eval_general'), agg('expert/eval_general')],
        }
    return charts


def _moe_hints(job):
    """Интерпретация результатов — «что с этим делать» (v5.2)."""
    hints = []
    scan = job.scan_report or {}
    if scan:
        if scan.get('gate') != 'PASS':
            hints.append(('danger', 'AUC низкий — направление темы не выделяется. '
                          'Проверьте валидацию темы (cross cos, баланс длин), добавьте промптов. Патч в таком случае бесполезен.'))
        else:
            top = max(scan.get('auc', {}).get('last', {}).get('test', [0]))
            if top >= 0.98:
                hints.append(('success', f'AUC {top} — направление выражено очень чётко; '
                              'α=0.2 обычно безопасна, при деградации general снижайте.'))
        late = [w for w in (scan.get('top_writers') or []) if w['layer'] >= scan.get('n_layers', 24) * 0.5]
        if late:
            hints.append(('info', f'Писатели сосредоточены в поздних слоях (например L{late[0]["layer"]}) — '
                          'это ожидаемо: доменное поведение записывается в конце residual stream.'))
    patch = job.patch_report or {}
    if patch and patch.get('patched_count', 0) == 0:
        hints.append(('danger', 'Ни один тензор не патчился: проверьте mean-входы (для steer) и слои.'))
    er = (job.eval_report or {}).get('summary') or {}
    g = er.get('general_len', {}).get('delta_pct')
    d = er.get('domain_len', {}).get('delta_pct')
    if g is not None and g < -15:
        hints.append(('warning', f'General просел на {abs(g)}% — уменьшите α (сетка 0.1/0.05) '
                      'или сузьте список слоёв; патч слишком широкий.'))
    if d is not None and d < -15:
        hints.append(('warning', 'Доменные ответы заметно короче базы — числовые метрики — это прокси: '
                      'обязательно просмотрите тексты ответов ниже руками.'))
    rr = job.router_report or {}
    if rr.get('accuracy') is not None and rr.get('gate') != 'PASS':
        hints.append(('warning', 'Роутер путает темы — добавьте контрастных negative-примеров '
                      'в scan-наборы или поднимите порог margin.'))
    if not hints:
        hints.append(('info', 'Гейты пройдены: результаты в отчётах ниже; тексты eval — для ручной проверки.'))
    return hints


@login_required
def moe_job_detail(request, pk):
    from .models import MoEExpertJob

    job = get_object_or_404(MoEExpertJob.objects.select_related(
        'theme', 'base_checkpoint', 'output_model'), pk=pk)

    def _j(d):
        return json.dumps(d, ensure_ascii=False, indent=2) if d else ''

    context = {
        'active_tab': 'soft_moe',
        'job': job,
        'is_running': job.status in ['PENDING', 'VALIDATING', 'SCANNING', 'PATCHING',
                                     'EVALUATING', 'ROUTING'],
        'scan_report_json': _j(job.scan_report),
        'patch_report_json': _j(job.patch_report),
        'eval_report_json': _j(job.eval_report),
        'router_report_json': _j(job.router_report),
        'eval_texts': _moe_eval_texts(job),
        'charts_json': json.dumps(_moe_charts(job), ensure_ascii=False),
        'hints': _moe_hints(job),
    }
    return render(request, 'ivanna_model_manager/moe/job_detail.html', context)


def _moe_eval_texts(job):
    """Тексты eval-гейта для ручной выборочной проверки (парные base/expert)."""
    if not job.eval_report:
        return []
    eval_dir = os.path.join(job.job_dir(), 'eval')
    if not os.path.isdir(eval_dir):
        return []
    texts = []
    for fname in sorted(os.listdir(eval_dir)):
        if fname.endswith('.txt'):
            try:
                content = open(os.path.join(eval_dir, fname), encoding='utf-8').read()
            except OSError:
                continue
            texts.append({'name': fname, 'content': content})
    return texts[:80]


@login_required
@require_GET
def moe_job_status(request, pk):
    """JSON-статус для polling-страницы джобы."""
    from .models import MoEExpertJob

    job = get_object_or_404(MoEExpertJob, pk=pk)
    return JsonResponse({
        'status': job.status,
        'status_display': job.get_status_display(),
        'progress': job.progress,
        'current_step': job.current_step,
        'is_running': job.status in ['PENDING', 'VALIDATING', 'SCANNING', 'PATCHING',
                                     'EVALUATING', 'ROUTING'],
        'error_message': job.error_message,
        'output_model_id': job.output_model_id,
        'hf_repo_id': job.hf_repo_id,
        'logs_tail': job.logs[-2000:],
    })


@login_required
@require_POST
def moe_job_push_hf(request, pk):
    """Публикация готового эксперта на Hugging Face."""
    from .models import MoEExpertJob
    from .tasks_moe import moe_push_to_hf_task

    job = get_object_or_404(MoEExpertJob, pk=pk)
    if job.status != 'COMPLETED' or job.output_model is None:
        messages.error(request, 'Публиковать можно только завершённую задачу с экспертом.')
        return redirect('moe_job_detail', pk=pk)
    repo_id = (request.POST.get('repo_id') or '').strip()
    if not re.match(r'^[\w.-]+/[\w.-]+$', repo_id):
        messages.error(request, 'Repo ID в формате username/repo-name.')
        return redirect('moe_job_detail', pk=pk)
    private = request.POST.get('private') == 'on'
    notes = (request.POST.get('notes') or '').strip()
    transaction.on_commit(lambda: moe_push_to_hf_task.delay(
        job.pk, repo_id, private=private, notes=notes))
    messages.info(request, f'Публикация на HF запущена: {repo_id}')
    return redirect('moe_job_detail', pk=pk)
