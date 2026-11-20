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
# MODELS
# ============================================================================
class ModelListView(LoginRequiredMixin, ListView):
    template_name = 'ivanna_model_manager/model_list.html'
    context_object_name = 'models'
    paginate_by = 20

    def get_queryset(self):
        qs = ModelCheckpoint.objects.filter(is_active=True)
        # Filters
        if self.request.GET.get('model_type'):
            qs = qs.filter(model_type=self.request.GET['model_type'])
        if self.request.GET.get('status'):
            qs = qs.filter(status=self.request.GET['status'])
        if self.request.GET.get('experiment_id'):
            qs = qs.filter(experiment_id=self.request.GET['experiment_id'])
        if self.request.GET.get('q'):
            qs = qs.filter(Q(name__icontains=self.request.GET['q']) | Q(display_name__icontains=self.request.GET['q']))
        # Sorting
        sort = self.request.GET.get('sort_by', '-created_at')
        if sort.lstrip('-') in ['name', 'display_name', 'created_at', 'parameter_count']:
            qs = qs.order_by(sort)
        else:
            qs = qs.order_by('-created_at')
        # HF repo the model was published to via the platform (soft MoE push):
        # Subquery on the newest MoE job with a non-empty repo_id — no N+1
        published_repo = (MoEExpertJob.objects.filter(output_model=OuterRef('pk'))
                          .exclude(hf_repo_id='').order_by('-id'))
        return qs.select_related('experiment', 'parent').annotate(
            published_hf=Subquery(published_repo.values('hf_repo_id')[:1]))

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

            # Check if the model is loaded in memory (Redis)
            model_id = model.huggingface_id or model.name
            info = redis_model_manager.get_model_info(model_id)
            model.is_loaded_in_memory = bool(info and info.get('loaded'))

        # Current sort value from the request (to keep in filters)
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
    """Model detail view with configuration display."""
    model = ModelCheckpoint
    template_name = 'ivanna_model_manager/model_detail.html'
    context_object_name = 'model'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        obj = self.object
        context['has_local_copy'] = bool(obj.local_path and os.path.exists(obj.local_path))

        # Load loader_config into JSON for display
        import json
        config_json = json.dumps(obj.loader_config, indent=2, ensure_ascii=False)
        context['config_json'] = config_json

        # Model versions
        context['versions'] = obj.versions.order_by('-created_at')[:10]

        # Evaluations
        context['evaluations'] = obj.evaluations.filter(is_valid=True).order_by('-evaluated_at')[:5]

        # v5.3: scores of this model per benchmark (latest completed run per pair)
        bench_scores = []
        for run in obj.benchmark_runs.filter(status='COMPLETED') \
                .prefetch_related('benchmarks').order_by('-completed_at'):
            scores = run.best_scores()
            for bench in run.benchmarks.all():
                if bench.name in scores and not any(bs['bench_pk'] == bench.pk for bs in bench_scores):
                    bench_scores.append({
                        'bench_pk': bench.pk,
                        'bench_name': bench.display_name or bench.name,
                        'score': scores[bench.name],
                        'run_pk': run.pk,
                        'when': run.completed_at,
                    })
        context['bench_scores'] = bench_scores

        # Active sessions
        context['active_sessions'] = obj.inference_sessions.filter(is_active=True).order_by('-last_activity')[:5]

        # Merge candidate models
        context['merge_candidates'] = ModelCheckpoint.objects.filter(
            status='LOADED',
            is_active=True,
            model_type=obj.model_type,
            parameter_count=obj.parameter_count
        ).exclude(id=obj.id)[:10]

        # Recent merge jobs
        context['merge_jobs'] = ModelMergeJob.objects.filter(
            Q(base_model=obj) | Q(merge_model=obj)
        ).order_by('-created_at')[:5]

        # Size and quantization
        context['size_gb'] = obj.size_gb
        context['is_quantized'] = obj.is_quantized

        return context


class LocalModelCreateView(LoginRequiredMixin, CreateView):
    """Create a model from a local path (no Hugging Face)."""
    template_name = 'ivanna_model_manager/local_model_form.html'
    form_class = LocalModelForm  # to be created
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
    """Start model download (synchronous task launch)."""
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
    """Soft-delete a model (deactivation)."""
    model = get_object_or_404(ModelCheckpoint, pk=pk)
    if request.method == 'POST':
        # Cancel active downloads
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
    """View model configuration."""
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
    """Create a new model version."""
    model = get_object_or_404(ModelCheckpoint, pk=pk, is_active=True)
    version_name = request.POST.get('version_name', f'v{model.versions.count() + 1}')
    notes = request.POST.get('notes', '')
    model.create_version(version_name, notes)
    messages.success(request, f'Версия {version_name} создана')
    return redirect('model_detail', pk=model.id)


@login_required
def restore_model_version_view(request, model_pk, version_pk):
    """Restore a model version."""
    model = get_object_or_404(ModelCheckpoint, pk=model_pk, is_active=True)
    version = get_object_or_404(ModelVersion, pk=version_pk, checkpoint=model)
    version.restore()
    messages.success(request, f'Версия {version.version} восстановлена')
    return redirect('model_detail', pk=model.id)


# ============================================================================
# MODEL TESTING (INFERENCE)
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
        # Check if the model is loaded
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
