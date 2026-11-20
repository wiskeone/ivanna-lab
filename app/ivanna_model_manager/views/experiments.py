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
