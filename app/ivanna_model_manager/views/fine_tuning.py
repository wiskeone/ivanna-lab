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
# FineTuning
# ============================================================================
@login_required
def fine_tuning_dashboard(request):
    """Fine-tuning dashboard"""
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
    Return training metrics as JSON for charting.
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

            # Check that data exists for instruction/raft
            if method in ('instruction', 'raft'):
                distillation_ids = request.POST.getlist('distillation_datasets')
                manual_ids = request.POST.getlist('manual_datasets')
                hf_ids = request.POST.getlist('hf_datasets')
                if not distillation_ids and not manual_ids and not hf_ids:
                    form.add_error(None, 'Для методов instruction и RAFT необходимы датасеты инструкций.')
                    return render(request, 'ivanna_model_manager/fine_tuning/create.html', _get_create_context(request, form))

            # Create the output model
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

            # Save selected collections and datasets into training_config
            training_config = job.training_config or {}
            # Collections
            collection_ids = request.POST.getlist('document_collections')
            if collection_ids:
                training_config['document_collections'] = [int(cid) for cid in collection_ids]
            # Distillation datasets
            distillation_ids = request.POST.getlist('distillation_datasets')
            if distillation_ids:
                training_config['distillation_datasets'] = [int(did) for did in distillation_ids]
            # Manual datasets
            manual_ids = request.POST.getlist('manual_datasets')
            if manual_ids:
                training_config['manual_datasets'] = [int(mid) for mid in manual_ids]
            # HF datasets
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
    from ..models import ModelCheckpoint, DocumentCollection, EvaluationDataset, DistillationDataset, ManualInstructionSet, HuggingFaceDataset

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
    """List all fine-tuning jobs"""
    from ..models import FineTuningJob
    
    # Filtering
    status = request.GET.get('status', '')
    method = request.GET.get('method', '')
    
    jobs = FineTuningJob.objects.filter(created_by=request.user.username)
    
    if status:
        jobs = jobs.filter(status=status)
    if method:
        jobs = jobs.filter(method=method)
    
    # Sorting
    sort = request.GET.get('sort', '-created_at')
    if sort.lstrip('-') in ['name', 'status', 'created_at', 'progress']:
        jobs = jobs.order_by(sort)
    else:
        jobs = jobs.order_by('-created_at')
    
    # Pagination
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
    """Fine-tuning job details"""
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
    """View job logs (AJAX)"""
    job = get_object_or_404(
        FineTuningJob,
        id=pk,
        created_by=request.user.username
    )
    return JsonResponse({'logs': job.logs})


@login_required
def fine_tuning_templates(request):
    """List fine-tuning templates"""
    from ..models import FineTuningTemplate
    
    # Filtering
    category = request.GET.get('category', '')
    method = request.GET.get('method', '')
    
    templates = FineTuningTemplate.objects.filter(is_public=True)
    
    if category:
        templates = templates.filter(category=category)
    if method:
        templates = templates.filter(method=method)
    
    # Per-category stats
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
    """Template details"""
    from ..models import FineTuningTemplate
    
    template = get_object_or_404(FineTuningTemplate, pk=pk, is_public=True)
    
    context = {
        'active_tab': 'fine_tuning',
        'template': template,
    }
    return render(request, 'ivanna_model_manager/fine_tuning/template_detail.html', context)


@login_required
def fine_tuning_template_create(request):
    """Create a new template"""
    from ..forms import FineTuningTemplateForm
    
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
    """Apply a template to job creation"""
    from ..models import FineTuningTemplate
    from ..forms import FineTuningJobForm
    
    template = get_object_or_404(FineTuningTemplate, pk=pk, is_public=True)
    template.increment_usage()
    
    # Pre-fill the form with template data
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
    
    # Redirect to job creation with the pre-filled form
    return redirect('fine_tuning_create_with_template', template_id=template.id)


@login_required
def conversion_dashboard(request):
    """Conversion dashboard"""
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
