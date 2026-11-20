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
def profile_view(request):
    # Create profile if it does not exist yet
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
# DASHBOARD
# ============================================================================
class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'ivanna_model_manager/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        user_id_str = str(user.id)

        # Global stats (models are shared)
        context['total_models'] = ModelCheckpoint.objects.filter(is_active=True).count()
        context['loaded_models'] = ModelCheckpoint.objects.filter(status='LOADED', is_active=True).count()
        context['downloading_models'] = ModelCheckpoint.objects.filter(status='DOWNLOADING', is_active=True).count()
        context['error_models'] = ModelCheckpoint.objects.filter(status='ERROR', is_active=True).count()
        context['active_experiments'] = Experiment.objects.filter(is_active=True).count()

        # User stats
        # My collections and documents (RAG)
        my_collections = DocumentCollection.objects.filter(owner_id=user_id_str, is_active=True)
        context['my_collections_count'] = my_collections.count()
        context['my_documents_count'] = Document.objects.filter(collection__in=my_collections, is_active=True).count()

        # My chat sessions
        context['my_chat_sessions'] = ChatSession.objects.filter(user_id=user_id_str, is_active=True).count()

        # My evaluation tasks
        my_evaluation_tasks = ModelEvaluationTask.objects.filter(created_by=user.username)
        context['my_evaluation_tasks_total'] = my_evaluation_tasks.count()
        context['my_evaluation_tasks_active'] = my_evaluation_tasks.filter(status__in=['PENDING', 'RUNNING']).count()
        context['my_evaluation_tasks_completed'] = my_evaluation_tasks.filter(status='COMPLETED').count()

        # My fine-tuning jobs
        my_ft_jobs = FineTuningJob.objects.filter(created_by=user.username)
        context['my_ft_jobs_total'] = my_ft_jobs.count()
        context['my_ft_jobs_active'] = my_ft_jobs.filter(status__in=['PREPARING', 'TRAINING']).count()
        context['my_ft_jobs_completed'] = my_ft_jobs.filter(status='COMPLETED').count()

        # My conversion jobs (GGUF)
        my_conv_jobs = ModelConversionJob.objects.filter(created_by=user.username)
        context['my_conv_jobs_total'] = my_conv_jobs.count()
        context['my_conv_jobs_active'] = my_conv_jobs.filter(status__in=['PENDING', 'PREPARING', 'CONVERTING']).count()
        context['my_conv_jobs_completed'] = my_conv_jobs.filter(status='COMPLETED').count()

        # My merge jobs (global, no per-user filtering yet)
        context['merge_jobs_active'] = ModelMergeJob.objects.filter(status__in=['PENDING', 'VALIDATING', 'RUNNING']).count()

        # Embedding stats
        embedding_models = EmbeddingModel.objects.filter(is_active=True)
        context['embedding_total'] = embedding_models.count()
        context['embedding_downloaded'] = embedding_models.filter(is_downloaded=True).count()
        context['embedding_downloading'] = embedding_models.filter(download_status='downloading').count()
        context['embedding_failed'] = embedding_models.filter(download_status='failed').count()

        # Recent embedding downloads (for the list)
        context['recent_embeddings'] = embedding_models.exclude(download_status='not_started').order_by('-updated_at')[:5]

        # ========== NEW BLOCKS FOR DISTILLATION ==========
        # My distillation question sets
        my_question_sets = DistillationQuestionSet.objects.filter(created_by=user.username)
        context['my_question_sets_total'] = my_question_sets.count()
        context['my_question_sets_active'] = my_question_sets.filter(status__in=['PENDING', 'RUNNING']).count()
        context['my_question_sets_completed'] = my_question_sets.filter(status='COMPLETED').count()

        # My answer generation jobs
        my_answer_jobs = DistillationAnswerJob.objects.filter(created_by=user.username)
        context['my_answer_jobs_total'] = my_answer_jobs.count()
        context['my_answer_jobs_active'] = my_answer_jobs.filter(status__in=['PENDING', 'RUNNING']).count()
        context['my_answer_jobs_completed'] = my_answer_jobs.filter(status='COMPLETED').count()

        # Merged active user tasks (for the recent list)
        recent_tasks = []

        # Evaluation tasks
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

        # Fine-tuning jobs
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

        # Conversion jobs
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

        # Distillation tasks (question generation)
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

        # Distillation tasks (answer generation)
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

        # Sort by creation date
        # v5.2: Soft MoE / WSP ablation (job.goal distinguishes the card type)
        my_moe_jobs = (MoEExpertJob.objects.all() if user.is_staff
                       else MoEExpertJob.objects.filter(created_by=user.username))
        for job in my_moe_jobs.order_by('-created_at')[:3]:
            recent_tasks.append({
                'id': job.id,
                'type': 'soft_moe_ablation' if job.goal == 'ablate' else 'soft_moe',
                'name': job.name,
                'status': job.status,
                'progress': job.progress,
                'created_at': job.created_at,
                'url': reverse('moe_job_detail', args=[job.id])
            })

        # v5.2: collection FAISS index builds (recent rebuilds)
        for coll in DocumentCollection.objects.filter(
                is_active=True, vector_index_created=True).order_by('-updated_at')[:3]:
            recent_tasks.append({
                'id': coll.id,
                'type': 'vector_index',
                'name': f"{coll.name} ({coll.total_vectors} vectors)",
                'status': 'COMPLETED',
                'progress': 100,
                'created_at': coll.updated_at,
                'url': reverse('rag_collection_detail', args=[coll.id])
            })

        # v5.3: benchmark runs (lm-eval-harness) — card title = benchmark
        # names, model shown un-truncated on its own line
        my_bench_runs = list((BenchmarkRun.objects.all() if user.is_staff
                              else BenchmarkRun.objects.filter(created_by=user.username))
                             .select_related('model')[:10])
        # no [:3] slice here — the global priority sort decides visibility;
        # slicing early crowded out RUNNING runs
        for run in my_bench_runs:
            bench_names = [b.display_name or b.name for b in run.benchmarks.all()[:3]]
            extra = run.benchmarks.count() - len(bench_names)
            label = ', '.join(bench_names) + (f' +{extra}' if extra > 0 else '')
            recent_tasks.append({
                'id': run.id,
                'type': 'benchmark',
                'name': label or run.name,
                'subtitle': run.model.display_name or run.model.name,
                'priority': 0 if run.status == 'RUNNING' else 1,
                'status': run.status,
                'progress': run.progress,
                'created_at': run.created_at,
                'url': reverse('benchmark_run_detail', args=[run.id])
            })

        # active tasks (priority 0) first, then newest across all features
        recent_tasks.sort(key=lambda x: (x.get('priority', 1),
                                         -x['created_at'].timestamp()))
        context['recent_tasks'] = recent_tasks[:5]

        # Recent models (global)
        context['recent_models'] = ModelCheckpoint.objects.filter(is_active=True).order_by('-created_at')[:10]

        # Active merge jobs (global)
        context['recent_jobs'] = ModelMergeJob.objects.filter(status__in=['PENDING', 'VALIDATING', 'RUNNING']).order_by('-created_at')[:5]

        # User metrics (total generated tokens)
        user_sessions = InferenceSession.objects.filter(api_key=user.username)
        total_tokens = user_sessions.aggregate(total=Sum('input_tokens') + Sum('output_tokens'))['total'] or 0
        context['user_total_tokens'] = total_tokens

        return context


# ============================================================================
# SYSTEM METRICS AND STATUS
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
    """GPU status page."""
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
    """Unified health-check endpoint (JSON)."""
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
        'celery': 'running',  # assumed
        'models': {
            'total': ModelCheckpoint.objects.filter(is_active=True).count(),
            'loaded': ModelCheckpoint.objects.filter(status='LOADED', is_active=True).count(),
        },
    })


@login_required
def task_status(request, task_id):
    """Celery task status (AJAX)."""
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
# STATIC PAGES
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
# TEMPLATE-FACING API-LIKE VIEWS (NOT DRF)
# ============================================================================
class SupportedModelTypesView(LoginRequiredMixin, TemplateView):
    """For templates, returns JSON."""
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
