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
        # Add stats
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
    # Do not allow deleting the default model while others exist
    if model.is_default and EmbeddingModel.objects.filter(is_default=True).count() <= 1:
        return JsonResponse({'success': False, 'error': 'Cannot delete the default model'}, status=400)

    model.delete()
    return JsonResponse({'success': True})


@login_required
@require_POST
def embedding_model_download(request, pk):
    """Start embedding model download."""
    model = get_object_or_404(EmbeddingModel, pk=pk, is_active=True)

    if model.is_downloaded:
        return JsonResponse({'success': False, 'error': 'Model already downloaded'})

    if model.download_status == 'downloading':
        return JsonResponse({'success': False, 'error': 'Model is already downloading'})

    # Reset status
    model.download_status = 'not_started'
    model.download_progress = 0
    model.download_error = ''
    model.save()

    # Launch the task with user_id
    from ..tasks_embedding import download_embedding_model_task
    task = download_embedding_model_task.delay(model.id, user_id=request.user.id)
    return JsonResponse({'success': True, 'task_id': task.id})


@login_required
def embedding_model_download_status(request, pk):
    """Return the model download status."""
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
    Reset the model download status and remove the Redis lock.
    """
    model = get_object_or_404(EmbeddingModel, pk=pk)
    # Reset status
    model.download_status = 'not_started'
    model.download_progress = 0
    model.download_error = ''
    model.save(update_fields=['download_status', 'download_progress', 'download_error'])
    # Remove the Redis lock
    from django.core.cache import cache
    cache.delete(f"download_embedding_model:{model.id}")
    logger.info(f"Reset download status for model {model.name} (ID={model.id})")
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
    """Fully rebuild embeddings and the FAISS index for a collection."""
    logger.info(f"rebuild_collection_embeddings called by {request.user}, headers: {request.headers}")
    collection = get_object_or_404(DocumentCollection, pk=pk, is_active=True)
    # Permission check
    if not request.user.is_staff and collection.owner_id != str(request.user.id):
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)

    from ..tasks_rag import rebuild_embeddings_and_index_task
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
    Change the embedding model for a collection.
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
        # Ensure the model exists and is active
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


@login_required
def reranker_list(request):
    """Reranker registry: list + create + download (modeled on embeddings)."""
    from ..models import RerankerModel
    from ..forms import RerankerModelForm

    if request.method == 'POST':
        form = RerankerModelForm(request.POST)
        if form.is_valid():
            r = form.save()
            messages.success(request, f'Реранкер «{r.name}» добавлен.')
            return redirect('reranker_list')
    else:
        form = RerankerModelForm()
    return render(request, 'ivanna_model_manager/embedding_models/rerankers.html',
                  {'form': form, 'rerankers': RerankerModel.objects.all(),
                   'active_tab': 'rag'})


@login_required
@require_POST
def reranker_download(request, pk):
    from ..models import RerankerModel
    from ..tasks_embedding import download_reranker_model_task
    r = get_object_or_404(RerankerModel, pk=pk)
    download_reranker_model_task.delay(r.pk, request.user.id)
    messages.info(request, f'Скачивание «{r.name}» запущено.')
    return redirect('reranker_list')


@login_required
@require_POST
def reranker_delete(request, pk):
    from ..models import RerankerModel
    get_object_or_404(RerankerModel, pk=pk).delete()
    return redirect('reranker_list')


@login_required
@require_POST
def change_collection_reranker(request, pk):
    """Select the collection reranker (null = auto by language)."""
    from ..models import DocumentCollection, RerankerModel

    coll = get_object_or_404(DocumentCollection, pk=pk)
    rid = request.POST.get('reranker_id')
    if rid in (None, '', 'auto'):
        coll.reranker_model = None
    else:
        coll.reranker_model = get_object_or_404(RerankerModel, pk=rid)
    coll.save(update_fields=['reranker_model'])
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'success': True,
                             'reranker': coll.reranker_model.name if coll.reranker_model else 'auto'})
    messages.success(request, 'Реранкер коллекции обновлён.')
    return redirect('rag_collection_detail', pk=pk)
