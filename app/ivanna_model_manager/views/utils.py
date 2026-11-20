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
# HELPER FUNCTIONS
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
