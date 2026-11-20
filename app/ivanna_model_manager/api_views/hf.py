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
# HUGGING FACE API (MODEL SEARCH)
# ============================================================================
class HuggingFaceAPI(APIView):
    """API for the Hugging Face Hub: search models, embeddings, datasets."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        action = request.GET.get('action', '')
        if action == 'search':
            return self.search_models(request)
        elif action == 'top_models':
            return self.get_top_models(request)
        elif action == 'model_info':
            return self.get_model_info(request)
        elif action == 'validate':
            return self.validate_model(request)
        elif action == 'search_embedding':
            return self.search_embedding_models(request)
        elif action == 'top_embedding_models':
            return self.get_top_embedding_models(request)
        elif action == 'validate_embedding':
            return self.validate_embedding_model(request)
        elif action == 'search_datasets':
            return self.search_datasets(request)
        elif action == 'top_datasets':
            return self.top_datasets(request)
        elif action == 'validate_dataset':
            return self.validate_dataset(request)
        else:
            return Response({'error': _('Неизвестное действие')}, status=400)

    # =========================================================================
    # Methods for regular (LLM) models
    # =========================================================================
    def search_models(self, request):
        from huggingface_hub import HfApi
        query = request.GET.get('q', '')
        model_type = request.GET.get('type', '')
        limit = int(request.GET.get('limit', 10))

        filter_tags = None
        if model_type == 'QWEN':
            filter_tags = 'qwen,text-generation'
        elif model_type == 'LLAMA':
            filter_tags = 'llama,text-generation'
        elif model_type == 'MISTRAL':
            filter_tags = 'mistral,text-generation'
        elif model_type == 'GEMMA':
            filter_tags = 'gemma,text-generation'
        elif model_type == 'DEEPSEEK':
            filter_tags = 'text-generation'
        elif model_type == 'PHI':
            filter_tags = 'text-generation'
        elif model_type == 'YI':
            filter_tags = 'text-generation'

        api = HfApi()
        try:
            models = api.list_models(
                search=query or None,
                filter=filter_tags,
                sort="downloads",
                limit=limit,
                full=True
            )
            results = []
            for m in models:
                results.append({
                    'id': m.id,
                    'name': m.id.split('/')[-1],
                    'author': m.id.split('/')[0],
                    'downloads': m.downloads,
                    'likes': m.likes,
                    'tags': m.tags[:5] if m.tags else [],
                    'pipeline_tag': m.pipeline_tag,
                    'created_at': m.created_at.isoformat() if m.created_at else None,
                })
            return Response({'results': results, 'total': len(results)})
        except Exception as e:
            return Response({'error': str(e), 'results': []}, status=500)

    def get_top_models(self, request):
        from huggingface_hub import HfApi
        model_type = request.GET.get('type', 'QWEN')
        limit = int(request.GET.get('limit', 5))

        type_config = {
            'QWEN': {'search': 'qwen', 'filter': 'qwen,text-generation'},
            'LLAMA': {'search': 'llama', 'filter': 'llama,text-generation'},
            'MISTRAL': {'search': 'mistral', 'filter': 'mistral,text-generation'},
            'GEMMA': {'search': 'gemma', 'filter': 'gemma,text-generation'},
            'DEEPSEEK': {'search': 'deepseek', 'filter': 'text-generation'},
            'PHI': {'search': 'phi-3', 'filter': 'text-generation'},
            'YI': {'search': 'yi-1.5', 'filter': 'text-generation'},
            'ALL': {'search': '', 'filter': 'text-generation'},
        }
        cfg = type_config.get(model_type, type_config['ALL'])

        api = HfApi()
        try:
            models = api.list_models(
                search=cfg['search'],
                filter=cfg['filter'],
                sort="downloads",
                limit=limit
            )
            results = []
            for m in models:
                results.append({
                    'id': m.id,
                    'name': m.id.split('/')[-1],
                    'full_name': m.id,
                    'author': m.id.split('/')[0],
                    'downloads': m.downloads,
                    'likes': m.likes,
                    'created_at': m.created_at.isoformat()[:10] if m.created_at else 'N/A',
                })
            return Response({'type': model_type, 'models': results})
        except Exception as e:
            logger.error(f"Error in get_top_models: {e}", exc_info=True)
            return Response({'error': str(e), 'models': []}, status=500)

    def get_model_info(self, request):
        from huggingface_hub import HfApi
        model_id = request.GET.get('model_id', '')
        if not model_id:
            return Response({'error': 'model_id required'}, status=400)
        try:
            api = HfApi()
            info = api.model_info(model_id)
            return Response({
                'id': info.id,
                'author': info.author,
                'downloads': info.downloads,
                'likes': info.likes,
                'tags': info.tags[:10] if info.tags else [],
                'pipeline_tag': info.pipeline_tag,
                'private': info.private,
                'gated': info.gated,
            })
        except Exception as e:
            return Response({'error': str(e), 'model_id': model_id}, status=404)

    def validate_model(self, request):
        model_id = request.GET.get('model_id', '')
        if not model_id:
            return Response({'error': _('Не указан ID модели')}, status=400)

        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.model_info(model_id, timeout=10)
            
            # Default values
            context_length = 4096
            vocab_size = 32000
            parameter_count = 0
            
            # Try to fetch the config via a direct request to raw config.json
            try:
                config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
                headers = {"User-Agent": "IvannaModelManager/1.0"}
                resp = requests.get(config_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    config = resp.json()
                    context_length = config.get('max_position_embeddings', 4096)
                    vocab_size = config.get('vocab_size', 32000)
                    if 'num_parameters' in config:
                        parameter_count = config['num_parameters']
            except Exception as e:
                logger.warning(f"Could not fetch config for {model_id}: {e}")
            
            # If parameter_count is still 0, try to extract it from tags or the name
            if parameter_count == 0 and info.tags:
                for tag in info.tags:
                    tag_lower = tag.lower()
                    if tag_lower.endswith('b') and tag_lower[:-1].isdigit():
                        parameter_count = int(tag_lower[:-1]) * 1_000_000_000
                        break
                    elif tag_lower.endswith('m') and tag_lower[:-1].isdigit():
                        parameter_count = int(tag_lower[:-1]) * 1_000_000
                        break
            
            # If not found, try the model name (e.g. "7b")
            if parameter_count == 0:
                match = re.search(r'(\d+\.?\d*)[bB]', model_id)
                if match:
                    parameter_count = int(float(match.group(1)) * 1_000_000_000)
            
            return Response({
                'valid': True,
                'exists': True,
                'model_id': info.id,
                'author': info.author,
                'downloads': info.downloads,
                'private': info.private,
                'gated': info.gated,
                'context_length': context_length,
                'vocab_size': vocab_size,
                'parameter_count': parameter_count,
            })
        except Exception as e:
            logger.error(f"Validate model error for {model_id}: {e}", exc_info=True)
            return Response({
                'valid': False,
                'exists': False,
                'model_id': model_id,
                'error': str(e)
            }, status=404)

    # =========================================================================
    # Methods for embedding models
    # =========================================================================
    def search_embedding_models(self, request):
        from huggingface_hub import HfApi
        import logging
        logger = logging.getLogger(__name__)

        query = request.GET.get('q', '')
        page = int(request.GET.get('page', 1))
        limit = int(request.GET.get('limit', 10))

        api = HfApi()
        try:
            # Fetch up to 100 models (to avoid overloading the API)
            all_models = list(api.list_models(
                search=query or None,
                filter="sentence-transformers,feature-extraction",
                sort="downloads",
                limit=100
            ))
            total = len(all_models)
            start = (page - 1) * limit
            end = start + limit
            page_models = all_models[start:end]

            results = []
            for m in page_models:
                dim = None
                if m.tags:
                    for tag in m.tags:
                        if tag.startswith('dim'):
                            dim = tag.replace('dim', '')
                            break
                results.append({
                    'id': m.id,
                    'name': m.id.split('/')[-1],
                    'author': m.id.split('/')[0],
                    'downloads': m.downloads,
                    'likes': m.likes,
                    'tags': m.tags[:10] if m.tags else [],
                    'dimension': dim,
                })
            return Response({'results': results, 'total': total})
        except Exception as e:
            logger.error(f"search_embedding_models error: {e}", exc_info=True)
            return Response({'error': str(e), 'results': []}, status=500)

    def get_top_embedding_models(self, request):
        from huggingface_hub import HfApi
        import logging
        logger = logging.getLogger(__name__)

        limit = int(request.GET.get('limit', 5))
        api = HfApi()
        try:
            models = list(api.list_models(
                filter="sentence-transformers,feature-extraction",
                sort="downloads",
                limit=limit
            ))
            results = []
            for m in models:
                dim = None
                if m.tags:
                    for tag in m.tags:
                        if tag.startswith('dim'):
                            dim = tag.replace('dim', '')
                            break
                results.append({
                    'id': m.id,
                    'name': m.id.split('/')[-1],
                    'author': m.id.split('/')[0],
                    'downloads': m.downloads,
                    'likes': m.likes,
                    'tags': m.tags[:10] if m.tags else [],
                    'dimension': dim,
                })
            return Response({'models': results})
        except Exception as e:
            logger.error(f"get_top_embedding_models error: {e}", exc_info=True)
            return Response({'error': str(e), 'models': []}, status=500)

    def validate_embedding_model(self, request):
        model_id = request.GET.get('model_id', '')
        if not model_id:
            return Response({'error': 'model_id required'}, status=400)
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            info = api.model_info(model_id, timeout=10)
            if info.pipeline_tag not in ['feature-extraction', 'sentence-similarity']:
                return Response({'valid': False, 'exists': True, 'error': 'Not an embedding model'})
            dim = None
            if info.tags:
                for tag in info.tags:
                    if tag.startswith('dim'):
                        dim = tag.replace('dim', '')
                        break
            return Response({
                'valid': True,
                'exists': True,
                'id': info.id,
                'author': info.author,
                'downloads': info.downloads,
                'likes': info.likes,
                'tags': info.tags[:10],
                'dimension': dim,
                'license': info.cardData.get('license', 'unknown'),
            })
        except Exception as e:
            return Response({'valid': False, 'exists': False, 'error': str(e)}, status=404)

    # =========================================================================
    # Methods for datasets
    # =========================================================================
    def search_datasets(self, request):
        """
        Search datasets on the Hugging Face Hub with manual sorting and pagination.
        Limits the number of fetched datasets for performance.
        """
        from huggingface_hub import HfApi
        import logging
        from datetime import datetime

        logger = logging.getLogger(__name__)

        query = request.GET.get('q', '').strip()
        limit = int(request.GET.get('limit', 10))
        page = int(request.GET.get('page', 1))
        sort_by = request.GET.get('sort', 'downloads')

        api = HfApi()
        try:
            # Fetch a limited number of datasets (500) - enough for top-20 and first search pages
            all_datasets = list(api.list_datasets(search=query or None, limit=500))

            # Manual sorting (the sort parameter is not always supported)
            if sort_by == 'downloads':
                all_datasets.sort(key=lambda x: getattr(x, 'downloads', 0), reverse=True)
            elif sort_by == 'likes':
                all_datasets.sort(key=lambda x: getattr(x, 'likes', 0), reverse=True)
            elif sort_by == 'created':
                all_datasets.sort(key=lambda x: x.created_at if x.created_at else datetime.min, reverse=True)

            total = len(all_datasets)
            start = (page - 1) * limit
            end = start + limit
            paginated = all_datasets[start:end]

            results = []
            for ds in paginated:
                description = ''
                if hasattr(ds, 'cardData') and ds.cardData:
                    description = ds.cardData.get('description', '')
                elif hasattr(ds, 'description'):
                    description = ds.description or ''

                results.append({
                    'id': ds.id,
                    'name': ds.id.split('/')[-1],
                    'author': ds.id.split('/')[0],
                    'downloads': getattr(ds, 'downloads', 0) or 0,
                    'likes': getattr(ds, 'likes', 0) or 0,
                    'tags': ds.tags[:5] if ds.tags else [],
                    'description': description[:200],
                    'created_at': ds.created_at.isoformat() if ds.created_at else None,
                })

            return Response({'results': results, 'total': total})
        except Exception as e:
            logger.error(f"Error in search_datasets: {e}", exc_info=True)
            return Response({'error': str(e), 'results': []}, status=500)

    def top_datasets(self, request):
        """Return top datasets by downloads (empty search)."""
        # Delegate to search_datasets with empty q
        new_request = request
        # Copy the GET params and set q=''
        if hasattr(request, 'GET') and isinstance(request.GET, dict):
            # Just call search_datasets with the same params but q replaced by ''
            get_params = request.GET.copy()
            get_params['q'] = ''
            request.GET = get_params
        else:
            # Fallback
            request.GET = request.GET.copy()
            request.GET['q'] = ''
        return self.search_datasets(request)

    def validate_dataset(self, request):
        repo_id = request.GET.get('repo_id', '')
        if not repo_id:
            return Response({'error': 'repo_id required'}, status=400)
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            info = api.dataset_info(repo_id, timeout=10)
            return Response({
                'valid': True,
                'exists': True,
                'id': info.id,
                'author': info.id.split('/')[0],
                'downloads': info.downloads,
                'likes': info.likes,
                'tags': info.tags[:10] if info.tags else [],
                'description': info.cardData.get('description', '')[:200] if info.cardData else '',
            })
        except Exception as e:
            return Response({'valid': False, 'exists': False, 'error': str(e)}, status=404)
