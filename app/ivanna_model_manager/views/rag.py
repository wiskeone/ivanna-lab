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

from .utils import get_document_type, sanitize_filename



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
    from ..models import DocumentProcessor
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
        '.md': 'MARKDOWN', '.csv': 'CSV', '.xlsx': 'EXCEL', '.xls': 'EXCEL',
        '.rtf': 'RTF', '.pptx': 'PPTX', '.ppt': 'PPTX', '.epub': 'EPUB'
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
    from ..models import DocumentCollection, SearchQuery, DocumentChunk
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
        from ..models import SavedSearch
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
    from ..services.rag_service import EnhancedSearchService
    from ..models import DocumentCollection, DocumentChunk, SearchQuery, SearchResult
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
            # Audit fix: the service signature is collection_ids — passing
            # collection_id raised TypeError (500) on every hybrid search
            results = search_service.enhanced_hybrid_search(
                query=query,
                collection_ids=collection.id,
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
                        # Audit fix: the status save raced with the task — the
                        # worker could overwrite INDEXED back to PROCESSING
                        # (stuck forever) or hit the PROCESSING guard itself.
                        # The task sets PROCESSING atomically on start.
                        process_document_task.delay(doc.id)
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
    from ..models import DocumentProcessor
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
    from ..services.faiss_service import FAISSService
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
    from ..services.faiss_service import FAISSService
    faiss = FAISSService()
    index_path = faiss._get_index_path(collection_id)
    meta_path = faiss._get_metadata_path(collection_id)
    return JsonResponse({
        'index_exists': os.path.exists(index_path),
        'metadata_exists': os.path.exists(meta_path),
        'index_size_mb': os.path.getsize(index_path) / (1024 * 1024) if os.path.exists(index_path) else 0,
    })


@login_required
def document_page_render(request, pk, page):
    """PNG-рендер страницы PDF для превью чанков (PyMuPDF)."""
    from ..models import Document

    doc = get_object_or_404(Document, pk=pk)
    path = doc.full_file_path
    if not path.lower().endswith('.pdf') or not os.path.exists(path):
        raise Http404('Не PDF или файл отсутствует')
    import fitz
    d = fitz.open(path)
    if not (1 <= page <= d.page_count):
        d.close()
        raise Http404('Страница вне диапазона')
    pix = d[page - 1].get_pixmap(dpi=110)
    d.close()
    buf = io.BytesIO(pix.tobytes('png'))
    resp = HttpResponse(buf.getvalue(), content_type='image/png')
    resp['Cache-Control'] = 'max-age=86400'
    return resp
