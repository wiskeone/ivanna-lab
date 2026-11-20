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
# ЧАТ
# ============================================================================
@login_required
def chat_view(request, session_id=None):
    """Основное представление чата с поддержкой создания новой сессии и проверки статуса модели."""
    from ..models import ChatSession, ModelCheckpoint
    from ..services.chat_service import chat_service
    from ..services.redis_model_service import redis_model_manager
    from ..tasks import sync_model_statuses_task
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
