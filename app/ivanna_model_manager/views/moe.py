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
    from ..models import MoETheme, MoEExpertJob

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
    from ..services.soft_moe import prompts as moe_prompts
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
    from ..forms import MoEThemeForm

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
    from ..forms import MoEThemeForm
    from ..models import MoETheme

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
    from ..models import MoETheme

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
    from ..models import MoETheme
    from ..tasks_moe import moe_validate_theme_task

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
    from ..models import MoETheme
    from ..tasks_moe import moe_router_build_task

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
    from ..models import MoETheme
    from ..services.soft_moe import router as moe_router

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
@require_POST
def moe_job_repatch(request, pk):
    """Click-tool: create a re-patch job from the heatmap selection on this job.

    Reuses the parent scan (npz + report); the selection (layers / column
    groups / alpha) travels as job.patch_config. Non-destructive: a NEW job
    is created and dispatched."""
    from ..models import MoEExpertJob
    from ..tasks_moe import moe_build_expert_task
    import json as _json

    parent = get_object_or_404(MoEExpertJob, pk=pk)
    if parent.status != 'COMPLETED':
        messages.error(request, _('Родительская джоба не завершена — выделение делается на готовых теплокартах.'))
        return redirect('moe_job_detail', pk=pk)
    try:
        cfg = _json.loads(request.POST.get('patch_config') or '{}')
    except ValueError:
        cfg = {}
    layers = (cfg.get('layers') or [])
    col_masks = cfg.get('col_masks') or {}
    alpha = cfg.get('alpha')
    if not layers and not col_masks:
        messages.error(request, _('Пустое выделение: выберите слои или группы размерностей.'))
        return redirect('moe_job_detail', pk=pk)

    job = MoEExpertJob.objects.create(
        name=f"repatch_{parent.name[:120]}_{timezone.now().strftime('%H%M%S')}",
        theme=parent.theme, base_checkpoint=parent.base_checkpoint,
        mode=parent.mode, alpha=float(alpha) if alpha not in (None, '') else parent.alpha,
        layers=','.join(str(l) for l in layers) if layers else parent.layers,
        position=parent.position, holdout_seed=parent.holdout_seed,
        load_mode=parent.load_mode, do_eval=False, do_router=parent.do_router,
        goal=parent.goal, do_patch=True, created_by=request.user.username,
        patch_config={'layers': layers, 'col_masks': col_masks},
        reuse_scan_from=parent,
    )
    transaction.on_commit(lambda: moe_build_expert_task.delay(job.id))
    messages.success(request, _('Re-patch задача "%(name)s" создана (скан переиспользуется).')
                     % {'name': job.name})
    return redirect('moe_job_detail', pk=job.pk)


@login_required
def moe_job_create(request):
    """Создание задачи сборки эксперта: валидации → джоба → GPU-очередь."""
    from ..forms import MoEExpertJobForm
    from ..models import MoEExpertJob, MoETheme
    from ..tasks_moe import moe_build_expert_task

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
        charts['heatmaps'] = [{'name': t['name'], 'z': t['heatmap'],
                               'ih': t.get('ih'), 'oh': t.get('oh'),
                               'suffix': '.'.join(t['name'].split('.')[-2:])}
                              for t in heatmaps]

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
    from django.utils.translation import gettext as _

    hints = []
    scan = job.scan_report or {}
    if scan:
        if scan.get('gate') != 'PASS':
            hints.append(('danger', _('AUC низкий — направление темы не выделяется. '
                          'Проверьте валидацию темы (cross cos, баланс длин), добавьте промптов. Патч в таком случае бесполезен.')))
        else:
            top = max(scan.get('auc', {}).get('last', {}).get('test', [0]))
            if top >= 0.98:
                hints.append(('success', _('AUC %(top)s — направление выражено очень чётко; '
                              'α=0.2 обычно безопасна, при деградации general снижайте.') % {'top': top}))
        late = [w for w in (scan.get('top_writers') or []) if w['layer'] >= scan.get('n_layers', 24) * 0.5]
        if late:
            hints.append(('info', _('Писатели сосредоточены в поздних слоях (например L%(layer)s) — '
                          'это ожидаемо: доменное поведение записывается в конце residual stream.')
                          % {'layer': late[0]["layer"]}))
    patch = job.patch_report or {}
    if patch and patch.get('patched_count', 0) == 0:
        hints.append(('danger', _('Ни один тензор не патчился: проверьте mean-входы (для steer) и слои.')))
    er = (job.eval_report or {}).get('summary') or {}
    g = er.get('general_len', {}).get('delta_pct')
    d = er.get('domain_len', {}).get('delta_pct')
    if g is not None and g < -15:
        hints.append(('warning', _('General просел на %(g)s%% — уменьшите α (сетка 0.1/0.05) '
                      'или сузьте список слоёв; патч слишком широкий.') % {'g': abs(g)}))
    if d is not None and d < -15:
        hints.append(('warning', _('Доменные ответы заметно короче базы — числовые метрики — это прокси: '
                      'обязательно просмотрите тексты ответов ниже руками.')))
    rr = job.router_report or {}
    if rr.get('accuracy') is not None and rr.get('gate') != 'PASS':
        hints.append(('warning', _('Роутер путает темы — добавьте контрастных negative-примеров '
                      'в scan-наборы или поднимите порог margin.')))
    if not hints:
        hints.append(('info', _('Гейты пройдены: результаты в отчётах ниже; тексты eval — для ручной проверки.')))
    return hints


@login_required
def moe_job_detail(request, pk):
    from ..models import MoEExpertJob

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
        'charts': _moe_charts(job),
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
    from ..models import MoEExpertJob

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
    from ..models import MoEExpertJob
    from ..tasks_moe import moe_push_to_hf_task

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


@login_required
@require_POST
def moe_interpret(request, pk):
    """Кайллер-фича: отправить отчёт прогона в модель джобы на интерпретацию."""
    from ..models import MoEExpertJob
    from ..tasks_moe import moe_interpret_task

    job = get_object_or_404(MoEExpertJob, pk=pk)
    if job.status != 'COMPLETED':
        messages.error(request, 'Интерпретация доступна для завершённых задач.')
        return redirect('moe_job_detail', pk=pk)
    transaction.on_commit(lambda: moe_interpret_task.delay(job.pk))
    messages.info(request, 'Модель анализирует отчёт прогона — обнови страницу через пару минут.')
    return redirect('moe_job_detail', pk=pk)
