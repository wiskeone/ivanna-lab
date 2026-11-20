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
def benchmark_list(request):
    """Catalog: popular curated cards + full-registry live search."""
    from ..models import Benchmark

    popular = Benchmark.objects.filter(is_popular=True)
    total = Benchmark.objects.count()
    installed = Benchmark.objects.filter(install_status='completed').count()
    context = {
        'popular': popular,
        'total_benchmarks': total,
        'installed_count': installed,
        'active_tab': 'benchmarks',
    }
    return render(request, 'ivanna_model_manager/benchmarks/list.html', context)


@login_required
def benchmark_custom_create(request):
    """Builder for custom benchmarks: name + QA generation from a RAG
    collection (LLM) + manual QA editor."""
    from ..models import Benchmark, DocumentCollection, ModelCheckpoint
    from ..tasks_benchmarks import generate_custom_qa_task

    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'create':
            bench = Benchmark.objects.create(
                name=request.POST.get('name', '').strip() or
                     f"custom_{timezone.now().strftime('%Y%m%d_%H%M%S')}",
                display_name=request.POST.get('display_name', '').strip(),
                description=request.POST.get('description', '').strip(),
                category='other', source='custom', is_popular=False,
            )
            collection_id = request.POST.get('collection')
            n_pairs = int(request.POST.get('n_pairs') or 10)
            model_id = request.POST.get('generator_model')
            if collection_id and model_id:
                bench.install_status = 'downloading'
                bench.save(update_fields=['install_status'])
                transaction.on_commit(lambda: generate_custom_qa_task.delay(
                    bench.pk, int(collection_id), n_pairs, int(model_id)))
            messages.success(request, _('Собственный бенчмарк "%(name)s" создан')
                             % {'name': bench.name})
            return redirect('benchmark_detail', pk=bench.pk)
        if action == 'save_qa':
            bench = get_object_or_404(Benchmark, pk=request.POST.get('bench_pk'))
            qa = []
            for line in (request.POST.get('qa_text') or '').splitlines():
                line = line.strip()
                if not line or '|' not in line:
                    continue
                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 2:
                    continue
                qa.append({'prompt': parts[0], 'reference': parts[1],
                           'keywords': [k.strip().lower() for k in (parts[2].split(',') if len(parts) > 2 else [])
                                        if k.strip()]})
            bench.qa_data = qa
            bench.n_samples = len(qa)
            bench.install_status = 'completed' if qa else bench.install_status
            bench.save(update_fields=['qa_data', 'n_samples', 'install_status'])
            messages.success(request, _('QA-набор сохранен: %(n)s пар') % {'n': len(qa)})
            return redirect('benchmark_detail', pk=bench.pk)
    from ..models import DocumentCollection
    collections = DocumentCollection.objects.filter(is_active=True)
    generator_models = ModelCheckpoint.objects.filter(
        is_active=True, status='DOWNLOADED', local_path__isnull=False).exclude(local_path='')
    return render(request, 'ivanna_model_manager/benchmarks/custom_create.html',
                  {'collections': collections, 'generator_models': generator_models,
                   'active_tab': 'benchmarks'})


@login_required
def benchmark_custom_qa_text(request, pk):
    """QA pairs as editable text (prompt | reference | kw1,kw2)."""
    from django.http import JsonResponse
    from ..models import Benchmark

    bench = get_object_or_404(Benchmark, pk=pk)
    lines = ['{} | {} | {}'.format(i['prompt'].replace('\n', ' '),
                                   i['reference'].replace('\n', ' '),
                                   ','.join(i.get('keywords', [])))
             for i in bench.qa_data]
    return JsonResponse({'text': '\n'.join(lines), 'n': len(lines)})


@login_required
def benchmark_search_api(request):
    """JSON search over the synced registry (name/group/description)."""
    from django.http import JsonResponse
    from ..models import Benchmark

    q = (request.GET.get('q') or '').strip()
    category = request.GET.get('category') or ''
    qs = Benchmark.objects.all()
    if q:
        qs = qs.filter(Q(name__icontains=q) | Q(display_name__icontains=q) |
                       Q(description__icontains=q) | Q(group__icontains=q))
    if category:
        qs = qs.filter(category=category)
    rows = list(qs.values('id', 'name', 'display_name', 'description', 'category',
                          'metrics', 'group', 'source', 'is_popular',
                          'install_status')[:50])
    for r in rows:
        r['is_installed'] = r.pop('install_status') == 'completed'
    return JsonResponse({'results': rows, 'total': qs.count()})


@login_required
def benchmark_detail(request, pk):
    """Benchmark page: per-model leaderboard chart + all model metrics.

    Cell semantics = latest completed run per model (as in the leaderboard)."""
    from ..models import Benchmark, BenchmarkRun

    bench = get_object_or_404(Benchmark, pk=pk)
    runs = BenchmarkRun.objects.filter(benchmarks=bench).select_related('model')[:20]

    # per-model: score from the latest completed run + its full metrics dict
    from ..models import MoEExpertJob
    per_model, order = {}, []
    for run in BenchmarkRun.objects.filter(benchmarks=bench, status='COMPLETED') \
            .select_related('model').order_by('-completed_at'):
        entry = per_model.setdefault(run.model_id, {'runs_count': 0})
        entry['runs_count'] += 1
        if run.model_id not in order:
            metrics = (run.summary or {}).get(bench.name, {})
            score = run.best_scores().get(bench.name)
            if score is not None:
                # MoE rebuilds share display_name — a variant tag + run id keep
                # chart labels unique (Plotly merges identical categories)
                job = MoEExpertJob.objects.filter(output_model=run.model).select_related('theme').first()
                variant = f"α={job.alpha}, {job.position}, {job.theme.slug}" if job else 'vanilla/base'
                entry.update({'model': run.model, 'score': score,
                              'metrics': metrics, 'run': run,
                              'variant': variant})
                order.append(run.model_id)
    rows = sorted((per_model[i] for i in order),
                  key=lambda r: -r['score'])

    top_metric = ''
    if rows and rows[0]['metrics']:
        for key in rows[0]['metrics']:
            if key != 'sample_len':
                top_metric = key
                break
    chart = {
        'models': [f"{(r['model'].display_name or r['model'].name)[:36]} · {r['variant']} · run{r['run'].pk}"
                   for r in rows],
        'scores': [round(r['score'], 4) for r in rows],
        'metric': top_metric,
    }
    context = {
        'bench': bench, 'runs': runs, 'model_rows': rows,
        'chart': chart, 'active_tab': 'benchmarks',
    }
    return render(request, 'ivanna_model_manager/benchmarks/detail.html', context)


@login_required
@require_POST
def benchmark_install(request, pk):
    """Kick off dataset prefetch for one benchmark."""
    from django.http import JsonResponse
    from ..models import Benchmark
    from ..tasks_benchmarks import install_benchmark_task

    bench = get_object_or_404(Benchmark, pk=pk)
    if not bench.dataset_path:
        return JsonResponse({'success': False, 'error': 'No dataset to install'})
    if bench.install_status == 'downloading':
        return JsonResponse({'success': False, 'error': 'Already installing'})
    transaction.on_commit(lambda: install_benchmark_task.delay(bench.pk))
    return JsonResponse({'success': True})


@login_required
def benchmark_install_status(request, pk):
    from django.http import JsonResponse
    from ..models import Benchmark

    bench = get_object_or_404(Benchmark, pk=pk)
    return JsonResponse({'install_status': bench.install_status,
                         'progress': bench.install_progress,
                         'is_installed': bench.is_installed,
                         'error': bench.install_error})


@login_required
def benchmark_run_create(request):
    """Create a run: model + installed benchmarks + knobs."""
    from ..models import Benchmark, ModelCheckpoint
    from ..tasks_benchmarks import run_benchmark_task

    models_qs = ModelCheckpoint.objects.filter(
        is_active=True, status__in=['DOWNLOADED', 'LOADED'],
        local_path__isnull=False).exclude(local_path='')

    if request.method == 'POST':
        model_id = request.POST.get('model')
        bench_ids = request.POST.getlist('benchmarks')
        run = BenchmarkRun(
            model_id=model_id,
            limit=int(request.POST['limit']) if request.POST.get('limit') else None,
            num_fewshot=int(request.POST['num_fewshot']) if request.POST.get('num_fewshot', '').strip() else None,
            batch_size=request.POST.get('batch_size') or 'auto',
            created_by=request.user.username,
        )
        base = run.model
        # same quantization policy as MoE scan: >= 1.5B on a small GPU needs nf4
        run.quantized = (base.parameter_count or 0) >= 1_500_000_000
        run.save()
        run.benchmarks.set(bench_ids)
        transaction.on_commit(lambda: run_benchmark_task.delay(run.pk))
        messages.success(request, _('Задача бенчмарка "%(name)s" создана') % {'name': run.name})
        return redirect('benchmark_run_detail', pk=run.pk)

    installed = Benchmark.objects.filter(install_status='completed').order_by('category', 'name')
    context = {
        'models': models_qs.order_by('name'),
        'installed_benchmarks': installed,
        'preselect': request.GET.get('benchmark') or '',
        'active_tab': 'benchmarks',
    }
    return render(request, 'ivanna_model_manager/benchmarks/run_create.html', context)


def _parse_run_log(logs):
    """Extract the executed command, live activity line and lm-eval phase
    progress from the run log."""
    command, activity = '', ''
    phases = {}  # ordered dict: phase label -> last seen percent
    phase_re = re.compile(r'Running (generate_until|loglikelihood) requests:\s+(\d{1,3})%')
    for line in logs.splitlines():
        if '] Command: ' in line:
            command = line.split('] Command: ', 1)[1].strip()
        m = phase_re.search(line)
        if m:
            label = 'Generation (beams)' if m.group(1) == 'generate_until' \
                else 'Scoring (loglikelihood)'
            phases[label] = int(m.group(2))
            activity = line.split('] ', 1)[-1].strip()
    return command, list(phases.items()), activity


@login_required
def benchmark_run_detail(request, pk):
    from ..models import BenchmarkRun

    run = get_object_or_404(BenchmarkRun.objects.select_related('model'), pk=pk)
    results_pretty = ''
    if run.results:
        try:
            results_pretty = json.dumps(run.results, indent=2, ensure_ascii=False)[:40000]
        except (TypeError, ValueError):
            results_pretty = str(run.results)[:40000]
    command, phases, activity = _parse_run_log(run.logs)
    return render(request, 'ivanna_model_manager/benchmarks/run_detail.html',
                  {'run': run, 'results_pretty': results_pretty,
                   'command': command, 'phases': phases, 'activity': activity,
                   'active_tab': 'benchmarks'})


@login_required
def benchmark_run_status(request, pk):
    """Polling JSON: DB fields merged with the Celery state (pattern
    api_evaluation_task_status)."""
    from django.http import JsonResponse
    from celery.result import AsyncResult
    from ..models import BenchmarkRun

    run = get_object_or_404(BenchmarkRun, pk=pk)
    data = {
        'status': run.status,
        'progress': run.progress,
        'current_step': run.current_step,
        'is_running': run.status in ('PENDING', 'RUNNING'),
        'error_message': run.error_message,
        'logs_tail': run.logs[-2500:],
    }
    if run.celery_task_id:
        res = AsyncResult(run.celery_task_id)
        if res.state == 'PROGRESS' and isinstance(res.info, dict):
            data['progress'] = res.info.get('progress', run.progress)
            data['current_step'] = res.info.get('current_step', run.current_step)
    from ..tasks_benchmarks import _PROGRESS_RE
    command, phases, activity = _parse_run_log(run.logs)
    data['phases'] = phases
    data['activity'] = activity
    return JsonResponse(data)


@login_required
def benchmark_leaderboard(request):
    """Cross-run leaderboard: checkpoints × benchmarks.

    Cell = score from the LATEST completed run of that pair (a limited debug
    run does not shadow a later full run). Rows sorted by mean score.
    Filters: model, benchmark column, free-text search over names."""
    from ..models import BenchmarkRun

    runs = (BenchmarkRun.objects.filter(status='COMPLETED')
            .select_related('model').order_by('-created_at'))
    bench_names, table = [], {}
    for run in runs:
        row = table.setdefault(run.model_id, {'model': run.model, 'scores': {}})
        for bench_name, value in run.best_scores().items():
            if bench_name not in row['scores']:
                row['scores'][bench_name] = value
            if bench_name not in bench_names:
                bench_names.append(bench_name)
    bench_names.sort()

    model_id = request.GET.get('model') or ''
    bench_filter = request.GET.get('benchmark') or ''
    q = (request.GET.get('q') or '').strip()

    all_rows = [r for r in table.values() if r['scores']]
    filter_models = sorted(all_rows, key=lambda r: (r['model'].display_name or r['model'].name).lower())

    rows = all_rows
    if model_id:
        rows = [r for r in rows if str(r['model'].id) == model_id]
    if q:
        ql = q.lower()
        rows = [r for r in rows
                if ql in (r['model'].display_name or '').lower()
                or ql in (r['model'].name or '').lower()]
        bench_names = [b for b in bench_names if ql in b.lower()]
    if bench_filter:
        bench_names = [b for b in bench_names if b == bench_filter]

    rows.sort(key=lambda r: -(sum(r['scores'].values()) / len(r['scores'])))
    for row in rows:
        row['max_score'] = max(row['scores'].values(), default=0)

    # right panel: benchmarks that have completed runs (models count + best score)
    from ..models import Benchmark, Benchmark as B
    bench_cards = []
    for b in bench_names:
        bench_obj = Benchmark.objects.filter(name=b).first()
        models_count = sum(1 for r in table.values() if b in r['scores'])
        best = max((r['scores'][b] for r in table.values() if b in r['scores']), default=0)
        bench_cards.append({'name': b,
                            'pk': bench_obj.pk if bench_obj else None,
                            'models_count': models_count,
                            'best': best})
    return render(request, 'ivanna_model_manager/benchmarks/leaderboard.html',
                  {'bench_names': bench_names, 'rows': rows,
                   'bench_cards': bench_cards,
                   'filter_models': filter_models,
                   'filter_benchmarks': sorted(table and {b for r in table.values() for b in r['scores']} or []),
                   'model_id': model_id, 'benchmark_filter': bench_filter, 'q': q,
                   'active_tab': 'benchmarks'})
