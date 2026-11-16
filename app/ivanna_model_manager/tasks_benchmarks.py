"""Benchmark tasks (v5.3): catalog sync, dataset install, lm-eval runs.

The lm-eval-harness engine lives in an ISOLATED venv (/opt/lmeval-venv,
built in the GPU Dockerfile) and is invoked as a subprocess — the pinned
main-environment stack (transformers/accelerate/bnb) is never touched.
Patterns: GGUF conversion subprocess streaming (tasks.py:1568), MoE stage
progress (tasks_moe.py), EmbeddingModel install state (tasks_embedding.py).
"""
import json
import logging
import os
import re
import subprocess
import traceback

from celery import shared_task
from django.conf import settings
from django.utils import timezone

from .models import Benchmark, BenchmarkRun
from .services.benchmarks.catalog import curated_entries

logger = logging.getLogger(__name__)

LMEVAL_PYTHON = getattr(settings, 'LMEVAL_PYTHON', '/opt/lmeval-venv/bin/python')

# Dumps the full lm-eval task registry as JSON (ids + group membership).
REGISTRY_DUMP_CODE = """
import json
from lm_eval.tasks import TaskManager
tm = TaskManager()
tasks = sorted(set(tm.all_tasks))
groups = sorted(set(tm.all_groups)) if hasattr(tm, 'all_groups') else []
print(json.dumps({'tasks': tasks, 'groups': groups}))
"""

# Progress lines in lm-eval output: "Running loglikelihood requests:  12%|"
_PROGRESS_RE = re.compile(r'(\d{1,3})%')


def _sync_curated():
    """Upsert curated rows; install state is preserved. Returns count."""
    known = 0
    for raw in curated_entries():
        entry = dict(raw)  # never mutate the module-level catalog
        popular = entry.pop('popular', False)
        Benchmark.objects.update_or_create(
            name=entry['name'],
            defaults={**entry, 'source': 'curated', 'is_popular': popular},
        )
        known += 1
    return known


@shared_task(bind=True, queue='cpu_tasks', name='ivanna_model_manager.tasks_benchmarks.refresh_benchmark_catalog_task')
def refresh_benchmark_catalog_task(self):
    """Sync curated top + full lm-eval registry into Benchmark rows."""
    curated_count = _sync_curated()

    registry_count = 0
    note = ''
    if os.path.exists(LMEVAL_PYTHON):
        try:
            r = subprocess.run([LMEVAL_PYTHON, '-c', REGISTRY_DUMP_CODE],
                               capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                data = json.loads(r.stdout.strip().splitlines()[-1])
                names = data.get('tasks', [])
                groups = set(data.get('groups', []))
                for name in names:
                    Benchmark.objects.get_or_create(
                        name=name,
                        defaults={'source': 'lm_eval',
                                  'group': name if name in groups else ''},
                    )
                registry_count = len(names)
            else:
                note = f'registry dump failed: {r.stderr[-300:]}'
        except Exception as e:
            note = f'registry dump error: {e}'
    else:
        note = f'lm-eval venv not found at {LMEVAL_PYTHON} — curated only'

    result = {'curated': curated_count, 'registry': registry_count, 'note': note}
    logger.info(f"benchmark catalog refreshed: {result}")
    return result


@shared_task(bind=True, queue='cpu_tasks', autoretry_for=(Exception,),
             retry_backoff=True, max_retries=3, retry_jitter=True,
             name='ivanna_model_manager.tasks_benchmarks.install_benchmark_task')
def install_benchmark_task(self, benchmark_id):
    """Prefetch a benchmark's dataset from the HF Hub into the datasets cache.

    'Installing' a benchmark = having its dataset locally so a run does not
    start with a cold download. Uses the main environment (datasets is in
    requirements); no lm-eval needed for prefetching.
    """
    from django.core.cache import cache

    bench = Benchmark.objects.get(pk=benchmark_id)
    if not bench.dataset_path:
        bench.install_status = 'failed'
        bench.install_error = 'No dataset_path for this benchmark — install is not applicable'
        bench.save(update_fields=['install_status', 'install_error'])
        return {'success': False, 'reason': 'no dataset_path'}

    lock_key = f'bench_install:{bench.pk}'
    if not cache.add(lock_key, '1', timeout=3600):
        return {'success': False, 'reason': 'already installing'}
    try:
        bench.install_status = 'downloading'
        bench.install_progress = 5.0
        bench.install_error = ''
        bench.save(update_fields=['install_status', 'install_progress', 'install_error'])

        import datasets as hf_datasets
        ds = hf_datasets.load_dataset(
            bench.dataset_path, bench.dataset_name or None,
            trust_remote_code=True,
        )
        bench.install_progress = 90.0
        bench.save(update_fields=['install_progress'])

        # count examples if the split layout is standard
        total = 0
        try:
            for split in ds.values():
                total += len(split)
        except Exception:
            pass
        bench.n_samples = bench.n_samples or (total or None)
        bench.install_status = 'completed'
        bench.install_progress = 100.0
        bench.installed_at = timezone.now()
        bench.save(update_fields=['install_status', 'install_progress',
                                  'installed_at', 'n_samples', 'install_error'])
        logger.info(f"benchmark {bench.name} installed ({total} examples)")
        return {'success': True, 'benchmark': bench.name, 'examples': total}
    except Exception as e:
        bench.install_status = 'failed'
        bench.install_error = str(e)[:2000]
        bench.save(update_fields=['install_status', 'install_error'])
        logger.warning(f"benchmark {bench.name} install failed: {e}")
        return {'success': False, 'reason': str(e)[:300]}
    finally:
        cache.delete(lock_key)


def _prepare_quantized_model(model_path, out_dir):
    """Create a run-local model dir whose config.json carries the nf4
    quantization_config.

    lm-eval 0.4.13 reads bnb settings from the MODEL CONFIG only (CLI
    quantization args were dropped in transformers 5.x — bare load_in_4bit
    reaches the model __init__ and crashes). Everything except config.json
    is symlinked, so no weight copying."""
    import glob
    import json as _json

    dst = os.path.join(out_dir, 'model_quantized')
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(model_path, 'config.json'), encoding='utf-8') as fh:
        cfg = _json.load(fh)
    cfg['quantization_config'] = {
        'quant_method': 'bitsandbytes',
        'load_in_4bit': True,
        'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_compute_dtype': 'bfloat16',
        # hybrid-arch modules (conv1d / gated delta net) crash bnb's 4-bit
        # weight fixup; keep them and the head/embeddings in bf16
        'llm_int8_skip_modules': ['conv1d', 'linear_attn', 'lm_head', 'embed_tokens'],
    }
    for f in os.listdir(model_path):
        if f == 'config.json':
            continue
        src = os.path.join(model_path, f)
        link = os.path.join(dst, f)
        if not os.path.exists(link):
            os.symlink(os.path.abspath(src), link)
    with open(os.path.join(dst, 'config.json'), 'w', encoding='utf-8') as fh:
        _json.dump(cfg, fh, indent=2)
    return dst


def _build_lm_eval_command(run: BenchmarkRun, task_names, out_dir, model_path=None):
    """Build the lm-eval CLI for a run. Quantized runs load through
    _prepare_quantized_model (same nf4 policy as MoE scan: >= 1.5B on a 4GB
    GPU cannot load in bf16)."""
    model_args = [f'pretrained={model_path or run.model.local_path}',
                  'trust_remote_code=True', 'dtype=bfloat16']

    cmd = [
        LMEVAL_PYTHON, '-m', 'lm_eval',
        '--model', 'hf',
        '--tasks', ','.join(task_names),
        '--model_args', ','.join(model_args),
        '--batch_size', str(run.batch_size or 'auto'),
        '--output_path', out_dir,
    ]
    if run.num_fewshot is not None:
        cmd += ['--num_fewshot', str(run.num_fewshot)]
    if run.limit:
        cmd += ['--limit', str(run.limit)]
    # humaneval-class tasks are marked unsafe (code execution); the flag only
    # gates those, other tasks run unchanged
    cmd += ['--confirm_run_unsafe_code']
    return cmd


@shared_task(bind=True, queue='gpu_tasks', time_limit=86400, soft_time_limit=None,
             name='ivanna_model_manager.tasks_benchmarks.run_benchmark_task')
def run_benchmark_task(self, run_id):
    """Run lm-eval-harness for one checkpoint over the selected benchmarks.

    subprocess streams stdout into run.add_log; percentages parsed from
    lm-eval tqdm lines drive coarse progress. Results JSON is normalized
    into run.summary {benchmark: {metric: value}}.
    """
    from django.utils import timezone
    from ivanna_model_manager.services.redis_model_service import redis_model_manager

    run = BenchmarkRun.objects.select_related('model').get(pk=run_id)

    lock_key = f'bench_run:{run_id}'
    if not redis_model_manager.redis.set(lock_key, '1', nx=True, ex=86400):
        return {'success': False, 'reason': 'already running'}

    run.status = 'RUNNING'
    run.started_at = timezone.now()
    run.celery_task_id = self.request.id
    run.results = {}
    run.summary = {}
    run.error_message = ''
    run.error_details = {}
    run.save(update_fields=['status', 'started_at', 'celery_task_id',
                            'results', 'summary', 'error_message', 'error_details'])
    run.add_log("🚀 Benchmark run started")

    try:
        if not run.model.local_path or not os.path.exists(run.model.local_path):
            raise FileNotFoundError(f"Model not found locally: {run.model.local_path}")
        if not os.path.exists(LMEVAL_PYTHON):
            raise FileNotFoundError(
                f"lm-eval venv not found at {LMEVAL_PYTHON} — rebuild the GPU image")

        benches = list(run.benchmarks.all())
        if not benches:
            raise ValueError("No benchmarks selected")

        # prefetch any datasets that were not installed yet (cheap safety net)
        missing = [b for b in benches if not b.is_installed and b.dataset_path]
        if missing:
            run.add_log(f"Prefetching {len(missing)} dataset(s)...")
            for b in missing:
                try:
                    import datasets as hf_datasets
                    hf_datasets.load_dataset(b.dataset_path, b.dataset_name or None,
                                             trust_remote_code=True)
                    b.install_status = 'completed'
                    b.installed_at = timezone.now()
                    b.save(update_fields=['install_status', 'installed_at'])
                except Exception as e:
                    run.add_log(f"⚠️ Prefetch failed for {b.name}: {e} — lm-eval will retry itself")

        task_names = [b.name for b in benches]
        out_dir = run.output_path()
        os.makedirs(out_dir, exist_ok=True)
        run.output_dir = out_dir
        run.save(update_fields=['output_dir'])

        model_path = run.model.local_path
        if run.quantized:
            model_path = _prepare_quantized_model(run.model.local_path, out_dir)
            run.add_log("nf4 quantized load via config-injected model copy")
        cmd = _build_lm_eval_command(run, task_names, out_dir, model_path)
        run.add_log(f"Command: {' '.join(cmd)}")
        run.update_progress(5, "Starting lm-eval")

        n_tasks = len(task_names)
        # humaneval-class tasks execute generated code and require the
        # disclaimer acknowledgement; the variable is ignored elsewhere
        env = {**os.environ, 'HF_ALLOW_CODE_EVAL': '1'}
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   env=env)
        try:
            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if len(run.logs) < 380_000:
                    run.add_log(line)
                m = _PROGRESS_RE.search(line)
                if m and ('Running' in line or 'it/s' in line or 's/it' in line):
                    # coarse: stage progress inside the running phase
                    run.update_progress(10 + 80 * int(m.group(1)) / 100.0,
                                        "Running lm-eval")
        finally:
            process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"lm-eval exited with code {process.returncode}")

        run.update_progress(90, "Collecting results")

        # lm-eval writes <outdir>/<model>/results_<ts>.json (v0.4 layout:
        # possibly nested one level deeper). Take the newest results_*.json.
        results_path = None
        for root, _dirs, files in os.walk(out_dir):
            for f in files:
                if f.startswith('results_') and f.endswith('.json'):
                    p = os.path.join(root, f)
                    if results_path is None or os.path.getmtime(p) > os.path.getmtime(results_path):
                        results_path = p
        if not results_path:
            raise FileNotFoundError("lm-eval results_*.json not found in output dir")

        with open(results_path, encoding='utf-8') as fh:
            report = json.load(fh)

        run.results = _json_safe(report)
        summary = {}
        for task_name, metrics in (report.get('results') or {}).items():
            summary[task_name] = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and 'stderr' not in k and k != 'alias'
            }
        run.summary = _json_safe(summary)
        run.save(update_fields=['results', 'summary'])

        scores = run.best_scores()
        run.add_log(f"Results: " + ', '.join(f"{k}={v:.3f}" for k, v in scores.items()))
        run.complete()
        return {'success': True, 'run_id': run.pk, 'scores': scores}
    except Exception as e:
        run.fail(str(e), {'traceback': traceback.format_exc()})
        return {'success': False, 'error': str(e)}
    finally:
        redis_model_manager.redis.delete(lock_key)


def _json_safe(obj):
    """Recursively replace NaN/Inf floats (PG JSON rejects them) — same
    cleaner as tasks_moe."""
    import math

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj
