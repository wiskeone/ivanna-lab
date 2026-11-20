import os
import re
from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.db.models import Count, Avg, Max, Min, Q, Sum
from django.core.validators import MinValueValidator, MaxValueValidator
from django.core.exceptions import ValidationError
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from django.utils import timezone
import json
import random
import hashlib
import fnmatch
from pathlib import Path
from django.db.models import Q
import cryptography
import traceback

import logging

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)



class Benchmark(models.Model):
    """A benchmark in the catalog.

    source='curated' rows come from services/benchmarks/catalog.py (the
    hand-picked top list); source='lm_eval' rows are synced from the full
    lm-eval-harness task registry (600+ tasks, TaskManager introspection).
    Install state lives on the row (EmbeddingModel pattern): 'installing'
    a benchmark = prefetching its dataset from the HF Hub.
    """

    CATEGORY_CHOICES = [
        ('knowledge', _('Знания')),
        ('reasoning', _('Рассуждения')),
        ('math', _('Математика')),
        ('code', _('Код')),
        ('multilingual', _('Мультиязычность')),
        ('safety', _('Безопасность')),
        ('instruction', _('Следование инструкциям')),
        ('other', _('Другое')),
    ]

    name = models.CharField(max_length=200, unique=True,
                            help_text=_("Идентификатор задачи lm-eval (например 'gsm8k')"))
    display_name = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES,
                                default='other')
    metrics = models.JSONField(default=list, blank=True,
                               help_text=_("Метрики задачи (acc, exact_match, ...)"))
    group = models.CharField(max_length=100, blank=True,
                             help_text=_("Группа lm-eval (mmlu, bbh, ...)"))
    dataset_path = models.CharField(max_length=300, blank=True,
                                    help_text=_("HF datasets id для prefetch (например 'gsm8k')"))
    dataset_name = models.CharField(max_length=200, blank=True,
                                    help_text=_("Имя конфига датасета (subset), если есть"))
    n_samples = models.IntegerField(null=True, blank=True,
                                    help_text=_("Число примеров в датасете (для оценки длительности)"))

    source = models.CharField(max_length=20, default='lm_eval',
                              choices=[('curated', _('Курируемый')), ('lm_eval', _('Реестр lm-eval')),
                                       ('custom', _('Собственный'))])
    # v5.3 phase 2: custom benchmark QA storage (engine='custom' runner)
    qa_data = models.JSONField(default=list, blank=True,
                               help_text=_("[{prompt, reference, keywords[]}] для source=custom"))
    scoring = models.CharField(max_length=20, default='keywords',
                               choices=[('keywords', _('Содержит ключевое слово')),
                                        ('exact', _('Точное совпадение'))])
    is_popular = models.BooleanField(default=False,
                                     help_text=_("Показывать карточкой в «Популярных»"))

    install_status = models.CharField(max_length=20, default='not_started',
                                      choices=[('not_started', _('Не установлено')),
                                               ('downloading', _('Загрузка')),
                                               ('completed', _('Установлено')),
                                               ('failed', _('Ошибка'))])
    install_progress = models.FloatField(default=0.0)
    install_error = models.TextField(blank=True)
    installed_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-is_popular', 'category', 'name']
        indexes = [models.Index(fields=['source', 'category']),
                   models.Index(fields=['install_status'])]

    def __str__(self):
        return self.display_name or self.name

    @property
    def is_installed(self):
        return self.install_status == 'completed'


class BenchmarkRun(models.Model):
    """A benchmark run of one checkpoint over one or more benchmarks.

    engine='lm_eval' executes via the isolated lm-eval-harness venv
    (/opt/lmeval-venv) as a subprocess; engine='custom' is reserved for the
    future in-house runner. Pattern: MoEExpertJob.
    """

    ENGINE_CHOICES = [
        ('lm_eval', 'lm-eval-harness'),
        ('custom', _('Собственный раннер')),
    ]
    STATUS_CHOICES = [
        ('PENDING', _('В очереди')),
        ('RUNNING', _('Выполняется')),
        ('COMPLETED', _('Завершено')),
        ('FAILED', _('Ошибка')),
        ('CANCELLED', _('Отменено')),
    ]

    name = models.CharField(max_length=255, blank=True,
                            help_text=_("Автогенерируется, если пусто"))
    model = models.ForeignKey('ModelCheckpoint', on_delete=models.CASCADE,
                              related_name='benchmark_runs',
                              help_text=_("Чекпойнт для прогона"))
    benchmarks = models.ManyToManyField(Benchmark, related_name='runs')
    engine = models.CharField(max_length=20, choices=ENGINE_CHOICES,
                              default='lm_eval')
    limit = models.IntegerField(null=True, blank=True,
                                help_text=_("Максимум примеров на бенчмарк (пусто = весь датасет)"))
    num_fewshot = models.IntegerField(null=True, blank=True,
                                      help_text=_("Число few-shot примеров (пусто = по умолчанию задачи)"))
    batch_size = models.CharField(max_length=20, default='auto',
                                  help_text=_("auto или число"))
    quantized = models.BooleanField(default=False,
                                    help_text=_("nf4-квантизация модели для прогона (авто для моделей от 1.5B)"))

    status = models.CharField(max_length=20, choices=STATUS_CHOICES,
                              default='PENDING')
    progress = models.FloatField(default=0.0)
    current_step = models.CharField(max_length=255, blank=True)

    results = models.JSONField(default=dict, blank=True,
                               help_text=_("Сырой JSON-отчёт lm-eval"))
    summary = models.JSONField(default=dict, blank=True,
                               help_text=_("Нормализованные метрики: {benchmark: {metric: value}}"))
    logs = models.TextField(blank=True)
    error_message = models.TextField(blank=True)
    error_details = models.JSONField(default=dict, blank=True)
    celery_task_id = models.CharField(max_length=255, blank=True)
    output_dir = models.CharField(max_length=500, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)

    created_by = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['model', 'status']),
                   models.Index(fields=['status', 'created_at'])]

    def __str__(self):
        return self.name or f"BenchmarkRun #{self.pk}"

    def save(self, *args, **kwargs):
        if not self.name and self.model_id:
            self.name = f"bench_{self.model.safe_name}_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
        super().save(*args, **kwargs)

    def output_path(self):
        from django.conf import settings
        return os.path.join(settings.MEDIA_ROOT, 'benchmarks', 'runs', str(self.pk))

    def update_progress(self, progress, step=""):
        self.progress = max(0.0, min(100.0, progress))
        if step:
            self.current_step = step
        self.save(update_fields=['progress', 'current_step'])

    def add_log(self, message):
        timestamp = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logs += f"[{timestamp}] {message}\n"
        # keep the tail bounded: long lm-eval logs can be megabytes
        if len(self.logs) > 400_000:
            self.logs = self.logs[-350_000:]
        self.save(update_fields=['logs'])

    def fail(self, error_message, error_details=None):
        self.status = 'FAILED'
        self.error_message = error_message
        if error_details:
            self.error_details = error_details
        self.completed_at = timezone.now()
        self.save()
        self.add_log(f"❌ ERROR: {error_message}")

    def complete(self):
        self.status = 'COMPLETED'
        self.progress = 100.0
        self.completed_at = timezone.now()
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self.save()
        self.add_log("✅ Benchmark run completed")

    def best_scores(self):
        """{benchmark_name: primary_metric_value} — for the leaderboard."""
        priority = ('acc_norm,none', 'acc,none', 'exact_match,strict-match',
                    'exact_match,none', 'exact_match', 'acc_norm', 'acc',
                    'pass@1,create_test', 'pass@1', 'prompt_level_strict_acc,none',
                    'inst_level_strict_acc,none')
        non_scores = {'sample_len', 'alias'}
        out = {}
        for bench_name, metrics in (self.summary or {}).items():
            primary = None
            for metric in priority:
                if metric in metrics and isinstance(metrics[metric], (int, float)):
                    primary = metrics[metric]
                    break
            if primary is None:
                # fallback: first plausible score (exclude bookkeeping keys)
                primary = next((v for k, v in metrics.items()
                                if k not in non_scores and 'stderr' not in k
                                and isinstance(v, (int, float))), None)
            if primary is not None:
                out[bench_name] = primary
        return out
