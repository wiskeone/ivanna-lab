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



class GenerationPreset(models.Model):
    """
    v5.0: named generation parameter profile.

    Reused in chat sessions and comparisons: applying a preset merges
    its config into the session generation_config (does not replace it).
    """
    name = models.CharField(
        max_length=100,
        unique=True,
        help_text=_("Уникальное имя пресета")
    )
    description = models.TextField(blank=True, help_text=_("Описание пресета"))
    config = models.JSONField(
        default=dict,
        help_text=_("Параметры генерации: temperature, max_tokens, top_p, top_k, repetition_penalty")
    )
    created_by = models.ForeignKey(
        User, on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='generation_presets',
        help_text=_("Кто создал пресет")
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name


class MoETheme(models.Model):
    """Theme (domain) for soft MoE: prompt sets + router artifacts.

    Sets are txt files (1 prompt per line) in MEDIA_ROOT/moe_themes/<slug>/.
    Validation gate: cross cos(domain, contrast) < 0.5 (embeddings).
    """
    VALIDATION_CHOICES = [
        ('NONE', _('Не проверена')),
        ('PENDING', _('Проверяется')),
        ('PASSED', _('Пройдена')),
        ('FAILED', _('Не пройдена')),
    ]

    name = models.CharField(max_length=100, unique=True, verbose_name=_("Название темы"))
    slug = models.SlugField(max_length=120, unique=True, allow_unicode=False,
                            verbose_name=_("Слаг (для файлов)"))
    description = models.TextField(blank=True, verbose_name=_("Описание домена"))

    # Prompt set files (relative paths from MEDIA_ROOT)
    domain_prompts_file = models.CharField(max_length=500, blank=True,
                                           verbose_name=_("Файл доменных промптов"))
    contrast_prompts_file = models.CharField(max_length=500, blank=True,
                                             verbose_name=_("Файл контрастных промптов"))
    eval_domain_file = models.CharField(max_length=500, blank=True,
                                        verbose_name=_("Файл eval-промптов домена"))
    eval_contrast_file = models.CharField(max_length=500, blank=True,
                                          verbose_name=_("Файл eval-общих промптов"))

    # Set validation (Stage 0 gate)
    validation_status = models.CharField(max_length=10, choices=VALIDATION_CHOICES,
                                         default='NONE')
    validation_report = models.JSONField(default=dict, blank=True)

    # Embedding model for validation and the router
    embedding_model = models.ForeignKey(
        'EmbeddingModel', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='moe_themes', verbose_name=_("Модель эмбеддингов"))

    # Router (Stage 4)
    router_threshold = models.FloatField(default=0.08,
                                         help_text=_("margin cos(тема)−cos(контраст) > порога → эксперт"))
    router_report = models.JSONField(default=dict, blank=True)
    router_built_at = models.DateTimeField(null=True, blank=True)

    created_by = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Мягкий MoE: тема"
        verbose_name_plural = "Мягкий MoE: темы"
        indexes = [models.Index(fields=['validation_status'])]

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.slug:
            from django.utils.text import slugify
            base = slugify(self.name, allow_unicode=False) or 'theme'
            slug, i = base, 2
            while MoETheme.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                slug = f"{base}-{i}"
                i += 1
            self.slug = slug
        super().save(*args, **kwargs)

    # Prompt set file paths
    def _media_path(self, rel):
        from django.conf import settings
        return os.path.join(settings.MEDIA_ROOT, rel) if rel else None

    def domain_prompts_path(self):
        return self._media_path(self.domain_prompts_file)

    def contrast_prompts_path(self):
        return self._media_path(self.contrast_prompts_file)

    def eval_domain_path(self):
        return self._media_path(self.eval_domain_file)

    def eval_contrast_path(self):
        return self._media_path(self.eval_contrast_file)

    def theme_dir(self):
        from django.conf import settings
        return os.path.join(settings.MEDIA_ROOT, 'moe_themes', self.slug)

    def get_embedding_model_id(self):
        return self.embedding_model.pk if self.embedding_model else None


class MoEExpertJob(models.Model):
    """Job building a soft MoE expert from a dense model (scan → patch → gates)."""
    STATUS_CHOICES = [
        ('PENDING', _('В очереди')),
        ('VALIDATING', _('Проверка входа')),
        ('SCANNING', _('Скан активаций')),
        ('PATCHING', _('Патч весов')),
        ('EVALUATING', _('Eval-гейт')),
        ('ROUTING', _('Построение роутера')),
        ('COMPLETED', _('Завершено')),
        ('FAILED', _('Ошибка')),
    ]
    MODE_CHOICES = [
        # mode names stay technical (latin); RU notes are help_text-only
        ('steer', 'steer'),
        ('amplify', 'amplify'),
    ]
    LOAD_MODE_CHOICES = [
        ('auto', _('Авто (bf16 + CPU-offload при нехватке VRAM)')),
        ('int4', _('INT4/nf4 (скан больших моделей на малой GPU)')),
        ('cpu', _('Только CPU')),
    ]

    name = models.CharField(max_length=255, verbose_name=_("Название задачи"))
    theme = models.ForeignKey('MoETheme', on_delete=models.CASCADE,
                              related_name='jobs', verbose_name=_("Тема"))
    base_checkpoint = models.ForeignKey(
        'ModelCheckpoint', on_delete=models.CASCADE,
        related_name='moe_expert_jobs', verbose_name=_("Базовая плотная модель"))

    # Patch parameters (defaults verified in the lab)
    mode = models.CharField(max_length=10, choices=MODE_CHOICES, default='steer')
    # v5.2: goal of the weight operation — amplify the theme (soft MoE) or
    # suppress it (WSP ablation, like Exp. 1 with the think direction: W ← (I−α·d̂d̂ᵀ)W)
    goal = models.CharField(max_length=10, choices=[
        ('amplify', _('amplify — усилить тему (мягкий MoE)')),
        ('ablate', _('ablate — подавить тему (WSP-абляция, как Эксп. 1)')),
    ], default='amplify', verbose_name=_("Цель операции"))
    # v5.2: analysis mode — scan and reports only (basis of the "pre-build" decision)
    do_patch = models.BooleanField(default=True,
                                   help_text=_("Снять флажок = только скан и визуализация, без патча и чекпойнта"))
    alpha = models.FloatField(default=0.2,
                              help_text=_("Сила патча; сетка 0.1/0.2/0.4, подбирается eval-гейтом"))
    layers = models.CharField(max_length=255, default='auto',
                              help_text=_("'auto' — по каузальным писателям, или CSV: 19,20,21,22,23"))
    position = models.CharField(max_length=10, default='last', choices=[
        ('last', _('last — последний токен (проверено в лабе)')),
        ('mean', _('mean — среднее по токенам')),
    ], help_text=_("Позиция скрытых состояний для направления d̂"))
    holdout_seed = models.IntegerField(default=7)
    load_mode = models.CharField(max_length=10, choices=LOAD_MODE_CHOICES, default='auto')

    # Optional stages
    do_eval = models.BooleanField(default=True,
                                  help_text=_("Eval-гейт: генерация base vs expert на eval-наборах темы"))
    do_router = models.BooleanField(default=True,
                                    help_text=_("Построить центроиды роутера и прогнать тест точности"))

    # Status and progress (FineTuningJob pattern)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    progress = models.FloatField(default=0.0)
    current_step = models.CharField(max_length=255, blank=True)

    # Stage reports
    scan_report = models.JSONField(default=dict, blank=True)
    patch_report = models.JSONField(default=dict, blank=True)
    eval_report = models.JSONField(default=dict, blank=True)
    router_report = models.JSONField(default=dict, blank=True)

    # v5.2: LLM interpretation of results (killer feature) — text generated
    # by the model from a digest of the scan/patch/eval/router reports
    interpretation = models.TextField(blank=True, default='',
                                      help_text=_("LLM-интерпретация результатов прогона"))
    # v5.3: re-patch click tool — config from the heatmap selection
    patch_config = models.JSONField(default=dict, blank=True,
                                    help_text=_("Слои/группы размерностей из выделения на теплокартах"))
    reuse_scan_from = models.ForeignKey(
        'self', null=True, blank=True, on_delete=models.SET_NULL,
        related_name='repatch_children',
        help_text=_("Родительская джоба: переиспользовать её скан (npz/scan_report), пропустить скан-фазу"))

    # Output
    output_model = models.ForeignKey(
        'ModelCheckpoint', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='moe_output_of', verbose_name=_("Модель-эксперт"))
    hf_repo_id = models.CharField(max_length=300, blank=True,
                                  verbose_name=_("HF-репозиторий (после публикации)"))

    # Logs and errors
    logs = models.TextField(blank=True)
    error_message = models.TextField(blank=True)
    error_details = models.JSONField(default=dict, blank=True)
    celery_task_id = models.CharField(max_length=255, blank=True)

    created_by = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Мягкий MoE: задача"
        verbose_name_plural = "Мягкий MoE: задачи"
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['base_checkpoint', 'status']),
        ]

    def __str__(self):
        return f"{self.name} ({self.get_status_display()})"

    def job_dir(self):
        """Job artifact directory: scan npz/report + output model."""
        from django.conf import settings
        return os.path.join(settings.MEDIA_ROOT, 'moe_experts', str(self.pk))

    def update_progress(self, progress, step=""):
        self.progress = max(0.0, min(100.0, progress))
        if step:
            self.current_step = step
        self.save(update_fields=['progress', 'current_step'])

    def add_log(self, message):
        timestamp = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logs += f"[{timestamp}] {message}\n"
        self.save(update_fields=['logs'])

    def fail(self, error_message, error_details=None):
        self.status = 'FAILED'
        self.error_message = error_message
        if error_details:
            self.error_details = error_details
        self.completed_at = timezone.now()
        self.save()
        self.add_log(f"❌ ERROR: {error_message}")

    def complete(self, output_model):
        self.status = 'COMPLETED'
        self.output_model = output_model
        self.progress = 100.0
        self.completed_at = timezone.now()
        self.save()
        self.add_log(f"✅ Expert built: {output_model.name}")
