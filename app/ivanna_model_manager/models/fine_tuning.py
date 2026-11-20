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

from .registry import ModelCheckpoint



class FineTuningMethod(models.TextChoices):
    """Available fine-tuning methods."""
    DOCUMENT = 'document', 'Обучение на документах (LoRA/QLoRA)'
    INSTRUCTION = 'instruction', 'Обучение на инструкциях'
    RAFT = 'raft', 'RAFT (Retrieval-Augmented Fine-Tuning)'


class FineTuningJob(models.Model):
    """Model fine-tuning job."""

    STATUS_CHOICES = [
        ('PENDING', 'В ожидании'),
        ('PREPARING', 'Подготовка данных'),
        ('TRAINING', 'Обучение'),
        ('EVALUATING', 'Оценка'),
        ('COMPLETED', 'Завершено'),
        ('FAILED', 'Ошибка'),
        ('PAUSED', 'Приостановлено'),
        ('CANCELLED', 'Отменено'),
    ]
    
    # Basic info
    name = models.CharField(max_length=255, verbose_name=_("Название задачи"))
    description = models.TextField(blank=True, verbose_name=_("Описание"))

    # Training method
    method = models.CharField(
        max_length=20,
        choices=FineTuningMethod.choices,
        default=FineTuningMethod.RAFT,
        verbose_name=_("Метод обучения")
    )
    
    # Base model
    base_model = models.ForeignKey(
        'ModelCheckpoint',
        on_delete=models.CASCADE,
        related_name='fine_tuning_jobs',
        verbose_name=_("Базовая модель")
    )
    
    # Output model (after training)
    output_model = models.ForeignKey(
        'ModelCheckpoint',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='tuned_from',
        verbose_name=_("Результирующая модель")
    )
    
    # Training data sources
    document_collections = models.ManyToManyField(
        'DocumentCollection',
        through='FineTuningJobCollection',
        blank=True,
        related_name='fine_tuning_jobs',
        verbose_name=_("Коллекции документов")
    )

    instruction_datasets = models.ManyToManyField(
        'EvaluationDataset',
        through='FineTuningJobDataset',
        blank=True,
        related_name='fine_tuning_jobs',
        verbose_name=_("Датасеты инструкций")
    )
    
    # Training parameters
    lora_r = models.IntegerField(
        default=16,
        help_text=_("Ранг LoRA (рекомендуется 8-32)")
    )
    lora_alpha = models.IntegerField(
        default=32,
        help_text=_("Alpha параметр LoRA")
    )
    lora_dropout = models.FloatField(
        default=0.05,
        help_text=_("Dropout в LoRA слоях")
    )
    
    # Hyperparameters
    learning_rate = models.FloatField(default=5e-5)
    batch_size = models.IntegerField(default=4)
    gradient_accumulation_steps = models.IntegerField(default=4)
    num_epochs = models.IntegerField(default=3)
    max_seq_length = models.IntegerField(default=2048)
    # v5.3 Tier-1: LoRA++ pack
    use_dora = models.BooleanField(default=False, help_text=_("DoRA: разложение веса на magnitude+direction — качество выше на низких рангах"))
    use_rslora = models.BooleanField(default=False, help_text=_("rsLoRA: масштабирование alpha на sqrt(r) — стабильнее на малых рангах"))
    lora_plus_lr_ratio = models.FloatField(null=True, blank=True, help_text=_("LoRA+: lr матриц B = lr × это отношение (пусто = выключено, типово 16)"))
    neftune_alpha = models.FloatField(null=True, blank=True, help_text=_("NEFTune: шум на эмбеддингах — буст instruction-tuning (типово 5-15)"))
    target_modules = models.JSONField(default=list, blank=True, help_text=_("Явный список модулей LoRA (пусто = пресет)"))
    target_preset = models.CharField(max_length=30, default='hybrid-full', blank=True,
                                     choices=[('hybrid-full', _('Гибрид: attention+MLP+linear_attn')),
                                              ('attention+mlp', _('Attention + MLP')),
                                              ('attention', _('Только attention'))],
                                     help_text=_("Пресет целевых модулей (conv1d исключён)"))    
    # RAFT parameters
    distractors_per_example = models.IntegerField(
        default=3,
        help_text=_("Количество дистракторов на пример (для RAFT)")
    )
    oracle_ratio = models.FloatField(
        default=0.3,
        help_text=_("Доля примеров только с релевантным контекстом")
    )
    distractor_ratio = models.FloatField(
        default=0.5,
        help_text=_("Доля примеров с дистракторами")
    )
    no_context_ratio = models.FloatField(
        default=0.2,
        help_text=_("Доля примеров без контекста")
    )
    
    # Anti-forgetting parameters
    use_kl_regularization = models.BooleanField(
        default=True,
        help_text=_("Использовать KL-дивергенцию против забывания")
    )
    kl_lambda = models.FloatField(
        default=0.1,
        help_text=_("Вес KL-дивергенции")
    )
    general_instructions_ratio = models.FloatField(
        default=0.2,
        help_text=_("Доля общих инструкций в обучающей выборке")
    )

    quantization = models.CharField(
        max_length=50,
        choices=ModelCheckpoint.QUANTIZATION_CHOICES,
        blank=True,
        default='int4',
        verbose_name=_("Квантизация"),
        help_text=_("Тип квантизации для обучения (будет применена к модели)")
    )
    
    # Configuration (JSON)
    training_config = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Дополнительные параметры обучения")
    )
    
    # Status and progress
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='PENDING'
    )

    # v5.1: pause/resume — the flag is read by PauseCallback on every step
    pause_requested = models.BooleanField(
        default=False,
        help_text=_("Пользователь запросил паузу: тренер остановится на ближайшем шаге")
    )

    # v5.1: GPU selection — null = auto (first available), -1 = CPU, 0..N = specific GPU
    gpu_device = models.IntegerField(
        null=True, blank=True,
        help_text=_("GPU-устройство для обучения (null = авто, -1 = CPU, 0..N = конкретный GPU)")
    )
    progress = models.FloatField(default=0.0)
    current_step = models.CharField(max_length=255, blank=True)
    
    # Logs and errors
    logs = models.TextField(blank=True)
    error_message = models.TextField(blank=True)
    error_details = models.JSONField(default=dict, blank=True)

    # Training metrics
    training_metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Метрики в процессе обучения (loss, grad_norm)")
    )
    
    # Evaluation results
    evaluation_results = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Результаты оценки на валидации")
    )

    # Error records
    training_errors = models.JSONField(
        default=list, blank=True,
        help_text=_("Список ошибок, возникших во время обучения (каждая ошибка – словарь с timestamp, message, traceback)")
    )
    
    # Celery task reference
    celery_task_id = models.CharField(max_length=255, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    created_by = models.CharField(max_length=255, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Задача fine-tuning"
        verbose_name_plural = "Задачи fine-tuning"
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['base_model', 'status']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.get_method_display()})"
    
    def update_progress(self, progress, step=""):
        """Update job progress."""
        self.progress = max(0.0, min(100.0, progress))
        if step:
            self.current_step = step
        self.save(update_fields=['progress', 'current_step'])
    

    # v5.1: pause/resume
    def request_pause(self):
        """Request a pause: the trainer will stop at the nearest step."""
        self.pause_requested = True
        self.save(update_fields=['pause_requested'])
        self.add_log("⏸ Pause requested — trainer will stop after the current step")

    def resume(self):
        """Resume: reset the flag and re-queue (the checkpoint will be picked up)."""
        self.pause_requested = False
        self.status = 'PENDING'
        self.save(update_fields=['pause_requested', 'status'])
        self.add_log("▶ Resumed — continuing from the last checkpoint")

    def add_log(self, message):
        """Append an entry to the log."""
        timestamp = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logs += f"[{timestamp}] {message}\n"
        self.save(update_fields=['logs'])  # removed 'updated_at'

    def fail(self, error_message, error_details=None):
        """Mark the job as failed."""
        import traceback
        self.status = 'FAILED'
        self.error_message = error_message
        if error_details:
            self.error_details = error_details
        self.completed_at = timezone.now()
        self.save()
        self.add_log(f"❌ ERROR: {error_message}")
    
    def complete(self, output_model):
        """Mark the job as completed."""
        self.status = 'COMPLETED'
        self.output_model = output_model
        self.progress = 100.0
        self.completed_at = timezone.now()
        self.save()
        self.add_log(f"✅ Fine-tuning completed, output model: {output_model.name}")

    def add_error(self, error_message, traceback_str=None):
        """Append an error entry to the training_errors list."""
        entry = {
            'timestamp': timezone.now().isoformat(),
            'message': error_message,
        }
        if traceback_str:
            entry['traceback'] = traceback_str
        self.training_errors.append(entry)
        self.save(update_fields=['training_errors'])
        # Also write to the regular log
        self.add_log(f"❌ ERROR: {error_message}")


class FineTuningJobCollection(models.Model):
    """Through model linking FineTuningJob and DocumentCollection."""
    job = models.ForeignKey('FineTuningJob', on_delete=models.CASCADE)
    collection = models.ForeignKey('DocumentCollection', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['job', 'collection']


class FineTuningJobDataset(models.Model):
    """Through model linking FineTuningJob and EvaluationDataset."""
    job = models.ForeignKey('FineTuningJob', on_delete=models.CASCADE)
    dataset = models.ForeignKey('EvaluationDataset', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['job', 'dataset']


class FineTuningCheckpoint(models.Model):
    """Intermediate checkpoints during training."""
    job = models.ForeignKey(
        FineTuningJob,
        on_delete=models.CASCADE,
        related_name='checkpoints'
    )
    
    step = models.IntegerField(help_text=_("Номер шага/эпохи"))
    path = models.CharField(max_length=500, help_text=_("Путь к файлу чекпоинта"))
    
    # Metrics at save time
    loss = models.FloatField(null=True, blank=True)
    grad_norm = models.FloatField(null=True, blank=True)
    learning_rate = models.FloatField(null=True, blank=True)

    # Additional metrics (if computed)
    metrics = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['job', 'step']
        unique_together = ['job', 'step']
    
    def __str__(self):
        return f"Checkpoint {self.step} for {self.job.name}"


class FineTuningEvaluation(models.Model):
    """Evaluation of the fine-tuned model on test data."""

    job = models.ForeignKey(
        FineTuningJob,
        on_delete=models.CASCADE,
        related_name='evaluations'
    )

    # Evaluation dataset
    dataset = models.ForeignKey(
        'EvaluationDataset',
        on_delete=models.CASCADE,
        related_name='fine_tuning_evaluations'
    )
    
    # Metrics
    metrics = models.JSONField(
        default=dict,
        help_text=_("Метрики оценки (rouge, bleu, bertscore)")
    )

    # Detailed results
    detailed_results = models.JSONField(
        default=list,
        help_text=_("Результаты по каждому примеру")
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        unique_together = ['job', 'dataset']
    
    def __str__(self):
        return f"Evaluation of {self.job.name} on {self.dataset.name}"


class FineTuningTemplate(models.Model):
    """Fine-tuning configuration template."""
    
    name = models.CharField(max_length=255, verbose_name=_("Название шаблона"))
    description = models.TextField(blank=True, verbose_name=_("Описание"))
    
    # Method parameters
    method = models.CharField(
        max_length=20,
        choices=FineTuningMethod.choices,
        default=FineTuningMethod.RAFT,
        verbose_name=_("Метод обучения")
    )

    # Recommended models (JSON list of HF IDs)
    recommended_models = models.JSONField(
        default=list,
        help_text=_("Список рекомендуемых моделей (HF IDs)")
    )
    
    # LoRA parameters
    lora_r = models.IntegerField(default=16)
    lora_alpha = models.IntegerField(default=32)
    lora_dropout = models.FloatField(default=0.05)
    
    # Hyperparameters
    learning_rate = models.FloatField(default=5e-5)
    batch_size = models.IntegerField(default=4)
    gradient_accumulation_steps = models.IntegerField(default=4)
    num_epochs = models.IntegerField(default=3)
    max_seq_length = models.IntegerField(default=2048)

    # RAFT parameters
    distractors_per_example = models.IntegerField(default=3)
    oracle_ratio = models.FloatField(default=0.3)
    distractor_ratio = models.FloatField(default=0.5)
    no_context_ratio = models.FloatField(default=0.2)
    use_kl_regularization = models.BooleanField(default=True)
    kl_lambda = models.FloatField(default=0.1)
    general_instructions_ratio = models.FloatField(default=0.2)
    
    # Category and tags
    category = models.CharField(
        max_length=50,
        choices=[
            ('quick', _('Быстрый старт')),
            ('quality', _('Максимальное качество')),
            ('rag', _('RAG-оптимизированный')),
            ('experimental', _('Экспериментальный')),
        ],
        default='quick'
    )
    tags = models.JSONField(default=list, blank=True)
    
    # Usage stats
    usage_count = models.IntegerField(default=0)
    avg_rating = models.FloatField(null=True, blank=True)

    # Author
    created_by = models.CharField(max_length=255, blank=True)
    is_public = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-usage_count', '-created_at']
        verbose_name = "Шаблон fine-tuning"
        verbose_name_plural = "Шаблоны fine-tuning"
    
    def __str__(self):
        return f"{self.name} ({self.get_method_display()})"
    
    def increment_usage(self):
        """Increment the usage counter."""
        self.usage_count += 1
        self.save(update_fields=['usage_count'])


class TrainingMetric(models.Model):
    """
    Per-step training metrics for detailed monitoring.
    """
    job = models.ForeignKey('FineTuningJob', on_delete=models.CASCADE, related_name='metrics')
    step = models.IntegerField(help_text=_("Номер шага (итерации)"))
    epoch = models.FloatField(null=True, blank=True, help_text=_("Текущая эпоха"))
    train_loss = models.FloatField(null=True, blank=True, help_text=_("Loss на тренировочном батче"))
    eval_loss = models.FloatField(null=True, blank=True, help_text=_("Loss на валидации"))
    grad_norm = models.FloatField(null=True, blank=True, help_text=_("Норма градиента (L2)"))
    learning_rate = models.FloatField(null=True, blank=True, help_text=_("Текущая скорость обучения"))
    weights_histograms = models.JSONField(
        default=dict, blank=True,
        help_text=_("Гистограммы весов ключевых слоёв (словарь: имя_параметра -> список частот)")
    )
    grad_histograms = models.JSONField(
        default=dict, blank=True,
        help_text=_("Гистограммы градиентов (аналогично)")
    )
    time_elapsed = models.FloatField(null=True, blank=True, help_text=_("Секунд от начала обучения"))
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['step']
        unique_together = ['job', 'step']
        verbose_name = "Метрика обучения"
        verbose_name_plural = "Метрики обучения"

    def __str__(self):
        return f"Шаг {self.step} задачи {self.job.id}"


#############################################################################
# Convertion
#############################################################################
class ModelConversionJob(models.Model):
    """
    Model conversion job to GGUF format.
    """
    STATUS_CHOICES = [
        ('PENDING', 'В ожидании'),
        ('PREPARING', 'Подготовка'),
        ('CONVERTING', 'Конвертация'),
        ('COMPLETED', 'Завершено'),
        ('FAILED', 'Ошибка'),
        ('CANCELLED', 'Отменено'),
    ]

    QUANTIZATION_CHOICES = [
        ('q4_0', 'Q4_0 (4-bit, минимальный)'),
        ('q4_1', 'Q4_1 (4-bit, чуть точнее)'),
        ('q5_0', 'Q5_0 (5-bit)'),
        ('q5_1', 'Q5_1 (5-bit, точнее)'),
        ('q8_0', 'Q8_0 (8-bit)'),
        ('f16', 'FP16'),
        ('f32', 'FP32'),
    ]

    name = models.CharField(max_length=255, verbose_name=_("Название задачи"))
    description = models.TextField(blank=True, verbose_name=_("Описание"))

    # Source model (ModelCheckpoint)
    source_model = models.ForeignKey(
        'ModelCheckpoint',
        on_delete=models.CASCADE,
        related_name='conversion_jobs',
        verbose_name=_("Исходная модель")
    )

    # Output model (after conversion) — also a ModelCheckpoint
    output_model = models.ForeignKey(
        'ModelCheckpoint',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='converted_from',
        verbose_name=_("Результирующая модель")
    )

    # Conversion parameters
    quantization = models.CharField(
        max_length=10,
        choices=QUANTIZATION_CHOICES,
        default='q4_0',
        verbose_name=_("Тип квантизации")
    )

    # Extra parameters (extensible)
    config = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Дополнительные параметры конвертации")
    )

    # Status and progress
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='PENDING'
    )
    progress = models.FloatField(default=0.0)
    current_step = models.CharField(max_length=255, blank=True)

    # Logs and errors
    logs = models.TextField(blank=True)
    error_message = models.TextField(blank=True)
    error_details = models.JSONField(default=dict, blank=True)

    # Celery task reference
    celery_task_id = models.CharField(max_length=255, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    created_by = models.CharField(max_length=255, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Задача конвертации в GGUF"
        verbose_name_plural = "Задачи конвертации в GGUF"

    def __str__(self):
        return f"{self.name} ({self.source_model.name} -> {self.get_quantization_display()})"

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
        import traceback
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
        self.add_log(f"✅ Conversion completed. Output model: {output_model.name}")
