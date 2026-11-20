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
    """Доступные методы fine-tuning"""
    DOCUMENT = 'document', 'Обучение на документах (LoRA/QLoRA)'
    INSTRUCTION = 'instruction', 'Обучение на инструкциях'
    RAFT = 'raft', 'RAFT (Retrieval-Augmented Fine-Tuning)'


class FineTuningJob(models.Model):
    """Задача fine-tuning модели"""
    
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
    
    # Основная информация
    name = models.CharField(max_length=255, verbose_name=_("Название задачи"))
    description = models.TextField(blank=True, verbose_name=_("Описание"))
    
    # Метод обучения
    method = models.CharField(
        max_length=20,
        choices=FineTuningMethod.choices,
        default=FineTuningMethod.RAFT,
        verbose_name=_("Метод обучения")
    )
    
    # Базовая модель
    base_model = models.ForeignKey(
        'ModelCheckpoint',
        on_delete=models.CASCADE,
        related_name='fine_tuning_jobs',
        verbose_name=_("Базовая модель")
    )
    
    # Выходная модель (после обучения)
    output_model = models.ForeignKey(
        'ModelCheckpoint',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='tuned_from',
        verbose_name=_("Результирующая модель")
    )
    
    # Источники данных для обучения
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
    
    # Параметры обучения
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
    
    # Гиперпараметры
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
    # Параметры для RAFT
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
    
    # Параметры для борьбы с забыванием
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
    
    # Конфигурация (JSON)
    training_config = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Дополнительные параметры обучения")
    )
    
    # Статус и прогресс
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='PENDING'
    )

    # v5.1: пауза/резюме — флаг читается PauseCallback-ом на каждом шаге
    pause_requested = models.BooleanField(
        default=False,
        help_text=_("Пользователь запросил паузу: тренер остановится на ближайшем шаге")
    )

    # v5.1: выбор GPU — null = авто (первый доступный), -1 = CPU, 0..N = конкретный GPU
    gpu_device = models.IntegerField(
        null=True, blank=True,
        help_text=_("GPU-устройство для обучения (null = авто, -1 = CPU, 0..N = конкретный GPU)")
    )
    progress = models.FloatField(default=0.0)
    current_step = models.CharField(max_length=255, blank=True)
    
    # Логи и ошибки
    logs = models.TextField(blank=True)
    error_message = models.TextField(blank=True)
    error_details = models.JSONField(default=dict, blank=True)
    
    # Метрики обучения
    training_metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Метрики в процессе обучения (loss, grad_norm)")
    )
    
    # Результаты оценки
    evaluation_results = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Результаты оценки на валидации")
    )

    # Запись ошибок
    training_errors = models.JSONField(
        default=list, blank=True,
        help_text=_("Список ошибок, возникших во время обучения (каждая ошибка – словарь с timestamp, message, traceback)")
    )
    
    # Ссылка на задачу Celery
    celery_task_id = models.CharField(max_length=255, blank=True)
    
    # Временные метки
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
        """Обновить прогресс выполнения"""
        self.progress = max(0.0, min(100.0, progress))
        if step:
            self.current_step = step
        self.save(update_fields=['progress', 'current_step'])
    

    # v5.1: пауза/резюме
    def request_pause(self):
        """Запросить паузу: тренер остановится на ближайшем шаге."""
        self.pause_requested = True
        self.save(update_fields=['pause_requested'])
        self.add_log("⏸ Пауза запрошена — тренер остановится после текущего шага")

    def resume(self):
        """Резюме: сброс флага и возврат в очередь (чекпоинт подхватится)."""
        self.pause_requested = False
        self.status = 'PENDING'
        self.save(update_fields=['pause_requested', 'status'])
        self.add_log("▶ Возобновлено — продолжение с последнего чекпоинта")

    def add_log(self, message):
        """Добавить запись в лог"""
        timestamp = timezone.now().strftime('%Y-%m-%d %H:%M:%S')
        self.logs += f"[{timestamp}] {message}\n"
        self.save(update_fields=['logs'])  # убрали 'updated_at'
    
    def fail(self, error_message, error_details=None):
        """Отметить задачу как ошибочную"""
        import traceback
        self.status = 'FAILED'
        self.error_message = error_message
        if error_details:
            self.error_details = error_details
        self.completed_at = timezone.now()
        self.save()
        self.add_log(f"❌ ERROR: {error_message}")
    
    def complete(self, output_model):
        """Завершить задачу успешно"""
        self.status = 'COMPLETED'
        self.output_model = output_model
        self.progress = 100.0
        self.completed_at = timezone.now()
        self.save()
        self.add_log(f"✅ Fine-tuning completed, output model: {output_model.name}")

    def add_error(self, error_message, traceback_str=None):
        """Добавить запись об ошибке в список training_errors."""
        entry = {
            'timestamp': timezone.now().isoformat(),
            'message': error_message,
        }
        if traceback_str:
            entry['traceback'] = traceback_str
        self.training_errors.append(entry)
        self.save(update_fields=['training_errors'])
        # Также запишем в обычный лог
        self.add_log(f"❌ ERROR: {error_message}")


class FineTuningJobCollection(models.Model):
    """Промежуточная модель для связи FineTuningJob и DocumentCollection"""
    job = models.ForeignKey('FineTuningJob', on_delete=models.CASCADE)
    collection = models.ForeignKey('DocumentCollection', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['job', 'collection']


class FineTuningJobDataset(models.Model):
    """Промежуточная модель для связи FineTuningJob и EvaluationDataset"""
    job = models.ForeignKey('FineTuningJob', on_delete=models.CASCADE)
    dataset = models.ForeignKey('EvaluationDataset', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['job', 'dataset']


class FineTuningCheckpoint(models.Model):
    """Промежуточные чекпоинты в процессе обучения"""
    
    job = models.ForeignKey(
        FineTuningJob,
        on_delete=models.CASCADE,
        related_name='checkpoints'
    )
    
    step = models.IntegerField(help_text=_("Номер шага/эпохи"))
    path = models.CharField(max_length=500, help_text=_("Путь к файлу чекпоинта"))
    
    # Метрики на момент сохранения
    loss = models.FloatField(null=True, blank=True)
    grad_norm = models.FloatField(null=True, blank=True)
    learning_rate = models.FloatField(null=True, blank=True)
    
    # Дополнительные метрики (если считали)
    metrics = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['job', 'step']
        unique_together = ['job', 'step']
    
    def __str__(self):
        return f"Checkpoint {self.step} for {self.job.name}"


class FineTuningEvaluation(models.Model):
    """Оценка дообученной модели на тестовых данных"""
    
    job = models.ForeignKey(
        FineTuningJob,
        on_delete=models.CASCADE,
        related_name='evaluations'
    )
    
    # Датасет для оценки
    dataset = models.ForeignKey(
        'EvaluationDataset',
        on_delete=models.CASCADE,
        related_name='fine_tuning_evaluations'
    )
    
    # Метрики
    metrics = models.JSONField(
        default=dict,
        help_text=_("Метрики оценки (rouge, bleu, bertscore)")
    )
    
    # Детальные результаты
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
    """Шаблон конфигурации fine-tuning"""
    
    name = models.CharField(max_length=255, verbose_name=_("Название шаблона"))
    description = models.TextField(blank=True, verbose_name=_("Описание"))
    
    # Параметры метода
    method = models.CharField(
        max_length=20,
        choices=FineTuningMethod.choices,
        default=FineTuningMethod.RAFT,
        verbose_name=_("Метод обучения")
    )
    
    # Рекомендуемые модели (JSON список HF IDs)
    recommended_models = models.JSONField(
        default=list,
        help_text=_("Список рекомендуемых моделей (HF IDs)")
    )
    
    # Параметры LoRA
    lora_r = models.IntegerField(default=16)
    lora_alpha = models.IntegerField(default=32)
    lora_dropout = models.FloatField(default=0.05)
    
    # Гиперпараметры
    learning_rate = models.FloatField(default=5e-5)
    batch_size = models.IntegerField(default=4)
    gradient_accumulation_steps = models.IntegerField(default=4)
    num_epochs = models.IntegerField(default=3)
    max_seq_length = models.IntegerField(default=2048)
    
    # Параметры RAFT
    distractors_per_example = models.IntegerField(default=3)
    oracle_ratio = models.FloatField(default=0.3)
    distractor_ratio = models.FloatField(default=0.5)
    no_context_ratio = models.FloatField(default=0.2)
    use_kl_regularization = models.BooleanField(default=True)
    kl_lambda = models.FloatField(default=0.1)
    general_instructions_ratio = models.FloatField(default=0.2)
    
    # Категория и теги
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
    
    # Статистика использования
    usage_count = models.IntegerField(default=0)
    avg_rating = models.FloatField(null=True, blank=True)
    
    # Кто создал
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
        """Увеличить счетчик использования"""
        self.usage_count += 1
        self.save(update_fields=['usage_count'])


class TrainingMetric(models.Model):
    """
    Пошаговые метрики обучения для детального мониторинга.
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
    Задача конвертации модели в формат GGUF.
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

    # Исходная модель (ModelCheckpoint)
    source_model = models.ForeignKey(
        'ModelCheckpoint',
        on_delete=models.CASCADE,
        related_name='conversion_jobs',
        verbose_name=_("Исходная модель")
    )

    # Выходная модель (после конвертации) – тоже будет ModelCheckpoint
    output_model = models.ForeignKey(
        'ModelCheckpoint',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='converted_from',
        verbose_name=_("Результирующая модель")
    )

    # Параметры конвертации
    quantization = models.CharField(
        max_length=10,
        choices=QUANTIZATION_CHOICES,
        default='q4_0',
        verbose_name=_("Тип квантизации")
    )

    # Дополнительные параметры (можно расширить)
    config = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Дополнительные параметры конвертации")
    )

    # Статус и прогресс
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='PENDING'
    )
    progress = models.FloatField(default=0.0)
    current_step = models.CharField(max_length=255, blank=True)

    # Логи и ошибки
    logs = models.TextField(blank=True)
    error_message = models.TextField(blank=True)
    error_details = models.JSONField(default=dict, blank=True)

    # Ссылка на задачу Celery
    celery_task_id = models.CharField(max_length=255, blank=True)

    # Временные метки
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
