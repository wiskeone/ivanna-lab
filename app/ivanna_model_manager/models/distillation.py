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



class DistillationQuestionSet(models.Model):
    """Набор групп чанков для генерации вопросов."""
    STATUS_CHOICES = [
        ('PENDING', 'В ожидании'),
        ('RUNNING', 'Выполняется'),
        ('COMPLETED', 'Завершено'),
        ('FAILED', 'Ошибка'),
        ('CANCELLED', 'Отменено'),
    ]

    name = models.CharField(max_length=255, verbose_name=_("Название"))
    description = models.TextField(blank=True, verbose_name=_("Описание"))

    collection = models.ForeignKey(
        'DocumentCollection',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='distillation_sets'
    )

    default_generation_config = models.JSONField(default=dict, blank=True)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    progress = models.FloatField(default=0.0)
    current_step = models.CharField(max_length=255, blank=True)
    logs = models.TextField(blank=True)

    created_by = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    celery_task_id = models.CharField(max_length=255, blank=True, null=True)
    checkpoint_data = models.JSONField(default=dict, blank=True, help_text=_("Данные чекпоинта для возобновления"))
    last_processed = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Набор для дистилляции"
        verbose_name_plural = "Наборы для дистилляции"

    def __str__(self):
        return self.name


class DistillationChunkGroup(models.Model):
    """Группа чанков, для которой будут генерироваться вопросы."""
    question_set = models.ForeignKey(
        DistillationQuestionSet,
        on_delete=models.CASCADE,
        related_name='groups'
    )

    name = models.CharField(max_length=255, blank=True, help_text=_("Название группы (опционально)"))
    chunk_ids = models.JSONField(default=list, help_text=_("Список ID чанков для этой группы"))
    num_questions = models.PositiveIntegerField(default=3, help_text=_("Сколько вопросов сгенерировать"))
    generation_config = models.JSONField(default=dict, blank=True)

    order = models.PositiveIntegerField(default=0, help_text=_("Порядок обработки"))
    logs = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['order', 'id']
        verbose_name = "Группа чанков"
        verbose_name_plural = "Группы чанков"

    def __str__(self):
        return self.name or f"Группа {self.id} (набор {self.question_set.name})"


class DistillationQuestion(models.Model):
    """Сгенерированный вопрос по группе чанков."""
    STATUS_CHOICES = [
        ('pending', _('Ожидает ответа')),
        ('answered', _('Ответ получен')),
        ('reviewed', _('Проверено')),
    ]

    group = models.ForeignKey(
        DistillationChunkGroup,
        on_delete=models.CASCADE,
        related_name='questions'
    )

    question_text = models.TextField()
    answer_text = models.TextField(blank=True)   # для обратной совместимости (можно убрать)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    metadata = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['group', 'id']
        verbose_name = "Вопрос дистилляции"
        verbose_name_plural = "Вопросы дистилляции"

    def __str__(self):
        return self.question_text[:50]


class DistillationAnswer(models.Model):
    """Ответ на вопрос, сгенерированный конкретной задачей."""
    question = models.ForeignKey(
        DistillationQuestion,
        on_delete=models.CASCADE,
        related_name='answers'
    )
    answer_job = models.ForeignKey(
        'DistillationAnswerJob',
        on_delete=models.CASCADE,
        related_name='answers'
    )
    answer_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Ответ дистилляции"
        verbose_name_plural = "Ответы дистилляции"

    def __str__(self):
        return f"Ответ к вопросу {self.question.id} (задача {self.answer_job.id})"


class DistillationAnswerJob(models.Model):
    """Задача на генерацию ответов для вопросов."""
    STATUS_CHOICES = [
        ('PENDING', 'В ожидании'),
        ('RUNNING', 'Выполняется'),
        ('COMPLETED', 'Завершено'),
        ('FAILED', 'Ошибка'),
        ('CANCELLED', 'Отменено'),
    ]

    name = models.CharField(max_length=255, verbose_name=_("Название"))
    description = models.TextField(blank=True, verbose_name=_("Описание"))

    question_set = models.ForeignKey(
        DistillationQuestionSet,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='answer_jobs'
    )
    question_ids = models.JSONField(default=list, blank=True, help_text=_("Список ID вопросов для обработки"))

    generation_config = models.JSONField(default=dict, blank=True)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    progress = models.FloatField(default=0.0)
    current_step = models.CharField(max_length=255, blank=True)

    processed_count = models.IntegerField(default=0)
    planned_count = models.PositiveIntegerField(default=0, help_text=_("Запланированное количество вопросов"))

    logs = models.TextField(blank=True)

    created_by = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    celery_task_id = models.CharField(max_length=255, blank=True, null=True)
    checkpoint_data = models.JSONField(default=dict, blank=True, help_text=_("Данные чекпоинта для возобновления"))
    last_processed = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Задача генерации ответов"
        verbose_name_plural = "Задачи генерации ответов"

    def __str__(self):
        return self.name


class DistillationDataset(models.Model):
    """Итоговый датасет для дообучения, собранный из вопросов и ответов."""
    name = models.CharField(max_length=255, verbose_name=_("Название"))
    description = models.TextField(blank=True, verbose_name=_("Описание"))

    # Исходные данные (для трассировки)
    source_question_set = models.ForeignKey(
        DistillationQuestionSet,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='exported_datasets'
    )
    source_answer_jobs = models.ManyToManyField(
        DistillationAnswerJob,
        blank=True,
        related_name='exported_datasets'
    )

    # Данные датасета: список объектов
    items = models.JSONField(
        default=list,
        help_text=_("Список элементов: [{'question_id': 1, 'question_text': '...', 'answer_text': '...', 'context_chunks': [1,2,3], 'selected_model_id': 5}]")
    )

    created_by = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Итоговый датасет дистилляции"
        verbose_name_plural = "Итоговые датасеты дистилляции"

    def __str__(self):
        return self.name


class ManualInstructionSet(models.Model):
    """Набор ручных инструкций (вопрос-ответ) для дообучения."""
    name = models.CharField(max_length=255, verbose_name=_("Название"))
    description = models.TextField(blank=True, verbose_name=_("Описание"))
    created_by = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Ручной набор инструкций"
        verbose_name_plural = "Ручные наборы инструкций"

    def __str__(self):
        return self.name


class ManualInstruction(models.Model):
    """Отдельная инструкция (вопрос-ответ) с контекстными чанками."""
    instruction_set = models.ForeignKey(
        ManualInstructionSet,
        on_delete=models.CASCADE,
        related_name='instructions'
    )
    question = models.TextField(verbose_name=_("Вопрос"))
    answer = models.TextField(verbose_name=_("Ответ"))
    context_chunks = models.JSONField(
        default=list,
        blank=True,
        help_text=_("Список ID чанков, используемых в качестве контекста")
    )
    order = models.PositiveIntegerField(default=0, help_text=_("Порядок отображения"))
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['order', 'id']
        verbose_name = "Ручная инструкция"
        verbose_name_plural = "Ручные инструкции"

    def __str__(self):
        return self.question[:50]


class HuggingFaceDataset(models.Model):
    """Модель для отслеживания загрузки датасетов с Hugging Face Hub."""
    STATUS_CHOICES = [
        ('PENDING', 'В очереди'),
        ('DOWNLOADING', 'Загрузка'),
        ('PROCESSING', 'Обработка'),
        ('COMPLETED', 'Завершено'),
        ('FAILED', 'Ошибка'),
    ]

    repo_id = models.CharField(max_length=255, verbose_name=_("ID репозитория на HF"), help_text=_("Например: 'tatsu-lab/alpaca'"))
    config = models.CharField(max_length=100, blank=True, help_text=_("Конфигурация датасета (если есть)"))
    split = models.CharField(max_length=50, default='train', help_text=_("Разделы (train, validation, test)"))
    name = models.CharField(max_length=255, blank=True, help_text=_("Название датасета в системе (если не указано, берётся из repo_id)"))
    description = models.TextField(blank=True)
    file_path = models.CharField(max_length=500, blank=True, help_text=_("Путь к сохранённому JSONL файлу"))
    total_entries = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    error_message = models.TextField(blank=True)
    celery_task_id = models.CharField(max_length=255, blank=True, null=True)
    created_by = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Датасет Hugging Face"
        verbose_name_plural = "Датасеты Hugging Face"

    def __str__(self):
        return self.repo_id
