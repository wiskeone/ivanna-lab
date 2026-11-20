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



class EvaluationDataset(models.Model):
    """Датасет вопросов для оценки моделей"""
    name = models.CharField(max_length=255, unique=True, help_text=_("Уникальное имя датасета"))
    description = models.TextField(blank=True, help_text=_("Описание датасета"))
    file_path = models.CharField(max_length=500, blank=True, help_text=_("Путь к файлу с вопросами"))
    questions = models.JSONField(
        default=list,
        help_text=_("Список вопросов в формате [{'id': 1, 'text': '...'}]")
    )
    tags = models.JSONField(
        default=list,
        help_text=_("Теги датасета ['russian', 'general', 'code', 'science']")
    )
    source_type = models.CharField(
        max_length=20,
        choices=[
            ('FILE', 'Файл'),
            ('MANUAL', 'Ручной ввод'),
            ('API', 'API импорт'),
        ],
        default='MANUAL'
    )
    total_questions = models.IntegerField(default=0, help_text=_("Общее количество вопросов"))
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.CharField(max_length=100, blank=True, help_text=_("Кто создал датасет"))
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Датасет для оценки'
        verbose_name_plural = 'Датасеты для оценки'
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['tags']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.total_questions} вопросов)"
    
    def save(self, *args, **kwargs):
        if self.questions:
            self.total_questions = len(self.questions)
        super().save(*args, **kwargs)
    
    def add_questions_from_file(self, file_path, file_format='auto'):
        """Добавить вопросы из файла"""
        import json
        import csv

        questions = []

        # Audit fix: utf-8-sig (BOM silently broke json.load / csv headers),
        # NUL stripped from extracted texts, JSONL accepted (export/import
        # and the edit view all speak JSONL while upload did not)
        enc = 'utf-8-sig'

        def _clean(text):
            return (text or '').replace('\x00', '').strip()

        try:
            if file_format == 'json' or (file_format == 'auto' and file_path.endswith('.json')):
                with open(file_path, 'r', encoding=enc) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        questions = data
                    elif isinstance(data, dict) and 'questions' in data:
                        questions = data['questions']

            elif file_format == 'jsonl' or (file_format == 'auto' and file_path.endswith('.jsonl')):
                with open(file_path, 'r', encoding=enc) as f:
                    for i, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(item, dict):
                            item['text'] = _clean(
                                item.get('instruction') or item.get('text')
                                or item.get('question') or '')
                            item['answer'] = _clean(
                                item.get('output') or item.get('answer')
                                or item.get('response') or '')
                            questions.append(item)

            elif file_format == 'csv' or (file_format == 'auto' and file_path.endswith('.csv')):
                with open(file_path, 'r', encoding=enc) as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader, 1):
                        if 'text' in row:
                            questions.append({
                                'id': i,
                                'text': _clean(row['text']),
                                'category': row.get('category', ''),
                                'difficulty': row.get('difficulty', 'medium')
                            })

            elif file_format == 'txt' or (file_format == 'auto' and file_path.endswith('.txt')):
                with open(file_path, 'r', encoding=enc) as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines, 1):
                        line = _clean(line)
                        if line and not line.startswith('#'):
                            questions.append({
                                'id': i,
                                'text': line
                            })

            if questions:
                self.questions = questions
                self.total_questions = len(questions)
                self.file_path = file_path
                self.source_type = 'FILE'
                self.save()
                return True
            
        except Exception as e:
            logger.error(f"Error loading questions from file: {str(e)}")
        
        return False


class ModelEvaluationTask(models.Model):
    """Задача оценки модели на датасете"""
    STATUS_CHOICES = [
        ('PENDING', 'В ожидании'),
        ('RUNNING', 'Выполняется'),
        ('COMPLETED', 'Завершено'),
        ('FAILED', 'Ошибка'),
        ('CANCELLED', 'Отменено'),
    ]
    
    name = models.CharField(max_length=255, help_text=_("Название задачи оценки"))
    description = models.TextField(blank=True, help_text=_("Описание задачи"))
    
    # Модели для оценки
    evaluation_models = models.ManyToManyField(  # Изменили имя с models на evaluation_models
        ModelCheckpoint,
        related_name='evaluation_tasks',
        help_text=_("Модели для оценки")
    )
    
    # Датасет для оценки
    dataset = models.ForeignKey(
        EvaluationDataset,
        on_delete=models.CASCADE,
        related_name='evaluation_tasks'
    )
    
    # Конфигурация генерации
    generation_config = models.JSONField(
        default=dict,
        help_text=_("Параметры генерации для всех моделей")
    )
    
    # Статус и прогресс
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    progress = models.FloatField(default=0.0, help_text=_("Прогресс выполнения (0-100)"))
    current_step = models.CharField(max_length=100, blank=True, help_text=_("Текущий шаг"))
    
    # Результаты
    results = models.JSONField(
        default=dict,
        help_text=_("Результаты оценки {'model_id': {'question_id': {'response': '...', 'metrics': {...}}}}")
    )
    
    # Оценки
    ratings = models.JSONField(
        default=dict,
        help_text=_("Ручные оценки {'model_id': {'question_id': rating}}")
    )
    
    # Сводные метрики
    summary_metrics = models.JSONField(
        default=dict,
        help_text=_("Сводные метрики по моделям {'model_id': {'total_score': X, 'avg_score': Y}}")
    )

    # задачи Celery для отслеживания
    celery_task_id = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text=_("ID задачи Celery для отслеживания")
    )
    # чекпоинты
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Метаданные задачи, включая чекпоинты")
    )
    
    # Логи и ошибки
    logs = models.TextField(blank=True)
    error_message = models.TextField(blank=True)
    
    # Временные метки
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_by = models.CharField(max_length=100, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Задача оценки моделей'
        verbose_name_plural = 'Задачи оценки моделей'
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['dataset', 'status']),
        ]
    
    def __str__(self):
        return f"Evaluation: {self.name} ({self.get_status_display()})"
    
    @property
    def total_models(self):
        return self.evaluation_models.count()  # Исправлено с models на evaluation_models
    
    @property
    def total_questions(self):
        return self.dataset.total_questions if self.dataset else 0
    
    @property
    def estimated_time_minutes(self):
        """Примерное время выполнения"""
        return self.total_questions * self.total_models * 2  # ~2 секунды на ответ
    
    def start_evaluation(self):
        """Начать выполнение задачи"""
        self.status = 'RUNNING'
        self.started_at = timezone.now()
        self.current_step = 'Инициализация'
        self.save()
    
    def complete_evaluation(self):
        """Завершить задачу успешно"""
        self.status = 'COMPLETED'
        self.completed_at = timezone.now()
        self.progress = 100.0
        self.current_step = 'Завершено'
        self.save()
    
    def fail_evaluation(self, error_message):
        """Завершить задачу с ошибкой"""
        self.status = 'FAILED'
        self.completed_at = timezone.now()
        self.error_message = error_message
        self.save()
    
    def update_progress(self, progress, step=""):
        """Обновить прогресс выполнения"""
        self.progress = max(0.0, min(100.0, progress))
        if step:
            self.current_step = step
        self.save(update_fields=['progress', 'current_step'])
    
    def add_model_result(self, model_id, question_id, response, metadata=None):
        """Добавить результат модели для вопроса"""
        if str(model_id) not in self.results:
            self.results[str(model_id)] = {}
        
        result = {
            'response': response,
            'timestamp': timezone.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.results[str(model_id)][str(question_id)] = result
        self.save()
    
    def add_rating(self, model_id, question_id, rating, user_id=None):
        """Добавить оценку для ответа модели"""
        if str(model_id) not in self.ratings:
            self.ratings[str(model_id)] = {}
        
        rating_data = {
            'rating': rating,
            'rated_at': timezone.now().isoformat(),
            'rated_by': user_id or 'anonymous'
        }
        
        self.ratings[str(model_id)][str(question_id)] = rating_data
        self.save()
        
        # Пересчитать сводные метрики
        self.calculate_summary_metrics()
    
    def calculate_summary_metrics(self):
        """Вычислить сводные метрики"""
        summary = {}
        
        for model_id, ratings_dict in self.ratings.items():
            if ratings_dict:
                ratings = []
                for rating_data in ratings_dict.values():
                    if isinstance(rating_data, dict) and 'rating' in rating_data:
                        ratings.append(rating_data['rating'])
                
                if ratings:
                    total_score = sum(ratings)
                    avg_score = total_score / len(ratings)
                    
                    summary[str(model_id)] = {
                        'total_score': total_score,
                        'avg_score': round(avg_score, 2),
                        'max_score': max(ratings),
                        'min_score': min(ratings),
                        'rated_questions': len(ratings),
                        'total_questions': self.total_questions,
                        'completion_percentage': round(len(ratings) / self.total_questions * 100, 1) if self.total_questions > 0 else 0
                    }
        
        self.summary_metrics = summary
        self.save()
        return summary


class EvaluationResult(models.Model):
    """Детализированные результаты оценки для конкретной модели и вопроса"""
    task = models.ForeignKey(ModelEvaluationTask, on_delete=models.CASCADE, related_name='detailed_results')
    model = models.ForeignKey(ModelCheckpoint, on_delete=models.CASCADE, related_name='evaluation_results')
    
    question_id = models.IntegerField(help_text=_("ID вопроса из датасета"))
    question_text = models.TextField(help_text=_("Текст вопроса"))
    
    # Ответы и метрики
    model_response = models.TextField(help_text=_("Ответ модели"))
    response_time = models.FloatField(default=0.0, help_text=_("Время генерации в секундах"))
    tokens_used = models.IntegerField(default=0, help_text=_("Количество использованных токенов"))
    
    # Автоматические метрики
    automatic_metrics = models.JSONField(
        default=dict,
        help_text=_("Автоматически вычисленные метрики")
    )
    
    # Ручные оценки
    manual_rating = models.IntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text=_("Ручная оценка (1-5)")
    )
    rating_comment = models.TextField(blank=True, help_text=_("Комментарий к оценке"))
    rated_by = models.CharField(max_length=100, blank=True, help_text=_("Кто оценил"))
    rated_at = models.DateTimeField(null=True, blank=True)
    
    # ДОБАВЛЯЕМ ПОЛЕ METADATA
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Дополнительные метаданные (RAG флаги, источники и т.д.)")
    )
    
    # Статистика
    retry_count = models.IntegerField(default=0, help_text=_("Количество попыток генерации"))
    error_log = models.TextField(blank=True, help_text=_("Лог ошибок при генерации"))
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['task', 'model', 'question_id']
        unique_together = ['task', 'model', 'question_id']
        verbose_name = 'Результат оценки'
        verbose_name_plural = 'Результаты оценок'
        indexes = [
            models.Index(fields=['task', 'model']),
            models.Index(fields=['model', 'manual_rating']),
            models.Index(fields=['question_id']),
        ]
    
    def __str__(self):
        return f"Result: {self.model.display_name} - Question {self.question_id}"
