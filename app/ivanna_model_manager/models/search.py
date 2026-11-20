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

from .rag import Document, DocumentChunk, DocumentCollection, EmbeddingModel



# v5.2: reranker registry (cross-encoders) — modeled after EmbeddingModel
class RerankerModel(models.Model):
    """Reranking model (cross-encoder) for hybrid RAG search."""
    name = models.CharField(max_length=255, unique=True, help_text=_("Название модели"))
    display_name = models.CharField(max_length=255, blank=True)
    model_id = models.CharField(max_length=500, help_text=_("Идентификатор на HF (например, DiTy/cross-encoder-russian-msmarco)"))
    max_length = models.IntegerField(default=512, help_text=_("Максимальная длина пары запрос/документ"))
    is_active = models.BooleanField(default=True)
    is_default = models.BooleanField(default=False)
    is_downloaded = models.BooleanField(default=False)
    download_progress = models.IntegerField(default=0)
    download_status = models.CharField(max_length=20, default='not_started', choices=[
        ('not_started', _('Не начата')), ('downloading', _('Загружается')),
        ('completed', _('Завершена')), ('failed', _('Ошибка')),
    ])
    download_error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-is_default', '-is_active', 'name']
        verbose_name = 'Модель реранкинга'
        verbose_name_plural = 'Модели реранкинга'

    def __str__(self):
        return self.display_name or self.name

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if self.is_default:
            RerankerModel.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)


class SearchQuery(models.Model):
    """
    User search query.
    """
    SEARCH_TYPE_CHOICES = [
        ('SEMANTIC', 'Семантический (по эмбеддингам)'),
        ('KEYWORD', 'По ключевым словам'),
        ('HYBRID', 'Гибридный'),
        ('FILTER', 'Фильтрация по метаданным'),
    ]
    
    # Basic info
    query_text = models.TextField(help_text=_("Текст поискового запроса"))
    normalized_query = models.TextField(blank=True, help_text=_("Нормализованный текст запроса"))

    # Search type and config
    search_type = models.CharField(max_length=20, choices=SEARCH_TYPE_CHOICES, default='HYBRID')
    search_config = models.JSONField(default=dict, blank=True, help_text=_("Конфигурация поиска"))

    # Query context
    session_id = models.CharField(max_length=100, blank=True, help_text=_("ID сессии пользователя"))
    user_id = models.CharField(max_length=100, blank=True, help_text=_("ID пользователя"))
    context = models.JSONField(default=dict, blank=True, help_text=_("Дополнительный контекст запроса"))

    # Filters
    collection_filter = models.ForeignKey(
        DocumentCollection,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text=_("Фильтр по коллекции")
    )
    metadata_filters = models.JSONField(default=dict, blank=True, help_text=_("Фильтры по метаданным"))
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    processing_time_ms = models.FloatField(null=True, blank=True, help_text=_("Время обработки запроса в мс"))
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['session_id', 'created_at']),
            models.Index(fields=['user_id', 'created_at']),
            models.Index(fields=['search_type', 'created_at']),
        ]
        verbose_name = 'Поисковый запрос'
        verbose_name_plural = 'Поисковые запросы'
    
    def __str__(self):
        return f"'{self.query_text[:50]}...' ({self.get_search_type_display()})"


class SearchResult(models.Model):
    """
    Search result for a query.
    Links the query, matched chunks, and their relevance.
    """
    # Relations
    query = models.ForeignKey(
        SearchQuery,
        on_delete=models.CASCADE,
        related_name='results'
    )
    chunk = models.ForeignKey(
        DocumentChunk,
        on_delete=models.CASCADE,
        related_name='search_results'
    )
    
    # Relevance and ranking
    relevance_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Оценка релевантности (0-1)")
    )
    rank_position = models.IntegerField(help_text=_("Позиция в результатах поиска"))
    
    # Relevance scoring method
    scoring_method = models.CharField(
        max_length=50,
        default='cosine_similarity',
        choices=[
            ('cosine_similarity', _('Косинусное сходство')),
            ('dot_product', _('Скалярное произведение')),
            ('bm25', 'BM25'),
            ('hybrid', _('Гибридный')),
            ('neural', _('Нейросетевой')),
        ],
        help_text=_("Метод вычисления релевантности")
    )
    
    # Additional metrics
    semantic_score = models.FloatField(null=True, blank=True, help_text=_("Семантическая оценка"))
    keyword_score = models.FloatField(null=True, blank=True, help_text=_("Оценка по ключевым словам"))
    
    # Result context
    context_before = models.TextField(blank=True, help_text=_("Контекст перед чанком"))
    context_after = models.TextField(blank=True, help_text=_("Контекст после чанка"))
    
    # Metadata
    metadata = models.JSONField(default=dict, blank=True, help_text=_("Метаданные результата"))
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['query', 'rank_position']
        indexes = [
            models.Index(fields=['query', 'relevance_score']),
            models.Index(fields=['chunk', 'query']),
            models.Index(fields=['scoring_method']),
        ]
        unique_together = ['query', 'chunk']
        verbose_name = 'Результат поиска'
        verbose_name_plural = 'Результаты поиска'
    
    def __str__(self):
        return f"Результат {self.rank_position} для запроса #{self.query.id}"
    
    @property
    def document_title(self):
        """Document title of the chunk."""
        return self.chunk.document.title

    @property
    def chunk_preview(self):
        """Chunk content preview."""
        return self.chunk.preview


class RAGSession(models.Model):
    """
    RAG session: combines the query, search, answer generation, and feedback.
    """
    SESSION_STATUS_CHOICES = [
        ('SEARCHING', 'Поиск документов'),
        ('GENERATING', 'Генерация ответа'),
        ('COMPLETED', 'Завершена'),
        ('FAILED', 'Ошибка'),
        ('EVALUATING', 'Оценка качества'),
    ]
    
    # Identifiers
    session_id = models.CharField(max_length=100, unique=True, help_text=_("Уникальный ID сессии"))
    conversation_id = models.CharField(max_length=100, blank=True, help_text=_("ID беседы (если часть диалога)"))

    # User info
    user_id = models.CharField(max_length=100, blank=True)
    user_ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)

    # Context and prompts
    original_query = models.TextField(help_text=_("Исходный запрос пользователя"))
    system_prompt = models.TextField(blank=True, help_text=_("Системный промпт для LLM"))
    augmented_prompt = models.TextField(blank=True, help_text=_("Промпт с контекстом из поиска"))
    
    # Models
    embedding_model = models.ForeignKey(
        EmbeddingModel,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='rag_sessions'
    )
    llm_model = models.ForeignKey(
        'ModelCheckpoint',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='rag_sessions'
    )
    
    # Search results
    search_query = models.ForeignKey(
        SearchQuery,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='rag_sessions'
    )
    retrieved_chunks = models.ManyToManyField(
        DocumentChunk,
        through='RetrievedChunk',
        related_name='rag_sessions'
    )
    
    # Answer generation
    generated_answer = models.TextField(blank=True, help_text=_("Сгенерированный ответ"))
    generation_metadata = models.JSONField(default=dict, blank=True, help_text=_("Метаданные генерации"))

    # Status and performance
    status = models.CharField(max_length=20, choices=SESSION_STATUS_CHOICES, default='SEARCHING')
    total_time_ms = models.FloatField(default=0.0, help_text=_("Общее время выполнения (мс)"))
    search_time_ms = models.FloatField(null=True, blank=True, help_text=_("Время поиска (мс)"))
    generation_time_ms = models.FloatField(null=True, blank=True, help_text=_("Время генерации (мс)"))
    
    # Tokens and cost
    input_tokens = models.IntegerField(default=0, help_text=_("Входные токены"))
    output_tokens = models.IntegerField(default=0, help_text=_("Выходные токены"))
    estimated_cost = models.FloatField(null=True, blank=True, help_text=_("Примерная стоимость"))
    
    # Answer quality
    confidence_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Уверенность модели в ответе")
    )
    
    # Feedback
    has_feedback = models.BooleanField(default=False, help_text=_("Есть ли обратная связь"))
    feedback_score = models.IntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text=_("Оценка пользователя (1-5)")
    )
    feedback_comment = models.TextField(blank=True, help_text=_("Комментарий пользователя"))
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    feedback_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['session_id']),
            models.Index(fields=['user_id', 'created_at']),
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['has_feedback', 'feedback_score']),
        ]
        verbose_name = 'RAG сессия'
        verbose_name_plural = 'RAG сессии'
    
    def __str__(self):
        return f"RAG сессия {self.session_id}"
    
    @property
    def is_successful(self):
        """Whether the session succeeded."""
        return self.status == 'COMPLETED' and self.generated_answer

    @property
    def total_tokens(self):
        """Total token count."""
        return self.input_tokens + self.output_tokens

    def mark_search_completed(self, search_time_ms, search_query):
        """Mark search as completed."""
        self.status = 'GENERATING'
        self.search_time_ms = search_time_ms
        self.search_query = search_query
        self.save()
    
    def mark_generation_completed(self, answer, metadata, generation_time_ms, tokens):
        """Mark generation as completed."""
        self.status = 'COMPLETED'
        self.generated_answer = answer
        self.generation_metadata = metadata
        self.generation_time_ms = generation_time_ms
        self.input_tokens = tokens.get('input', 0)
        self.output_tokens = tokens.get('output', 0)
        self.completed_at = timezone.now()
        self.total_time_ms = (self.search_time_ms or 0) + generation_time_ms
        self.save()
    
    def add_feedback(self, score, comment=""):
        """Add user feedback."""
        self.has_feedback = True
        self.feedback_score = score
        self.feedback_comment = comment
        self.feedback_at = timezone.now()
        self.save()


class RetrievedChunk(models.Model):
    """
    Through model linking RAGSession and DocumentChunk.
    Stores extra info on how the chunk was used in the session.
    """
    session = models.ForeignKey(RAGSession, on_delete=models.CASCADE)
    chunk = models.ForeignKey(DocumentChunk, on_delete=models.CASCADE)

    # Position in results
    retrieval_rank = models.IntegerField(help_text=_("Позиция при извлечении"))
    relevance_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Релевантность для этого запроса")
    )
    
    # How the chunk was used
    used_in_context = models.BooleanField(default=False, help_text=_("Был ли использован в контексте для LLM"))
    context_position = models.IntegerField(null=True, blank=True, help_text=_("Позиция в контексте"))

    # Usage quality
    attribution_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Насколько ответ основан на этом чанке")
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['session', 'retrieval_rank']
        unique_together = ['session', 'chunk']
        verbose_name = 'Извлеченный фрагмент'
        verbose_name_plural = 'Извлеченные фрагменты'
    
    def __str__(self):
        return f"Чанк #{self.chunk.id} в сессии {self.session.session_id}"


class SearchQualityFeedback(models.Model):
    """
    User feedback on search quality.
    Used to improve ranking and relevance.
    """
    FEEDBACK_TYPE_CHOICES = [
        ('RELEVANCE', 'Релевантность'),
        ('COMPLETENESS', 'Полнота'),
        ('FRESHNESS', 'Актуальность'),
        ('DIVERSITY', 'Разнообразие'),
        ('OTHER', 'Другое'),
    ]
    
    # Relations
    search_result = models.ForeignKey(
        SearchResult,
        on_delete=models.CASCADE,
        related_name='feedback',
        null=True,
        blank=True
    )
    rag_session = models.ForeignKey(
        RAGSession,
        on_delete=models.CASCADE,
        related_name='quality_feedback',
        null=True,
        blank=True
    )
    
    # Rating
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_TYPE_CHOICES, default='RELEVANCE')
    rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text=_("Оценка качества (1-5)")
    )
    
    # Feedback details
    comment = models.TextField(blank=True, help_text=_("Комментарий пользователя"))
    expected_document = models.ForeignKey(
        Document,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text=_("Какой документ ожидался")
    )
    
    # Context
    user_id = models.CharField(max_length=100, blank=True)
    user_expertise = models.CharField(
        max_length=50,
        blank=True,
        choices=[
            ('NOVICE', 'Новичок'),
            ('INTERMEDIATE', 'Средний'),
            ('EXPERT', 'Эксперт'),
        ]
    )
    
    # System info
    is_resolved = models.BooleanField(default=False, help_text=_("Обработана ли обратная связь"))
    resolution_notes = models.TextField(blank=True, help_text=_("Заметки по обработке"))
    
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['feedback_type', 'rating']),
            models.Index(fields=['user_id', 'created_at']),
            models.Index(fields=['is_resolved', 'created_at']),
        ]
        verbose_name = 'Оценка качества поиска'
        verbose_name_plural = 'Оценки качества поиска'
    
    def __str__(self):
        return f"Оценка {self.rating}/5 от {self.user_id or 'анонима'}"
    
    def mark_resolved(self, notes=""):
        """Mark the feedback as resolved."""
        self.is_resolved = True
        self.resolution_notes = notes
        self.resolved_at = timezone.now()
        self.save()


class RAGEvaluation(models.Model):
    """
    Systematic evaluation of RAG system quality.
    """
    EVALUATION_TYPE_CHOICES = [
        ('AUTOMATED', 'Автоматическая'),
        ('MANUAL', 'Ручная'),
        ('AB_TEST', 'A/B тест'),
        ('USER_STUDY', 'Исследование с пользователями'),
    ]
    
    name = models.CharField(max_length=255, help_text=_("Название оценки"))
    description = models.TextField(blank=True, help_text=_("Описание оценки"))
    
    # Configuration
    evaluation_type = models.CharField(max_length=20, choices=EVALUATION_TYPE_CHOICES, default='AUTOMATED')
    test_queries = models.JSONField(
        default=list,
        help_text=_("Тестовые запросы [{'id': 1, 'text': '...', 'expected_docs': [doc_ids]}]")
    )
    
    # Metrics
    target_metrics = models.JSONField(
        default=dict,
        help_text=_("Целевые метрики {'recall@5': 0.8, 'precision@3': 0.9}")
    )
    
    # Results
    actual_metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Фактические метрики")
    )
    detailed_results = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Детальные результаты по каждому запросу")
    )
    
    # Comparison
    baseline_evaluation = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text=_("Базовая оценка для сравнения")
    )
    improvement_percentage = models.FloatField(null=True, blank=True, help_text=_("Улучшение в %"))
    
    # Status
    status = models.CharField(
        max_length=20,
        default='PENDING',
        choices=[
            ('PENDING', 'Ожидает'),
            ('RUNNING', 'Выполняется'),
            ('COMPLETED', 'Завершена'),
            ('FAILED', 'Ошибка'),
        ]
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Оценка RAG системы'
        verbose_name_plural = 'Оценки RAG системы'
    
    def __str__(self):
        return f"Оценка: {self.name}"
    
    @property
    def total_queries(self):
        """Number of test queries."""
        return len(self.test_queries) if self.test_queries else 0

    def start_evaluation(self):
        """Start the evaluation."""
        self.status = 'RUNNING'
        self.started_at = timezone.now()
        self.save()
    
    def complete_evaluation(self, metrics, detailed_results):
        """Complete the evaluation."""
        self.status = 'COMPLETED'
        self.completed_at = timezone.now()
        self.actual_metrics = metrics
        self.detailed_results = detailed_results

        # Compute improvement if a baseline exists
        if self.baseline_evaluation and self.baseline_evaluation.actual_metrics:
            self._calculate_improvement()

        self.save()

    def _calculate_improvement(self):
        """Compute improvement relative to the baseline evaluation."""
        try:
            baseline_metrics = self.baseline_evaluation.actual_metrics
            
            improvements = {}
            for metric, value in self.actual_metrics.items():
                if metric in baseline_metrics and baseline_metrics[metric] != 0:
                    baseline = baseline_metrics[metric]
                    improvement = ((value - baseline) / baseline) * 100
                    improvements[metric] = round(improvement, 2)
            
            if improvements:
                # Average improvement across all metrics
                self.improvement_percentage = sum(improvements.values()) / len(improvements)
                self.actual_metrics['improvements'] = improvements
        except Exception as e:
            logger.error(f"Error calculating improvement: {e}")
