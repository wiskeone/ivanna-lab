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

from .rag import Document, DocumentCollection
from .search import SearchQuery



class ProcessingPipeline(models.Model):
    """
    Document processing pipeline configuration.
    """
    name = models.CharField(max_length=255, unique=True, help_text=_("Название пайплайна"))
    description = models.TextField(blank=True, help_text=_("Описание пайплайна"))
    
    # Processing steps
    steps = models.JSONField(
        default=list,
        help_text=_("Список этапов обработки [{'name': 'extract', 'module': '...', 'config': {...}}]")
    )
    
    # Configuration
    config = models.JSONField(default=dict, blank=True, help_text=_("Общая конфигурация пайплайна"))

    # Performance
    avg_processing_time_ms = models.FloatField(null=True, blank=True, help_text=_("Среднее время обработки документа"))
    success_rate = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Процент успешной обработки")
    )
    
    # Status
    is_active = models.BooleanField(default=True)
    is_default = models.BooleanField(default=False, help_text=_("Пайплайн по умолчанию"))
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-is_default', '-is_active', 'name']
        verbose_name = 'Пайплайн обработки'
        verbose_name_plural = 'Пайплайны обработки'
    
    def __str__(self):
        return f"Пайплайн: {self.name}"
    
    def clean(self):
        """Validate the pipeline."""
        if self.is_default and self.pk:
            ProcessingPipeline.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)


class DocumentProcessor(models.Model):
    """
    Document processing log for monitoring and debugging.
    """
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='processing_logs'
    )
    pipeline = models.ForeignKey(
        ProcessingPipeline,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='processing_logs'
    )
    
    # Status and progress
    status = models.CharField(
        max_length=20,
        choices=[
            ('STARTED', 'Начата'),
            ('EXTRACTING', 'Извлечение'),
            ('CHUNKING', 'Разделение'),
            ('EMBEDDING', 'Векторизация'),
            ('INDEXING', 'Индексация'),
            ('COMPLETED', 'Завершена'),
            ('FAILED', 'Ошибка'),
        ],
        default='STARTED'
    )
    progress = models.FloatField(default=0.0, help_text=_("Прогресс обработки (0-100)"))
    
    # Resources and performance
    worker_id = models.CharField(max_length=100, blank=True, help_text=_("ID воркера"))
    memory_usage_mb = models.FloatField(null=True, blank=True, help_text=_("Использование памяти (МБ)"))
    processing_time_ms = models.FloatField(null=True, blank=True, help_text=_("Время обработки (мс)"))
    
    # Results
    chunks_created = models.IntegerField(default=0, help_text=_("Создано чанков"))
    embeddings_created = models.IntegerField(default=0, help_text=_("Создано эмбеддингов"))
    
    # Errors
    error_message = models.TextField(blank=True)
    error_details = models.JSONField(default=dict, blank=True, help_text=_("Детали ошибки"))

    # Timestamps
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-started_at']
        indexes = [
            models.Index(fields=['document', 'status']),
            models.Index(fields=['status', 'started_at']),
        ]
        verbose_name = 'Лог обработки документа'
        verbose_name_plural = 'Логи обработки документов'
    
    def __str__(self):
        return f"Обработка {self.document.title} - {self.get_status_display()}"


class DeletedDocumentLog(models.Model):
    """Log of deleted documents."""
    original_name = models.CharField(max_length=500)
    title = models.CharField(max_length=500, blank=True)
    document_type = models.CharField(max_length=20, choices=Document.DOCUMENT_TYPE_CHOICES)
    file_size_mb = models.FloatField(default=0.0)
    collection_name = models.CharField(max_length=255, blank=True)
    total_chunks = models.IntegerField(default=0)
    deleted_at = models.DateTimeField()
    
    class Meta:
        ordering = ['-deleted_at']
        verbose_name = 'Лог удаленного документа'
        verbose_name_plural = 'Логи удаленных документов'
    
    def __str__(self):
        return f"{self.title} (удален {self.deleted_at.strftime('%d.%m.%Y %H:%M')})"


def fix_document_file_paths():
    """Fix file paths for existing documents."""
    for document in Document.objects.filter(is_active=True):
        full_path = document.full_file_path

        # Check whether the file exists
        if not os.path.exists(full_path):
            print(f"Файл не найден: {full_path}")

            # Try to find the file by name in the documents folder
            possible_locations = [
                os.path.join(settings.MEDIA_ROOT, 'documents', document.original_filename),
                os.path.join(settings.MEDIA_ROOT, 'documents', str(document.id), document.original_filename),
                os.path.join(settings.MEDIA_ROOT, document.original_filename),
            ]
            
            for location in possible_locations:
                if os.path.exists(location):
                    # Update file_path
                    relative_path = os.path.relpath(location, settings.MEDIA_ROOT)
                    document.file_path = relative_path
                    document.save()
                    print(f"Исправлен путь для документа {document.id}: {relative_path}")
                    break


class SavedSearch(models.Model):
    """Saved search queries."""
    name = models.CharField(max_length=255, verbose_name=_("Название"))
    description = models.TextField(blank=True, verbose_name=_("Описание"))
    query_text = models.TextField(verbose_name=_("Текст запроса"))
    search_type = models.CharField(
        max_length=20,
        choices=[
            ('SEMANTIC', 'Семантический'),
            ('KEYWORD', 'Ключевой'),
            ('HYBRID', 'Гибридный'),
            ('HYBRID_ENHANCED', 'Улучшенный гибридный'),
        ],
        default='HYBRID',
        verbose_name=_("Тип поиска")
    )
    collection = models.ForeignKey(
        DocumentCollection,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name=_("Коллекция")
    )
    user_id = models.CharField(max_length=255, db_index=True, verbose_name=_("ID пользователя"))
    search_config = models.JSONField(default=dict, verbose_name=_("Конфигурация поиска"))
    original_query = models.ForeignKey(
        SearchQuery,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name=_("Исходный запрос")
    )
    is_favorite = models.BooleanField(default=False, verbose_name=_("Избранное"))
    usage_count = models.IntegerField(default=0, verbose_name=_("Количество использований"))
    last_used_at = models.DateTimeField(null=True, blank=True, verbose_name=_("Последнее использование"))
    created_at = models.DateTimeField(auto_now_add=True, verbose_name=_("Создан"))
    updated_at = models.DateTimeField(auto_now=True, verbose_name=_("Обновлен"))
    
    class Meta:
        db_table = 'saved_searches'
        verbose_name = "Сохраненный поиск"
        verbose_name_plural = "Сохраненные поиски"
        ordering = ['-is_favorite', '-last_used_at', '-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.user_id})"
    
    def use(self):
        """Increment the usage counter."""
        self.usage_count += 1
        self.last_used_at = timezone.now()
        self.save(update_fields=['usage_count', 'last_used_at'])
