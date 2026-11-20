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



class ChatSession(models.Model):
    """Сессия чата с моделью"""
    session_id = models.CharField(max_length=100, unique=True)
    user_id = models.CharField(max_length=100, blank=True)
    checkpoint = models.ForeignKey(
        ModelCheckpoint,
        on_delete=models.CASCADE,
        related_name='chat_sessions',
        null=True,
        blank=True
    )
    
    # Поле для переименования сессии
    notes = models.CharField(max_length=200, blank=True, help_text=_("Пользовательское название сессии"))
    
    # Контекст чата
    system_prompt = models.TextField(blank=True, help_text=_("Системный промпт для сессии"))
    conversation_history = models.JSONField(
        default=list,
        help_text=_("История диалога в формате [{'role': 'user', 'content': '...'}, ...]")
    )
    
    # Техническая информация
    model_config = models.JSONField(
        default=dict,
        help_text=_("Конфигурация модели при создании сессии")
    )
    generation_config = models.JSONField(
        default=dict,
        help_text=_("Параметры генерации")
    )
    
    # Статистика
    total_messages = models.IntegerField(default=0)
    total_input_tokens = models.IntegerField(default=0)
    total_output_tokens = models.IntegerField(default=0)
    
    # Статус
    is_active = models.BooleanField(default=True)
    
    # Временные метки
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_activity = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-last_activity']
        indexes = [
            models.Index(fields=['session_id', 'is_active']),
            models.Index(fields=['checkpoint', 'last_activity']),
            models.Index(fields=['user_id', 'is_active']),  # Добавим индекс для быстрого поиска
        ]
    
    def __str__(self):
        display_name = self.notes or f"Сессия {self.session_id[:10]}..."
        return f"Chat {display_name} with {self.checkpoint.name if self.checkpoint else 'no model'}"

    def add_message(self, role: str, content: str, metadata: dict = None):
        """Добавить сообщение в историю диалога"""
        message = {
            'role': role,
            'content': content,
            'timestamp': timezone.now().isoformat()
        }
        
        if metadata:
            message['metadata'] = metadata
        
        # Получаем текущую историю
        current_history = self.conversation_history or []
        current_history.append(message)
        
        # Обновляем историю
        self.conversation_history = current_history
        
        # Обновляем счетчики, извлекая токены из metadata['tokens']
        if metadata and 'tokens' in metadata:
            tokens = metadata['tokens']
            if role == 'user':
                self.total_input_tokens += tokens.get('input', 0)
            elif role == 'assistant':
                self.total_output_tokens += tokens.get('output', 0)
        
        self.total_messages = len(current_history)
        self.last_activity = timezone.now()
        self.save()

    @property
    def total_tokens(self):
        """Общее количество токенов в сессии"""
        return self.total_input_tokens + self.total_output_tokens


class ChatMessage(models.Model):
    """Детализированное сообщение чата с технической информацией"""
    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name='detailed_messages'
    )
    
    # Основное содержание
    role = models.CharField(
        max_length=20,
        choices=[
            ('system', 'System'),
            ('user', 'User'),
            ('assistant', 'Assistant'),
            ('function', 'Function'),
            ('tool', 'Tool'),
        ]
    )
    content = models.TextField()
    
    # Техническая информация о генерации
    generation_metadata = models.JSONField(
        default=dict,
        help_text=_("Метаданные генерации: токены, время, температура и т.д.")
    )
    
    # Родительское сообщение (для древовидных структур)
    parent = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='children'
    )
    
    # Временные метки
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['session', 'created_at']
        indexes = [
            models.Index(fields=['session', 'created_at']),
        ]
    
    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."


class ModelComparison(models.Model):
    """Сравнение ответов разных моделей на одни и те же промпты"""
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    # Набор промптов для сравнения
    prompts = models.JSONField(
        default=list,
        help_text=_("Список промптов для сравнения [{'id': 1, 'text': '...'}]")
    )
    
    # Результаты сравнения
    results = models.JSONField(
        default=dict,
        help_text=_("Результаты по моделям {model_id: {prompt_id: {...}}}")
    )
    
    # Метрики сравнения
    metrics = models.JSONField(
        default=dict,
        help_text=_("Вычисленные метрики сравнения")
    )
    
    # Статус
    is_complete = models.BooleanField(default=False)
    
    # Временные метки
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Comparison: {self.name}"
    
    def add_model_result(self, model, prompt_id, response, metadata=None):
        """Добавить результат модели для промпта"""
        if str(model.id) not in self.results:
            self.results[str(model.id)] = {}
        
        result = {
            'response': response,
            'model_name': model.name,
            'timestamp': timezone.now().isoformat()
        }
        
        if metadata:
            result['metadata'] = metadata
        
        self.results[str(model.id)][str(prompt_id)] = result
        self.save()
    
    def calculate_metrics(self):
        """Вычислить метрики сравнения (заглушка)"""
        # Здесь можно реализовать вычисление BLEU, ROUGE, косинусного сходства и т.д.
        self.metrics = {
            'calculated_at': timezone.now().isoformat(),
            'note': 'Метрики требуют реализации'
        }
        self.save()
