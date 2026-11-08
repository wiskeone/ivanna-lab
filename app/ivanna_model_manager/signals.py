import os
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.core.exceptions import ValidationError
from django.utils import timezone
from django.db.models import Count, Avg, Max, Min, Q, Sum
import logging
from datetime import timedelta
from django.contrib.auth.models import User

from .models import Document, DeletedDocumentLog, UserProfile

logger = logging.getLogger(__name__)

# ============================================================================
# СИГНАЛЫ (добавить в signals.py или в конец models.py)
# ============================================================================

from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver

@receiver(post_save, sender=Document)
def update_collection_stats(sender, instance, **kwargs):
    """Обновить статистику коллекции при изменении документа"""
    if instance.collection:
        instance.collection.update_statistics()

@receiver(pre_delete, sender=Document)
def log_deleted_document(sender, instance, **kwargs):
    """Записать лог при удалении документа"""
    try:
        DeletedDocumentLog.objects.create(
            original_name=instance.original_filename,
            title=instance.title,
            document_type=instance.document_type,
            file_size_mb=instance.file_size_mb,
            collection_name=instance.collection.name if instance.collection else '',
            total_chunks=instance.chunk_count,
            deleted_at=timezone.now()
        )
    except Exception as e:
        logger.error(f"Error logging deleted document: {e}")

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Создаёт профиль пользователя сразу после создания учётной записи."""
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """Сохраняет профиль при сохранении пользователя."""
    # Используем try/except на случай, если профиль ещё не создан (например, при миграции)
    try:
        instance.profile.save()
    except UserProfile.DoesNotExist:
        # Если профиля нет, создаём (на всякий случай)
        UserProfile.objects.create(user=instance)

