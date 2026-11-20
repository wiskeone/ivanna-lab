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
# SIGNALS (add to signals.py or to the end of models.py)
# ============================================================================

from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver

@receiver(post_save, sender=Document)
def update_collection_stats(sender, instance, **kwargs):
    """Update collection statistics when a document changes."""
    if instance.collection:
        instance.collection.update_statistics()

@receiver(pre_delete, sender=Document)
def log_deleted_document(sender, instance, **kwargs):
    """Log a document deletion."""
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
    """Create a user profile right after account creation."""
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """Save the profile when the user is saved."""
    # try/except in case the profile has not been created yet (e.g. during a migration)
    try:
        instance.profile.save()
    except UserProfile.DoesNotExist:
        # Profile missing; create it as a fallback
        UserProfile.objects.create(user=instance)

