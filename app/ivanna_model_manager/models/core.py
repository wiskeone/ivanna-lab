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



class SecurityError(Exception):
    """Custom security error."""
    pass


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    huggingface_token = models.CharField(
        max_length=255,
        blank=True,
        help_text=_("Токен Hugging Face для доступа к gated моделям")
    )
    # Additional fields can be added here: avatar, settings, etc.

    def __str__(self):
        return f"Профиль {self.user.username}"


class Experiment(models.Model):
    """
    Container for isolated model experiments.
    Allows grouping models and easily rolling back changes.
    """
    name = models.CharField(
        max_length=255,
        unique=True,
        help_text=_("Уникальное имя эксперимента")
    )
    description = models.TextField(blank=True)
    base_checkpoint = models.ForeignKey(
        'ModelCheckpoint',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='base_checkpoint_experiments',
        help_text=_("Базовая модель эксперимента")
    )
    is_active = models.BooleanField(default=True)
    tags = models.JSONField(
        default=list,
        help_text=_("Теги для фильтрации (например, ['ru', 'code', 'medical'])")
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['is_active', 'created_at']),
        ]
    
    def __str__(self):
        return f"{self.name} ({'active' if self.is_active else 'inactive'})"
    
    def deactivate(self):
        """Deactivates the experiment and all its related models."""
        self.is_active = False
        self.save()
        self.checkpoints.update(is_active=False)
