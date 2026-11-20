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

from .moe import MoEExpertJob



class ModelCheckpoint(models.Model):
    """Model for storing checkpoint metadata."""
    MODEL_TYPE_CHOICES = [
        ('QWEN', 'Qwen'),
        ('LLAMA', 'Llama'),
        ('MISTRAL', 'Mistral'),
        ('GEMMA', 'Gemma'),
        ('CUSTOM', 'Custom'),
        ('DEEPSEEK', 'DeepSeek'),
        ('PHI', 'Phi'),
        ('YI', 'Yi'),
    ]
    
    STATUS_CHOICES = [
        # lazy labels: Django 5+ autodetector compares them by msgid, so no
        # spurious migrations; UI translates per active language
        ('PENDING', _('Ожидание')),
        ('DOWNLOADING', _('Загрузка')),
        ('DOWNLOADED', _('Загружена на диск')),
        ('LOADING', _('Загрузка в память')),
        ('LOADED', _('Загружена')),
        ('MERGED', _('Слитая')),
        ('ERROR', _('Ошибка')),
        ('ARCHIVED', _('В архиве')),
    ]
    
    QUANTIZATION_CHOICES = [
        ('float32', 'FP32'),
        ('float16', 'FP16'),
        ('bfloat16', 'BF16'),
        ('int8', 'INT8'),
        ('int4', 'INT4'),
        ('gptq', 'GPTQ'),
        ('awq', 'AWQ'),
        ('quanto_int4', 'Quanto INT4 (CPU)'),
        ('quanto_int8', 'Quanto INT8 (CPU)'), 
    ]

    # v5.0: inference backend — in-process transformers or external vLLM
    SERVING_BACKEND_CHOICES = [
        ('transformers', _('Transformers (in-process)')),
        ('vllm', _('vLLM (OpenAI API)')),
    ]
    
    # Base fields
    name = models.CharField(
        max_length=255,
        unique=True,
        help_text=_("Уникальное имя модели в системе")
    )
    display_name = models.CharField(
        max_length=255,
        blank=True,
        help_text=_("Человекочитаемое имя для интерфейса")
    )
    model_type = models.CharField(
        max_length=20, 
        choices=MODEL_TYPE_CHOICES,
        default='CUSTOM'
    )
    
    # Safe Hugging Face download
    huggingface_id = models.CharField(
        max_length=300,
        blank=True,
        help_text=_("ID на Hugging Face Hub (например, 'Qwen/Qwen2.5-7B-Instruct')")
    )
    huggingface_revision = models.CharField(
        max_length=100,
        blank=True,
        default='main',
        help_text=_("Ревизия/ветка на Hugging Face")
    )
    
    # Loader configuration with validation
    loader_config = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Конфигурация для загрузки модели в формате JSON")
    )
    
    # Paths and status
    local_path = models.CharField(
        max_length=500,
        blank=True,
        help_text=_("Локальный путь к модели в контейнере")
    )
    relative_path = models.CharField(
        max_length=500,
        blank=True,
        help_text=_("Относительный путь для переносимости")
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='PENDING'
    )

    # v5.0: inference backend
    serving_backend = models.CharField(
        max_length=20,
        choices=SERVING_BACKEND_CHOICES,
        default='transformers',
        help_text=_("Как запускать инференс: в этом процессе (transformers) или на внешнем vLLM-сервере (OpenAI API)")
    )
    error_message = models.TextField(blank=True)
    
    # Model metadata
    parameter_count = models.BigIntegerField(
        default=0,
        help_text=_("Количество параметров (например, 7000000000 для 7B)")
    )
    quantization = models.CharField(
        max_length=50,
        choices=QUANTIZATION_CHOICES,
        blank=True
    )
    context_length = models.IntegerField(default=4096)
    vocab_size = models.IntegerField(null=True, blank=True)
    
    # License and safety
    license_type = models.CharField(
        max_length=100,
        blank=True,
        help_text=_("Тип лицензии (Apache 2.0, MIT, Commercial и т.д.)")
    )
    is_safe = models.BooleanField(
        default=False,
        help_text=_("Прошла ли модель проверку безопасности")
    )
    safety_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Оценка безопасности (чем выше, тем безопаснее)")
    )
    
    # Performance and metrics
    inference_speed = models.FloatField(
        null=True,
        blank=True,
        help_text=_("Скорость инференса (токенов/сек) на тестовом оборудовании")
    )
    accuracy_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Оценка точности на валидации")
    )
    evaluation_metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Расширенные метрики оценки {'mmlu': 0.75, 'hellaswag': 0.82}")
    )
    
    # Integrity check hashes
    config_hash = models.CharField(max_length=64, blank=True)
    weights_hash = models.CharField(max_length=64, blank=True)
    
    # Relations
    experiment = models.ForeignKey(
        'Experiment',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='checkpoints',
        help_text=_("Эксперимент, к которому относится модель")
    )
    parent = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='children',
        help_text=_("Родительская модель (для отслеживания происхождения)")
    )
    
    # System fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    is_public = models.BooleanField(
        default=False,
        help_text=_("Доступна ли модель другим пользователям")
    )
    
    # Tags for filtering
    tags = models.JSONField(
        default=list,
        blank=True,
        help_text=_("Теги модели ['ru', 'code', 'instruct', '7b']")
    )
    
    class Meta:
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['model_type', 'is_active']),
            models.Index(fields=['created_at']),
            models.Index(fields=['parameter_count']),
            models.Index(fields=['experiment', 'is_active']),
            models.Index(fields=['tags']),
        ]
        ordering = ['-created_at']
        permissions = [
            ("can_download_external", "Может загружать модели из внешних источников"),
            ("can_merge_models", "Может сливать модели"),
            ("can_delete_protected", "Может удалять защищенные модели"),
        ]
    
    def __str__(self):
        display = self.display_name or self.name
        return f"{display} ({self.get_model_type_display()}, {self.size_gb}GB)"
    
    def fill_missing_loader_config(self):
        """Fills missing loader_config fields with default values."""
        if not self.loader_config:
            self.loader_config = {}

        default_config = self.get_default_config()

        # Fill only missing fields or fields set to None
        for key, value in default_config.items():
            if key not in self.loader_config or self.loader_config[key] is None:
                self.loader_config[key] = value
        
        return self
    
    def clean(self):
        """Validate the model before saving."""
        super().clean()

        # Validate Hugging Face ID if provided
        if self.huggingface_id and not self._validate_huggingface_id():
            raise ValidationError(
                f"Hugging Face ID '{self.huggingface_id}' не соответствует формату или не в белом списке"
            )
        
        # Auto-fill loader_config before validation
        self.fill_missing_loader_config()

        # Config validation (now only fills in, never raises)
        self._validate_loader_config()

        # License check for public models
        if self.is_public and not self.license_type:
            raise ValidationError(
                "Публичные модели должны иметь указанную лицензию"
            )
        
        # Ensure evaluation_metrics and tags are non-empty
        if not self.evaluation_metrics:
            self.evaluation_metrics = {}
        if not self.tags:
            self.tags = []

    def _validate_huggingface_id(self) -> bool:
        """Validate Hugging Face ID format: owner/name with safe characters.

        v5.2: the organizational allowlist was removed — a self-hosted lab
        assumes its own repos (including its own HF uploads).
        Format: 'owner/name', safe characters, no path traversal.
        """
        if not self.huggingface_id:
            return True

        parts = self.huggingface_id.split('/')
        if len(parts) != 2:
            return False
        owner, name = parts
        safe = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,60}$')
        return bool(safe.match(owner) and safe.match(name))
    
    def _validate_loader_config(self):
        """Loader config validation — now only fills in missing fields."""
        self.fill_missing_loader_config()

    def save(self, *args, **kwargs):
        """save() override with hash generation."""
        # Fill missing loader_config fields
        self.fill_missing_loader_config()

        # Generate config hash
        if self.loader_config:
            config_str = json.dumps(self.loader_config, sort_keys=True)
            self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:64]

        # Auto-create display_name
        if not self.display_name:
            self.display_name = self.name.replace('_', ' ').title()

        # Default value for model_type
        if not self.model_type:
            self.model_type = 'CUSTOM'

        # Ensure evaluation_metrics and tags are non-empty
        if not self.evaluation_metrics:
            self.evaluation_metrics = {}
        if not self.tags:
            self.tags = []

        # Run validation
        self.full_clean()
        super().save(*args, **kwargs)
    
    @property
    def size_gb(self):
        """Approximate model size in GB."""
        if self.parameter_count <= 0:
            return 0

        # Bytes per parameter depending on quantization
        bytes_per_param = {
            'float32': 4,
            'float16': 2,
            'bfloat16': 2,
            'int8': 1,
            'int4': 0.5,
            'gptq': 0.5,  # approximate
            'awq': 0.5,   # approximate
        }.get(self.quantization, 2)  # default: float16
        
        size_bytes = self.parameter_count * bytes_per_param
        return round(size_bytes / (1024**3), 2)
    
    @property
    def is_quantized(self):
        """Whether the model is quantized."""
        return self.quantization in ['int8', 'int4', 'gptq', 'awq']

    @property
    def huggingface_url(self):
        """Model URL on Hugging Face."""
        if self.huggingface_id:
            return f"https://huggingface.co/{self.huggingface_id}"
        return None

    @property
    def published_hf_repo(self):
        """HF repo this checkpoint was published to from the platform
        (soft MoE push, v5.2). None if it has never been published.
        Uses the `published_hf` queryset annotation when available
        (model list) to avoid a per-row query. MoEExpertJob is defined
        later in this module — resolved at call time."""
        annotated = getattr(self, 'published_hf', None)
        if annotated:
            return annotated
        job = (MoEExpertJob.objects.filter(output_model=self)
               .exclude(hf_repo_id='').order_by('-id').first())
        return job.hf_repo_id if job else None

    @property
    def published_hf_url(self):
        """URL of the platform-published HF repo, if any"""
        repo = self.published_hf_repo
        return f"https://huggingface.co/{repo}" if repo else None

    @property
    def safe_name(self):
        """Filesystem-safe name."""
        import re
        name = self.name or self.display_name or ''
        # Strip special characters
        safe_name = re.sub(r'[^\w\-_]', '_', name)
        return safe_name.lower()

    def get_default_config(self):
        """Return the default config for the model type."""
        defaults = {
            'QWEN': {
                'architecture': 'Qwen2ForCausalLM',
                'trust_remote_code': True,
                'auto_map': {'auto_class': 'AutoModelForCausalLM'},
                'tokenizer_class': 'Qwen2Tokenizer',
                'use_flash_attention_2': True,
                'torch_dtype': 'auto',
                'device_map': 'auto',
                'generation_config': {
                    'max_tokens': 512,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'repetition_penalty': 1.1
                }
            },
            'LLAMA': {
                'architecture': 'LlamaForCausalLM',
                'trust_remote_code': False,
                'auto_map': {'auto_class': 'AutoModelForCausalLM'},
                'tokenizer_class': 'LlamaTokenizer',
                'use_flash_attention_2': True,
                'torch_dtype': 'auto',
                'device_map': 'auto',
                'generation_config': {
                    'max_tokens': 512,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'repetition_penalty': 1.1
                }
            },
            'MISTRAL': {
                'architecture': 'MistralForCausalLM',
                'trust_remote_code': False,
                'auto_map': {'auto_class': 'AutoModelForCausalLM'},
                'tokenizer_class': 'MistralTokenizer',
                'use_flash_attention_2': True,
                'torch_dtype': 'auto',
                'device_map': 'auto',
                'generation_config': {
                    'max_tokens': 512,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'repetition_penalty': 1.1
                }
            },
            'GEMMA': {
                'architecture': 'GemmaForCausalLM',
                'trust_remote_code': False,
                'auto_map': {'auto_class': 'AutoModelForCausalLM'},
                'tokenizer_class': 'GemmaTokenizer',
                'torch_dtype': 'bfloat16',
                'device_map': 'auto',
                'generation_config': {
                    'max_tokens': 512,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'repetition_penalty': 1.1
                }
            },
            'DEEPSEEK': {
                'architecture': 'AutoModelForCausalLM',
                'trust_remote_code': True,
                'auto_map': {'auto_class': 'AutoModelForCausalLM'},
                'tokenizer_class': 'AutoTokenizer',
                'use_flash_attention_2': True,
                'torch_dtype': 'auto',
                'device_map': 'auto',
                'generation_config': {
                    'max_tokens': 512,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'repetition_penalty': 1.1
                }
            },
            'PHI': {
                'architecture': 'AutoModelForCausalLM',
                'trust_remote_code': True,
                'auto_map': {'auto_class': 'AutoModelForCausalLM'},
                'tokenizer_class': 'AutoTokenizer',
                'use_flash_attention_2': False,
                'torch_dtype': 'auto',
                'device_map': 'auto',
                'generation_config': {
                    'max_tokens': 512,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'repetition_penalty': 1.1
                }
            },
            'YI': {
                'architecture': 'AutoModelForCausalLM',
                'trust_remote_code': True,
                'auto_map': {'auto_class': 'AutoModelForCausalLM'},
                'tokenizer_class': 'AutoTokenizer',
                'use_flash_attention_2': True,
                'torch_dtype': 'auto',
                'device_map': 'auto',
                'generation_config': {
                    'max_tokens': 512,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'repetition_penalty': 1.1
                }
            },
            'CUSTOM': {
                'architecture': 'AutoModelForCausalLM',
                'trust_remote_code': True,
                'auto_map': {},
                'tokenizer_class': 'AutoTokenizer',
                'torch_dtype': 'auto',
                'device_map': 'auto',
                'generation_config': {
                    'max_tokens': 512,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'repetition_penalty': 1.1
                }
            }
        }
        return defaults.get(self.model_type, defaults['CUSTOM'])
    
    def update_config(self, **kwargs):
        """Updates the model config with change logging."""
        old_config = self.loader_config.copy()
        self.loader_config.update(kwargs)

        # Create a change-history entry
        if self.pk:  # Only if the model is already saved
            ModelVersion.objects.create(
                checkpoint=self,
                version=f"config_update_{timezone.now().strftime('%Y%m%d_%H%M%S')}",
                changes={
                    'old': old_config,
                    'new': self.loader_config,
                    'updated_fields': list(kwargs.keys())
                },
                notes=f"Обновление конфигурации через update_config"
            )
        
        self.save(update_fields=['loader_config'])
    
    def mark_as_safe(self, score: float, reason: str = ""):
        """Mark the model as safe."""
        self.is_safe = True
        self.safety_score = score
        if reason:
            self.tags.append('safety_checked')
        self.save()
    
    def create_version(self, version_name: str, notes: str = "") -> 'ModelVersion':
        """Create a new model version."""
        return ModelVersion.objects.create(
            checkpoint=self,
            version=version_name,
            notes=notes,
            config_snapshot=self.loader_config.copy(),
            file_hash=self.weights_hash or self._calculate_file_hash()
        )
    
    def _calculate_file_hash(self) -> str:
        """Computes the model file hash (simplified version)."""
        # In reality all model files should be hashed
        return hashlib.sha256(self.local_path.encode()).hexdigest()[:64]

    def get_download_size_mb(self) -> float:
        """Estimated download size in MB."""
        size_gb = self.size_gb
        if self.is_quantized:
            size_gb *= 0.5  # Quantized models are smaller
        return round(size_gb * 1024, 2)

    def download_model(self):
        """Start downloading the model in the background."""
        try:
            from ..tasks import download_model_task
            return download_model_task.delay(self.id)
        except ImportError as e:
            logger.warning(f"Tasks module not available for download_model: {e}")
            return None
    
    def load_to_memory(self):
        """Load the model into memory."""
        try:
            from ..tasks import load_model_to_memory_task
            return load_model_to_memory_task.delay(self.id)
        except ImportError as e:
            logger.warning(f"Tasks module not available for load_to_memory: {e}")
            return None
    
    def unload_from_memory(self):
        """Unload the model from memory."""
        try:
            from ..tasks import unload_model_from_memory_task
            model_id = self.huggingface_id or self.name
            return unload_model_from_memory_task.delay(model_id)
        except ImportError as e:
            logger.warning(f"Tasks module not available for unload_from_memory: {e}")
            return None
    
    def is_loaded_in_memory(self):
        """Check whether the model is loaded in memory, via Redis."""
        try:
            from ..services.redis_model_service import redis_model_manager
        except ImportError as e:
            logger.warning(f"redis_model_service not available: {e}")
            return None
        
        model_id = self.huggingface_id or self.name
        
        # Check via Redis
        info = redis_model_manager.get_model_info(model_id)

        if info and info.get('loaded', False):
            device_info = info.get('device', 'unknown')

            # Determine device type for display
            device_str = str(device_info).lower()
            is_on_cpu = 'cpu' in device_str
            is_on_gpu = 'cuda' in device_str or 'gpu' in device_str
            is_mixed = 'mixed' in device_str
            
            return {
                'loaded': True,
                'device': device_info,
                'dtype': info.get('dtype', 'unknown'),
                'parameters': info.get('parameters', 0),
                'gpu_memory': info.get('gpu_memory', {}),
                'model_info': info,
                'is_on_cpu': is_on_cpu,
                'is_on_gpu': is_on_gpu,
                'is_mixed': is_mixed,
                'device_display': self._get_device_display(device_info, is_on_cpu, is_on_gpu, is_mixed),
                'loaded_by': info.get('loaded_by', {}),
                'loaded_at': info.get('loaded_at'),
            }
        
        return None

    def get_loading_status(self):
        """Get the current model loading status."""
        try:
            from ..services.redis_model_service import redis_model_manager
            model_id = self.huggingface_id or self.name
            return redis_model_manager.get_loading_status(model_id)
        except ImportError as e:
            logger.warning(f"redis_model_service not available: {e}")
            return {'status': 'ERROR', 'error': str(e)}
        except Exception as e:
            logger.error(f"Error getting loading status for model {self.id}: {str(e)}")
            return {'status': 'ERROR', 'error': str(e)}

    def _get_device_display(self, device_info, is_on_cpu, is_on_gpu, is_mixed):
        """Get a human-readable device label."""
        if is_mixed:
            if is_on_gpu and is_on_cpu:
                return "Смешанно (CPU/GPU)"
            elif is_on_gpu:
                return "Смешанно (только GPU)"
            elif is_on_cpu:
                return "Смешанно (только CPU)"
            else:
                return f"Смешанно: {device_info}"
        elif is_on_gpu:
            return "GPU"
        elif is_on_cpu:
            return "CPU"
        else:
            return str(device_info)[:30]

    def _get_device_summary(self, device_info, is_on_cpu, is_on_gpu, is_mixed):
        """Get a human-readable device description."""
        if is_mixed:
            if is_on_gpu and is_on_cpu:
                return "mixed (CPU/GPU)"
            elif is_on_gpu:
                return "mixed (GPU only)"
            elif is_on_cpu:
                return "mixed (CPU only)"
            else:
                return "mixed"
        elif is_on_gpu:
            return "gpu"
        elif is_on_cpu:
            return "cpu"
        else:
            return str(device_info)[:30]
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model."""
        # Import here to avoid circular dependencies
        try:
            from ..services.redis_model_service import redis_model_manager
            return redis_model_manager.get_model_info(model_id)
        except ImportError:
            return None
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error getting model info for {model_id}: {str(e)}")
            return None
    
    def generate_text(self, prompt: str, **kwargs):
        """Generate text with the model."""
        try:
            from ..services.redis_model_service import redis_model_manager
        except ImportError as e:
            logger.warning(f"redis_model_service not available: {e}")
            return {
                "response": "Сервис генерации недоступен.",
                "success": False,
                "error": "SERVICE_UNAVAILABLE"
            }
        
        model_id = self.huggingface_id or self.name

        # Check whether the model is loaded
        if not self.is_loaded_in_memory():
            return {
                "response": "Модель не загружена в память. Загрузите модель сначала.",
                "success": False,
                "error": "MODEL_NOT_LOADED"
            }
        
        # Prepare parameters
        generation_params = {
            'max_tokens': kwargs.get('max_tokens', 512),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
            'top_k': kwargs.get('top_k', 50),
            'repetition_penalty': kwargs.get('repetition_penalty', 1.0),
        }
        
        # Prepend the system prompt if provided
        system_prompt = kwargs.get('system_prompt', '')
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        
        # Generate text
        result = redis_model_manager.generate_text(
            model_id=model_id,
            prompt=prompt,
            **generation_params
        )
        
        return result

    def sync_loading_status(self):
        """Sync the DB loading status with the actual state."""
        try:
            from ..services.redis_model_service import redis_model_manager
        except ImportError as e:
            logger.warning(f"redis_model_service not available: {e}")
            return False
        
        memory_info = self.is_loaded_in_memory()
        actually_loaded = bool(memory_info)
        current_status = self.status
        
        if actually_loaded and current_status != 'LOADED':
            self.status = 'LOADED'
            self.save(update_fields=['status'])
            logger.info(f"Synced status for model {self.name}: {current_status} -> LOADED")
            return True
        elif not actually_loaded and current_status == 'LOADED':
            new_status = 'DOWNLOADED' if self.local_path else 'PENDING'
            self.status = new_status
            self.save(update_fields=['status'])
            logger.info(f"Synced status for model {self.name}: {current_status} -> {new_status}")
            return True
        
        return False  # Status unchanged


class ModelVersion(models.Model):
    """
    Model versioning for tracking changes.
    Each version is a snapshot of the model state.
    """
    checkpoint = models.ForeignKey(
        ModelCheckpoint,
        on_delete=models.CASCADE,
        related_name='versions'
    )
    version = models.CharField(
        max_length=50,
        help_text=_("Идентификатор версии (v1.0, experimental-merge)")
    )
    parent_version = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='children'
    )
    
    # State snapshot
    config_snapshot = models.JSONField(
        help_text=_("Полная конфигурация на момент создания версии")
    )
    file_hash = models.CharField(
        max_length=64,
        help_text=_("Хеш файлов модели для проверки целостности")
    )
    
    # Metadata
    notes = models.TextField(blank=True)
    changes = models.JSONField(
        default=dict,
        help_text=_("Изменения относительно предыдущей версии")
    )
    is_stable = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.CharField(
        max_length=100,
        blank=True,
        help_text=_("Кто создал версию (пользователь или система)")
    )
    
    class Meta:
        ordering = ['checkpoint', '-created_at']
        unique_together = ['checkpoint', 'version']
        indexes = [
            models.Index(fields=['checkpoint', 'is_stable']),
            models.Index(fields=['file_hash']),
        ]
    
    def __str__(self):
        return f"{self.checkpoint.name} - {self.version}"
    
    def restore(self):
        """Restore this version as the current one."""
        self.checkpoint.loader_config = self.config_snapshot
        self.checkpoint.save()

        # Create a new version to track the restore
        ModelVersion.objects.create(
            checkpoint=self.checkpoint,
            version=f"restore_{timezone.now().strftime('%Y%m%d_%H%M%S')}",
            parent_version=self,
            config_snapshot=self.config_snapshot,
            file_hash=self.file_hash,
            changes={'action': 'restore', 'from_version': self.version},
            notes=f"Восстановление версии {self.version}",
            created_by='system'
        )


class ModelMergeJob(models.Model):
    """Model merge jobs with extended tracking."""
    MERGE_METHOD_CHOICES = [
        ('SLERP', 'SLERP (Spherical Linear Interpolation)'),
        ('TIES', 'TIES (Task Arithmetic)'),
        ('DARE', 'DARE (Drop And Rescale)'),
        ('LINEAR', 'Linear Interpolation'),
        ('PASSTHROUGH', 'Pass-through (экспериментальный)'),
    ]
    
    STATUS_CHOICES = [
        ('PENDING', 'В ожидании'),
        ('VALIDATING', 'Валидация'),
        ('RUNNING', 'Выполняется'),
        ('COMPLETED', 'Завершено'),
        ('FAILED', 'Ошибка'),
        ('CANCELLED', 'Отменено'),
    ]
    
    # Main fields
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    # Input models
    base_model = models.ForeignKey(
        ModelCheckpoint,
        related_name='base_merges',
        on_delete=models.CASCADE
    )
    merge_model = models.ForeignKey(
        ModelCheckpoint,
        related_name='merge_merges',
        on_delete=models.CASCADE
    )
    
    # Output model (created after the merge)
    output_model = models.ForeignKey(
        ModelCheckpoint,
        related_name='output_merges',
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    
    # Merge settings
    merge_method = models.CharField(
        max_length=20,
        choices=MERGE_METHOD_CHOICES,
        default='TIES'
    )
    merge_ratio = models.FloatField(
        default=0.5,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Доля merge_model в результате (0 = только base, 1 = только merge)")
    )
    
    # Advanced configuration
    merge_config = models.JSONField(
        default=dict,
        help_text=_("Дополнительные параметры слияния")
    )
    
    # Resources and limits
    gpu_memory_limit_gb = models.FloatField(
        default=8.0,
        help_text=_("Лимит памяти GPU для слияния (GB)")
    )
    timeout_minutes = models.IntegerField(
        default=120,
        help_text=_("Таймаут выполнения в минутах")
    )
    
    # Progress and status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    progress = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
    )
    current_step = models.CharField(max_length=100, blank=True)
    
    # Logs and results
    logs = models.TextField(blank=True)
    error_details = models.JSONField(default=dict, help_text=_("Детали ошибки"))
    performance_metrics = models.JSONField(
        default=dict,
        help_text=_("Метрики производительности слияния")
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    validated_at = models.DateTimeField(null=True, blank=True)

    # Version control
    parent_job = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='retries'
    )
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['base_model', 'merge_model']),
        ]
        permissions = [
            ("can_cancel_merge", "Может отменять слияние"),
            ("can_priority_merge", "Высокий приоритет слияния"),
        ]
    
    def __str__(self):
        return f"Merge: {self.base_model.name} + {self.merge_model.name} ({self.get_status_display()})"
    
    def clean(self):
        """Validate the merge job."""
        super().clean()

        # Model compatibility check
        if self.base_model and self.merge_model:
            if self.base_model.model_type != self.merge_model.model_type:
                raise ValidationError(
                    f"Модели разных типов: {self.base_model.model_type} и {self.merge_model.model_type}"
                )
            
            if self.base_model.parameter_count != self.merge_model.parameter_count:
                raise ValidationError(
                    "Модели имеют разное количество параметров, слияние невозможно"
                )
        
        # Merge method check
        if self.merge_method == 'SLERP' and not (0 <= self.merge_ratio <= 1):
            raise ValidationError("Для SLERP merge_ratio должен быть между 0 и 1")
    
    def save(self, *args, **kwargs):
        """save() override with automatic naming."""
        if not self.name:
            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
            self.name = f"merge_{timestamp}"
        
        super().save(*args, **kwargs)
    
    @property
    def duration_seconds(self):
        """Job duration in seconds."""
        if not self.started_at or not self.completed_at:
            return None
        return (self.completed_at - self.started_at).total_seconds()
    
    @property
    def is_compatible(self):
        """Whether the models are compatible for merging."""
        if not self.base_model or not self.merge_model:
            return False
        
        return (
            self.base_model.model_type == self.merge_model.model_type and
            self.base_model.parameter_count == self.merge_model.parameter_count and
            self.base_model.quantization == self.merge_model.quantization
        )
    
    def start(self):
        """Start the job."""
        self.status = 'RUNNING'
        self.started_at = timezone.now()
        self.current_step = 'initialization'
        self.save()
    
    def complete(self, output_checkpoint: ModelCheckpoint):
        """Mark the job as completed."""
        self.status = 'COMPLETED'
        self.completed_at = timezone.now()
        self.output_model = output_checkpoint
        self.progress = 100.0
        self.current_step = 'completed'
        self.save()
    
    def fail(self, error_message: str, error_details: dict = None):
        """Mark the job as failed."""
        self.status = 'FAILED'
        self.completed_at = timezone.now()
        self.error_message = error_message
        self.error_details = error_details or {}
        self.save()
    
    def update_progress(self, progress: float, step: str = ""):
        """Update job progress."""
        self.progress = max(0.0, min(100.0, progress))
        if step:
            self.current_step = step
        self.save(update_fields=['progress', 'current_step', 'updated_at'])


class InferenceSession(models.Model):
    """
    Inference sessions for usage and performance monitoring.
    """
    checkpoint = models.ForeignKey(
        ModelCheckpoint,
        on_delete=models.CASCADE,
        related_name='inference_sessions'
    )
    session_id = models.CharField(
        max_length=100,
        unique=True,
        help_text=_("Уникальный ID сессии")
    )
    
    # User info
    user_ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    api_key = models.CharField(max_length=100, blank=True)
    
    # Usage stats
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    total_requests = models.IntegerField(default=0)
    
    # Performance
    total_duration = models.FloatField(default=0.0, help_text=_("Общая длительность в секундах"))
    avg_response_time = models.FloatField(default=0.0, help_text=_("Среднее время ответа в секундах"))
    
    # Errors
    error_count = models.IntegerField(default=0)
    last_error = models.TextField(blank=True)
    
    # Status
    is_active = models.BooleanField(default=True)

    # Timestamps
    started_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-started_at']
        indexes = [
            models.Index(fields=['checkpoint', 'started_at']),
            models.Index(fields=['is_active', 'last_activity']),
        ]
    
    def __str__(self):
        return f"Session {self.session_id} for {self.checkpoint.name}"
    
    @property
    def tokens_per_second(self):
        """Token processing speed per second."""
        if self.total_duration > 0:
            return (self.input_tokens + self.output_tokens) / self.total_duration
        return 0.0
    
    @property
    def duration_minutes(self):
        """Duration of the session in minutes (0 until the session starts)."""
        if self.ended_at and self.started_at:
            return (self.ended_at - self.started_at).total_seconds() / 60
        if self.started_at:
            return (timezone.now() - self.started_at).total_seconds() / 60
        return 0.0
    
    def add_request(self, input_tokens: int, output_tokens: int, duration: float):
        """Add a request to the statistics."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_requests += 1
        self.total_duration += duration

        # Recompute the average
        if self.total_requests > 0:
            self.avg_response_time = self.total_duration / self.total_requests
        
        self.save()
    
    def end_session(self):
        """End the session."""
        self.is_active = False
        self.ended_at = timezone.now()
        self.save()


class ModelEvaluation(models.Model):
    """
    Model evaluation results on standard datasets.
    """
    checkpoint = models.ForeignKey(
        ModelCheckpoint,
        on_delete=models.CASCADE,
        related_name='evaluations'
    )
    
    # Evaluation configuration
    dataset_name = models.CharField(max_length=100)
    dataset_split = models.CharField(max_length=50, default='validation')
    evaluation_type = models.CharField(
        max_length=50,
        choices=[
            ('AUTOMATIC', 'Автоматическая'),
            ('MANUAL', 'Ручная'),
            ('COMPARATIVE', 'Сравнительная'),
        ],
        default='AUTOMATIC'
    )
    
    # Results
    metrics = models.JSONField(
        default=dict,
        help_text=_("Метрики оценки {'accuracy': 0.85, 'f1': 0.82}")
    )
    raw_results = models.JSONField(
        default=dict,
        help_text=_("Сырые результаты для анализа")
    )
    
    # System info
    evaluated_at = models.DateTimeField(auto_now_add=True)
    evaluator_version = models.CharField(max_length=50, blank=True)
    hardware_info = models.JSONField(
        default=dict,
        help_text=_("Информация о железе {'gpu': 'RTX 4090', 'memory_gb': 24}")
    )
    
    # Validation
    is_valid = models.BooleanField(default=True)
    validation_notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-evaluated_at']
        unique_together = ['checkpoint', 'dataset_name', 'dataset_split']
        indexes = [
            models.Index(fields=['checkpoint', 'evaluated_at']),
        ]
    
    def __str__(self):
        return f"{self.checkpoint.name} on {self.dataset_name}"
    
    @property
    def primary_metric(self):
        """Primary metric for this dataset."""
        # Different datasets use different primary metrics
        metric_map = {
            'mmlu': 'accuracy',
            'hellaswag': 'accuracy',
            'winogrande': 'accuracy',
            'russian_superglue': 'average',
            'humaneval': 'pass@1',
        }
        
        primary = metric_map.get(self.dataset_name, 'accuracy')
        return self.metrics.get(primary, 0.0)


class ModelDownloadJob(models.Model):
    """Model download job."""
    STATUS_CHOICES = [
        ('PENDING', 'Ожидает'),
        ('DOWNLOADING', 'Загружается'),
        ('COMPLETED', 'Завершена'),
        ('FAILED', 'Ошибка'),
        ('CANCELLED', 'Отменена'),
    ]
    
    checkpoint = models.ForeignKey(ModelCheckpoint, on_delete=models.CASCADE, related_name='download_jobs')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    local_path = models.TextField(blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)
    task_id = models.CharField(max_length=100, blank=True, null=True, help_text=_("ID задачи Celery"))
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Download job for {self.checkpoint.name} - {self.status}"


class DeletedModelLog(models.Model):
    """Log of deleted models."""
    original_name = models.CharField(max_length=255)
    display_name = models.CharField(max_length=255)
    model_type = models.CharField(max_length=20, choices=ModelCheckpoint.MODEL_TYPE_CHOICES)
    huggingface_id = models.CharField(max_length=300, blank=True)
    
    # File info
    local_path = models.CharField(max_length=500, blank=True)
    relative_path = models.CharField(max_length=500, blank=True)
    had_local_files = models.BooleanField(default=False)
    
    # Technical info
    size_gb = models.FloatField(default=0.0)
    parameter_count = models.BigIntegerField(default=0)
    
    # Deletion info
    deleted_by = models.CharField(max_length=100)
    deletion_reason = models.TextField(blank=True)
    deleted_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-deleted_at']
        verbose_name = 'Лог удаленной модели'
        verbose_name_plural = 'Логи удаленных моделей'
    
    def __str__(self):
        return f"{self.display_name} (удалена {self.deleted_at.strftime('%d.%m.%Y %H:%M')})"
