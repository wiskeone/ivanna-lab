import os
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

class SecurityError(Exception):
    """Кастомная ошибка безопасности"""
    pass

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    huggingface_token = models.CharField(
        max_length=255,
        blank=True,
        help_text=_("Токен Hugging Face для доступа к gated моделям")
    )
    # Можно добавить другие поля: аватар, настройки и т.д.

    def __str__(self):
        return f"Профиль {self.user.username}"

class Experiment(models.Model):
    """
    Контейнер для изолированных экспериментов с моделями.
    Позволяет группировать модели и легко откатывать изменения.
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
        """Деактивирует эксперимент и все связанные модели"""
        self.is_active = False
        self.save()
        self.checkpoints.update(is_active=False)


class ModelCheckpoint(models.Model):
    """Модель для хранения метаинформации о чекпоинтах"""
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
        ('PENDING', 'Ожидание'),
        ('DOWNLOADING', 'Загрузка'),
        ('DOWNLOADED', 'Загружена на диск'), 
        ('LOADING', 'Загрузка в память'),
        ('LOADED', 'Загружена'),
        ('MERGED', 'Слитая'),
        ('ERROR', 'Ошибка'),
        ('ARCHIVED', 'В архиве'),
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

    # v5.0: бэкенд инференса — in-process transformers или внешний vLLM
    SERVING_BACKEND_CHOICES = [
        ('transformers', _('Transformers (in-process)')),
        ('vllm', _('vLLM (OpenAI API)')),
    ]
    
    # Базовые поля
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
    
    # Безопасная загрузка с Hugging Face
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
    
    # Конфигурация загрузки с валидацией
    loader_config = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Конфигурация для загрузки модели в формате JSON")
    )
    
    # Пути и статус
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

    # v5.0: бэкенд инференса
    serving_backend = models.CharField(
        max_length=20,
        choices=SERVING_BACKEND_CHOICES,
        default='transformers',
        help_text=_("Как запускать инференс: в этом процессе (transformers) или на внешнем vLLM-сервере (OpenAI API)")
    )
    error_message = models.TextField(blank=True)
    
    # Метаданные модели
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
    
    # Лицензия и безопасность
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
    
    # Производительность и метрики
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
    
    # Хеши для проверки целостности
    config_hash = models.CharField(max_length=64, blank=True)
    weights_hash = models.CharField(max_length=64, blank=True)
    
    # Связи
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
    
    # Системные поля
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    is_public = models.BooleanField(
        default=False,
        help_text=_("Доступна ли модель другим пользователям")
    )
    
    # Теги для фильтрации
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
        """Заполняет недостающие поля в loader_config значениями по умолчанию"""
        if not self.loader_config:
            self.loader_config = {}
        
        default_config = self.get_default_config()
        
        # Заполняем только отсутствующие поля или поля со значением None
        for key, value in default_config.items():
            if key not in self.loader_config or self.loader_config[key] is None:
                self.loader_config[key] = value
        
        return self
    
    def clean(self):
        """Валидация модели перед сохранением"""
        super().clean()
        
        # Проверка Hugging Face ID если указан
        if self.huggingface_id and not self._validate_huggingface_id():
            raise ValidationError(
                f"Hugging Face ID '{self.huggingface_id}' не соответствует формату или не в белом списке"
            )
        
        # Автоматическое заполнение loader_config перед валидацией
        self.fill_missing_loader_config()
        
        # Проверка конфигурации (теперь только проверяет, не вызывает ошибок)
        self._validate_loader_config()
        
        # Проверка лицензии для публичных моделей
        if self.is_public and not self.license_type:
            raise ValidationError(
                "Публичные модели должны иметь указанную лицензию"
            )
        
        # Убеждаемся, что evaluation_metrics и tags не пустые
        if not self.evaluation_metrics:
            self.evaluation_metrics = {}
        if not self.tags:
            self.tags = []
    
    def _validate_huggingface_id(self) -> bool:
        """Проверяет, что Hugging Face ID безопасен и в белом списке"""
        if not self.huggingface_id:
            return True
        
        # Обновленный белый список разрешенных репозиториев с новыми типами моделей
        ALLOWED_PATTERNS = [
            "Qwen/*",
            "qwen/*",
            "meta-llama/*",
            "mistralai/*",
            "google/*",
            "microsoft/*",
            "bigcode/*",
            "codellama/*",
            "NousResearch/*",
            "OpenHermes/*",
            "deepseek-ai/*",
            "01-ai/*",
        ]
        
        # Проверка по шаблонам
        for pattern in ALLOWED_PATTERNS:
            if fnmatch.fnmatch(self.huggingface_id, pattern):
                return True
        
        # Дополнительная проверка формата
        parts = self.huggingface_id.split('/')
        if len(parts) != 2:
            return False
        
        return False
    
    def _validate_loader_config(self):
        """Валидация конфигурации загрузки - теперь только заполняет недостающие поля"""
        self.fill_missing_loader_config()
    
    def save(self, *args, **kwargs):
        """Переопределение save с генерацией хешей"""
        # Заполняем недостающие поля в loader_config
        self.fill_missing_loader_config()
        
        # Генерация хеша конфигурации
        if self.loader_config:
            config_str = json.dumps(self.loader_config, sort_keys=True)
            self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:64]
        
        # Автоматическое создание display_name
        if not self.display_name:
            self.display_name = self.name.replace('_', ' ').title()
        
        # Установка значения по умолчанию для параметра model_type
        if not self.model_type:
            self.model_type = 'CUSTOM'
        
        # Убеждаемся, что evaluation_metrics и tags не пустые
        if not self.evaluation_metrics:
            self.evaluation_metrics = {}
        if not self.tags:
            self.tags = []
        
        # Вызов валидации
        self.full_clean()
        super().save(*args, **kwargs)
    
    @property
    def size_gb(self):
        """Приблизительный размер модели в GB"""
        if self.parameter_count <= 0:
            return 0
        
        # Байт на параметр в зависимости от квантизации
        bytes_per_param = {
            'float32': 4,
            'float16': 2,
            'bfloat16': 2,
            'int8': 1,
            'int4': 0.5,
            'gptq': 0.5,  # примерно
            'awq': 0.5,   # примерно
        }.get(self.quantization, 2)  # по умолчанию float16
        
        size_bytes = self.parameter_count * bytes_per_param
        return round(size_bytes / (1024**3), 2)
    
    @property
    def is_quantized(self):
        """Проверка, квантизирована ли модель"""
        return self.quantization in ['int8', 'int4', 'gptq', 'awq']
    
    @property
    def huggingface_url(self):
        """URL модели на Hugging Face"""
        if self.huggingface_id:
            return f"https://huggingface.co/{self.huggingface_id}"
        return None
    
    @property
    def safe_name(self):
        """Безопасное имя для файловой системы"""
        import re
        name = self.name or self.display_name or ''
        # Удаляем специальные символы
        safe_name = re.sub(r'[^\w\-_]', '_', name)
        return safe_name.lower()
    
    def get_default_config(self):
        """Возвращает конфигурацию по умолчанию для типа модели"""
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
        """Обновляет конфигурацию модели с логированием изменений"""
        old_config = self.loader_config.copy()
        self.loader_config.update(kwargs)
        
        # Создание записи в истории изменений
        if self.pk:  # Только если модель уже сохранена
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
        """Пометить модель как безопасную"""
        self.is_safe = True
        self.safety_score = score
        if reason:
            self.tags.append('safety_checked')
        self.save()
    
    def create_version(self, version_name: str, notes: str = "") -> 'ModelVersion':
        """Создать новую версию модели"""
        return ModelVersion.objects.create(
            checkpoint=self,
            version=version_name,
            notes=notes,
            config_snapshot=self.loader_config.copy(),
            file_hash=self.weights_hash or self._calculate_file_hash()
        )
    
    def _calculate_file_hash(self) -> str:
        """Вычисляет хеш файлов модели (упрощенная версия)"""
        # В реальности нужно хешировать все файлы модели
        return hashlib.sha256(self.local_path.encode()).hexdigest()[:64]
    
    def get_download_size_mb(self) -> float:
        """Предполагаемый размер загрузки в MB"""
        size_gb = self.size_gb
        if self.is_quantized:
            size_gb *= 0.5  # Квантизированные модели меньше
        return round(size_gb * 1024, 2)

    def download_model(self):
        """Запускаем загрузку модели в фоне"""
        try:
            from .tasks import download_model_task
            return download_model_task.delay(self.id)
        except ImportError as e:
            logger.warning(f"Tasks module not available for download_model: {e}")
            return None
    
    def load_to_memory(self):
        """Загружаем модель в память"""
        try:
            from .tasks import load_model_to_memory_task
            return load_model_to_memory_task.delay(self.id)
        except ImportError as e:
            logger.warning(f"Tasks module not available for load_to_memory: {e}")
            return None
    
    def unload_from_memory(self):
        """Выгружаем модель из памяти"""
        try:
            from .tasks import unload_model_from_memory_task
            model_id = self.huggingface_id or self.name
            return unload_model_from_memory_task.delay(model_id)
        except ImportError as e:
            logger.warning(f"Tasks module not available for unload_from_memory: {e}")
            return None
    
    def is_loaded_in_memory(self):
        """Проверяем, загружена ли модель в память - используя Redis"""
        try:
            from .services.redis_model_service import redis_model_manager
        except ImportError as e:
            logger.warning(f"redis_model_service not available: {e}")
            return None
        
        model_id = self.huggingface_id or self.name
        
        # Проверяем через Redis
        info = redis_model_manager.get_model_info(model_id)
        
        if info and info.get('loaded', False):
            device_info = info.get('device', 'unknown')
            
            # Определяем тип устройства для отображения
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
        """Получает текущий статус загрузки модели"""
        try:
            from .services.redis_model_service import redis_model_manager
            model_id = self.huggingface_id or self.name
            return redis_model_manager.get_loading_status(model_id)
        except ImportError as e:
            logger.warning(f"redis_model_service not available: {e}")
            return {'status': 'ERROR', 'error': str(e)}
        except Exception as e:
            logger.error(f"Error getting loading status for model {self.id}: {str(e)}")
            return {'status': 'ERROR', 'error': str(e)}

    def _get_device_display(self, device_info, is_on_cpu, is_on_gpu, is_mixed):
        """Получаем понятное отображение устройства"""
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
        """Получаем понятное описание устройства"""
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
        """Получает информацию о загруженной модели"""
        # Используем импорт для избежания циклических зависимостей
        try:
            from .services.redis_model_service import redis_model_manager
            return redis_model_manager.get_model_info(model_id)
        except ImportError:
            return None
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error getting model info for {model_id}: {str(e)}")
            return None
    
    def generate_text(self, prompt: str, **kwargs):
        """Генерируем текст с помощью модели"""
        try:
            from .services.redis_model_service import redis_model_manager
        except ImportError as e:
            logger.warning(f"redis_model_service not available: {e}")
            return {
                "response": "Сервис генерации недоступен.",
                "success": False,
                "error": "SERVICE_UNAVAILABLE"
            }
        
        model_id = self.huggingface_id or self.name
        
        # Проверяем, загружена ли модель
        if not self.is_loaded_in_memory():
            return {
                "response": "Модель не загружена в память. Загрузите модель сначала.",
                "success": False,
                "error": "MODEL_NOT_LOADED"
            }
        
        # Подготавливаем параметры
        generation_params = {
            'max_tokens': kwargs.get('max_tokens', 512),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
            'top_k': kwargs.get('top_k', 50),
            'repetition_penalty': kwargs.get('repetition_penalty', 1.0),
        }
        
        # Добавляем системный промпт если есть
        system_prompt = kwargs.get('system_prompt', '')
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        
        # Генерируем текст
        result = redis_model_manager.generate_text(
            model_id=model_id,
            prompt=prompt,
            **generation_params
        )
        
        return result

    def sync_loading_status(self):
        """Синхронизирует статус загрузки в базе с фактическим состоянием"""
        try:
            from .services.redis_model_service import redis_model_manager
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
        
        return False  # Статус не изменился


class ModelVersion(models.Model):
    """
    Версионирование моделей для отслеживания изменений.
    Каждая версия - снимок состояния модели.
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
    
    # Снимок состояния
    config_snapshot = models.JSONField(
        help_text=_("Полная конфигурация на момент создания версии")
    )
    file_hash = models.CharField(
        max_length=64,
        help_text=_("Хеш файлов модели для проверки целостности")
    )
    
    # Метаданные
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
        """Восстанавливает эту версию как текущую"""
        self.checkpoint.loader_config = self.config_snapshot
        self.checkpoint.save()
        
        # Создаем новую версию для отслеживания восстановления
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
    """Задачи слияния моделей с расширенным отслеживанием"""
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
    
    # Основные поля
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    # Входные модели
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
    
    # Выходная модель (создается после слияния)
    output_model = models.ForeignKey(
        ModelCheckpoint,
        related_name='output_merges',
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    
    # Настройки слияния
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
    
    # Расширенная конфигурация
    merge_config = models.JSONField(
        default=dict,
        help_text=_("Дополнительные параметры слияния")
    )
    
    # Ресурсы и лимиты
    gpu_memory_limit_gb = models.FloatField(
        default=8.0,
        help_text=_("Лимит памяти GPU для слияния (GB)")
    )
    timeout_minutes = models.IntegerField(
        default=120,
        help_text=_("Таймаут выполнения в минутах")
    )
    
    # Прогресс и статус
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    progress = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
    )
    current_step = models.CharField(max_length=100, blank=True)
    
    # Логи и результаты
    logs = models.TextField(blank=True)
    error_details = models.JSONField(default=dict, help_text=_("Детали ошибки"))
    performance_metrics = models.JSONField(
        default=dict,
        help_text=_("Метрики производительности слияния")
    )
    
    # Временные метки
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    validated_at = models.DateTimeField(null=True, blank=True)
    
    # Контроль версий
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
        """Валидация задачи слияния"""
        super().clean()
        
        # Проверка совместимости моделей
        if self.base_model and self.merge_model:
            if self.base_model.model_type != self.merge_model.model_type:
                raise ValidationError(
                    f"Модели разных типов: {self.base_model.model_type} и {self.merge_model.model_type}"
                )
            
            if self.base_model.parameter_count != self.merge_model.parameter_count:
                raise ValidationError(
                    "Модели имеют разное количество параметров, слияние невозможно"
                )
        
        # Проверка метода слияния
        if self.merge_method == 'SLERP' and not (0 <= self.merge_ratio <= 1):
            raise ValidationError("Для SLERP merge_ratio должен быть между 0 и 1")
    
    def save(self, *args, **kwargs):
        """Переопределение save с автоматическим именем"""
        if not self.name:
            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
            self.name = f"merge_{timestamp}"
        
        super().save(*args, **kwargs)
    
    @property
    def duration_seconds(self):
        """Длительность выполнения в секундах"""
        if not self.started_at or not self.completed_at:
            return None
        return (self.completed_at - self.started_at).total_seconds()
    
    @property
    def is_compatible(self):
        """Проверка совместимости моделей для слияния"""
        if not self.base_model or not self.merge_model:
            return False
        
        return (
            self.base_model.model_type == self.merge_model.model_type and
            self.base_model.parameter_count == self.merge_model.parameter_count and
            self.base_model.quantization == self.merge_model.quantization
        )
    
    def start(self):
        """Начать выполнение задачи"""
        self.status = 'RUNNING'
        self.started_at = timezone.now()
        self.current_step = 'initialization'
        self.save()
    
    def complete(self, output_checkpoint: ModelCheckpoint):
        """Завершить задачу успешно"""
        self.status = 'COMPLETED'
        self.completed_at = timezone.now()
        self.output_model = output_checkpoint
        self.progress = 100.0
        self.current_step = 'completed'
        self.save()
    
    def fail(self, error_message: str, error_details: dict = None):
        """Завершить задачу с ошибкой"""
        self.status = 'FAILED'
        self.completed_at = timezone.now()
        self.error_message = error_message
        self.error_details = error_details or {}
        self.save()
    
    def update_progress(self, progress: float, step: str = ""):
        """Обновить прогресс выполнения"""
        self.progress = max(0.0, min(100.0, progress))
        if step:
            self.current_step = step
        self.save(update_fields=['progress', 'current_step', 'updated_at'])


class InferenceSession(models.Model):
    """
    Сессии инференса для мониторинга использования и производительности.
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
    
    # Пользовательская информация
    user_ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    api_key = models.CharField(max_length=100, blank=True)
    
    # Статистика использования
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    total_requests = models.IntegerField(default=0)
    
    # Производительность
    total_duration = models.FloatField(default=0.0, help_text=_("Общая длительность в секундах"))
    avg_response_time = models.FloatField(default=0.0, help_text=_("Среднее время ответа в секундах"))
    
    # Ошибки
    error_count = models.IntegerField(default=0)
    last_error = models.TextField(blank=True)
    
    # Статус
    is_active = models.BooleanField(default=True)
    
    # Временные метки
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
        """Скорость обработки токенов в секунду"""
        if self.total_duration > 0:
            return (self.input_tokens + self.output_tokens) / self.total_duration
        return 0.0
    
    @property
    def duration_minutes(self):
        """Длительность сессии в минутах"""
        if self.ended_at and self.started_at:
            return (self.ended_at - self.started_at).total_seconds() / 60
        return (timezone.now() - self.started_at).total_seconds() / 60
    
    def add_request(self, input_tokens: int, output_tokens: int, duration: float):
        """Добавить запрос к статистике"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_requests += 1
        self.total_duration += duration
        
        # Пересчет среднего
        if self.total_requests > 0:
            self.avg_response_time = self.total_duration / self.total_requests
        
        self.save()
    
    def end_session(self):
        """Завершить сессию"""
        self.is_active = False
        self.ended_at = timezone.now()
        self.save()


class ModelEvaluation(models.Model):
    """
    Результаты оценки моделей на стандартных датасетах.
    """
    checkpoint = models.ForeignKey(
        ModelCheckpoint,
        on_delete=models.CASCADE,
        related_name='evaluations'
    )
    
    # Конфигурация оценки
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
    
    # Результаты
    metrics = models.JSONField(
        default=dict,
        help_text=_("Метрики оценки {'accuracy': 0.85, 'f1': 0.82}")
    )
    raw_results = models.JSONField(
        default=dict,
        help_text=_("Сырые результаты для анализа")
    )
    
    # Системная информация
    evaluated_at = models.DateTimeField(auto_now_add=True)
    evaluator_version = models.CharField(max_length=50, blank=True)
    hardware_info = models.JSONField(
        default=dict,
        help_text=_("Информация о железе {'gpu': 'RTX 4090', 'memory_gb': 24}")
    )
    
    # Валидация
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
        """Основная метрика для данного датасета"""
        # Для разных датасетов разные основные метрики
        metric_map = {
            'mmlu': 'accuracy',
            'hellaswag': 'accuracy',
            'winogrande': 'accuracy',
            'russian_superglue': 'average',
            'humaneval': 'pass@1',
        }
        
        primary = metric_map.get(self.dataset_name, 'accuracy')
        return self.metrics.get(primary, 0.0)


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

class ModelDownloadJob(models.Model):
    """Задача загрузки модели"""
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
    """Лог удаленных моделей"""
    original_name = models.CharField(max_length=255)
    display_name = models.CharField(max_length=255)
    model_type = models.CharField(max_length=20, choices=ModelCheckpoint.MODEL_TYPE_CHOICES)
    huggingface_id = models.CharField(max_length=300, blank=True)
    
    # Информация о файлах
    local_path = models.CharField(max_length=500, blank=True)
    relative_path = models.CharField(max_length=500, blank=True)
    had_local_files = models.BooleanField(default=False)
    
    # Техническая информация
    size_gb = models.FloatField(default=0.0)
    parameter_count = models.BigIntegerField(default=0)
    
    # Информация об удалении
    deleted_by = models.CharField(max_length=100)
    deletion_reason = models.TextField(blank=True)
    deleted_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-deleted_at']
        verbose_name = 'Лог удаленной модели'
        verbose_name_plural = 'Логи удаленных моделей'
    
    def __str__(self):
        return f"{self.display_name} (удалена {self.deleted_at.strftime('%d.%m.%Y %H:%M')})"

# ============================================================================
# Модели для системы оценки (Model Evaluation System)
# ============================================================================

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
        
        try:
            if file_format == 'json' or (file_format == 'auto' and file_path.endswith('.json')):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        questions = data
                    elif isinstance(data, dict) and 'questions' in data:
                        questions = data['questions']
            
            elif file_format == 'csv' or (file_format == 'auto' and file_path.endswith('.csv')):
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader, 1):
                        if 'text' in row:
                            questions.append({
                                'id': i,
                                'text': row['text'].strip(),
                                'category': row.get('category', ''),
                                'difficulty': row.get('difficulty', 'medium')
                            })
            
            elif file_format == 'txt' or (file_format == 'auto' and file_path.endswith('.txt')):
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines, 1):
                        line = line.strip()
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

# ============================================================================
# МЕНЕДЖЕРЫ МОДЕЛЕЙ
# ============================================================================

class DocumentManager(models.Manager):
    """Кастомный менеджер для документов"""
    
    def get_queryset(self):
        """Возвращает только активные документы по умолчанию"""
        # Безопасная проверка наличия поля is_active
        try:
            # Проверяем, есть ли поле is_active в модели
            if hasattr(self.model, '_meta') and 'is_active' in [f.name for f in self.model._meta.get_fields()]:
                return super().get_queryset().filter(is_active=True)
        except Exception:
            pass
        return super().get_queryset()
    
    def all_documents(self):
        """Получить ВСЕ документы (включая неактивные)"""
        return super().get_queryset()
    
    def find_similar_by_content(self, content, limit=5, min_similarity=0.7):
        """
        Найти документы с похожим содержимым.
        В реальной системе здесь будет поиск по векторным эмбеддингам.
        """
        # Пока используем простой текстовый поиск
        documents = []
        content_words = set(content.lower().split())
        
        for doc in self.filter(status='INDEXED'):  # get_queryset уже фильтрует is_active=True
            if doc.processed_content:
                doc_words = set(doc.processed_content.lower().split())
                common_words = content_words.intersection(doc_words)
                
                if common_words:
                    similarity = len(common_words) / max(len(content_words), 1)
                    if similarity >= min_similarity:
                        documents.append({
                            'document': doc,
                            'similarity': similarity
                        })
        
        # Сортировка по схожести
        documents.sort(key=lambda x: x['similarity'], reverse=True)
        return [doc['document'] for doc in documents[:limit]]
    
    def get_by_collection_and_type(self, collection_id=None, doc_type=None):
        """Получить документы по коллекции и типу"""
        queryset = self.all()  # Используем self.all() чтобы получить активные
        
        if collection_id:
            queryset = queryset.filter(collection_id=collection_id)
        
        if doc_type:
            queryset = queryset.filter(document_type=doc_type)
        
        return queryset
    
    def get_statistics(self):
        """Получить статистику по документам"""
        from django.db.models import Count, Avg, Sum
        
        stats = self.all().aggregate(  # Используем self.all() чтобы получить активные
            total=Count('id'),
            by_type=Count('document_type'),
            avg_size=Avg('file_size_mb'),
            total_size=Sum('file_size_mb'),
            indexed=Count('id', filter=Q(status='INDEXED')),
            processing=Count('id', filter=Q(status='PROCESSING')),
            error=Count('id', filter=Q(status='ERROR')),
        )
        
        return stats

# ============================================================================
# МОДЕЛИ ДЛЯ ДОКУМЕНТОВ И ИХ ПРЕДСТАВЛЕНИЙ
# ============================================================================

class DocumentCollection(models.Model):
    """
    Коллекция документов для RAG.
    Группирует связанные документы (например, по проекту, пользователю, теме).
    """
    name = models.CharField(max_length=255, help_text=_("Название коллекции"))
    description = models.TextField(blank=True, help_text=_("Описание коллекции"))
    
    # Владелец и доступ
    owner_id = models.CharField(max_length=100, blank=True, help_text=_("ID владельца (пользователя/организации)"))
    is_public = models.BooleanField(default=False, help_text=_("Доступна ли коллекция другим пользователям"))
    
    # Модель эмбеддингов
    embedding_model = models.ForeignKey(
        'EmbeddingModel',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='collections',
        help_text=_("Модель эмбеддингов для этой коллекции")
    )
    
    # Статистика векторного индекса
    vector_index_created = models.BooleanField(default=False, help_text=_("Создан ли векторный индекс"))
    total_vectors = models.IntegerField(default=0, help_text=_("Количество векторов в индексе"))
    vector_dimension = models.IntegerField(null=True, blank=True, help_text=_("Размерность векторов"))
    index_metadata = models.JSONField(default=dict, blank=True, help_text=_("Метаданные векторного индекса"))
    
    # Настройки обработки
    chunking_strategy = models.CharField(
        max_length=50,
        default='recursive',
        choices=[
            ('recursive', _('Рекурсивное разделение')),
            ('semantic', _('Семантическое разделение')),
            ('fixed', _('Фиксированный размер')),
            ('paragraph', _('По параграфам')),
        ],
        help_text=_("Стратегия разделения на чанки")
    )
    
    chunk_size = models.IntegerField(default=1000, help_text=_("Размер чанка в токенах/символах"))
    chunk_overlap = models.IntegerField(default=200, help_text=_("Перекрытие между чанками"))
    
    # Метаданные фильтрации
    metadata_filters = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Фильтры по метаданным для этой коллекции {'doc_type': 'legal', 'year': 2023}")
    )
    
    # Статистика
    total_documents = models.IntegerField(default=0, help_text=_("Общее количество документов"))
    total_chunks = models.IntegerField(default=0, help_text=_("Общее количество чанков"))
    total_size_mb = models.FloatField(default=0.0, help_text=_("Общий размер документов в МБ"))
    
    # Системные поля
    is_active = models.BooleanField(default=True)
    auto_process = models.BooleanField(default=True, help_text=_("Автоматически обрабатывать новые документы"))
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['owner_id', 'is_active']),
            models.Index(fields=['is_public', 'is_active']),
            models.Index(fields=['embedding_model', 'is_active']),
            models.Index(fields=['vector_index_created']),
        ]
        verbose_name = 'Коллекция документов'
        verbose_name_plural = 'Коллекции документов'
    
    def __str__(self):
        return f"{self.name} ({self.total_documents} док.)"
    
    def clean(self):
        """Валидация коллекции"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValidationError("Перекрытие чанков должно быть меньше их размера")
    
    def update_statistics(self):
        """Обновить статистику коллекции"""
        from django.db.models import Count, Sum
        
        docs = self.documents.filter(is_active=True)
        self.total_documents = docs.count()
        
        # Считаем общее количество чанков и размер
        chunks_info = docs.aggregate(
            total_chunks=Count('chunks', filter=Q(chunks__is_active=True)),
            total_size=Sum('file_size_mb')
        )
        
        self.total_chunks = chunks_info['total_chunks'] or 0
        self.total_size_mb = chunks_info['total_size'] or 0.0
        
        # Считаем количество векторов (чанков с эмбеддингами)
        chunks_with_embeddings = DocumentChunk.objects.filter(
            document__collection=self,
            has_embedding=True,
            is_active=True
        ).count()
        
        self.total_vectors = chunks_with_embeddings
        
        # Если есть векторы, устанавливаем размерность
        if chunks_with_embeddings > 0 and not self.vector_dimension:
            chunk_with_embedding = DocumentChunk.objects.filter(
                document__collection=self,
                has_embedding=True,
                is_active=True
            ).first()
            if chunk_with_embedding and chunk_with_embedding.embedding_model:
                self.vector_dimension = chunk_with_embedding.embedding_model.embedding_dim
        
        self.save(update_fields=['total_documents', 'total_chunks', 'total_size_mb', 
                                'total_vectors', 'vector_dimension', 'updated_at'])
    
    def get_embedding_progress(self):
        """Получить прогресс создания эмбеддингов"""
        chunks_with_embeddings = DocumentChunk.objects.filter(
            document__collection=self,
            has_embedding=True,
            is_active=True
        ).count()
        
        total_chunks = DocumentChunk.objects.filter(
            document__collection=self,
            is_active=True
        ).count()
        
        documents_with_embeddings = Document.objects.filter(
            collection=self,
            is_active=True,
            chunks__has_embedding=True
        ).distinct().count()
        
        total_documents = self.documents.filter(is_active=True).count()
        
        return {
            'chunks': {
                'with_embeddings': chunks_with_embeddings,
                'total': total_chunks,
                'percent': (chunks_with_embeddings / total_chunks * 100) if total_chunks > 0 else 0
            },
            'documents': {
                'with_embeddings': documents_with_embeddings,
                'total': total_documents,
                'percent': (documents_with_embeddings / total_documents * 100) if total_documents > 0 else 0
            }
        }

    def build_vector_index(self, index_type='ivf'):
        """Запустить создание векторного индекса"""
        try:
            from .tasks import build_faiss_index_task
            task = build_faiss_index_task.delay(self.id, index_type)
            return task.id
        except Exception as e:
            logger.error(f"Error starting index build for collection {self.id}: {str(e)}")
            return None

    def search_in_index(self, query_vector: list, k: int = 10):
        """Поиск в векторном индексе"""
        try:
            from .tasks import search_faiss_index_task
            result = search_faiss_index_task.delay(self.id, query_vector, k)
            return result
        except Exception as e:
            logger.error(f"Error starting search for collection {self.id}: {str(e)}")
            return None

    def update_vector_index(self):
        """Обновить векторный индекс"""
        try:
            from .tasks import update_faiss_index_task
            task = update_faiss_index_task.delay(self.id)
            return task.id
        except Exception as e:
            logger.error(f"Error starting index update for collection {self.id}: {str(e)}")
            return None

    def get_index_stats(self):
        """Получить статистику векторного индекса"""
        try:
            from .services.faiss_service import FAISSService
            faiss_service = FAISSService()
            return faiss_service.get_index_stats(self.id)
        except Exception as e:
            logger.error(f"Error getting index stats for collection {self.id}: {str(e)}")
            return {'error': str(e)}

    @property
    def owner(self):
        """Получить объект владельца"""
        from django.contrib.auth.models import User
        try:
            if self.owner_id:
                return User.objects.get(id=int(self.owner_id))
        except (ValueError, User.DoesNotExist):
            return None
    
    @property
    def owner_username(self):
        """Имя владельца"""
        owner = self.owner
        return owner.username if owner else self.owner_id or "Система"
    
    def get_owner_display(self):
        """Отображаемое имя владельца"""
        if self.owner:
            return f"{self.owner.username} ({self.owner.get_full_name() or self.owner.email})"
        return self.owner_id or "Система"

    def get_effective_embedding_model(self):
        """
        Возвращает модель эмбеддингов, которая будет использоваться для этой коллекции.
        Если у коллекции явно не задана модель, возвращается глобальная модель по умолчанию.
        """
        if self.embedding_model:
            return self.embedding_model
        return EmbeddingModel.objects.filter(is_default=True).first()
        

class Document(models.Model):
    """
    Основная модель документа для RAG.
    """
    DOCUMENT_STATUS_CHOICES = [
        ('UPLOADED', 'Загружен'),
        ('PROCESSING', 'В обработке'),
        ('CHUNKED', 'Разделен на чанки'),
        ('EMBEDDED', 'Эмбеддинги созданы'),
        ('INDEXED', 'Проиндексирован'),
        ('ERROR', 'Ошибка'),
        ('ARCHIVED', 'В архиве'),
    ]
    
    DOCUMENT_TYPE_CHOICES = [
        ('TEXT', 'Текст'),
        ('PDF', 'PDF'),
        ('DOCX', 'Word'),
        ('HTML', 'HTML'),
        ('MARKDOWN', 'Markdown'),
        ('CODE', 'Исходный код'),
        ('CSV', 'Таблица CSV'),
        ('EXCEL', 'Excel'),
        ('UNKNOWN', 'Неизвестный'),
    ]
    
    # Основная информация
    original_filename = models.CharField(max_length=500, help_text=_("Оригинальное имя файла"))
    title = models.CharField(max_length=500, blank=True, help_text=_("Название документа"))
    description = models.TextField(blank=True, help_text=_("Описание документа"))
    
    # Тип и содержимое
    document_type = models.CharField(max_length=20, choices=DOCUMENT_TYPE_CHOICES, default='UNKNOWN')
    content_hash = models.CharField(max_length=64, help_text=_("SHA256 хеш содержимого для дедупликации"))
    raw_content = models.TextField(blank=True, help_text=_("Исходное текстовое содержимое"))
    processed_content = models.TextField(blank=True, help_text=_("Очищенное содержимое после обработки"))
    
    # Пути и размеры
    file_path = models.CharField(max_length=1000, blank=True, help_text=_("Путь к исходному файлу"))
    file_size_mb = models.FloatField(default=0.0, help_text=_("Размер файла в МБ"))
    file_encoding = models.CharField(max_length=50, default='utf-8', help_text=_("Кодировка файла"))
    
    # Метаданные
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Произвольные метаданные документа")
    )
    
    # Теги и категории
    tags = models.JSONField(default=list, blank=True, help_text=_("Теги документа"))
    category = models.CharField(max_length=100, blank=True, help_text=_("Категория документа"))
    
    # Статус и обработка
    status = models.CharField(max_length=20, choices=DOCUMENT_STATUS_CHOICES, default='UPLOADED')
    processing_progress = models.FloatField(default=0.0, help_text=_("Прогресс обработки (0-100)"))
    error_message = models.TextField(blank=True, help_text=_("Сообщение об ошибке обработки"))
    
    # Качество и релевантность
    quality_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Оценка качества документа (0-1)")
    )
    
    # Связи
    collection = models.ForeignKey(
        DocumentCollection,
        on_delete=models.CASCADE,
        related_name='documents',
        null=True,
        blank=True
    )
    
    # Временные метки
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    indexed_at = models.DateTimeField(null=True, blank=True)
    
    # Системные поля
    is_active = models.BooleanField(default=True)
    requires_reindexing = models.BooleanField(default=False, help_text=_("Требуется переиндексация"))
    objects = DocumentManager()

    class Meta:
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['content_hash']),  # Для дедупликации
            models.Index(fields=['status', 'uploaded_at']),
            models.Index(fields=['collection', 'status']),
            models.Index(fields=['document_type', 'is_active']),
            models.Index(fields=['quality_score']),
            models.Index(fields=['requires_reindexing']),  # Для поиска документов для переиндексации
        ]
        verbose_name = 'Документ'
        verbose_name_plural = 'Документы'
    
    def __str__(self):
        return f"{self.title or self.original_filename} ({self.get_status_display()})"
    
    def clean(self):
        """Валидация документа"""
        # Автоматическое определение типа по расширению
        if not self.document_type or self.document_type == 'UNKNOWN':
            ext = os.path.splitext(self.original_filename)[1].lower()
            ext_to_type = {
                '.txt': 'TEXT',
                '.pdf': 'PDF',
                '.docx': 'DOCX',
                '.doc': 'DOCX',
                '.html': 'HTML',
                '.htm': 'HTML',
                '.md': 'MARKDOWN',
                '.py': 'CODE',
                '.js': 'CODE',
                '.java': 'CODE',
                '.cpp': 'CODE',
                '.csv': 'CSV',
                '.xlsx': 'EXCEL',
                '.xls': 'EXCEL',
            }
            self.document_type = ext_to_type.get(ext, 'UNKNOWN')
    
    def save(self, *args, **kwargs):
        """Переопределение save с автоматическим вычислением хеша"""
        # Вычисляем хеш содержимого если есть raw_content
        if self.raw_content and not self.content_hash:
            self.content_hash = hashlib.sha256(self.raw_content.encode()).hexdigest()
        
        # Автоматически задаем заголовок если не задан
        if not self.title and self.original_filename:
            self.title = os.path.splitext(self.original_filename)[0]
        
        super().save(*args, **kwargs)
        
        # Обновляем статистику коллекции
        if self.collection:
            self.collection.update_statistics()
    
    def file_url(self):
        """URL для доступа к файлу через медиа"""
        if self.file_path:
            # Убираем дублирование 'documents/' если оно есть
            if self.file_path.startswith('documents/'):
                # Уже содержит папку documents - используем как есть
                return f"{settings.MEDIA_URL}{self.file_path}"
            else:
                # Добавляем папку documents
                return f"{settings.MEDIA_URL}documents/{self.file_path}"
        return None

    @property
    def full_file_path(self):
        """Полный путь к файлу на диске"""
        if not self.file_path:
            return None
        
        # Логирование для отладки
        import logging
        logger = logging.getLogger(__name__)
        
        # Если путь уже абсолютный
        if os.path.isabs(self.file_path):
            logger.debug(f"Absolute path for doc {self.id}: {self.file_path}")
            return self.file_path
        
        # Если начинается с 'documents/'
        if self.file_path.startswith('documents/'):
            # Убираем возможное дублирование 'media/' если есть
            path = self.file_path
            if path.startswith('media/') and path.count('media/') > 1:
                path = 'media/' + path.split('media/')[-1]
            
            # Строим путь относительно MEDIA_ROOT
            full_path = os.path.join(settings.MEDIA_ROOT, path)
            logger.debug(f"Path with 'documents/' for doc {self.id}: {full_path}")
            return full_path
        
        # Просто имя файла - добавляем 'documents/'
        full_path = os.path.join(settings.MEDIA_ROOT, 'documents', self.file_path)
        logger.debug(f"Path with added 'documents/' for doc {self.id}: {full_path}")
        return full_path
    
    def file_exists(self):
        """Проверяет существование файла"""
        full_path = self.full_file_path
        return full_path and os.path.exists(full_path)
    
    def get_file_response(self):
        """Возвращает FileResponse для скачивания"""
        full_path = self.full_file_path
        if not full_path or not os.path.exists(full_path):
            # Попробуем найти файл в других возможных местах
            possible_paths = [
                full_path,
                os.path.join(settings.MEDIA_ROOT, self.file_path) if self.file_path else None,
                os.path.join(settings.MEDIA_ROOT, 'documents', str(self.collection.id), self.original_filename) if self.collection else None,
                os.path.join(settings.MEDIA_ROOT, 'documents', self.original_filename),
            ]
            
            for path in possible_paths:
                if path and os.path.exists(path):
                    full_path = path
                    break
            
            if not full_path or not os.path.exists(full_path):
                return None
        
        try:
            from django.http import FileResponse
            return FileResponse(
                open(full_path, 'rb'),
                as_attachment=True,
                filename=self.original_filename
            )
        except Exception as e:
            logger.error(f"Error creating file response: {str(e)}")
            return None
            
    @property
    def extension(self):
        """Расширение файла"""
        return os.path.splitext(self.original_filename)[1].lower()
    
    @property
    def has_embeddings(self):
        """Есть ли у документа эмбеддинги"""
        return self.chunks.filter(has_embedding=True).exists()
    
    @property
    def chunk_count(self):
        """Количество чанков документа"""
        return self.chunks.filter(is_active=True).count()
    
    def get_similar_documents(self, limit=5):
        """Найти похожие документы (по эмбеддингам чанков)"""
        # Находим эмбеддинги этого документа
        chunk_embeddings = list(self.chunks.filter(has_embedding=True).values_list('embedding_vector', flat=True))
        
        if not chunk_embeddings:
            return Document.objects.none()
        
        # Здесь будет логика поиска похожих документов через векторную БД
        # Пока возвращаем заглушку
        return Document.objects.filter(
            collection=self.collection,
            is_active=True
        ).exclude(id=self.id)[:limit]
    
    def start_processing(self):
        """Запустить обработку документа"""
        try:
            from .tasks import process_document_task
            return process_document_task.delay(self.id)
        except ImportError as e:
            logger.warning(f"Tasks module not available for start_processing: {e}")
            return None

    @property
    def chunks_with_embeddings(self):
        """Количество чанков с эмбеддингами"""
        return self.chunks.filter(has_embedding=True, is_active=True).count()
    
    @property
    def chunks_with_embeddings_from_collection_model(self):
        """Количество чанков с эмбеддингами от модели коллекции"""
        if not self.collection or not self.collection.embedding_model:
            return 0
        return self.chunks.filter(
            has_embedding=True,
            embedding_model=self.collection.embedding_model,
            is_active=True
        ).count()
    
    @property
    def embedding_model_used(self):
        """Модель эмбеддингов, используемая для этого документа."""
        # Проверяем, есть ли чанки с эмбеддингами
        chunk_with_emb = self.chunks.filter(
            has_embedding=True,
            is_active=True
        ).first()
        
        if chunk_with_emb and chunk_with_emb.embedding_model:
            return chunk_with_emb.embedding_model
        
        # Если чанков с эмбеддингами нет, возвращаем модель коллекции
        return self.collection.embedding_model if self.collection else None
    
    @property
    def embedding_dimension(self):
        """Размерность эмбеддингов документа"""
        model = self.embedding_model_used
        return model.embedding_dim if model else None
    
    @property
    def has_embeddings(self):
        """Есть ли у документа эмбеддинги (любые)"""
        return self.chunks.filter(has_embedding=True, is_active=True).exists()
    
    @property
    def has_complete_embeddings(self):
        """Все ли чанки имеют эмбеддинги"""
        total_chunks = self.chunks.filter(is_active=True).count()
        if total_chunks == 0:
            return False
        return self.chunks_with_embeddings == total_chunks


class DocumentChunk(models.Model):
    """
    Фрагмент (чанк) документа для векторного поиска.
    """
    CHUNK_STRATEGY_CHOICES = [
        ('recursive', _('Рекурсивный')),
        ('semantic', _('Семантический')),
        ('fixed', _('Фиксированный')),
        ('paragraph', _('Абзацный')),
        ('table', _('Табличный')),
        ('code', _('Код')),
    ]
    
    # Связь с документом
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='chunks'
    )
    
    # Содержимое и позиция
    chunk_index = models.IntegerField(help_text=_("Порядковый номер чанка в документе"))
    content = models.TextField(help_text=_("Текстовое содержимое чанка"))
    token_count = models.IntegerField(default=0, help_text=_("Количество токенов в чанке"))
    
    # Позиция в исходном документе
    start_char = models.IntegerField(help_text=_("Начальная позиция в исходном тексте"))
    end_char = models.IntegerField(help_text=_("Конечная позиция в исходном тексте"))
    start_line = models.IntegerField(null=True, blank=True, help_text=_("Начальная строка (если применимо)"))
    end_line = models.IntegerField(null=True, blank=True, help_text=_("Конечная строка (если применимо)"))
    
    # Метаданные чанка
    metadata = models.JSONField(default=dict, blank=True, help_text=_("Метаданные чанка"))
    chunk_strategy = models.CharField(
        max_length=20,
        choices=CHUNK_STRATEGY_CHOICES,
        default='recursive'
    )
    
    # Эмбеддинги
    has_embedding = models.BooleanField(default=False, help_text=_("Создан ли эмбеддинг для чанка"))
    embedding_model = models.ForeignKey(
        'EmbeddingModel',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='chunks'
    )
    embedding_version = models.CharField(max_length=50, blank=True, help_text=_("Версия модели эмбеддингов"))
    
    # Векторное представление (в реальной системе это может храниться в векторной БД)
    embedding_vector = models.JSONField(
        null=True,
        blank=True,
        help_text=_("Вектор эмбеддинга (JSON список чисел)")
    )
    embedding_dim = models.IntegerField(null=True, blank=True, help_text=_("Размерность эмбеддинга"))
    
    # Качество и релевантность
    relevance_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Оценка релевантности чанка")
    )
    is_key_chunk = models.BooleanField(default=False, help_text=_("Ключевой чанк документа"))
    
    # Статистика использования
    usage_count = models.IntegerField(default=0, help_text=_("Сколько раз чанк был использован в поиске"))
    last_used = models.DateTimeField(null=True, blank=True)
    
    # Системные поля
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['document', 'chunk_index']
        indexes = [
            models.Index(fields=['document', 'chunk_index']),
            models.Index(fields=['has_embedding', 'is_active']),
            models.Index(fields=['embedding_model', 'has_embedding']),
            models.Index(fields=['relevance_score']),
            models.Index(fields=['is_key_chunk']),
        ]
        verbose_name = 'Фрагмент документа'
        verbose_name_plural = 'Фрагменты документов'
    
    def __str__(self):
        return f"Чанк {self.chunk_index} документа '{self.document.title}'"
    
    @property
    def preview(self):
        """Короткий превью содержимого"""
        return self.content[:100] + ('...' if len(self.content) > 100 else '')
    
    @property
    def document_title(self):
        """Название родительского документа"""
        return self.document.title
    
    def mark_used(self):
        """Отметить использование чанка"""
        self.usage_count += 1
        self.last_used = timezone.now()
        self.save(update_fields=['usage_count', 'last_used', 'updated_at'])
    
    def get_similar_chunks(self, limit=5):
        """Найти похожие чанки (по эмбеддингам)"""
        if not self.has_embedding or not self.embedding_vector:
            return DocumentChunk.objects.none()
        
        # В реальной системе здесь будет поиск по векторной БД
        # Пока возвращаем чанки из того же документа
        return DocumentChunk.objects.filter(
            document=self.document,
            is_active=True,
            has_embedding=True
        ).exclude(id=self.id).order_by('chunk_index')[:limit]


class EmbeddingModel(models.Model):
    """
    Модель для создания эмбеддингов (векторных представлений текста).
    """
    MODEL_TYPE_CHOICES = [
        ('LOCAL', 'Локальная модель'),
        ('OPENAI', 'OpenAI API'),
        ('COHERE', 'Cohere API'),
        ('HUGGINGFACE', 'Hugging Face'),
        ('SENTENCE_TRANSFORMERS', 'Sentence Transformers'),
    ]
    
    name = models.CharField(max_length=255, help_text=_("Название модели"))
    display_name = models.CharField(max_length=255, blank=True, help_text=_("Человекочитаемое имя для интерфейса"))
    model_type = models.CharField(max_length=30, choices=MODEL_TYPE_CHOICES, default='HUGGINGFACE')
    
    # Идентификаторы моделей
    model_id = models.CharField(max_length=500, help_text=_("Идентификатор модели (на HF, в API и т.д.)"))
    model_version = models.CharField(max_length=100, default='latest', help_text=_("Версия модели"))
    
    # Технические характеристики
    embedding_dim = models.IntegerField(help_text=_("Размерность эмбеддингов"))
    max_tokens = models.IntegerField(default=512, help_text=_("Максимальное количество токенов на вход"))
    is_multilingual = models.BooleanField(default=False, help_text=_("Поддерживает ли несколько языков"))
    
    # Производительность
    speed_ms_per_token = models.FloatField(null=True, blank=True, help_text=_("Скорость обработки (мс/токен)"))
    accuracy_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Оценка точности на тестовых данных")
    )
    
    # Конфигурация
    config = models.JSONField(default=dict, blank=True, help_text=_("Конфигурация модели"))
    
    # Лицензия и стоимость
    license_type = models.CharField(max_length=100, blank=True, help_text=_("Тип лицензии"))
    cost_per_1k_tokens = models.FloatField(null=True, blank=True, help_text=_("Стоимость за 1000 токенов (если API)"))
    
    # Статистика использования
    total_tokens_processed = models.BigIntegerField(default=0, help_text=_("Всего обработано токенов"))
    total_requests = models.IntegerField(default=0, help_text=_("Всего запросов к модели"))
    
    # Статус
    is_active = models.BooleanField(default=True, help_text=_("Доступна для использования"))
    is_default = models.BooleanField(default=False, help_text=_("Модель по умолчанию для новых коллекций"))
    is_downloaded = models.BooleanField(default=False, help_text=_("Скачана ли модель локально"))
    
    # Информация о загрузке
    download_progress = models.IntegerField(default=0, help_text=_("Прогресс загрузки (0-100)"))
    download_status = models.CharField(
        max_length=20,
        default='not_started',
        choices=[
            ('not_started', _('Не начата')),
            ('downloading', _('Загружается')),
            ('completed', _('Завершена')),
            ('failed', _('Ошибка')),
        ]
    )
    download_error = models.TextField(blank=True, help_text=_("Ошибка загрузки, если была"))
    
    # Временные метки
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-is_default', '-is_active', 'name']
        indexes = [
            models.Index(fields=['model_type', 'is_active']),
            models.Index(fields=['is_default']),
            models.Index(fields=['is_downloaded']),
        ]
        verbose_name = 'Модель эмбеддингов'
        verbose_name_plural = 'Модели эмбеддингов'
    
    def __str__(self):
        return f"{self.display_name or self.name} ({self.embedding_dim}d)"
    
    def clean(self):
        """Валидация модели"""
        if self.is_default and self.pk:
            # Если делаем эту модель моделью по умолчанию, снимаем флаг с других
            EmbeddingModel.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)
    
    def save(self, *args, **kwargs):
        """Переопределение save для обработки is_default"""
        # Если это первая модель и is_default не задан, устанавливаем его
        if not self.pk and not EmbeddingModel.objects.filter(is_default=True).exists():
            self.is_default = True
        
        super().save(*args, **kwargs)
        
        # Если установлен is_default, снимаем его с других моделей
        if self.is_default:
            EmbeddingModel.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)


# ============================================================================
# 2. МОДЕЛИ ДЛЯ ПОИСКА И ИНФЕРЕНСА
# ============================================================================

class SearchQuery(models.Model):
    """
    Поисковый запрос пользователя.
    """
    SEARCH_TYPE_CHOICES = [
        ('SEMANTIC', 'Семантический (по эмбеддингам)'),
        ('KEYWORD', 'По ключевым словам'),
        ('HYBRID', 'Гибридный'),
        ('FILTER', 'Фильтрация по метаданным'),
    ]
    
    # Основная информация
    query_text = models.TextField(help_text=_("Текст поискового запроса"))
    normalized_query = models.TextField(blank=True, help_text=_("Нормализованный текст запроса"))
    
    # Тип поиска и конфигурация
    search_type = models.CharField(max_length=20, choices=SEARCH_TYPE_CHOICES, default='HYBRID')
    search_config = models.JSONField(default=dict, blank=True, help_text=_("Конфигурация поиска"))
    
    # Контекст запроса
    session_id = models.CharField(max_length=100, blank=True, help_text=_("ID сессии пользователя"))
    user_id = models.CharField(max_length=100, blank=True, help_text=_("ID пользователя"))
    context = models.JSONField(default=dict, blank=True, help_text=_("Дополнительный контекст запроса"))
    
    # Фильтры
    collection_filter = models.ForeignKey(
        DocumentCollection,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text=_("Фильтр по коллекции")
    )
    metadata_filters = models.JSONField(default=dict, blank=True, help_text=_("Фильтры по метаданным"))
    
    # Временные метки
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
    Результат поиска по запросу.
    Связывает запрос, найденные чанки и их релевантность.
    """
    # Связи
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
    
    # Релевантность и ранжирование
    relevance_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Оценка релевантности (0-1)")
    )
    rank_position = models.IntegerField(help_text=_("Позиция в результатах поиска"))
    
    # Метод вычисления релевантности
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
    
    # Дополнительные метрики
    semantic_score = models.FloatField(null=True, blank=True, help_text=_("Семантическая оценка"))
    keyword_score = models.FloatField(null=True, blank=True, help_text=_("Оценка по ключевым словам"))
    
    # Контекст результата
    context_before = models.TextField(blank=True, help_text=_("Контекст перед чанком"))
    context_after = models.TextField(blank=True, help_text=_("Контекст после чанка"))
    
    # Метаданные
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
        """Название документа чанка"""
        return self.chunk.document.title
    
    @property
    def chunk_preview(self):
        """Превью содержимого чанка"""
        return self.chunk.preview


class RAGSession(models.Model):
    """
    Сессия RAG: объединяет запрос, поиск, генерацию ответа и обратную связь.
    """
    SESSION_STATUS_CHOICES = [
        ('SEARCHING', 'Поиск документов'),
        ('GENERATING', 'Генерация ответа'),
        ('COMPLETED', 'Завершена'),
        ('FAILED', 'Ошибка'),
        ('EVALUATING', 'Оценка качества'),
    ]
    
    # Идентификаторы
    session_id = models.CharField(max_length=100, unique=True, help_text=_("Уникальный ID сессии"))
    conversation_id = models.CharField(max_length=100, blank=True, help_text=_("ID беседы (если часть диалога)"))
    
    # Пользовательская информация
    user_id = models.CharField(max_length=100, blank=True)
    user_ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    
    # Контекст и промпт
    original_query = models.TextField(help_text=_("Исходный запрос пользователя"))
    system_prompt = models.TextField(blank=True, help_text=_("Системный промпт для LLM"))
    augmented_prompt = models.TextField(blank=True, help_text=_("Промпт с контекстом из поиска"))
    
    # Модели
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
    
    # Поисковые результаты
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
    
    # Генерация ответа
    generated_answer = models.TextField(blank=True, help_text=_("Сгенерированный ответ"))
    generation_metadata = models.JSONField(default=dict, blank=True, help_text=_("Метаданные генерации"))
    
    # Статус и производительность
    status = models.CharField(max_length=20, choices=SESSION_STATUS_CHOICES, default='SEARCHING')
    total_time_ms = models.FloatField(default=0.0, help_text=_("Общее время выполнения (мс)"))
    search_time_ms = models.FloatField(null=True, blank=True, help_text=_("Время поиска (мс)"))
    generation_time_ms = models.FloatField(null=True, blank=True, help_text=_("Время генерации (мс)"))
    
    # Токены и стоимость
    input_tokens = models.IntegerField(default=0, help_text=_("Входные токены"))
    output_tokens = models.IntegerField(default=0, help_text=_("Выходные токены"))
    estimated_cost = models.FloatField(null=True, blank=True, help_text=_("Примерная стоимость"))
    
    # Качество ответа
    confidence_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Уверенность модели в ответе")
    )
    
    # Обратная связь
    has_feedback = models.BooleanField(default=False, help_text=_("Есть ли обратная связь"))
    feedback_score = models.IntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text=_("Оценка пользователя (1-5)")
    )
    feedback_comment = models.TextField(blank=True, help_text=_("Комментарий пользователя"))
    
    # Временные метки
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
        """Успешна ли сессия"""
        return self.status == 'COMPLETED' and self.generated_answer
    
    @property
    def total_tokens(self):
        """Общее количество токенов"""
        return self.input_tokens + self.output_tokens
    
    def mark_search_completed(self, search_time_ms, search_query):
        """Отметить завершение поиска"""
        self.status = 'GENERATING'
        self.search_time_ms = search_time_ms
        self.search_query = search_query
        self.save()
    
    def mark_generation_completed(self, answer, metadata, generation_time_ms, tokens):
        """Отметить завершение генерации"""
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
        """Добавить обратную связь от пользователя"""
        self.has_feedback = True
        self.feedback_score = score
        self.feedback_comment = comment
        self.feedback_at = timezone.now()
        self.save()


class RetrievedChunk(models.Model):
    """
    Промежуточная модель для связи RAGSession и DocumentChunk.
    Хранит дополнительную информацию о том, как чанк был использован в сессии.
    """
    session = models.ForeignKey(RAGSession, on_delete=models.CASCADE)
    chunk = models.ForeignKey(DocumentChunk, on_delete=models.CASCADE)
    
    # Позиция в результатах
    retrieval_rank = models.IntegerField(help_text=_("Позиция при извлечении"))
    relevance_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Релевантность для этого запроса")
    )
    
    # Как чанк был использован
    used_in_context = models.BooleanField(default=False, help_text=_("Был ли использован в контексте для LLM"))
    context_position = models.IntegerField(null=True, blank=True, help_text=_("Позиция в контексте"))
    
    # Качество использования
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


# ============================================================================
# 3. МОДЕЛИ ДЛЯ ОЦЕНКИ КАЧЕСТВА И ОБУЧЕНИЯ
# ============================================================================

class SearchQualityFeedback(models.Model):
    """
    Обратная связь по качеству поиска от пользователей.
    Используется для улучшения ранжирования и релевантности.
    """
    FEEDBACK_TYPE_CHOICES = [
        ('RELEVANCE', 'Релевантность'),
        ('COMPLETENESS', 'Полнота'),
        ('FRESHNESS', 'Актуальность'),
        ('DIVERSITY', 'Разнообразие'),
        ('OTHER', 'Другое'),
    ]
    
    # Связи
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
    
    # Оценка
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_TYPE_CHOICES, default='RELEVANCE')
    rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text=_("Оценка качества (1-5)")
    )
    
    # Детали обратной связи
    comment = models.TextField(blank=True, help_text=_("Комментарий пользователя"))
    expected_document = models.ForeignKey(
        Document,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text=_("Какой документ ожидался")
    )
    
    # Контекст
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
    
    # Системная информация
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
        """Пометить обратную связь как обработанную"""
        self.is_resolved = True
        self.resolution_notes = notes
        self.resolved_at = timezone.now()
        self.save()


class RAGEvaluation(models.Model):
    """
    Систематическая оценка качества RAG-системы.
    """
    EVALUATION_TYPE_CHOICES = [
        ('AUTOMATED', 'Автоматическая'),
        ('MANUAL', 'Ручная'),
        ('AB_TEST', 'A/B тест'),
        ('USER_STUDY', 'Исследование с пользователями'),
    ]
    
    name = models.CharField(max_length=255, help_text=_("Название оценки"))
    description = models.TextField(blank=True, help_text=_("Описание оценки"))
    
    # Конфигурация
    evaluation_type = models.CharField(max_length=20, choices=EVALUATION_TYPE_CHOICES, default='AUTOMATED')
    test_queries = models.JSONField(
        default=list,
        help_text=_("Тестовые запросы [{'id': 1, 'text': '...', 'expected_docs': [doc_ids]}]")
    )
    
    # Метрики
    target_metrics = models.JSONField(
        default=dict,
        help_text=_("Целевые метрики {'recall@5': 0.8, 'precision@3': 0.9}")
    )
    
    # Результаты
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
    
    # Сравнение
    baseline_evaluation = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text=_("Базовая оценка для сравнения")
    )
    improvement_percentage = models.FloatField(null=True, blank=True, help_text=_("Улучшение в %"))
    
    # Статус
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
    
    # Временные метки
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
        """Количество тестовых запросов"""
        return len(self.test_queries) if self.test_queries else 0
    
    def start_evaluation(self):
        """Начать оценку"""
        self.status = 'RUNNING'
        self.started_at = timezone.now()
        self.save()
    
    def complete_evaluation(self, metrics, detailed_results):
        """Завершить оценку"""
        self.status = 'COMPLETED'
        self.completed_at = timezone.now()
        self.actual_metrics = metrics
        self.detailed_results = detailed_results
        
        # Вычисляем улучшение если есть базовая оценка
        if self.baseline_evaluation and self.baseline_evaluation.actual_metrics:
            self._calculate_improvement()
        
        self.save()
    
    def _calculate_improvement(self):
        """Вычислить улучшение относительно базовой оценки"""
        try:
            baseline_metrics = self.baseline_evaluation.actual_metrics
            
            improvements = {}
            for metric, value in self.actual_metrics.items():
                if metric in baseline_metrics and baseline_metrics[metric] != 0:
                    baseline = baseline_metrics[metric]
                    improvement = ((value - baseline) / baseline) * 100
                    improvements[metric] = round(improvement, 2)
            
            if improvements:
                # Среднее улучшение по всем метрикам
                self.improvement_percentage = sum(improvements.values()) / len(improvements)
                self.actual_metrics['improvements'] = improvements
        except Exception as e:
            logger.error(f"Error calculating improvement: {e}")


# ============================================================================
# 4. СЕРВИСНЫЕ МОДЕЛИ И УТИЛИТЫ
# ============================================================================

class ProcessingPipeline(models.Model):
    """
    Конфигурация пайплайна обработки документов.
    """
    name = models.CharField(max_length=255, unique=True, help_text=_("Название пайплайна"))
    description = models.TextField(blank=True, help_text=_("Описание пайплайна"))
    
    # Этапы обработки
    steps = models.JSONField(
        default=list,
        help_text=_("Список этапов обработки [{'name': 'extract', 'module': '...', 'config': {...}}]")
    )
    
    # Конфигурация
    config = models.JSONField(default=dict, blank=True, help_text=_("Общая конфигурация пайплайна"))
    
    # Производительность
    avg_processing_time_ms = models.FloatField(null=True, blank=True, help_text=_("Среднее время обработки документа"))
    success_rate = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Процент успешной обработки")
    )
    
    # Статус
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
        """Валидация пайплайна"""
        if self.is_default and self.pk:
            ProcessingPipeline.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)


class DocumentProcessor(models.Model):
    """
    Лог обработки документов для мониторинга и отладки.
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
    
    # Статус и прогресс
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
    
    # Ресурсы и производительность
    worker_id = models.CharField(max_length=100, blank=True, help_text=_("ID воркера"))
    memory_usage_mb = models.FloatField(null=True, blank=True, help_text=_("Использование памяти (МБ)"))
    processing_time_ms = models.FloatField(null=True, blank=True, help_text=_("Время обработки (мс)"))
    
    # Результаты
    chunks_created = models.IntegerField(default=0, help_text=_("Создано чанков"))
    embeddings_created = models.IntegerField(default=0, help_text=_("Создано эмбеддингов"))
    
    # Ошибки
    error_message = models.TextField(blank=True)
    error_details = models.JSONField(default=dict, blank=True, help_text=_("Детали ошибки"))
    
    # Временные метки
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
    """Лог удаленных документов"""
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
    """Исправляет пути файлов для существующих документов"""
    for document in Document.objects.filter(is_active=True):
        full_path = document.full_file_path
        
        # Проверяем существование файла
        if not os.path.exists(full_path):
            print(f"Файл не найден: {full_path}")
            
            # Попробуем найти файл по имени в папке documents
            possible_locations = [
                os.path.join(settings.MEDIA_ROOT, 'documents', document.original_filename),
                os.path.join(settings.MEDIA_ROOT, 'documents', str(document.id), document.original_filename),
                os.path.join(settings.MEDIA_ROOT, document.original_filename),
            ]
            
            for location in possible_locations:
                if os.path.exists(location):
                    # Обновляем file_path
                    relative_path = os.path.relpath(location, settings.MEDIA_ROOT)
                    document.file_path = relative_path
                    document.save()
                    print(f"Исправлен путь для документа {document.id}: {relative_path}")
                    break

class SavedSearch(models.Model):
    """Сохраненные поисковые запросы"""
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
        """Увеличить счетчик использования"""
        self.usage_count += 1
        self.last_used_at = timezone.now()
        self.save(update_fields=['usage_count', 'last_used_at'])

# ============================================================================
# МОДЕЛИ ДЛЯ FINE-TUNING
# ============================================================================

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

# ============================================================================
# МОДЕЛЬ ДЛЯ ШАБЛОНОВ FINE-TUNING
# ============================================================================

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


# ============================================================================
# МОДЕЛИ ДЛЯ ДИСТИЛЛЯЦИИ
# ============================================================================

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

class GenerationPreset(models.Model):
    """
    v5.0: именованный профиль параметров генерации.

    Переиспользуется в чат-сессиях и сравнениях: применение пресета
    мержит его конфиг в generation_config сессии (не заменяет целиком).
    """
    name = models.CharField(
        max_length=100,
        unique=True,
        help_text=_("Уникальное имя пресета")
    )
    description = models.TextField(blank=True, help_text=_("Описание пресета"))
    config = models.JSONField(
        default=dict,
        help_text=_("Параметры генерации: temperature, max_tokens, top_p, top_k, repetition_penalty")
    )
    created_by = models.ForeignKey(
        User, on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='generation_presets',
        help_text=_("Кто создал пресет")
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name
