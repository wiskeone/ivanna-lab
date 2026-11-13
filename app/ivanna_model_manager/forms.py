from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.conf import settings
from django.db.models import Q, Count
from django.utils.translation import gettext_lazy as _
import json
import os
import re
from django.utils.text import slugify

from .models import (
    Experiment, ModelCheckpoint, ModelVersion,
    ModelMergeJob, InferenceSession, ModelEvaluation,
    ChatSession, ChatMessage, ModelComparison,
    FineTuningJob, ModelDownloadJob, EvaluationDataset,
    ModelEvaluationTask, EvaluationResult,
    Document, DocumentCollection, DocumentChunk,
    FineTuningJob, FineTuningTemplate, UserProfile,
    ModelConversionJob, EmbeddingModel, DistillationAnswerJob, DistillationQuestionSet,
    DistillationDataset, ManualInstructionSet, HuggingFaceDataset,
    MoETheme, MoEExpertJob,
)

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['huggingface_token']
        labels = {
            'huggingface_token': 'Токен Hugging Face',
        }
        help_texts = {
            'huggingface_token': 'Введите ваш токен с Hugging Face (https://huggingface.co/settings/tokens). Он будет использоваться для загрузки приватных моделей.',
        }
        widgets = {
            'huggingface_token': forms.PasswordInput(render_value=True),  # скрываем ввод, но показываем звёздочки
        }

class ModelUploadForm(forms.ModelForm):
    """Форма для загрузки модели с расширенными полями"""
    
    SOURCE_CHOICES = [
        ('hf', 'Hugging Face Hub'),
        ('local', 'Локальный путь'),
    ]
    
    source_type = forms.ChoiceField(
        choices=SOURCE_CHOICES,
        initial='hf',
        label=_('Источник модели'),
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    
    # Поля для Hugging Face
    huggingface_id = forms.CharField(
        required=False,
        label='Hugging Face ID',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'organization/model-name',
            'data-api-endpoint': '/api/huggingface/validate/'
        })
    )
    
    huggingface_revision = forms.CharField(
        required=False,
        initial='main',
        label=_('Ревизия/ветка'),
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    # Поля для локального пути
    local_path = forms.CharField(
        required=False,
        label=_('Локальный путь'),
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '/app/models/your_model'
        })
    )
    
    relative_path = forms.CharField(
        required=False,
        label=_('Относительный путь'),
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'models/your_model'
        })
    )
    
    # Квантизация
    QUANTIZATION_CHOICES = [
        ('none', 'Без квантизации'),
        ('4bit', '4-битная (NF4)'),
        ('8bit', '8-битная'),
        ('gptq', 'GPTQ'),
        ('awq', 'AWQ'),
    ]
    
    quantization = forms.ChoiceField(
        choices=QUANTIZATION_CHOICES,
        required=False,
        initial='none',
        label=_('Тип квантизации'),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Дополнительные настройки
    trust_remote_code = forms.BooleanField(
        required=False,
        initial=True,
        label='Trust Remote Code',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    use_flash_attention_2 = forms.BooleanField(
        required=False,
        initial=True,
        label=_('Использовать Flash Attention 2'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    start_download = forms.BooleanField(
        required=False,
        initial=True,
        label=_('Начать загрузку сразу'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    mark_safe = forms.BooleanField(
        required=False,
        initial=False,
        label=_('Пометить как безопасную'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    # Поля для auto-fill из API
    auto_fill_from_hf = forms.BooleanField(
        required=False,
        initial=True,
        label=_('Автоматически заполнить параметры из Hugging Face'),
        widget=forms.HiddenInput()
    )
    
    class Meta:
        model = ModelCheckpoint
        fields = [
            'name', 'display_name', 'model_type', 'huggingface_id',
            'experiment', 'parameter_count', 'context_length',
            'vocab_size', 'license_type', 'tags', 'is_public'
        ]
        widgets = {
            'model_type': forms.HiddenInput(),  # Скрываем, так как выбирается через карточки
            'parameter_count': forms.HiddenInput(),
            'context_length': forms.HiddenInput(),
            'vocab_size': forms.HiddenInput(),
            'license_type': forms.HiddenInput(),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Делаем эти поля необязательными - они заполнятся автоматически
        self.fields['parameter_count'].required = False
        self.fields['context_length'].required = False
        self.fields['vocab_size'].required = False
        self.fields['license_type'].required = False
        
        # Устанавливаем начальные значения
        self.fields['license_type'].initial = 'Apache 2.0'
    
    def clean(self):
        cleaned_data = super().clean()
        source_type = cleaned_data.get('source_type')
        model_type = cleaned_data.get('model_type')
        
        # Валидация в зависимости от источника
        if source_type == 'hf':
            hf_id = cleaned_data.get('huggingface_id')
            if not hf_id:
                raise ValidationError({
                    'huggingface_id': 'Укажите Hugging Face ID для загрузки с Hub'
                })
            
            # Проверка формата HF ID
            if not re.match(r'^[a-zA-Z0-9\-_]+\/[a-zA-Z0-9\-_.]+$', hf_id):
                raise ValidationError({
                    'huggingface_id': 'Некорректный формат ID. Используйте: автор/название-модели'
                })
        
        elif source_type == 'local':
            local_path = cleaned_data.get('local_path')
            if not local_path:
                raise ValidationError({
                    'local_path': 'Укажите локальный путь к модели'
                })
        
        # Автозаполнение параметров на основе типа модели
        if model_type and cleaned_data.get('auto_fill_from_hf', True):
            self.auto_fill_parameters(cleaned_data, model_type)
        
        return cleaned_data
    
    def auto_fill_parameters(self, cleaned_data, model_type):
        """Автоматическое заполнение параметров на основе типа модели"""
        defaults = {
            'QWEN': {
                'architecture': 'Qwen2ForCausalLM',
                'trust_remote_code': True,
                'tokenizer_class': 'Qwen2Tokenizer',
                'use_flash_attention_2': True,
                'torch_dtype': 'bfloat16'
            },
            'LLAMA': {
                'architecture': 'LlamaForCausalLM',
                'trust_remote_code': False,
                'tokenizer_class': 'LlamaTokenizer',
                'use_flash_attention_2': True,
                'torch_dtype': 'bfloat16'
            },
            'MISTRAL': {
                'architecture': 'MistralForCausalLM',
                'trust_remote_code': False,
                'tokenizer_class': 'MistralTokenizer',
                'use_flash_attention_2': True,
                'torch_dtype': 'bfloat16'
            },
            'GEMMA': {
                'architecture': 'GemmaForCausalLM',
                'trust_remote_code': False,
                'tokenizer_class': 'GemmaTokenizer',
                'use_flash_attention_2': False,
                'torch_dtype': 'bfloat16'
            }
        }
        
        if model_type in defaults:
            for key, value in defaults[model_type].items():
                if key in self.fields and not cleaned_data.get(key):
                    cleaned_data[key] = value
    
    def save(self, commit=True):
        instance = super().save(commit=False)
        
        # Обработка в зависимости от источника
        source_type = self.cleaned_data.get('source_type')
        
        if source_type == 'hf':
            instance.huggingface_revision = self.cleaned_data.get('huggingface_revision', 'main')
            
            # Автогенерация локального пути из HF ID
            if not instance.local_path:
                safe_name = instance.name.replace('/', '_').replace('.', '_')
                instance.local_path = f"/app/models/{safe_name}"
            
            if not instance.relative_path:
                safe_name = instance.name.replace('/', '_').replace('.', '_')
                instance.relative_path = f"models/{safe_name}"
        
        elif source_type == 'local':
            instance.local_path = self.cleaned_data.get('local_path', '')
            instance.relative_path = self.cleaned_data.get('relative_path', '')
            instance.huggingface_id = ''
            instance.huggingface_revision = ''
        
        # Установка квантизации
        quantization = self.cleaned_data.get('quantization')
        if quantization and quantization != 'none':
            instance.quantization = quantization
        
        # Установка безопасности
        if self.cleaned_data.get('mark_safe'):
            instance.is_safe = True
            instance.safety_score = 0.9
        
        # Создание loader_config
        loader_config = {
            'device_map': 'auto',
            'trust_remote_code': self.cleaned_data.get('trust_remote_code', True),
            'torch_dtype': self.cleaned_data.get('torch_dtype', 'auto')
        }
        
        if self.cleaned_data.get('use_flash_attention_2'):
            loader_config['use_flash_attention_2'] = True
        
        instance.loader_config = loader_config
        
        if commit:
            instance.save()
        
        return instance


class LocalModelForm(forms.ModelForm):
    """Форма для локального добавления модели (без Hugging Face)."""
    local_path = forms.CharField(
        required=True,
        label=_('Локальный путь'),
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '/app/models/your_model'
        })
    )
    relative_path = forms.CharField(
        required=False,
        label=_('Относительный путь (опционально)'),
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'models/your_model'
        })
    )
    quantization = forms.ChoiceField(
        choices=[('none', 'Без квантизации')] + list(ModelCheckpoint.QUANTIZATION_CHOICES),
        required=False,
        label=_('Тип квантизации'),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    trust_remote_code = forms.BooleanField(
        required=False,
        initial=True,
        label='Trust Remote Code',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    use_flash_attention_2 = forms.BooleanField(
        required=False,
        initial=True,
        label=_('Использовать Flash Attention 2'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    mark_safe = forms.BooleanField(
        required=False,
        initial=False,
        label=_('Пометить как безопасную'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

    class Meta:
        model = ModelCheckpoint
        fields = [
            'name', 'display_name', 'model_type', 'experiment',
            'parameter_count', 'context_length', 'vocab_size',
            'license_type', 'tags', 'is_public',
            'serving_backend',  # v5.0: transformers или vLLM
        ]
        widgets = {
            'model_type': forms.Select(attrs={'class': 'form-select'}),
            'parameter_count': forms.NumberInput(attrs={'class': 'form-control'}),
            'context_length': forms.NumberInput(attrs={'class': 'form-control'}),
            'vocab_size': forms.NumberInput(attrs={'class': 'form-control'}),
            'license_type': forms.TextInput(attrs={'class': 'form-control'}),
            'tags': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'ru, code, 7b'}),
            'is_public': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['model_type'].initial = 'CUSTOM'
        self.fields['parameter_count'].required = False
        self.fields['context_length'].required = False
        self.fields['vocab_size'].required = False
        self.fields['license_type'].initial = 'Apache 2.0'

    def clean_local_path(self):
        path = self.cleaned_data['local_path']
        # Можно добавить проверку существования пути, если нужно
        return path

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.local_path = self.cleaned_data['local_path']
        instance.relative_path = self.cleaned_data.get('relative_path', '')
        instance.huggingface_id = ''
        instance.huggingface_revision = ''
        if self.cleaned_data['quantization'] and self.cleaned_data['quantization'] != 'none':
            instance.quantization = self.cleaned_data['quantization']
        instance.loader_config = {
            'device_map': 'auto',
            'trust_remote_code': self.cleaned_data['trust_remote_code'],
            'use_flash_attention_2': self.cleaned_data['use_flash_attention_2'],
            'torch_dtype': 'auto'
        }
        if self.cleaned_data['mark_safe']:
            instance.is_safe = True
            instance.safety_score = 0.9
        instance.status = 'PENDING'  # Локальная модель не скачивается, сразу готова к загрузке? Можно 'DOWNLOADED'
        # Если файлы существуют, можно поменять статус на 'DOWNLOADED'
        if instance.local_path and os.path.exists(instance.local_path):
            instance.status = 'DOWNLOADED'
        if commit:
            instance.save()
        return instance
        
class ExperimentForm(forms.ModelForm):
    class Meta:
        model = Experiment
        fields = ['name', 'description', 'base_checkpoint', 'tags', 'is_active']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
            'tags': forms.TextInput(attrs={
                'placeholder': 'Введите теги через запятую: ru, code, medical'
            }),
        }
    
    def clean_tags(self):
        tags = self.cleaned_data.get('tags', [])
        if isinstance(tags, str):
            # Преобразуем строку в список
            tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        return tags


class ModelCheckpointForm(forms.ModelForm):
    # Дополнительные поля для удобства
    auto_generate_name = forms.BooleanField(
        required=False,
        initial=True,
        label=_('Автоматически генерировать имя из HF ID')
    )
    
    class Meta:
        model = ModelCheckpoint
        fields = [
            'name', 'display_name', 'model_type', 'huggingface_id',
            'huggingface_revision', 'experiment', 'parent',
            'parameter_count', 'quantization', 'context_length',
            'license_type', 'tags', 'is_public'
        ]
        widgets = {
            'huggingface_id': forms.TextInput(attrs={
                'placeholder': 'organization/model-name'
            }),
            'parameter_count': forms.NumberInput(attrs={
                'placeholder': '7000000000 для 7B модели'
            }),
            'tags': forms.TextInput(attrs={
                'placeholder': 'ru, code, 7b, instruct'
            }),
        }
    
    def clean(self):
        cleaned_data = super().clean()
        
        # Если выбран auto_generate_name и есть huggingface_id
        if cleaned_data.get('auto_generate_name') and cleaned_data.get('huggingface_id'):
            hf_id = cleaned_data['huggingface_id']
            # Генерируем имя из HF ID
            name = hf_id.replace('/', '_').replace('.', '_').lower()
            cleaned_data['name'] = name
            
            # Генерируем display_name
            display_parts = hf_id.split('/')
            if len(display_parts) == 2:
                org, model = display_parts
                cleaned_data['display_name'] = f"{org}/{model}"
        
        return cleaned_data
    
    def clean_huggingface_id(self):
        hf_id = self.cleaned_data.get('huggingface_id', '').strip()
        if hf_id:
            # Проверяем формат
            parts = hf_id.split('/')
            if len(parts) != 2:
                raise ValidationError(
                    'Hugging Face ID должен быть в формате "organization/model-name"'
                )
        return hf_id


class ModelConfigForm(forms.Form):
    config_json = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 20,
            'style': 'font-family: monospace; font-size: 12px;'
        }),
        label=_('Конфигурация (JSON)')
    )
    
    def clean_config_json(self):
        config_json = self.cleaned_data.get('config_json')
        try:
            return json.loads(config_json)
        except json.JSONDecodeError as e:
            raise ValidationError(f'Неверный JSON формат: {str(e)}')


class ModelMergeForm(forms.ModelForm):
    class Meta:
        model = ModelMergeJob
        fields = [
            'name', 'description', 'base_model', 'merge_model',
            'merge_method', 'merge_ratio', 'merge_config',
            'gpu_memory_limit_gb', 'timeout_minutes'
        ]
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'merge_config': forms.Textarea(attrs={
                'rows': 10,
                'placeholder': '{\n  "density": 0.3,\n  "weight": 0.8\n}'
            }),
            'merge_ratio': forms.NumberInput(attrs={
                'min': 0,
                'max': 1,
                'step': 0.1
            }),
        }
    
    def clean(self):
        cleaned_data = super().clean()
        
        base_model = cleaned_data.get('base_model')
        merge_model = cleaned_data.get('merge_model')
        
        if base_model and merge_model:
            if base_model == merge_model:
                raise ValidationError(_('Нельзя выбрать одну и ту же модель'))
            
            if base_model.model_type != merge_model.model_type:
                raise ValidationError(_('Модели должны быть одного типа'))
            
            if base_model.parameter_count != merge_model.parameter_count:
                raise ValidationError(_('Модели должны иметь одинаковое количество параметров'))
        
        return cleaned_data


class InferenceTestForm(forms.Form):
    """Форма для тестирования инференса модели"""
    prompt = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 6,
            'placeholder': 'Введите текст для тестирования модели...',
            'class': 'form-control'
        }),
        label=_('Промпт'),
        required=True,
        max_length=10000
    )
    
    system_prompt = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 3,
            'placeholder': 'Опционально: задайте системный промпт или роль...',
            'class': 'form-control'
        }),
        label=_('Системный промпт'),
        required=False,
        max_length=5000
    )
    
    max_tokens = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': 1,
            'max': 32768
        }),
        label=_('Максимальное количество токенов'),
        initial=512,
        required=True
    )
    
    temperature = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': 0.1,
            'min': 0.0,
            'max': 2.0
        }),
        label=_('Температура'),
        initial=0.7,
        required=True
    )
    
    top_p = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': 0.01,
            'min': 0.0,
            'max': 1.0
        }),
        label='Top P',
        initial=0.9,
        required=True
    )
    
    top_k = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': 1,
            'max': 100
        }),
        label='Top K',
        initial=50,
        required=True
    )
    
    def clean_max_tokens(self):
        max_tokens = self.cleaned_data['max_tokens']
        if max_tokens < 1:
            raise forms.ValidationError(_('Количество токенов должно быть положительным'))
        if max_tokens > 32768:
            raise forms.ValidationError(_('Слишком большое количество токенов (макс. 32768)'))
        return max_tokens
    
    def clean_temperature(self):
        temperature = self.cleaned_data['temperature']
        if temperature < 0.0 or temperature > 2.0:
            raise forms.ValidationError(_('Температура должна быть в диапазоне 0.0-2.0'))
        return temperature
    
    def clean_top_p(self):
        top_p = self.cleaned_data['top_p']
        if top_p < 0.0 or top_p > 1.0:
            raise forms.ValidationError(_('Top P должен быть в диапазоне 0.0-1.0'))
        return top_p


class EvaluationForm(forms.ModelForm):
    class Meta:
        model = ModelEvaluation
        fields = [
            'checkpoint', 'dataset_name', 'dataset_split',
            'evaluation_type', 'metrics', 'hardware_info',
            'validation_notes'
        ]
        widgets = {
            'metrics': forms.Textarea(attrs={
                'rows': 10,
                'placeholder': '{\n  "accuracy": 0.85,\n  "f1": 0.82\n}'
            }),
            'hardware_info': forms.Textarea(attrs={
                'rows': 4,
                'placeholder': '{\n  "gpu": "RTX 4090",\n  "memory_gb": 24\n}'
            }),
            'validation_notes': forms.Textarea(attrs={'rows': 3}),
        }
    
    def clean_metrics(self):
        metrics = self.cleaned_data.get('metrics', '{}')
        try:
            return json.loads(metrics) if isinstance(metrics, str) else metrics
        except json.JSONDecodeError:
            raise ValidationError(_('Неверный JSON формат в метриках'))
    
    def clean_hardware_info(self):
        hardware_info = self.cleaned_data.get('hardware_info', '{}')
        try:
            return json.loads(hardware_info) if isinstance(hardware_info, str) else hardware_info
        except json.JSONDecodeError:
            raise ValidationError(_('Неверный JSON формат в информации об оборудовании'))

# ============================================================================
# Формы для системы оценки
# ============================================================================

class EvaluationDatasetForm(forms.ModelForm):
    """Форма для создания/редактирования датасета"""
    class Meta:
        model = EvaluationDataset
        fields = ['name', 'description', 'tags', 'source_type']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'rows': 4, 'class': 'form-control'}),
            'tags': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'russian, general, code, science'
            }),
            'source_type': forms.HiddenInput(),  # Скрываем поле
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['tags'].required = False
        self.fields['description'].required = False
        # Устанавливаем значение по умолчанию для source_type
        self.fields['source_type'].initial = 'MANUAL'
    
    def clean_tags(self):
        """Преобразуем строку тегов в JSON массив"""
        tags = self.cleaned_data.get('tags', '')
        if isinstance(tags, str):
            # Если это строка, преобразуем в список
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            return tag_list
        elif isinstance(tags, list):
            # Если уже список, возвращаем как есть
            return tags
        return []


class QuestionUploadForm(forms.Form):
    """Форма для загрузки вопросов из файла"""
    name = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        label=_('Название датасета')
    )
    
    description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
        label=_('Описание (необязательно)')
    )
    
    file = forms.FileField(
        widget=forms.FileInput(attrs={'class': 'form-control'}),
        label=_('Файл с вопросами'),
        help_text=_('Поддерживаемые форматы: TXT, CSV, JSON')
    )
    
    file_format = forms.ChoiceField(
        choices=[
            ('auto', 'Автоопределение'),
            ('txt', 'Текстовый файл (TXT)'),
            ('csv', 'CSV файл'),
            ('json', 'JSON файл'),
        ],
        initial='auto',
        widget=forms.Select(attrs={'class': 'form-select'}),
        label=_('Формат файла')
    )
    
    tags = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'russian, general, code'
        }),
        label=_('Теги (через запятую)'),
        help_text=_('Теги для классификации датасета')
    )
    
    def clean_tags(self):
        tags = self.cleaned_data.get('tags', '')
        if tags:
            return [tag.strip() for tag in tags.split(',') if tag.strip()]
        return []
    
    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            # Проверяем размер файла (макс 10MB)
            if file.size > 10 * 1024 * 1024:
                raise ValidationError(_('Размер файла не должен превышать 10MB'))
            
            # Проверяем расширение
            valid_extensions = ['.txt', '.csv', '.json']
            if not any(file.name.lower().endswith(ext) for ext in valid_extensions):
                raise ValidationError(_('Поддерживаются только файлы .txt, .csv, .json'))
        
        return file


class CreateEvaluationTaskForm(forms.Form):
    """Форма для создания задачи оценки"""
    name = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        label=_('Название задачи')
    )
    
    description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
        label=_('Описание задачи')
    )
    
    models = forms.ModelMultipleChoiceField(
        queryset=ModelCheckpoint.objects.none(),
        widget=forms.SelectMultiple(attrs={'class': 'form-control', 'size': 6}),
        label=_('Модели для оценки'),
        help_text=_('Выберите загруженные модели')
    )
    
    dataset = forms.ModelChoiceField(
        queryset=EvaluationDataset.objects.all(),
        widget=forms.Select(attrs={'class': 'form-select'}),
        label=_('Датасет для оценки')
    )
    
    # Поле выбора квантизации
    QUANTIZATION_CHOICES = [
        ('', '--- Без квантизации (использовать настройки модели) ---'),
        ('float16', 'FP16 (половинная точность)'),
        ('bfloat16', 'BF16 (bfloat16)'),
        ('int8', 'INT8 (8‑бит)'),
        ('int4', 'INT4 (4‑бит)'),
    ]
    quantization = forms.ChoiceField(
        choices=QUANTIZATION_CHOICES,
        required=False,
        label=_('Квантизация для всех моделей'),
        help_text=_('Если выбрано, переопределит квантизацию моделей при загрузке в память'),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Параметры генерации
    max_tokens = forms.IntegerField(
        initial=512,
        min_value=1,
        max_value=8192,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label=_('Максимальное количество токенов')
    )
    
    temperature = forms.FloatField(
        initial=0.7,
        min_value=0.0,
        max_value=2.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
        label=_('Температура')
    )
    
    top_p = forms.FloatField(
        initial=0.9,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
        label='Top P'
    )
    
    top_k = forms.IntegerField(
        initial=50,
        min_value=1,
        max_value=100,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Top K'
    )
    
    repetition_penalty = forms.FloatField(
        initial=1.0,
        min_value=0.0,
        max_value=2.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
        label='Penalty'
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['models'].queryset = ModelCheckpoint.objects.filter(
            is_active=True,
            status='LOADED'  # только загруженные, чтобы не пришлось качать
        )
    
    def clean(self):
        cleaned_data = super().clean()
        models = cleaned_data.get('models')
        if not models:
            raise ValidationError(_('Выберите хотя бы одну модель для оценки'))
        dataset = cleaned_data.get('dataset')
        if dataset and not dataset.questions:
            raise ValidationError(_('Выбранный датасет не содержит вопросов'))
        return cleaned_data


class RatingForm(forms.Form):
    """Форма для оценки ответа"""
    rating = forms.ChoiceField(
        choices=[(i, str(i)) for i in range(1, 6)],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label=_('Оценка (1-5)')
    )
    
    comment = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'rows': 3, 'class': 'form-control'}),
        label=_('Комментарий')
    )

#############################################################################
# RAG
#############################################################################

# ----------------------------------------------------------------------
# Кастомные поля и виджеты
# ----------------------------------------------------------------------
class MultipleFileInput(forms.ClearableFileInput):
    """Кастомный виджет для множественного выбора файлов"""
    allow_multiple_selected = True
    
    def __init__(self, attrs=None):
        super().__init__(attrs)
        if attrs is None:
            attrs = {}
        attrs['multiple'] = 'multiple'
        self.attrs.update(attrs)


class MultipleFileField(forms.FileField):
    """Кастомное поле для множественной загрузки файлов"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result


# ----------------------------------------------------------------------
# Загрузка эмбендингов
# ----------------------------------------------------------------------
class EmbeddingModelForm(forms.ModelForm):
    class Meta:
        model = EmbeddingModel
        fields = [
            'name', 'display_name', 'model_type', 'model_id', 'model_version',
            'embedding_dim', 'max_tokens', 'is_multilingual',
            'license_type', 'is_default', 'is_active',
        ]
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'display_name': forms.TextInput(attrs={'class': 'form-control'}),
            'model_type': forms.Select(attrs={'class': 'form-select'}),
            'model_id': forms.TextInput(attrs={'class': 'form-control'}),
            'model_version': forms.TextInput(attrs={'class': 'form-control'}),
            'embedding_dim': forms.NumberInput(attrs={'class': 'form-control'}),
            'max_tokens': forms.NumberInput(attrs={'class': 'form-control'}),
            'is_multilingual': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'license_type': forms.TextInput(attrs={'class': 'form-control'}),
            'is_default': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }

    def clean_name(self):
        name = self.cleaned_data['name']
        # Если это редактирование (есть instance и pk), исключаем текущий объект
        if self.instance and self.instance.pk:
            if EmbeddingModel.objects.filter(name=name).exclude(pk=self.instance.pk).exists():
                raise ValidationError(_('Модель с таким именем уже существует.'))
        else:
            if EmbeddingModel.objects.filter(name=name).exists():
                raise ValidationError(_('Модель с таким именем уже существует.'))
        return name
        
# ----------------------------------------------------------------------
# Формы загрузки документов
# ----------------------------------------------------------------------
class MultipleDocumentUploadForm(forms.Form):
    """Форма для множественной загрузки документов"""
    
    collection = forms.ModelChoiceField(
        queryset=DocumentCollection.objects.none(),
        label=_('Коллекция'),
        help_text=_('Выберите коллекцию для документов'),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    files = MultipleFileField(
        label=_('Файлы документов'),
        help_text=_('Выберите один или несколько файлов'),
        widget=MultipleFileInput(attrs={
            'class': 'form-control',
            'accept': '.pdf,.docx,.doc,.txt,.html,.htm,.md,.csv,.xlsx,.xls'
        })
    )
    
    title_prefix = forms.CharField(
        required=False,
        label=_('Префикс названия'),
        help_text=_('Будет добавлен к названию каждого документа'),
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    category = forms.CharField(
        required=False,
        label=_('Категория'),
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    tags = forms.CharField(
        required=False,
        label=_('Теги'),
        help_text=_('Через запятую: legal, finance, report'),
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    auto_process = forms.BooleanField(
        initial=True,
        required=False,
        label=_('Автоматическая обработка'),
        help_text=_('Немедленно начать обработку после загрузки'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    chunk_size = forms.IntegerField(
        min_value=100,
        max_value=10000,
        initial=1000,
        label=_('Размер чанка (символов)'),
        help_text=_('Рекомендуется 500-2000 символов'),
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    chunk_overlap = forms.IntegerField(
        min_value=0,
        max_value=2000,                         # увеличено для больших чанков
        initial=200,
        label=_('Перекрытие чанков'),
        help_text=_('Рекомендуется 10-20% от размера чанка'),
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        if user:
            self.fields['collection'].queryset = DocumentCollection.objects.filter(
                Q(owner_id=str(user.id)) | Q(is_public=True),
                is_active=True
            ).order_by('name')
    
    def clean_files(self):
        files = self.cleaned_data.get('files', [])
        
        if not files:
            raise ValidationError(_('Выберите хотя бы один файл'))
        
        if not isinstance(files, (list, tuple)):
            files = [files]
        
        max_size = getattr(settings, 'MAX_UPLOAD_SIZE', 100 * 1024 * 1024)  # 100MB
        allowed_extensions = ['.pdf', '.txt', '.docx', '.doc', '.html', '.htm', '.md', '.csv', '.xlsx', '.xls']
        
        for file in files:
            if file.size > max_size:
                raise ValidationError(
                    f"Файл {file.name} слишком большой. Максимальный размер: {max_size//(1024*1024)}MB"
                )
            
            ext = os.path.splitext(file.name)[1].lower()
            if ext not in allowed_extensions:
                raise ValidationError(
                    f"Формат {ext} не поддерживается. Разрешены: {', '.join(allowed_extensions)}"
                )
        
        return files
    
    def clean_tags(self):
        tags = self.cleaned_data.get('tags', '')
        if tags:
            tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            return tags_list
        return []
    
    def clean(self):
        cleaned_data = super().clean()
        chunk_size = cleaned_data.get('chunk_size')
        chunk_overlap = cleaned_data.get('chunk_overlap')
        
        if chunk_size and chunk_overlap:
            if chunk_overlap >= chunk_size:
                raise ValidationError(
                    'Перекрытие чанков должно быть меньше их размера'
                )
            if chunk_overlap > chunk_size * 0.5:
                raise ValidationError(
                    'Перекрытие чанков не должно превышать 50% от их размера'
                )
        
        return cleaned_data


class DocumentUploadForm(forms.ModelForm):
    """Форма загрузки одного документа (для обратной совместимости)"""
    
    collection = forms.ModelChoiceField(
        queryset=DocumentCollection.objects.none(),
        label=_('Коллекция'),
        help_text=_('Выберите коллекцию для документа'),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    file = forms.FileField(
        label=_('Файл документа'),
        help_text=_('Выберите файл для загрузки'),
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.pdf,.docx,.doc,.txt,.html,.htm,.md'
        })
    )
    
    title = forms.CharField(
        required=False,
        label=_('Название документа'),
        help_text=_('Если не указано, будет использовано имя файла'),
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    description = forms.CharField(
        required=False,
        label=_('Описание'),
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Краткое описание документа'
        })
    )
    
    category = forms.CharField(
        required=False,
        label=_('Категория'),
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    tags = forms.CharField(
        required=False,
        label=_('Теги'),
        help_text=_('Через запятую: legal, finance, report'),
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    chunk_size = forms.IntegerField(
        min_value=100,
        max_value=10000,
        initial=1000,
        label=_('Размер чанка (символов)'),
        help_text=_('Рекомендуется 500-2000 символов'),
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    chunk_overlap = forms.IntegerField(
        min_value=0,
        max_value=2000,
        initial=200,
        label=_('Перекрытие чанков'),
        help_text=_('Рекомендуется 10-20% от размера чанка'),
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    auto_process = forms.BooleanField(
        initial=True,
        required=False,
        label=_('Автоматическая обработка'),
        help_text=_('Немедленно начать обработку после загрузки'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    class Meta:
        model = Document
        fields = ['title', 'description', 'category', 'collection']
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        if user:
            self.fields['collection'].queryset = DocumentCollection.objects.filter(
                Q(owner_id=str(user.id)) | Q(is_public=True),
                is_active=True
            ).order_by('name')
        
        # Преобразуем теги из JSON в строку для начального отображения
        if self.instance and self.instance.tags:
            self.initial['tags'] = ', '.join(self.instance.tags)
    
    def clean_file(self):
        file = self.cleaned_data.get('file')
        
        if not file:
            raise ValidationError(_('Выберите файл для загрузки'))
        
        max_size = getattr(settings, 'MAX_UPLOAD_SIZE', 100 * 1024 * 1024)  # 100MB
        allowed_extensions = ['.pdf', '.txt', '.docx', '.doc', '.html', '.htm', '.md']
        
        if file.size > max_size:
            raise ValidationError(
                f"Файл {file.name} слишком большой. Максимальный размер: {max_size//(1024*1024)}MB"
            )
        
        ext = os.path.splitext(file.name)[1].lower()
        if ext not in allowed_extensions:
            raise ValidationError(
                f"Формат {ext} не поддерживается. Разрешены: {', '.join(allowed_extensions)}"
            )
        
        return file
    
    def clean_tags(self):
        tags = self.cleaned_data.get('tags', '')
        if tags:
            tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            return tags_list
        return []
    
    def clean(self):
        cleaned_data = super().clean()
        
        chunk_size = cleaned_data.get('chunk_size')
        chunk_overlap = cleaned_data.get('chunk_overlap')
        
        if chunk_size and chunk_overlap:
            if chunk_overlap >= chunk_size:
                raise ValidationError(
                    'Перекрытие чанков должно быть меньше их размера'
                )
            if chunk_overlap > chunk_size * 0.5:
                raise ValidationError(
                    'Перекрытие чанков не должно превышать 50% от их размера'
                )
        
        return cleaned_data


class DocumentEditForm(forms.ModelForm):
    """Форма для редактирования документа с параметрами чанков"""
    
    chunk_size = forms.IntegerField(
        min_value=100,
        max_value=10000,
        initial=1000,
        label=_('Размер чанка (символов)'),
        help_text=_('Рекомендуется 500-2000 символов'),
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        required=False
    )
    
    chunk_overlap = forms.IntegerField(
        min_value=0,
        max_value=2000,
        initial=200,
        label=_('Перекрытие чанков'),
        help_text=_('Рекомендуется 10-20% от размера чанка'),
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        required=False
    )
    
    regenerate_chunks = forms.BooleanField(
        initial=False,
        required=False,
        label=_('Пересоздать чанки'),
        help_text=_('Удалить существующие чанки и создать новые с новыми параметрами'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    class Meta:
        model = Document
        fields = ['title', 'description', 'category', 'tags', 'collection', 'quality_score']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'category': forms.TextInput(attrs={'class': 'form-control'}),
            'tags': forms.TextInput(attrs={'class': 'form-control'}),
            'collection': forms.Select(attrs={'class': 'form-control'}),
            'quality_score': forms.NumberInput(attrs={'class': 'form-control', 'min': 0, 'max': 1, 'step': 0.1}),
        }
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        if self.user:
            self.fields['collection'].queryset = DocumentCollection.objects.filter(
                Q(owner_id=str(self.user.id)) | Q(is_public=True),
                is_active=True
            ).order_by('name')
        
        if self.instance and self.instance.tags:
            self.initial['tags'] = ', '.join(self.instance.tags)
        
        if self.instance and 'chunk_size' in self.instance.metadata:
            self.initial['chunk_size'] = self.instance.metadata['chunk_size']
        if self.instance and 'chunk_overlap' in self.instance.metadata:
            self.initial['chunk_overlap'] = self.instance.metadata['chunk_overlap']
    
    def clean_tags(self):
        tags = self.cleaned_data.get('tags', '')
        if tags:
            return [tag.strip() for tag in tags.split(',') if tag.strip()]
        return []
    
    def clean(self):
        cleaned_data = super().clean()
        
        chunk_size = cleaned_data.get('chunk_size')
        chunk_overlap = cleaned_data.get('chunk_overlap')
        regenerate_chunks = cleaned_data.get('regenerate_chunks')
        
        if chunk_size and chunk_overlap:
            if chunk_overlap >= chunk_size:
                raise ValidationError(_('Перекрытие чанков должно быть меньше их размера'))
            if chunk_overlap > chunk_size * 0.5:
                raise ValidationError(_('Перекрытие чанков не должно превышать 50% от их размера'))
        
        if regenerate_chunks and (not chunk_size or not chunk_overlap):
            raise ValidationError(_('Для пересоздания чанков укажите размер чанка и перекрытие'))
        
        return cleaned_data
    
    def save(self, commit=True):
        document = super().save(commit=False)
        
        if 'chunk_size' in self.cleaned_data:
            document.metadata['chunk_size'] = self.cleaned_data['chunk_size']
        if 'chunk_overlap' in self.cleaned_data:
            document.metadata['chunk_overlap'] = self.cleaned_data['chunk_overlap']
        
        if commit:
            document.save()
            
            if self.cleaned_data.get('regenerate_chunks'):
                from ..tasks import regenerate_document_chunks_task
                regenerate_document_chunks_task.delay(document.id)
        
        return document


class DocumentChunkEditForm(forms.ModelForm):
    """Форма редактирования чанка документа"""
    
    class Meta:
        model = DocumentChunk
        fields = ['content', 'metadata', 'relevance_score', 'is_key_chunk']
        widgets = {
            'content': forms.Textarea(attrs={'rows': 6, 'class': 'form-control'}),
            'metadata': forms.Textarea(attrs={'rows': 2, 'class': 'form-control', 'placeholder': 'JSON метаданные'}),
            'relevance_score': forms.NumberInput(attrs={'class': 'form-control', 'min': 0, 'max': 1, 'step': 0.01}),
            'is_key_chunk': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
    
    def clean_content(self):
        content = self.cleaned_data.get('content', '').strip()
        if len(content) < 10:
            raise ValidationError(_('Содержимое чанка слишком короткое'))
        return content
    
    def clean_metadata(self):
        metadata = self.cleaned_data.get('metadata', '{}')
        if metadata:
            try:
                json.loads(metadata)
            except json.JSONDecodeError:
                raise ValidationError(_('Неверный формат JSON'))
        return metadata


class DocumentBulkActionForm(forms.Form):
    """Форма массовых действий с документами"""
    ACTION_CHOICES = [
        ('delete', 'Удалить выбранные'),
        ('reprocess', 'Переобработать выбранные'),
        ('archive', 'Архивировать выбранные'),
        ('export', 'Экспортировать метаданные'),
    ]
    
    action = forms.ChoiceField(
        choices=ACTION_CHOICES,
        label=_('Действие'),
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'})
    )
    document_ids = forms.CharField(
        widget=forms.HiddenInput(),
        required=False
    )
    confirm = forms.BooleanField(
        required=False,
        label=_('Подтверждаю действие'),
        help_text=_('Это действие нельзя отменить'),
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    
    def clean_document_ids(self):
        ids_str = self.cleaned_data.get('document_ids', '')
        if ids_str:
            try:
                ids = [int(i) for i in ids_str.split(',') if i.strip()]
                return ids
            except ValueError:
                raise ValidationError(_('Некорректный формат ID документов'))
        return []
    
    def clean(self):
        cleaned_data = super().clean()
        action = cleaned_data.get('action')
        confirm = cleaned_data.get('confirm')
        
        if action in ['delete', 'archive'] and not confirm:
            raise ValidationError(_('Подтвердите действие для удаления или архивации'))
        
        return cleaned_data


# ----------------------------------------------------------------------
# Формы для коллекций
# ----------------------------------------------------------------------
class DocumentCollectionForm(forms.ModelForm):
    """Форма для создания и редактирования коллекций документов"""
    
    class Meta:
        model = DocumentCollection
        fields = [
            'name',
            'description',
            'is_public',
            'chunking_strategy',
            'chunk_size',
            'chunk_overlap',
            'auto_process'
        ]
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Например: Юридические документы'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Опишите назначение коллекции'}),
            'is_public': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'chunking_strategy': forms.Select(attrs={'class': 'form-select'}),
            'chunk_size': forms.NumberInput(attrs={'class': 'form-control', 'min': 100, 'max': 10000}),
            'chunk_overlap': forms.NumberInput(attrs={'class': 'form-control', 'min': 0, 'max': 2000}),
            'auto_process': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        self.fields['chunk_size'].help_text = 'Рекомендуется: 1000-2000 символов'
        self.fields['chunk_overlap'].help_text = 'Рекомендуется: 10-20% от chunk_size'
        self.fields['is_public'].help_text = 'Сделать коллекцию доступной другим пользователям'
    
    def clean(self):
        cleaned_data = super().clean()
        chunk_size = cleaned_data.get('chunk_size')
        chunk_overlap = cleaned_data.get('chunk_overlap')
        
        if chunk_size and chunk_overlap:
            if chunk_overlap >= chunk_size:
                raise ValidationError(_('Перекрытие чанков должно быть меньше их размера'))
            if chunk_overlap > chunk_size * 0.5:
                raise ValidationError(_('Перекрытие чанков не должно превышать 50% от их размера'))
        
        return cleaned_data


# ----------------------------------------------------------------------
# Формы для RAG-запросов и поиска
# ----------------------------------------------------------------------
class RAGQueryForm(forms.Form):
    """Форма для RAG запросов"""
    query = forms.CharField(
        label=_('Вопрос'),
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Введите ваш вопрос...'})
    )
    
    collection = forms.ModelChoiceField(
        queryset=DocumentCollection.objects.none(),
        label=_('Коллекция'),
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    system_prompt = forms.CharField(
        label=_('Системный промпт'),
        required=False,
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 2, 'placeholder': 'Необязательно: кастомизация поведения ассистента'})
    )
    
    top_k = forms.IntegerField(
        min_value=1,
        max_value=20,
        initial=5,
        label=_('Количество источников'),
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    temperature = forms.FloatField(
        min_value=0.1,
        max_value=2.0,
        initial=0.7,
        label=_('Температура генерации'),
        help_text=_('Случайность ответа (ниже = более детерминированный)'),
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': 0.1})
    )
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        if user:
            self.fields['collection'].queryset = DocumentCollection.objects.filter(
                Q(owner_id=str(user.id)) | Q(is_public=True),
                is_active=True
            ).order_by('name')


class DocumentSearchForm(forms.Form):
    """Форма поиска документов"""
    query = forms.CharField(
        required=False,
        label=_('Поиск'),
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Поиск по названию или содержимому...'})
    )
    
    collection = forms.ModelChoiceField(
        queryset=DocumentCollection.objects.none(),
        label=_('Коллекция'),
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    status = forms.ChoiceField(
        choices=[
            ('', 'Все статусы'),
            ('UPLOADED', 'Загружено'),
            ('PROCESSING', 'Обрабатывается'),
            ('INDEXED', 'Проиндексировано'),
            ('ERROR', 'Ошибка')
        ],
        required=False,
        label=_('Статус'),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    date_from = forms.DateField(
        required=False,
        label=_('Дата от'),
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    
    date_to = forms.DateField(
        required=False,
        label=_('Дата до'),
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        if user:
            self.fields['collection'].queryset = DocumentCollection.objects.filter(
                Q(owner_id=str(user.id)) | Q(is_public=True),
                is_active=True
            ).order_by('name')


#############################################################################
# FineTuning
#############################################################################
class FineTuningJobForm(forms.ModelForm):
    class Meta:
        model = FineTuningJob
        fields = [
            'name', 'description', 'base_model', 'method', 'quantization',
            'lora_r', 'lora_alpha', 'lora_dropout',
            'learning_rate', 'batch_size', 'gradient_accumulation_steps',
            'num_epochs', 'max_seq_length',
            # Параметры RAFT
            'distractors_per_example', 'oracle_ratio', 'distractor_ratio',
            'no_context_ratio', 'use_kl_regularization', 'kl_lambda',
            'general_instructions_ratio',
            # v5.1: выбор GPU
            'gpu_device'
        ]
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'learning_rate': forms.NumberInput(attrs={'step': '1e-6'}),
            'lora_dropout': forms.NumberInput(attrs={'step': '0.01'}),
            'oracle_ratio': forms.NumberInput(attrs={'step': '0.05'}),
            'distractor_ratio': forms.NumberInput(attrs={'step': '0.05'}),
            'no_context_ratio': forms.NumberInput(attrs={'step': '0.05'}),
            'kl_lambda': forms.NumberInput(attrs={'step': '0.01'}),
            'general_instructions_ratio': forms.NumberInput(attrs={'step': '0.01'}),
            'gpu_device': forms.Select(attrs={'class': 'form-select'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # v5.1: выбор GPU — варианты из фактического количества устройств
        gpu_choices = [('', _('Авто (первый доступный)')), ('-1', _('CPU'))]
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_choices.append((str(i), f'GPU {i}: {torch.cuda.get_device_name(i)}'))
        except Exception:
            pass
        self.fields['gpu_device'] = forms.TypedChoiceField(
            choices=gpu_choices, coerce=lambda v: int(v) if v != '' else None,
            required=False, empty_value=None,
            widget=forms.Select(attrs={'class': 'form-select'}),
            label=_('GPU-устройство'),
            help_text=_('Авто = первый доступный GPU; CPU = обучение без GPU')
        )

class FineTuningTemplateForm(forms.ModelForm):
    """Форма для создания/редактирования шаблонов"""
    
    class Meta:
        model = FineTuningTemplate
        exclude = ['usage_count', 'avg_rating', 'created_by']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'recommended_models': forms.Textarea(attrs={
                'rows': 2,
                'placeholder': '["Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.3"]'
            }),
        }
    
    def clean_recommended_models(self):
        data = self.cleaned_data['recommended_models']
        if isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return [m.strip() for m in data.split(',')]
        return data


class FineTuningTemplateSelectForm(forms.Form):
    """Форма для выбора шаблона при создании задачи"""
    
    template = forms.ModelChoiceField(
        queryset=FineTuningTemplate.objects.filter(is_public=True),
        empty_label=_('-- Выберите шаблон --'),
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )


#############################################################################
# Convertion
#############################################################################
class ModelConversionForm(forms.ModelForm):
    quantization = forms.ChoiceField(
        choices=ModelConversionJob.QUANTIZATION_CHOICES,
        initial='q4_0',
        label=_('Квантизация'),
        help_text=_('Тип квантизации для GGUF')
    )

    class Meta:
        model = ModelConversionJob
        fields = ['name', 'description', 'source_model', 'quantization', 'config']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'config': forms.Textarea(attrs={'rows': 5, 'class': 'font-monospace'}),
        }
        help_texts = {
            'config': 'JSON с дополнительными параметрами (опционально)',
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Показываем только модели, которые загружены локально (имеют local_path)
        self.fields['source_model'].queryset = ModelCheckpoint.objects.filter(
            is_active=True,
            local_path__isnull=False
        ).exclude(local_path='').order_by('name')
        # Поле config делаем необязательным
        self.fields['config'].required = False
        self.fields['config'].widget.attrs['placeholder'] = '{"param": "value"}'

    def clean_config(self):
        config = self.cleaned_data.get('config')
        if config:
            if isinstance(config, str):
                try:
                    config = json.loads(config)
                except json.JSONDecodeError:
                    raise forms.ValidationError(_('Неверный JSON формат'))
            if not isinstance(config, dict):
                raise forms.ValidationError(_('Конфигурация должна быть словарём'))
        return config or {}


#############################################################################
# Distillation
#############################################################################
class DistillationAnswerJobForm(forms.ModelForm):
    # Поле для ручного ввода ID вопросов (в виде строки)
    question_ids_str = forms.CharField(
        required=False,
        widget=forms.HiddenInput(),
        label=_('ID вопросов (через запятую)')
    )

    class Meta:
        model = DistillationAnswerJob
        fields = ['name', 'description', 'question_set', 'question_ids_str']

    def clean_question_set(self):
        value = self.cleaned_data.get('question_set')
        if value == '' or value is None:
            return None
        # Если пришло строковое представление ID, пытаемся получить объект
        if isinstance(value, str) and value.isdigit():
            return DistillationQuestionSet.objects.filter(id=int(value)).first()
        return value

    def clean_question_ids_str(self):
        value = self.cleaned_data.get('question_ids_str', '')
        if not value:
            return []
        # Разбираем строку вида "1,2,3"
        ids = [int(id.strip()) for id in value.split(',') if id.strip().isdigit()]
        return ids

    def save(self, commit=True):
        instance = super().save(commit=False)
        # Сохраняем обработанные ID в поле модели
        instance.question_ids = self.cleaned_data['question_ids_str']
        if commit:
            instance.save()
        return instance

# ═══════════════════════════════════════════════════════════════════════
# МЯГКИЙ MoE (v5.2): темы и задачи сборки эксперта
# ═══════════════════════════════════════════════════════════════════════

MOE_PROMPT_FIELD_HELP = _(
    "По одному промпту на строку. Рекомендуется 25–30 промптов; "
    "длины доменных и контрастных наборов должны быть выровнены "
    "(иначе длина протечёт в направление).")


class MoEThemeForm(forms.ModelForm):
    """Тема мягкого MoE: домен + контраст (и опционально eval-наборы)."""

    domain_prompts_text = forms.CharField(
        label=_("Доменные промпты"), required=False,
        widget=forms.Textarea(attrs={'rows': 10, 'placeholder': _('Реальные запросы/тексты домена, по одному на строку')}),
        help_text=MOE_PROMPT_FIELD_HELP)
    contrast_prompts_text = forms.CharField(
        label=_("Контрастные промпты"), required=False,
        widget=forms.Textarea(attrs={'rows': 10, 'placeholder': _('Быт/код/кулинария/право/спорт — всё НЕ из домена')}),
        help_text=MOE_PROMPT_FIELD_HELP)
    eval_domain_prompts_text = forms.CharField(
        label=_("Eval: доменные (свежие)"), required=False,
        widget=forms.Textarea(attrs={'rows': 5}),
        help_text=_("НЕ пересекаются со скановыми наборами — на них честный eval-гейт и тест роутера (10 шт)"))
    eval_contrast_prompts_text = forms.CharField(
        label=_("Eval: общие (свежие)"), required=False,
        widget=forms.Textarea(attrs={'rows': 5}),
        help_text=_("НЕ пересекаются со скановыми наборами (10 шт)"))

    class Meta:
        model = MoETheme
        fields = ['name', 'description', 'embedding_model', 'router_threshold']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'router_threshold': forms.NumberInput(attrs={'step': '0.01'}),
        }

    def clean_name(self):
        name = self.cleaned_data['name']
        if len(name) < 2:
            raise ValidationError(_("Название минимум 2 символа"))
        return name

    def save(self, commit=True):
        from .services.soft_moe import prompts as moe_prompts

        instance = super().save(commit=False)
        tdir = moe_prompts.theme_dir(instance.slug)
        mapping = [
            ('domain_prompts_text', 'domain_prompts.txt', 'domain_prompts_file'),
            ('contrast_prompts_text', 'contrast_prompts.txt', 'contrast_prompts_file'),
            ('eval_domain_prompts_text', 'eval_domain_prompts.txt', 'eval_domain_file'),
            ('eval_contrast_prompts_text', 'eval_general_prompts.txt', 'eval_contrast_file'),
        ]
        for field, fname, attr in mapping:
            raw = self.cleaned_data.get(field)
            if raw and raw.strip():
                texts = moe_prompts.clean_prompt_set(moe_prompts.parse_prompts(raw),
                                                     min_len=10, max_len=4000)
                if texts:
                    moe_prompts.save_prompt_set(tdir / fname, texts)
                    setattr(instance, attr, f"moe_themes/{instance.slug}/{fname}")
        if commit:
            instance.save()
        return instance


class MoEExpertJobForm(forms.ModelForm):
    """Задача сборки мягкого MoE-эксперта из плотной модели."""

    class Meta:
        model = MoEExpertJob
        fields = ['name', 'theme', 'base_checkpoint', 'goal', 'mode', 'alpha', 'layers',
                  'position', 'holdout_seed', 'load_mode', 'do_patch', 'do_eval', 'do_router']
        widgets = {
            'alpha': forms.NumberInput(attrs={'step': '0.05',
                                              'help_text': _('Сетка 0.1/0.2/0.4')}),
            'layers': forms.TextInput(attrs={'placeholder': 'auto'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Только плотные модели, выгруженные на диск
        self.fields['base_checkpoint'].queryset = ModelCheckpoint.objects.filter(
            is_active=True).exclude(local_path='').order_by('name')
        self.fields['theme'].queryset = MoETheme.objects.order_by('name')

    def clean_layers(self):
        layers = self.cleaned_data.get('layers', 'auto').strip()
        if layers.lower() == 'auto':
            return 'auto'
        ids = [x.strip() for x in layers.split(',') if x.strip()]
        if not ids:
            raise ValidationError(_("Укажите 'auto' или номера слоёв через запятую"))
        try:
            nums = [int(x) for x in ids]
        except ValueError:
            raise ValidationError(_("Слои — целые числа через запятую или 'auto'"))
        if any(n < 0 for n in nums):
            raise ValidationError(_("Номера слоёв неотрицательны"))
        return ','.join(str(n) for n in nums)

    def clean_alpha(self):
        alpha = self.cleaned_data.get('alpha', 0.2)
        if not (0.01 <= alpha <= 2.0):
            raise ValidationError(_("α в диапазоне 0.01–2.0 (сетка лабы: 0.1/0.2/0.4)"))
        return alpha


class RerankerModelForm(forms.ModelForm):
    class Meta:
        model = RerankerModel
        fields = ['name', 'display_name', 'model_id', 'max_length', 'is_active']
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'dipty-russian-crossencoder'}),
            'model_id': forms.TextInput(attrs={'placeholder': 'DiTy/cross-encoder-russian-msmarco'}),
        }
