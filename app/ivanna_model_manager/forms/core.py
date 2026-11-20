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

from ..models import (
    Experiment, ModelCheckpoint, ModelVersion,
    ModelMergeJob, InferenceSession, ModelEvaluation,
    ChatSession, ChatMessage, ModelComparison,
    FineTuningJob, ModelDownloadJob, EvaluationDataset,
    ModelEvaluationTask, EvaluationResult,
    Document, DocumentCollection, DocumentChunk,
    FineTuningJob, FineTuningTemplate, UserProfile,
    ModelConversionJob, EmbeddingModel, DistillationAnswerJob, DistillationQuestionSet,
    DistillationDataset, ManualInstructionSet, HuggingFaceDataset,
    MoETheme, MoEExpertJob, RerankerModel,
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
            'huggingface_token': forms.PasswordInput(render_value=True),  # mask input, show dots
        }


class ModelUploadForm(forms.ModelForm):
    """Model upload form with extended fields."""
    
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
    
    # Hugging Face fields
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
    
    # Local path fields
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
    
    # Quantization
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
    
    # Extra settings
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
    
    # Auto-fill fields from the API
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
            'model_type': forms.HiddenInput(),  # hidden: selected via model type cards
            'parameter_count': forms.HiddenInput(),
            'context_length': forms.HiddenInput(),
            'vocab_size': forms.HiddenInput(),
            'license_type': forms.HiddenInput(),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Make these fields optional — they are filled in automatically
        self.fields['parameter_count'].required = False
        self.fields['context_length'].required = False
        self.fields['vocab_size'].required = False
        self.fields['license_type'].required = False
        
        # Set initial values
        self.fields['license_type'].initial = 'Apache 2.0'
    
    def clean(self):
        cleaned_data = super().clean()
        source_type = cleaned_data.get('source_type')
        model_type = cleaned_data.get('model_type')
        
        # Validation depending on source type
        if source_type == 'hf':
            hf_id = cleaned_data.get('huggingface_id')
            if not hf_id:
                raise ValidationError({
                    'huggingface_id': 'Укажите Hugging Face ID для загрузки с Hub'
                })
            
            # Check HF ID format
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
        
        # Auto-fill parameters based on model type
        if model_type and cleaned_data.get('auto_fill_from_hf', True):
            self.auto_fill_parameters(cleaned_data, model_type)
        
        return cleaned_data
    
    def auto_fill_parameters(self, cleaned_data, model_type):
        """Auto-fill parameters based on model type."""
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
        
        # Handling depending on source type
        source_type = self.cleaned_data.get('source_type')
        
        if source_type == 'hf':
            instance.huggingface_revision = self.cleaned_data.get('huggingface_revision', 'main')
            
            # Auto-generate the local path from the HF ID
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
        
        # Set quantization
        quantization = self.cleaned_data.get('quantization')
        if quantization and quantization != 'none':
            instance.quantization = quantization
        
        # Set safety flags
        if self.cleaned_data.get('mark_safe'):
            instance.is_safe = True
            instance.safety_score = 0.9
        
        # Build loader_config
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
    """Local model add form (no Hugging Face)."""
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
            'serving_backend',  # v5.0: transformers or vLLM
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
        # A path existence check can be added here if needed
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
        instance.status = 'PENDING'  # local model needs no download; 'DOWNLOADED' if files exist
        # Switch the status to 'DOWNLOADED' when the files exist
        if instance.local_path and os.path.exists(instance.local_path):
            instance.status = 'DOWNLOADED'
        if commit:
            instance.save()
        return instance


class ModelCheckpointForm(forms.ModelForm):
    # Extra convenience fields
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Bootstrap classes (same as in MoE forms)
        for name, field in self.fields.items():
            w = field.widget
            css = ('form-check-input' if isinstance(w, forms.CheckboxInput)
                   else 'form-select' if isinstance(w, (forms.Select, forms.SelectMultiple))
                   else 'form-control')
            w.attrs['class'] = (w.attrs.get('class', '') + ' ' + css).strip()

    def clean(self):
        cleaned_data = super().clean()
        
        # If auto_generate_name is checked and huggingface_id is set
        if cleaned_data.get('auto_generate_name') and cleaned_data.get('huggingface_id'):
            hf_id = cleaned_data['huggingface_id']
            # Generate the name from the HF ID
            name = hf_id.replace('/', '_').replace('.', '_').lower()
            cleaned_data['name'] = name
            
            # Generate display_name
            display_parts = hf_id.split('/')
            if len(display_parts) == 2:
                org, model = display_parts
                cleaned_data['display_name'] = f"{org}/{model}"
        
        return cleaned_data
    
    def clean_huggingface_id(self):
        hf_id = self.cleaned_data.get('huggingface_id', '').strip()
        if hf_id:
            # Check the format
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
    """Model inference test form."""
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


# ----------------------------------------------------------------------
# Custom fields and widgets
# ----------------------------------------------------------------------
class MultipleFileInput(forms.ClearableFileInput):
    """Custom widget for multiple file selection."""
    allow_multiple_selected = True
    
    def __init__(self, attrs=None):
        super().__init__(attrs)
        if attrs is None:
            attrs = {}
        attrs['multiple'] = 'multiple'
        self.attrs.update(attrs)


class MultipleFileField(forms.FileField):
    """Custom field for multiple file upload."""
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
