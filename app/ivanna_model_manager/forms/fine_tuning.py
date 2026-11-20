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



#############################################################################
# FineTuning
#############################################################################
class FineTuningJobForm(forms.ModelForm):
    class Meta:
        model = FineTuningJob
        fields = [
            'name', 'description', 'base_model', 'method', 'quantization',
            'lora_r', 'lora_alpha', 'lora_dropout',
            # v5.3 Tier-1: LoRA++ pack
            'use_dora', 'use_rslora', 'lora_plus_lr_ratio', 'neftune_alpha',
            'target_preset', 'target_modules',
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
        # v5.3: bootstrap classes for the Tier-1 fields
        for fname in ['use_dora', 'use_rslora']:
            w = self.fields[fname].widget
            w.attrs['class'] = (w.attrs.get('class', '') + ' form-check-input').strip()
        for fname, css in [('lora_plus_lr_ratio', 'form-control'),
                           ('neftune_alpha', 'form-control'),
                           ('target_preset', 'form-select form-select-sm')]:
            w = self.fields[fname].widget
            w.attrs['class'] = (w.attrs.get('class', '') + ' ' + css).strip()

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
        # form-level lazy labels: the model's plain-string choices stay
        # migration-stable, the form translates them per active language
        choices=[(v, _(label)) for v, label in ModelConversionJob.QUANTIZATION_CHOICES],
        initial='q4_0',
        label=_('Квантизация'),
        help_text=_('Тип квантизации для GGUF')
    )

    class Meta:
        model = ModelConversionJob
        fields = ['name', 'description', 'source_model', 'quantization', 'config']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'config': forms.Textarea(attrs={'rows': 4, 'class': 'font-monospace'}),
        }
        help_texts = {
            'config': _('JSON с дополнительными параметрами (опционально)'),
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
        # Bootstrap-классы (как в MoE-формах)
        for name, field in self.fields.items():
            w = field.widget
            css = ('form-check-input' if isinstance(w, forms.CheckboxInput)
                   else 'form-select' if isinstance(w, (forms.Select, forms.SelectMultiple))
                   else 'form-control')
            w.attrs['class'] = (w.attrs.get('class', '') + ' ' + css).strip()

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
