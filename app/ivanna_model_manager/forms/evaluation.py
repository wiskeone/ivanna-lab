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


class EvaluationDatasetForm(forms.ModelForm):
    """Form for creating/editing a dataset."""
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
            'source_type': forms.HiddenInput(),  # Hide the field
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['tags'].required = False
        self.fields['description'].required = False
        # Set the default value for source_type
        self.fields['source_type'].initial = 'MANUAL'
    
    def clean_tags(self):
        """Convert the tags string into a JSON array."""
        tags = self.cleaned_data.get('tags', '')
        if isinstance(tags, str):
            # If it is a string, convert to a list
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            return tag_list
        elif isinstance(tags, list):
            # If already a list, return as is
            return tags
        return []


class QuestionUploadForm(forms.Form):
    """Form for uploading questions from a file."""
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
            # Check file size (max 10MB)
            if file.size > 10 * 1024 * 1024:
                raise ValidationError(_('Размер файла не должен превышать 10MB'))
            
            # Check file extension
            valid_extensions = ['.txt', '.csv', '.json']
            if not any(file.name.lower().endswith(ext) for ext in valid_extensions):
                raise ValidationError(_('Поддерживаются только файлы .txt, .csv, .json'))
        
        return file


class CreateEvaluationTaskForm(forms.Form):
    """Form for creating an evaluation task."""
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
    
    # Quantization selection field
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
    
    # Generation parameters
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
            status='LOADED'  # only loaded models to avoid extra downloads
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
    """Form for rating a response."""
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
