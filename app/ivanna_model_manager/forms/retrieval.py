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



# ----------------------------------------------------------------------
# Embedding model upload
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
        # On edit (instance with pk), exclude the current object
        if self.instance and self.instance.pk:
            if EmbeddingModel.objects.filter(name=name).exclude(pk=self.instance.pk).exists():
                raise ValidationError(_('Модель с таким именем уже существует.'))
        else:
            if EmbeddingModel.objects.filter(name=name).exists():
                raise ValidationError(_('Модель с таким именем уже существует.'))
        return name


class RerankerModelForm(forms.ModelForm):
    class Meta:
        model = RerankerModel
        fields = ['name', 'display_name', 'model_id', 'max_length', 'is_active']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            w = field.widget
            css = ('form-check-input' if isinstance(w, forms.CheckboxInput)
                   else 'form-select' if isinstance(w, (forms.Select, forms.SelectMultiple))
                   else 'form-control')
            w.attrs['class'] = (w.attrs.get('class', '') + ' ' + css).strip()
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'dipty-russian-crossencoder'}),
            'model_id': forms.TextInput(attrs={'placeholder': 'DiTy/cross-encoder-russian-msmarco'}),
        }
