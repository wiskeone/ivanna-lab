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



class ExperimentForm(forms.ModelForm):
    class Meta:
        model = Experiment
        fields = ['name', 'description', 'base_checkpoint', 'tags', 'is_active']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
            'tags': forms.TextInput(attrs={
                'placeholder': _('Теги через запятую: ru, code, medical')
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Bootstrap-классы (как в MoE-формах)
        for name, field in self.fields.items():
            w = field.widget
            css = ('form-check-input' if isinstance(w, forms.CheckboxInput)
                   else 'form-select' if isinstance(w, (forms.Select, forms.SelectMultiple))
                   else 'form-control')
            w.attrs['class'] = (w.attrs.get('class', '') + ' ' + css).strip()

    def clean_tags(self):
        tags = self.cleaned_data.get('tags', [])
        if isinstance(tags, str):
            # Преобразуем строку в список
            tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
        return tags
