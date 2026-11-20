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
