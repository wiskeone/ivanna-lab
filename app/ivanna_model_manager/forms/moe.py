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



MOE_PROMPT_FIELD_HELP = _(
    "По одному промпту на строку. Рекомендуется 25–30 промптов; "
    "длины доменных и контрастных наборов должны быть выровнены "
    "(иначе длина протечёт в направление).")


class MoEThemeForm(forms.ModelForm):
    """Soft MoE theme: domain + contrast (and optional eval sets)."""

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            w = field.widget
            css = ('form-check-input' if isinstance(w, forms.CheckboxInput)
                   else 'form-select' if isinstance(w, (forms.Select, forms.SelectMultiple))
                   else 'form-control')
            w.attrs['class'] = (w.attrs.get('class', '') + ' ' + css).strip()

    def clean_name(self):
        name = self.cleaned_data['name']
        if len(name) < 2:
            raise ValidationError(_("Название минимум 2 символа"))
        return name

    def save(self, commit=True):
        from ..services.soft_moe import prompts as moe_prompts

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
    """Job to build a soft MoE expert from a dense model."""

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
        for name, field in self.fields.items():
            w = field.widget
            css = ('form-check-input' if isinstance(w, forms.CheckboxInput)
                   else 'form-select' if isinstance(w, (forms.Select, forms.SelectMultiple))
                   else 'form-control')
            w.attrs['class'] = (w.attrs.get('class', '') + ' ' + css).strip()
        # Only dense models offloaded to disk
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
