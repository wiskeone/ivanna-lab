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

from .core import MultipleFileField, MultipleFileInput



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
