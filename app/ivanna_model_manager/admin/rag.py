from django.contrib import admin
from django.utils.translation import gettext_lazy as _
from django.contrib import messages
from django.contrib.admin import ModelAdmin
from django.utils.html import format_html
from django.db.models import Count, Avg, Q
from django.urls import reverse
from django.shortcuts import redirect
from django import forms
import json
from django.conf import settings
import os
from datetime import datetime, timedelta

from ..models import (
    Experiment, ModelCheckpoint, ModelVersion,
    ModelMergeJob, InferenceSession, ModelEvaluation,
    ChatSession, ChatMessage, ModelComparison,
    FineTuningJob, DocumentCollection, Document,
    DocumentChunk, EmbeddingModel,
    SearchQuery, SearchResult, RAGSession, RetrievedChunk,
    SearchQualityFeedback, RAGEvaluation, ProcessingPipeline,
    DocumentProcessor, DeletedDocumentLog,
    MoETheme, MoEExpertJob,
    Benchmark, BenchmarkRun
)



class DocumentCollectionAdmin(ModelAdmin):
    """Admin for document collections."""
    list_display = ('name', 'owner_id', 'is_public', 'is_active', 
                    'total_documents_display', 'total_size_display', 'created_at')
    list_filter = ('is_public', 'is_active', 'created_at', 'chunking_strategy')
    search_fields = ('name', 'description', 'owner_id')
    readonly_fields = ('created_at', 'updated_at', 'total_documents', 
                      'total_chunks', 'total_size_mb', 'total_documents_display',
                      'total_chunks_display', 'total_size_display')  # display methods added to readonly_fields
    fieldsets = (
        (_('Основная информация'), {
            'fields': ('name', 'description', 'owner_id', 'is_public', 'is_active')
        }),
        (_('Настройки обработки'), {
            'fields': ('chunking_strategy', 'chunk_size', 'chunk_overlap', 
                      'auto_process')
        }),
        (_('Фильтры и метаданные'), {
            'fields': ('metadata_filters',)
        }),
        (_('Статистика'), {
            'fields': ('total_documents_display', 'total_chunks_display', 
                      'total_size_display'),
            'classes': ('collapse',)
        }),
        (_('Системная информация'), {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def total_documents_display(self, obj):
        return f"{obj.total_documents} док."
    total_documents_display.short_description = 'Документов'
    
    def total_chunks_display(self, obj):
        return f"{obj.total_chunks} чанков"
    total_chunks_display.short_description = 'Чанков'
    
    def total_size_display(self, obj):
        return f"{obj.total_size_mb:.1f} МБ"
    total_size_display.short_description = 'Размер'


class DocumentChunkInline(admin.TabularInline):
    """Inline admin for document chunks."""
    model = DocumentChunk
    extra = 0
    fields = ('chunk_index', 'content_preview', 'token_count', 
              'has_embedding', 'relevance_score')
    readonly_fields = ('content_preview',)
    
    def content_preview(self, obj):
        return obj.content[:100] + ('...' if len(obj.content) > 100 else '')
    content_preview.short_description = 'Содержимое'


class DocumentAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'title_display', 'original_filename', 'document_type', 
        'status', 'collection_display', 'uploaded_at_short', 'file_size_display'
    ]
    list_filter = ['status', 'document_type', 'collection', 'is_active', 'uploaded_at']
    search_fields = ['title', 'original_filename', 'description', 'raw_content']
    readonly_fields = [
        'id', 'uploaded_at', 'processed_at', 'indexed_at', 
        'content_hash', 'raw_content_preview', 'file_size_display',
        'chunk_count_display', 'extension_display'
    ]
    fieldsets = (
        (_('Основная информация'), {
            'fields': ('title', 'original_filename', 'collection', 'description', 'category')
        }),
        (_('Тип и содержимое'), {
            'fields': ('document_type', 'content_hash', 'raw_content_preview', 'processed_content')
        }),
        (_('Файл'), {
            'fields': ('file_path', 'file_size_display', 'file_encoding')
        }),
        (_('Статус и обработка'), {
            'fields': ('status', 'processing_progress', 'error_message', 'quality_score')
        }),
        (_('Метаданные'), {
            'fields': ('metadata', 'tags'),
            'classes': ('collapse',)
        }),
        (_('Даты'), {
            'fields': ('uploaded_at', 'processed_at', 'indexed_at'),
            'classes': ('collapse',)
        }),
        (_('Системные поля'), {
            'fields': ('is_active', 'requires_reindexing', 'chunk_count_display'),
            'classes': ('collapse',)
        })
    )
    
    def title_display(self, obj):
        """Display title."""
        return obj.title or obj.original_filename
    title_display.short_description = 'Название'
    
    def collection_display(self, obj):
        """Display collection."""
        if obj.collection:
            return obj.collection.name
        return "Без коллекции"
    collection_display.short_description = 'Коллекция'
    
    def uploaded_at_short(self, obj):
        """Short uploaded-at format."""
        return obj.uploaded_at.strftime('%d.%m.%Y %H:%M') if obj.uploaded_at else ''
    uploaded_at_short.short_description = 'Загружен'
    
    def file_size_display(self, obj):
        """Display file size."""
        if obj.file_size_mb:
            return f"{obj.file_size_mb:.2f} MB"
        return "Неизвестно"
    file_size_display.short_description = 'Размер файла'
    
    def raw_content_preview(self, obj):
        """Content preview."""
        if obj.raw_content:
            content = obj.raw_content[:500]
            if len(obj.raw_content) > 500:
                content += "..."
            return format_html(
                '<div style="max-height: 200px; overflow-y: auto; white-space: pre-wrap; font-family: monospace;">{}</div>',
                content
            )
        return "Нет содержимого"
    raw_content_preview.short_description = 'Предпросмотр содержимого'
    
    def chunk_count_display(self, obj):
        """Display chunk count."""
        return obj.chunk_count
    chunk_count_display.short_description = 'Чанков'
    
    def extension_display(self, obj):
        """Display file extension."""
        try:
            return getattr(obj, 'extension', '')
        except:
            return ''
    extension_display.short_description = 'Расширение'
    
    def get_queryset(self, request):
        """Queryset optimization."""
        return super().get_queryset(request).select_related('collection')
    
    def save_model(self, request, obj, form, change):
        """Handle model save in admin."""
        if not obj.title and obj.original_filename:
            obj.title = os.path.splitext(obj.original_filename)[0]
        super().save_model(request, obj, form, change)


class DocumentChunkAdmin(ModelAdmin):
    """Admin for document chunks."""
    list_display = ('document_title', 'chunk_index', 'token_count', 
                    'has_embedding', 'relevance_score', 'usage_count')
    list_filter = ('has_embedding', 'chunk_strategy', 'is_active', 
                   'embedding_model')
    search_fields = ('content', 'document__title', 'document__original_filename')
    readonly_fields = ('created_at', 'updated_at', 'preview_display', 
                      'document_title')
    fieldsets = (
        (_('Связи'), {
            'fields': ('document', 'chunk_index')
        }),
        (_('Содержимое'), {
            'fields': ('content', 'token_count', 'preview_display')
        }),
        (_('Позиция в документе'), {
            'fields': ('start_char', 'end_char', 'start_line', 'end_line')
        }),
        (_('Эмбеддинги'), {
            'fields': ('has_embedding', 'embedding_model', 'embedding_version', 
                      'embedding_dim')
        }),
        (_('Качество'), {
            'fields': ('relevance_score', 'is_key_chunk')
        }),
        (_('Использование'), {
            'fields': ('usage_count', 'last_used')
        }),
        (_('Метаданные'), {
            'fields': ('metadata', 'chunk_strategy')
        }),
        (_('Системная информация'), {
            'fields': ('is_active', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def document_title(self, obj):
        return obj.document.title
    document_title.short_description = 'Документ'
    
    def preview_display(self, obj):
        return obj.preview
    preview_display.short_description = 'Превью'


class EmbeddingModelAdmin(ModelAdmin):
    """Admin for embedding models."""
    list_display = ('name', 'model_type', 'embedding_dim', 'is_default', 
                    'is_active', 'cost_per_1k_display')
    list_filter = ('model_type', 'is_active', 'is_default', 'is_multilingual')
    search_fields = ('name', 'model_id', 'description')
    readonly_fields = ('created_at', 'updated_at', 'total_tokens_display')
    
    def cost_per_1k_display(self, obj):
        if obj.cost_per_1k_tokens:
            return f"${obj.cost_per_1k_tokens:.4f}"
        return "Бесплатно"
    cost_per_1k_display.short_description = 'Стоимость/1K токенов'
    
    def total_tokens_display(self, obj):
        return f"{obj.total_tokens_processed:,}"
    total_tokens_display.short_description = 'Обработано токенов'
