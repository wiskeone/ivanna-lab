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



class RAGSessionAdmin(ModelAdmin):
    """Админка для RAG сессий"""
    list_display = ('session_id', 'user_id', 'llm_model', 'status', 
                    'total_time_display', 'has_feedback', 'created_at')
    list_filter = ('status', 'has_feedback', 'embedding_model', 'llm_model', 
                   'created_at')
    search_fields = ('session_id', 'user_id', 'original_query', 
                    'generated_answer')
    readonly_fields = ('created_at', 'started_at', 'completed_at', 
                      'feedback_at', 'total_tokens_display')
    fieldsets = (
        (_('Основная информация'), {
            'fields': ('session_id', 'conversation_id', 'user_id', 'status')
        }),
        (_('Запрос и контекст'), {
            'fields': ('original_query', 'system_prompt', 'augmented_prompt')
        }),
        (_('Модели'), {
            'fields': ('embedding_model', 'llm_model', 'search_query')
        }),
        (_('Ответ'), {
            'fields': ('generated_answer', 'generation_metadata', 
                      'confidence_score')
        }),
        (_('Производительность'), {
            'fields': ('total_time_display', 'search_time_ms', 
                      'generation_time_ms')
        }),
        (_('Ресурсы и стоимость'), {
            'fields': ('input_tokens', 'output_tokens', 'total_tokens_display', 
                      'estimated_cost')
        }),
        (_('Обратная связь'), {
            'fields': ('has_feedback', 'feedback_score', 'feedback_comment', 
                      'feedback_at')
        }),
        (_('Временные метки'), {
            'fields': ('created_at', 'started_at', 'completed_at'),
            'classes': ('collapse',)
        }),
    )
    
    def total_time_display(self, obj):
        return f"{obj.total_time_ms:.0f} мс"
    total_time_display.short_description = 'Общее время'
    
    def total_tokens_display(self, obj):
        return obj.total_tokens
    total_tokens_display.short_description = 'Всего токенов'


class SearchResultInline(admin.TabularInline):
    """Инлайн-админка для результатов поиска"""
    model = SearchResult
    extra = 0
    fields = ('chunk', 'relevance_score', 'rank_position', 'scoring_method')
    readonly_fields = ('chunk',)


class SearchQueryAdmin(ModelAdmin):
    """Админка для поисковых запросов"""
    list_display = ('query_preview', 'search_type', 'user_id', 
                    'collection_filter', 'created_at')
    list_filter = ('search_type', 'created_at')
    search_fields = ('query_text', 'normalized_query', 'user_id', 'session_id')
    readonly_fields = ('created_at', 'processing_time_display')
    
    inlines = [SearchResultInline]
    
    def query_preview(self, obj):
        return obj.query_text[:50] + ('...' if len(obj.query_text) > 50 else '')
    query_preview.short_description = 'Запрос'
    
    def processing_time_display(self, obj):
        if obj.processing_time_ms:
            return f"{obj.processing_time_ms:.0f} мс"
        return "—"
    processing_time_display.short_description = 'Время обработки'


class SearchResultAdmin(ModelAdmin):
    """Админка для результатов поиска"""
    list_display = ('query_preview', 'chunk_document', 'rank_position', 
                    'relevance_score', 'scoring_method')
    list_filter = ('scoring_method', 'query__search_type')
    search_fields = ('query__query_text', 'chunk__content', 
                    'chunk__document__title')
    
    def query_preview(self, obj):
        return obj.query.query_text[:50] + ('...' if len(obj.query.query_text) > 50 else '')
    query_preview.short_description = 'Запрос'
    
    def chunk_document(self, obj):
        return obj.chunk.document.title
    chunk_document.short_description = 'Документ'


class SearchQualityFeedbackAdmin(ModelAdmin):
    """Админка для обратной связи по качеству поиска"""
    list_display = ('feedback_type', 'rating', 'user_id', 'is_resolved', 
                    'created_at')
    list_filter = ('feedback_type', 'rating', 'is_resolved', 'created_at')
    search_fields = ('comment', 'user_id', 'search_result__chunk__content')
    
    actions = ['mark_as_resolved']
    
    def mark_as_resolved(self, request, queryset):
        """Пометить обратную связь как обработанную"""
        updated = queryset.update(is_resolved=True)
        self.message_user(request, f"{updated} отзывов помечены как обработанные.")
    mark_as_resolved.short_description = "Пометить как обработанные"


class RAGEvaluationAdmin(ModelAdmin):
    """Админка для оценки RAG системы"""
    list_display = ('name', 'evaluation_type', 'status', 'total_queries', 
                    'improvement_percentage', 'created_at')
    list_filter = ('evaluation_type', 'status', 'created_at')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at', 'started_at', 'completed_at')
    
    def total_queries(self, obj):
        """Отображение общего количества запросов"""
        return obj.total_queries
    total_queries.short_description = 'Количество запросов'
    
    def improvement_percentage_display(self, obj):
        """Отображение улучшения в процентах"""
        if obj.improvement_percentage:
            return f"{obj.improvement_percentage:.1f}%"
        return "—"
    improvement_percentage_display.short_description = 'Улучшение'
