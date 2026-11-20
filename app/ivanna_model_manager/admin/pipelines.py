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



class ProcessingPipelineAdmin(ModelAdmin):
    """Админка для пайплайнов обработки"""
    list_display = ('name', 'is_active', 'is_default', 'avg_processing_time_display', 
                    'success_rate_display', 'created_at')
    list_filter = ('is_active', 'is_default', 'created_at')
    search_fields = ('name', 'description')
    
    def avg_processing_time_display(self, obj):
        if obj.avg_processing_time_ms:
            return f"{obj.avg_processing_time_ms:.0f} мс"
        return "—"
    avg_processing_time_display.short_description = 'Среднее время'
    
    def success_rate_display(self, obj):
        if obj.success_rate:
            return f"{obj.success_rate*100:.1f}%"
        return "—"
    success_rate_display.short_description = 'Успешность'


class DocumentProcessorAdmin(ModelAdmin):
    """Админка для логов обработки документов"""
    list_display = ('document', 'status', 'progress_display', 
                    'processing_time_display', 'started_at')
    list_filter = ('status', 'pipeline', 'started_at')
    search_fields = ('document__title', 'document__original_filename', 
                    'worker_id')
    readonly_fields = ('started_at', 'completed_at')
    
    def progress_display(self, obj):
        return f"{obj.progress}%"
    progress_display.short_description = 'Прогресс'
    
    def processing_time_display(self, obj):
        if obj.processing_time_ms:
            return f"{obj.processing_time_ms:.0f} мс"
        return "—"
    processing_time_display.short_description = 'Время обработки'
