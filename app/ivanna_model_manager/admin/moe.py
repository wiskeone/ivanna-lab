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



@admin.register(MoETheme)
class MoEThemeAdmin(ModelAdmin):
    list_display = ('name', 'slug', 'validation_status', 'created_by', 'created_at')
    list_filter = ('validation_status',)
    search_fields = ('name', 'description')


@admin.register(MoEExpertJob)
class MoEExpertJobAdmin(ModelAdmin):
    list_display = ('name', 'theme', 'base_checkpoint', 'status', 'progress', 'created_at')
    list_filter = ('status', 'mode')
    search_fields = ('name',)
