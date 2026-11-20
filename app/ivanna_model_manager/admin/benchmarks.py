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



@admin.register(Benchmark)
class BenchmarkAdmin(ModelAdmin):
    list_display = ('name', 'display_name', 'category', 'source', 'is_popular',
                    'install_status', 'n_samples')
    list_filter = ('source', 'category', 'install_status', 'is_popular')
    search_fields = ('name', 'display_name', 'description')


@admin.register(BenchmarkRun)
class BenchmarkRunAdmin(ModelAdmin):
    list_display = ('name', 'model', 'engine', 'status', 'progress', 'created_by', 'created_at')
    list_filter = ('status', 'engine')
    search_fields = ('name', 'model__name')
