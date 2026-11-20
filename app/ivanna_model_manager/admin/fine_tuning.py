from django.contrib import admin
from django.utils.translation import gettext_lazy as _
from django.contrib import messages
from django.contrib.admin import ModelAdmin
from django.utils.html import format_html
from django.utils.safestring import mark_safe
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



class FineTuningJobAdminForm(forms.ModelForm):
    """Кастомная форма для задач дообучения"""
    class Meta:
        model = FineTuningJob
        fields = '__all__'
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'data_config': forms.Textarea(attrs={'rows': 6}),
            'hyperparameters': forms.Textarea(attrs={'rows': 8, 'style': 'font-family: monospace'}),
            'training_metrics': forms.Textarea(attrs={'rows': 10, 'style': 'font-family: monospace'}),
            'logs': forms.Textarea(attrs={'rows': 12}),
            'error_message': forms.Textarea(attrs={'rows': 4}),
        }


class FineTuningJobAdmin(admin.ModelAdmin):
    form = FineTuningJobAdminForm
    list_display = (
        'name', 'base_model_link', 'status_badge', 'progress_bar',
        'current_step_display', 'created_at', 'duration_display'
    )
    list_filter = ('status', 'created_at')
    search_fields = ('name', 'description', 'base_model__name', 'output_model__name')
    readonly_fields = (
        'created_at', 'started_at', 'completed_at', 'progress_bar',
        'status_history', 'logs_preview', 'training_metrics_display',
        'data_info', 'hyperparameters_display', 'job_stats'
    )
    actions = ['start_jobs', 'retry_failed_jobs', 'cancel_jobs']
    raw_id_fields = ('base_model', 'output_model')
    
    fieldsets = (
        (_('Основная информация'), {
            'fields': ('name', 'description', 'base_model')
        }),
        (_('Данные для обучения'), {
            'fields': ('data_info',),
            'classes': ('collapse',)
        }),
        (_('Гиперпараметры'), {
            'fields': ('hyperparameters_display',),
            'classes': ('collapse',)
        }),
        (_('Прогресс и статус'), {
            'fields': ('status', 'progress_bar', 'current_step', 'error_message')
        }),
        (_('Результаты'), {
            'fields': ('output_model', 'training_metrics_display'),
            'classes': ('collapse',)
        }),
        (_('Логи выполнения'), {
            'fields': ('logs_preview',),
            'classes': ('collapse',)
        }),
        (_('Временные метки'), {
            'fields': ('created_at', 'started_at', 'completed_at'),
            'classes': ('collapse',)
        }),
        (_('Системная информация'), {
            'fields': ('status_history', 'job_stats'),
            'classes': ('collapse',)
        }),
    )
    
    def base_model_link(self, obj):
        if obj.base_model:
            url = reverse('admin:ivanna_model_manager_modelcheckpoint_change', args=[obj.base_model.id])
            return format_html('<a href="{}">{}</a>', url, obj.base_model.name)
        return "-"
    base_model_link.short_description = 'Базовая модель'
    
    def status_badge(self, obj):
        """Бейдж статуса"""
        colors = {
            'PENDING': 'gray',
            'PREPROCESSING': 'blue',
            'TRAINING': 'orange',
            'EVALUATING': 'purple',
            'COMPLETED': 'green',
            'FAILED': 'red',
        }
        color = colors.get(obj.status, 'black')
        return format_html(
            '<span style="padding: 2px 8px; background: {}; color: white; '
            'border-radius: 10px; font-size: 12px;">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Статус'
    
    def progress_bar(self, obj):
        """Прогресс-бар"""
        color = 'green' if obj.status == 'COMPLETED' else 'orange' if obj.status in ['TRAINING', 'EVALUATING'] else 'blue'
        return format_html(
            '''
            <div style="width: 100px; background-color: #eee; border-radius: 3px; display: inline-block;">
                <div style="width: {}%; background-color: {}; height: 20px; border-radius: 3px; text-align: center; color: white; font-size: 12px; line-height: 20px;">
                    {}%
                </div>
            </div>
            ''',
            obj.progress, color, obj.progress
        )
    progress_bar.short_description = 'Прогресс'
    
    def current_step_display(self, obj):
        return obj.current_step[:50] + "..." if len(obj.current_step) > 50 else obj.current_step
    current_step_display.short_description = 'Текущий шаг'
    
    def duration_display(self, obj):
        if obj.started_at and obj.completed_at:
            duration = obj.completed_at - obj.started_at
            minutes = duration.total_seconds() // 60
            seconds = duration.total_seconds() % 60
            return f"{int(minutes)}:{int(seconds):02d}"
        elif obj.started_at:
            return "Выполняется..."
        return "Не начата"
    duration_display.short_description = 'Длительность'
    
    def data_info(self, obj):
        """Информация о данных"""
        info = []
        if obj.training_data_path:
            info.append(f"<strong>Данные обучения:</strong> {obj.training_data_path}")
        if obj.validation_data_path:
            info.append(f"<strong>Данные валидации:</strong> {obj.validation_data_path}")
        
        if info:
            return mark_safe('<br>'.join(info))
        return "Пути к данным не указаны"
    data_info.short_description = 'Информация о данных'
    
    def hyperparameters_display(self, obj):
        """Отображение гиперпараметров"""
        if not obj.hyperparameters:
            return "Гиперпараметры не заданы"
        
        return format_html(
            '<pre style="background: #f8f9fa; padding: 10px; '
            'border-radius: 5px; overflow: auto;">{}</pre>',
            json.dumps(obj.hyperparameters, indent=2, ensure_ascii=False)
        )
    hyperparameters_display.short_description = 'Гиперпараметры'
    
    def training_metrics_display(self, obj):
        """Отображение метрик обучения"""
        if not obj.training_metrics:
            return "Метрики не доступны"
        
        metrics_html = []
        for key, value in obj.training_metrics.items():
            if isinstance(value, (int, float)):
                metrics_html.append(f"<tr><td><strong>{key}</strong></td><td>{value:.4f}</td></tr>")
            elif isinstance(value, dict):
                # Для вложенных словарей
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        metrics_html.append(f"<tr><td><strong>{key}.{sub_key}</strong></td><td>{sub_value:.4f}</td></tr>")
            else:
                metrics_html.append(f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>")
        
        return format_html(
            '<table style="width: 100%; border-collapse: collapse;">'
            '<tr style="background: #f8f9fa;"><th>Метрика</th><th>Значение</th></tr>'
            '{}'
            '</table>',
            ''.join(metrics_html)
        )
    training_metrics_display.short_description = 'Метрики обучения'
    
    def logs_preview(self, obj):
        """Предпросмотр логов"""
        if not obj.logs:
            return "Нет логов"
        
        lines = obj.logs.strip().split('\n')[-20:]  # последние 20 строк
        preview = '\n'.join(lines)
        return format_html(
            '<textarea style="width: 100%; height: 200px; font-family: monospace; '
            'background: #f8f9fa;" readonly>{}</textarea>',
            preview
        )
    logs_preview.short_description = 'Логи выполнения'
    
    def status_history(self, obj):
        """История статусов"""
        history = []
        if obj.created_at:
            history.append(f"📅 <strong>Создана:</strong> {obj.created_at.strftime('%d.%m.%Y %H:%M')}")
        if obj.started_at:
            history.append(f"▶️ <strong>Начата:</strong> {obj.started_at.strftime('%d.%m.%Y %H:%M')}")
        if obj.completed_at:
            history.append(f"✅ <strong>Завершена:</strong> {obj.completed_at.strftime('%d.%m.%Y %H:%M')}")
        
        return mark_safe('<br>'.join(history))
    status_history.short_description = 'История'
    
    def job_stats(self, obj):
        """Статистика задачи"""
        stats = []
        
        if obj.started_at and obj.completed_at:
            duration = obj.completed_at - obj.started_at
            hours = duration.total_seconds() // 3600
            minutes = (duration.total_seconds() % 3600) // 60
            stats.append(f"<strong>Общее время:</strong> {int(hours)}ч {int(minutes)}м")
        
        stats.append(f"<strong>Прогресс:</strong> {obj.progress}%")
        stats.append(f"<strong>Текущий шаг:</strong> {obj.current_step}")
        
        if obj.training_metrics:
            stats.append(f"<strong>Метрики:</strong> {len(obj.training_metrics)} записей")
        
        return mark_safe('<br>'.join(stats))
    job_stats.short_description = 'Статистика задачи'
    
    @admin.action(description='Запустить выбранные задачи')
    def start_jobs(self, request, queryset):
        runnable = queryset.filter(status__in=['PENDING', 'FAILED'])
        count = runnable.update(status='PREPROCESSING', started_at=datetime.now())
        self.message_user(request, f'Запущено {count} задач')
    
    @admin.action(description='Повторить неудачные задачи')
    def retry_failed_jobs(self, request, queryset):
        failed_jobs = queryset.filter(status='FAILED')
        count = failed_jobs.count()
        
        for job in failed_jobs:
            # Создаем новую задачу как копию старой
            FineTuningJob.objects.create(
                name=f"{job.name}_retry",
                description=f"Повтор задачи {job.id}",
                base_model=job.base_model,
                training_data_path=job.training_data_path,
                validation_data_path=job.validation_data_path,
                data_config=job.data_config.copy(),
                hyperparameters=job.hyperparameters.copy()
            )
        
        self.message_user(request, f'Создано {count} задач для повторного выполнения')
    
    @admin.action(description='Отменить выбранные задачи')
    def cancel_jobs(self, request, queryset):
        cancellable = queryset.filter(status__in=['PENDING', 'PREPROCESSING', 'TRAINING', 'EVALUATING'])
        count = cancellable.update(status='CANCELLED')
        self.message_user(request, f'Отменено {count} задач')
