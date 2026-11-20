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



class IsActiveFilter(admin.SimpleListFilter):
    """Фильтр для активных/неактивных записей"""
    title = 'Статус активности'
    parameter_name = 'is_active'
    
    def lookups(self, request, model_admin):
        return (
            ('yes', 'Активные'),
            ('no', 'Неактивные'),
        )
    
    def queryset(self, request, queryset):
        if self.value() == 'yes':
            return queryset.filter(is_active=True)
        if self.value() == 'no':
            return queryset.filter(is_active=False)


class ModelTypeFilter(admin.SimpleListFilter):
    """Фильтр по типу модели"""
    title = 'Тип модели'
    parameter_name = 'model_type'
    
    def lookups(self, request, model_admin):
        return ModelCheckpoint.MODEL_TYPE_CHOICES
    
    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(model_type=self.value())


class StatusFilter(admin.SimpleListFilter):
    """Фильтр по статусу чекпоинта"""
    title = 'Статус'
    parameter_name = 'status'
    
    def lookups(self, request, model_admin):
        return ModelCheckpoint.STATUS_CHOICES
    
    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(status=self.value())


# Кастомная форма для админки ModelCheckpoint
class ModelCheckpointAdminForm(forms.ModelForm):
    class Meta:
        model = ModelCheckpoint
        fields = '__all__'
        widgets = {
            'loader_config': forms.Textarea(attrs={'rows': 15}),
            'evaluation_metrics': forms.Textarea(attrs={'rows': 10}),
            'description': forms.Textarea(attrs={'rows': 4}),
            'tags': forms.TextInput(attrs={
                'placeholder': 'Введите теги через запятую: ru, code, medical'
            }),
        }


class ExperimentAdmin(admin.ModelAdmin):
    list_display = ('name', 'is_active', 'model_count', 'created_at', 'updated_at')
    list_filter = (IsActiveFilter, 'created_at')
    search_fields = ('name', 'description', 'tags')
    readonly_fields = ('created_at', 'updated_at', 'model_count')
    filter_horizontal = ()
    actions = ['activate_experiments', 'deactivate_experiments']
    
    fieldsets = (
        (_('Основная информация'), {
            'fields': ('name', 'description', 'base_checkpoint', 'is_active')
        }),
        (_('Теги и классификация'), {
            'fields': ('tags',),
            'classes': ('collapse',)
        }),
        (_('Системная информация'), {
            'fields': ('created_at', 'updated_at', 'model_count'),
            'classes': ('collapse',)
        }),
    )
    
    def model_count(self, obj):
        """Количество моделей в эксперименте"""
        return obj.checkpoints.count()
    model_count.short_description = 'Количество моделей'
    
    @admin.action(description='Активировать выбранные эксперименты')
    def activate_experiments(self, request, queryset):
        updated = queryset.update(is_active=True)
        self.message_user(request, f'Активировано {updated} экспериментов')
    
    @admin.action(description='Деактивировать выбранные эксперименты')
    def deactivate_experiments(self, request, queryset):
        updated = queryset.update(is_active=False)
        # Деактивируем и связанные модели
        for experiment in queryset:
            experiment.checkpoints.update(is_active=False)
        self.message_user(request, f'Деактивировано {updated} экспериментов и их модели')


class ModelVersionInline(admin.TabularInline):
    """Inline для версий модели"""
    model = ModelVersion
    extra = 0
    readonly_fields = ('created_at', 'file_hash', 'is_stable')
    fields = ('version', 'parent_version', 'is_stable', 'created_at', 'file_hash')
    can_delete = False
    show_change_link = True
    
    def has_add_permission(self, request, obj=None):
        return False


class ModelEvaluationInline(admin.TabularInline):
    """Inline для оценок модели"""
    model = ModelEvaluation
    extra = 0
    readonly_fields = ('evaluated_at', 'primary_metric_display')
    fields = ('dataset_name', 'primary_metric_display', 'evaluated_at', 'is_valid')
    
    def primary_metric_display(self, obj):
        return f"{obj.primary_metric:.3f}" if hasattr(obj, 'primary_metric') else "N/A"
    primary_metric_display.short_description = 'Основная метрика'
    
    def has_add_permission(self, request, obj=None):
        return False


class ModelCheckpointAdmin(admin.ModelAdmin):
    form = ModelCheckpointAdminForm  # Используем кастомную форму
    list_display = (
        'name', 'display_name', 'model_type', 'status_badge', 
        'size_gb_display', 'is_active', 'is_safe', 'created_at'
    )
    list_filter = (ModelTypeFilter, StatusFilter, IsActiveFilter, 'is_safe', 'quantization', 'created_at')
    search_fields = ('name', 'display_name', 'huggingface_id', 'description', 'tags')
    readonly_fields = (
        'created_at', 'updated_at', 'size_gb_display', 'is_quantized', 
        'huggingface_url', 'safe_name', 'config_hash', 'weights_hash',
        'download_size_mb_display', 'model_info_card'
    )
    list_editable = ('is_active', 'is_safe')
    actions = [
        'mark_as_safe', 'mark_as_unsafe', 'activate_models', 
        'deactivate_models', 'update_config_from_huggingface'
    ]
    filter_horizontal = ()
    save_on_top = True
    
    fieldsets = (
        (_('Основная информация'), {
            'fields': (
                'name', 'display_name', 'model_type', 'experiment', 
                'parent', 'is_active', 'is_public'
            )
        }),
        ('Hugging Face', {
            'fields': ('huggingface_id', 'huggingface_revision', 'huggingface_url'),
            'classes': ('collapse',)
        }),
        (_('Конфигурация и пути'), {
            'fields': ('loader_config', 'local_path', 'relative_path'),
            'classes': ('wide',)
        }),
        (_('Характеристики модели'), {
            'fields': (
                'parameter_count', 'size_gb_display', 'download_size_mb_display',
                'quantization', 'context_length', 'vocab_size'
            )
        }),
        (_('Безопасность и лицензия'), {
            'fields': ('license_type', 'is_safe', 'safety_score'),
            'classes': ('collapse',)
        }),
        (_('Производительность и метрики'), {
            'fields': ('inference_speed', 'accuracy_score', 'evaluation_metrics'),
            'classes': ('collapse',)
        }),
        (_('Статус и отслеживание'), {
            'fields': ('status', 'error_message', 'config_hash', 'weights_hash'),
            'classes': ('collapse',)
        }),
        (_('Теги'), {
            'fields': ('tags',),
            'classes': ('collapse',)
        }),
        (_('Системная информация'), {
            'fields': ('model_info_card', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    inlines = [ModelVersionInline, ModelEvaluationInline]
    
    def status_badge(self, obj):
        """Красивое отображение статуса"""
        colors = {
            'DOWNLOADING': 'orange',
            'LOADED': 'green',
            'MERGED': 'blue',
            'ERROR': 'red',
            'ARCHIVED': 'gray',
        }
        color = colors.get(obj.status, 'black')
        return format_html(
            '<span style="display: inline-block; padding: 3px 8px; '
            'border-radius: 10px; background-color: {}; color: white; '
            'font-size: 12px;">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Статус'
    status_badge.admin_order_field = 'status'
    
    def size_gb_display(self, obj):
        """Отображение размера в GB"""
        return f"{obj.size_gb} GB"
    size_gb_display.short_description = 'Размер'
    
    def download_size_mb_display(self, obj):
        """Отображение размера загрузки в MB"""
        return f"{obj.get_download_size_mb()} MB"
    download_size_mb_display.short_description = 'Размер загрузки'
    
    def model_info_card(self, obj):
        """Карточка с информацией о модели"""
        if not obj.pk:
            return "Сначала сохраните модель"
        
        info = f"""
        <div style="padding: 10px; background: #f8f9fa; border-radius: 5px;">
            <h4>📊 Статистика модели</h4>
            <p><strong>ID:</strong> {obj.id}</p>
            <p><strong>Безопасное имя:</strong> {obj.safe_name}</p>
            <p><strong>Квантизация:</strong> {'Да' if obj.is_quantized else 'Нет'}</p>
            <p><strong>Размер:</strong> {obj.size_gb} GB</p>
            <p><strong>Параметры:</strong> {obj.parameter_count:,}</p>
        </div>
        """
        return format_html(info)
    model_info_card.short_description = 'Информация о модели'
    
    @admin.action(description='Пометить как безопасные')
    def mark_as_safe(self, request, queryset):
        updated = queryset.update(is_safe=True, safety_score=0.9)
        self.message_user(request, f'{updated} моделей помечены как безопасные')
    
    @admin.action(description='Пометить как небезопасные')
    def mark_as_unsafe(self, request, queryset):
        updated = queryset.update(is_safe=False, safety_score=None)
        self.message_user(request, f'{updated} моделей помечены как небезопасные')
    
    @admin.action(description='Активировать модели')
    def activate_models(self, request, queryset):
        updated = queryset.update(is_active=True)
        self.message_user(request, f'Активировано {updated} моделей')
    
    @admin.action(description='Деактивировать модели')
    def deactivate_models(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(request, f'Деактивировано {updated} моделей')
    
    @admin.action(description='Обновить конфигурацию из HF')
    def update_config_from_huggingface(self, request, queryset):
        # Заглушка для реальной логики загрузки конфигурации
        for model in queryset:
            if model.huggingface_id:
                # Здесь должна быть логика загрузки конфигурации
                model.loader_config = model.get_default_config()
                model.save()
        
        self.message_user(
            request, 
            f'Конфигурация обновлена для {queryset.count()} моделей',
            messages.SUCCESS
        )
    
    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        return queryset.select_related('experiment', 'parent')
    
    def save_model(self, request, obj, form, change):
        """Добавляем логирование при сохранении"""
        if change:
            # Создаем версию при изменении конфигурации
            old_config = None
            if 'loader_config' in form.changed_data:
                old_config = ModelCheckpoint.objects.get(pk=obj.pk).loader_config
            
            super().save_model(request, obj, form, change)
            
            if old_config is not None:
                obj.create_version(
                    version=f"admin_edit_{obj.versions.count() + 1}",
                    notes=f"Изменение через админку пользователем {request.user}"
                )
        else:
            super().save_model(request, obj, form, change)
            obj.create_version(
                version="v1.0",
                notes="Создание через админ-панель"
            )
    
    def change_view(self, request, object_id, form_url='', extra_context=None):
        """Добавляем дополнительные кнопки в форму редактирования"""
        extra_context = extra_context or {}
        obj = self.get_object(request, object_id)
        
        if obj:
            extra_context['has_huggingface'] = bool(obj.huggingface_id)
            extra_context['can_download'] = obj.status in ['DOWNLOADING', 'ERROR']
            extra_context['can_test'] = obj.status == 'LOADED'
            
            # Ссылки на связанные объекты
            extra_context['versions_count'] = obj.versions.count()
            extra_context['evaluations_count'] = obj.evaluations.count()
            extra_context['sessions_count'] = obj.inference_sessions.count()
        
        return super().change_view(
            request, object_id, form_url, extra_context=extra_context
        )


class ModelMergeJobAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'base_model_link', 'merge_model_link', 'merge_method', 
        'status_badge', 'progress_bar', 'created_at', 'duration_display'
    )
    list_filter = ('status', 'merge_method', 'created_at')
    search_fields = ('name', 'description', 'base_model__name', 'merge_model__name')
    readonly_fields = (
        'created_at', 'started_at', 'completed_at', 'duration_seconds',
        'progress_bar', 'performance_display', 'logs_preview',
        'is_compatible', 'status_history'
    )
    actions = ['retry_failed_jobs', 'cancel_jobs']
    raw_id_fields = ('base_model', 'merge_model', 'output_model', 'parent_job')
    
    fieldsets = (
        (_('Основная информация'), {
            'fields': ('name', 'description', 'parent_job')
        }),
        (_('Модели для слияния'), {
            'fields': ('base_model', 'merge_model', 'output_model', 'is_compatible')
        }),
        (_('Настройки слияния'), {
            'fields': ('merge_method', 'merge_ratio', 'merge_config')
        }),
        (_('Ресурсы и лимиты'), {
            'fields': ('gpu_memory_limit_gb', 'timeout_minutes'),
            'classes': ('collapse',)
        }),
        (_('Прогресс и статус'), {
            'fields': ('status', 'progress_bar', 'current_step', 'error_message')
        }),
        (_('Результаты и метрики'), {
            'fields': ('performance_display', 'performance_metrics', 'error_details'),
            'classes': ('collapse',)
        }),
        (_('Временные метки'), {
            'fields': ('created_at', 'started_at', 'completed_at', 'duration_seconds'),
            'classes': ('collapse',)
        }),
        (_('Логи выполнения'), {
            'fields': ('logs_preview',),
            'classes': ('collapse',)
        }),
        (_('История статусов'), {
            'fields': ('status_history',),
            'classes': ('collapse',)
        }),
    )
    
    def base_model_link(self, obj):
        if obj.base_model:
            url = reverse('admin:app_modelcheckpoint_change', args=[obj.base_model.id])
            return format_html('<a href="{}">{}</a>', url, obj.base_model.name)
        return "-"
    base_model_link.short_description = 'Базовая модель'
    
    def merge_model_link(self, obj):
        if obj.merge_model:
            url = reverse('admin:app_modelcheckpoint_change', args=[obj.merge_model.id])
            return format_html('<a href="{}">{}</a>', url, obj.merge_model.name)
        return "-"
    merge_model_link.short_description = 'Модель для слияния'
    
    def status_badge(self, obj):
        """Красивое отображение статуса"""
        colors = {
            'PENDING': 'gray',
            'VALIDATING': 'blue',
            'RUNNING': 'orange',
            'COMPLETED': 'green',
            'FAILED': 'red',
            'CANCELLED': 'darkgray',
        }
        color = colors.get(obj.status, 'black')
        return format_html(
            '<span style="display: inline-block; padding: 3px 8px; '
            'border-radius: 10px; background-color: {}; color: white; '
            'font-size: 12px;">{}</span>',
            color, obj.get_status_display()
        )
    status_badge.short_description = 'Статус'
    
    def progress_bar(self, obj):
        """Прогресс-бар для задачи"""
        color = 'green' if obj.status == 'COMPLETED' else 'blue' if obj.status == 'RUNNING' else 'orange'
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
    
    def duration_display(self, obj):
        """Отображение длительности"""
        if obj.duration_seconds:
            minutes = int(obj.duration_seconds // 60)
            seconds = int(obj.duration_seconds % 60)
            return f"{minutes}:{seconds:02d}"
        return "-"
    duration_display.short_description = 'Длительность'
    
    def performance_display(self, obj):
        """Отображение метрик производительности"""
        if not obj.performance_metrics:
            return "Нет данных"
        
        metrics_html = []
        for key, value in obj.performance_metrics.items():
            if isinstance(value, (int, float)):
                metrics_html.append(f"<li><strong>{key}:</strong> {value:.2f}</li>")
            else:
                metrics_html.append(f"<li><strong>{key}:</strong> {value}</li>")
        
        return format_html(f"<ul>{''.join(metrics_html)}</ul>")
    performance_display.short_description = 'Метрики производительности'
    
    def logs_preview(self, obj):
        """Предпросмотр логов"""
        if not obj.logs:
            return "Нет логов"
        
        # Показываем только последние 10 строк
        lines = obj.logs.strip().split('\n')[-10:]
        preview = '\n'.join(lines)
        return format_html(
            '<textarea style="width: 100%; height: 150px; font-family: monospace;" readonly>{}</textarea>',
            preview
        )
    logs_preview.short_description = 'Предпросмотр логов'
    
    def status_history(self, obj):
        """История изменения статусов"""
        # В реальном приложении можно добавить модель для отслеживания истории статусов
        history = [
            f"{obj.created_at}: PENDING",
            f"{obj.started_at}: RUNNING" if obj.started_at else None,
            f"{obj.completed_at}: {obj.status}" if obj.completed_at else None,
        ]
        history = [h for h in history if h]
        return format_html("<br>".join(history))
    status_history.short_description = 'История статусов'
    
    @admin.action(description='Повторить неудачные задачи')
    def retry_failed_jobs(self, request, queryset):
        failed_jobs = queryset.filter(status='FAILED')
        count = failed_jobs.count()
        
        for job in failed_jobs:
            # Создаем новую задачу как копию старной
            ModelMergeJob.objects.create(
                name=f"{job.name}_retry",
                description=f"Повтор задачи {job.id}",
                base_model=job.base_model,
                merge_model=job.merge_model,
                merge_method=job.merge_method,
                merge_ratio=job.merge_ratio,
                merge_config=job.merge_config,
                parent_job=job
            )
        
        self.message_user(request, f'Создано {count} задач для повторного выполнения')
    
    @admin.action(description='Отменить выбранные задачи')
    def cancel_jobs(self, request, queryset):
        cancellable = queryset.filter(status__in=['PENDING', 'VALIDATING', 'RUNNING'])
        count = cancellable.update(status='CANCELLED')
        self.message_user(request, f'Отменено {count} задач')


class InferenceSessionAdmin(admin.ModelAdmin):
    list_display = (
        'session_id', 'checkpoint_link', 'user_ip', 
        'tokens_total', 'requests_count', 'duration_minutes_display',
        'is_active_badge', 'started_at'
    )
    list_filter = ('is_active', 'started_at', 'checkpoint__model_type')
    search_fields = ('session_id', 'user_ip', 'api_key', 'checkpoint__name')
    readonly_fields = (
        'session_id', 'started_at', 'last_activity', 'ended_at',
        'tokens_total', 'tokens_per_second', 'duration_minutes',
        'performance_stats'
    )
    actions = ['close_sessions']
    
    fieldsets = (
        (_('Идентификация'), {
            'fields': ('session_id', 'checkpoint', 'is_active')
        }),
        (_('Пользовательская информация'), {
            'fields': ('user_ip', 'user_agent', 'api_key'),
            'classes': ('collapse',)
        }),
        (_('Статистика использования'), {
            'fields': ('input_tokens', 'output_tokens', 'tokens_total', 'total_requests')
        }),
        (_('Производительность'), {
            'fields': ('performance_stats', 'total_duration', 'avg_response_time', 'tokens_per_second')
        }),
        (_('Ошибки'), {
            'fields': ('error_count', 'last_error'),
            'classes': ('collapse',)
        }),
        (_('Временные метки'), {
            'fields': ('started_at', 'last_activity', 'ended_at', 'duration_minutes'),
            'classes': ('collapse',)
        }),
    )
    
    def checkpoint_link(self, obj):
        if obj.checkpoint:
            url = reverse('admin:app_modelcheckpoint_change', args=[obj.checkpoint.id])
            return format_html('<a href="{}">{}</a>', url, obj.checkpoint.name)
    checkpoint_link.short_description = 'Модель'
    
    def tokens_total(self, obj):
        return obj.input_tokens + obj.output_tokens
    tokens_total.short_description = 'Всего токенов'
    
    def requests_count(self, obj):
        return obj.total_requests
    requests_count.short_description = 'Запросы'
    
    def duration_minutes_display(self, obj):
        return f"{obj.duration_minutes:.1f} мин"
    duration_minutes_display.short_description = 'Длительность'
    
    def is_active_badge(self, obj):
        color = 'green' if obj.is_active else 'red'
        text = 'Активна' if obj.is_active else 'Завершена'
        return format_html(
            '<span style="padding: 2px 6px; background: {}; color: white; border-radius: 3px; font-size: 11px;">{}</span>',
            color, text
        )
    is_active_badge.short_description = 'Статус'
    
    def performance_stats(self, obj):
        """Статистика производительности"""
        stats = [
            f"<strong>Токенов/сек:</strong> {obj.tokens_per_second:.1f}",
            f"<strong>Среднее время ответа:</strong> {obj.avg_response_time:.2f}с",
            f"<strong>Ошибок:</strong> {obj.error_count}",
        ]
        return format_html("<br>".join(stats))
    performance_stats.short_description = 'Статистика производительности'
    
    @admin.action(description='Завершить выбранные сессии')
    def close_sessions(self, request, queryset):
        active_sessions = queryset.filter(is_active=True)
        count = 0
        
        for session in active_sessions:
            session.end_session()
            count += 1
        
        self.message_user(request, f'Завершено {count} сессий')


class ModelEvaluationAdmin(admin.ModelAdmin):
    list_display = (
        'checkpoint_link', 'dataset_name', 'dataset_split', 
        'primary_metric_display', 'is_valid', 'evaluated_at'
    )
    list_filter = ('dataset_name', 'evaluation_type', 'is_valid', 'evaluated_at')
    search_fields = ('checkpoint__name', 'dataset_name', 'validation_notes')
    readonly_fields = ('evaluated_at', 'primary_metric_display', 'metrics_display')
    list_editable = ('is_valid',)  # Теперь is_valid есть в list_display
    
    fieldsets = (
        (_('Основная информация'), {
            'fields': ('checkpoint', 'dataset_name', 'dataset_split', 'evaluation_type')
        }),
        (_('Результаты'), {
            'fields': ('metrics_display', 'raw_results'),
            'classes': ('wide',)
        }),
        (_('Системная информация'), {
            'fields': ('evaluator_version', 'hardware_info'),
            'classes': ('collapse',)
        }),
        (_('Валидация'), {
            'fields': ('is_valid', 'validation_notes'),
            'classes': ('collapse',)
        }),
        (_('Метаданные'), {
            'fields': ('evaluated_at',),
            'classes': ('collapse',)
        }),
    )
    
    def checkpoint_link(self, obj):
        if obj.checkpoint:
            url = reverse('admin:app_modelcheckpoint_change', args=[obj.checkpoint.id])
            return format_html('<a href="{}">{}</a>', url, obj.checkpoint.name)
    checkpoint_link.short_description = 'Модель'
    
    def primary_metric_display(self, obj):
        return f"{obj.primary_metric:.3f}"
    primary_metric_display.short_description = 'Основная метрика'
    
    def metrics_display(self, obj):
        """Красивое отображение метрик"""
        if not obj.metrics:
            return "Нет данных"
        
        metrics_html = []
        for key, value in obj.metrics.items():
            if isinstance(value, (int, float)):
                metrics_html.append(f"<tr><td><strong>{key}</strong></td><td>{value:.4f}</td></tr>")
            else:
                metrics_html.append(f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>")
        
        return format_html(
            '<table style="width: 100%; border-collapse: collapse;">'
            '<tr style="background: #f8f9fa;"><th>Метрика</th><th>Значение</th></tr>'
            '{}'
            '</table>',
            ''.join(metrics_html)
        )
    metrics_display.short_description = 'Метрики оценки'


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ('checkpoint', 'version', 'is_stable', 'created_at', 'created_by')
    list_filter = ('is_stable', 'created_at')
    search_fields = ('checkpoint__name', 'version', 'notes')
    readonly_fields = ('created_at', 'file_hash', 'config_snapshot_preview')
    raw_id_fields = ('checkpoint', 'parent_version')
    
    def config_snapshot_preview(self, obj):
        """Предпросмотр конфигурации"""
        return format_html(
            '<pre style="background: #f5f5f5; padding: 10px; border-radius: 5px; overflow: auto;">{}</pre>',
            json.dumps(obj.config_snapshot, indent=2, ensure_ascii=False)
        )
    config_snapshot_preview.short_description = 'Конфигурация'
    
    actions = ['mark_as_stable', 'mark_as_unstable', 'restore_version']
    
    @admin.action(description='Пометить как стабильные')
    def mark_as_stable(self, request, queryset):
        updated = queryset.update(is_stable=True)
        self.message_user(request, f'{updated} версий помечены как стабильные')
    
    @admin.action(description='Пометить как нестабильные')
    def mark_as_unstable(self, request, queryset):
        updated = queryset.update(is_stable=False)
        self.message_user(request, f'{updated} версий помечены как нестабильные')
    
    @admin.action(description='Восстановить выбранные версии')
    def restore_version(self, request, queryset):
        for version in queryset:
            version.restore()
        self.message_user(request, f'Восстановлено {queryset.count()} версий')
