from django.contrib import admin
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

from .models import (
    Experiment, ModelCheckpoint, ModelVersion, 
    ModelMergeJob, InferenceSession, ModelEvaluation,
    ChatSession, ChatMessage, ModelComparison, 
    FineTuningJob, DocumentCollection, Document, 
    DocumentChunk, EmbeddingModel,
    SearchQuery, SearchResult, RAGSession, RetrievedChunk,
    SearchQualityFeedback, RAGEvaluation, ProcessingPipeline,
    DocumentProcessor, DeletedDocumentLog
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
        ('Основная информация', {
            'fields': ('name', 'description', 'base_checkpoint', 'is_active')
        }),
        ('Теги и классификация', {
            'fields': ('tags',),
            'classes': ('collapse',)
        }),
        ('Системная информация', {
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
        ('Основная информация', {
            'fields': (
                'name', 'display_name', 'model_type', 'experiment', 
                'parent', 'is_active', 'is_public'
            )
        }),
        ('Hugging Face', {
            'fields': ('huggingface_id', 'huggingface_revision', 'huggingface_url'),
            'classes': ('collapse',)
        }),
        ('Конфигурация и пути', {
            'fields': ('loader_config', 'local_path', 'relative_path'),
            'classes': ('wide',)
        }),
        ('Характеристики модели', {
            'fields': (
                'parameter_count', 'size_gb_display', 'download_size_mb_display',
                'quantization', 'context_length', 'vocab_size'
            )
        }),
        ('Безопасность и лицензия', {
            'fields': ('license_type', 'is_safe', 'safety_score'),
            'classes': ('collapse',)
        }),
        ('Производительность и метрики', {
            'fields': ('inference_speed', 'accuracy_score', 'evaluation_metrics'),
            'classes': ('collapse',)
        }),
        ('Статус и отслеживание', {
            'fields': ('status', 'error_message', 'config_hash', 'weights_hash'),
            'classes': ('collapse',)
        }),
        ('Теги', {
            'fields': ('tags',),
            'classes': ('collapse',)
        }),
        ('Системная информация', {
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
        ('Основная информация', {
            'fields': ('name', 'description', 'parent_job')
        }),
        ('Модели для слияния', {
            'fields': ('base_model', 'merge_model', 'output_model', 'is_compatible')
        }),
        ('Настройки слияния', {
            'fields': ('merge_method', 'merge_ratio', 'merge_config')
        }),
        ('Ресурсы и лимиты', {
            'fields': ('gpu_memory_limit_gb', 'timeout_minutes'),
            'classes': ('collapse',)
        }),
        ('Прогресс и статус', {
            'fields': ('status', 'progress_bar', 'current_step', 'error_message')
        }),
        ('Результаты и метрики', {
            'fields': ('performance_display', 'performance_metrics', 'error_details'),
            'classes': ('collapse',)
        }),
        ('Временные метки', {
            'fields': ('created_at', 'started_at', 'completed_at', 'duration_seconds'),
            'classes': ('collapse',)
        }),
        ('Логи выполнения', {
            'fields': ('logs_preview',),
            'classes': ('collapse',)
        }),
        ('История статусов', {
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
        ('Идентификация', {
            'fields': ('session_id', 'checkpoint', 'is_active')
        }),
        ('Пользовательская информация', {
            'fields': ('user_ip', 'user_agent', 'api_key'),
            'classes': ('collapse',)
        }),
        ('Статистика использования', {
            'fields': ('input_tokens', 'output_tokens', 'tokens_total', 'total_requests')
        }),
        ('Производительность', {
            'fields': ('performance_stats', 'total_duration', 'avg_response_time', 'tokens_per_second')
        }),
        ('Ошибки', {
            'fields': ('error_count', 'last_error'),
            'classes': ('collapse',)
        }),
        ('Временные метки', {
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
        ('Основная информация', {
            'fields': ('checkpoint', 'dataset_name', 'dataset_split', 'evaluation_type')
        }),
        ('Результаты', {
            'fields': ('metrics_display', 'raw_results'),
            'classes': ('wide',)
        }),
        ('Системная информация', {
            'fields': ('evaluator_version', 'hardware_info'),
            'classes': ('collapse',)
        }),
        ('Валидация', {
            'fields': ('is_valid', 'validation_notes'),
            'classes': ('collapse',)
        }),
        ('Метаданные', {
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


class ChatSessionStatusFilter(admin.SimpleListFilter):
    """Фильтр по статусу чат-сессий"""
    title = 'Статус чата'
    parameter_name = 'chat_status'
    
    def lookups(self, request, model_admin):
        return (
            ('active', 'Активные'),
            ('recent', 'За последние 7 дней'),
            ('old', 'Старые (более 30 дней)'),
        )
    
    def queryset(self, request, queryset):
        now = datetime.now()
        if self.value() == 'active':
            return queryset.filter(is_active=True)
        elif self.value() == 'recent':
            week_ago = now - timedelta(days=7)
            return queryset.filter(last_activity__gte=week_ago)
        elif self.value() == 'old':
            month_ago = now - timedelta(days=30)
            return queryset.filter(last_activity__lte=month_ago)


class ChatMessageInline(admin.TabularInline):
    """Inline для сообщений чата"""
    model = ChatMessage
    extra = 0
    readonly_fields = ('created_at', 'message_preview')
    fields = ('role', 'message_preview', 'created_at')
    can_delete = True
    show_change_link = True
    
    def message_preview(self, obj):
        """Предпросмотр сообщения"""
        preview = obj.content[:100] + "..." if len(obj.content) > 100 else obj.content
        return format_html(
            '<span title="{}" style="cursor: help;">{}</span>',
            obj.content.replace('"', '&quot;'),
            preview
        )
    message_preview.short_description = 'Сообщение'
    
    def has_add_permission(self, request, obj=None):
        return False


class ChatSessionAdminForm(forms.ModelForm):
    """Кастомная форма для чат-сессий"""
    class Meta:
        model = ChatSession
        fields = '__all__'
        widgets = {
            'system_prompt': forms.Textarea(attrs={'rows': 4}),
            'conversation_history': forms.Textarea(attrs={'rows': 10, 'style': 'font-family: monospace'}),
            'model_config': forms.Textarea(attrs={'rows': 6}),
            'generation_config': forms.Textarea(attrs={'rows': 4}),
        }


class ChatSessionAdmin(admin.ModelAdmin):
    form = ChatSessionAdminForm
    list_display = (
        'session_id', 'checkpoint_link', 'user_id', 
        'messages_count', 'tokens_total', 'is_active_badge',
        'last_activity_display', 'created_at'
    )
    list_filter = (ChatSessionStatusFilter, 'is_active', 'checkpoint__model_type', 'created_at')
    search_fields = ('session_id', 'user_id', 'checkpoint__name', 'system_prompt')
    readonly_fields = (
        'created_at', 'updated_at', 'last_activity', 'messages_count',
        'tokens_total', 'conversation_preview', 'model_info',
        'generation_info', 'session_stats'
    )
    actions = ['close_sessions', 'clear_inactive_sessions', 'export_conversations']
    raw_id_fields = ('checkpoint',)
    
    fieldsets = (
        ('Основная информация', {
            'fields': ('session_id', 'user_id', 'checkpoint', 'is_active')
        }),
        ('Конфигурация', {
            'fields': ('system_prompt', 'model_config', 'generation_config'),
            'classes': ('collapse',)
        }),
        ('История диалога', {
            'fields': ('conversation_preview',),
            'classes': ('collapse',)
        }),
        ('Статистика', {
            'fields': ('session_stats', 'total_messages', 'total_input_tokens', 'total_output_tokens'),
            'classes': ('collapse',)
        }),
        ('Системная информация', {
            'fields': ('created_at', 'updated_at', 'last_activity'),
            'classes': ('collapse',)
        }),
    )
    
    inlines = [ChatMessageInline]
    
    def checkpoint_link(self, obj):
        if obj.checkpoint:
            url = reverse('admin:app_modelcheckpoint_change', args=[obj.checkpoint.id])
            return format_html('<a href="{}">{}</a>', url, obj.checkpoint.name)
        return "-"
    checkpoint_link.short_description = 'Модель'
    
    def messages_count(self, obj):
        return obj.total_messages
    messages_count.short_description = 'Сообщений'
    
    def tokens_total(self, obj):
        return obj.total_tokens
    tokens_total.short_description = 'Всего токенов'
    
    def last_activity_display(self, obj):
        now = datetime.now()
        delta = now - obj.last_activity.replace(tzinfo=None)
        
        if delta.days == 0:
            hours = delta.seconds // 3600
            if hours > 0:
                return f"{hours}ч назад"
            minutes = (delta.seconds % 3600) // 60
            return f"{minutes}м назад"
        elif delta.days == 1:
            return "вчера"
        elif delta.days < 7:
            return f"{delta.days}д назад"
        else:
            return obj.last_activity.strftime("%d.%m.%Y")
    last_activity_display.short_description = 'Активность'
    
    def is_active_badge(self, obj):
        color = 'green' if obj.is_active else 'red'
        text = 'Активен' if obj.is_active else 'Неактивен'
        return format_html(
            '<span style="padding: 2px 8px; background: {}; color: white; '
            'border-radius: 10px; font-size: 12px; font-weight: bold;">{}</span>',
            color, text
        )
    is_active_badge.short_description = 'Статус'
    
    def conversation_preview(self, obj):
        """Предпросмотр истории диалога"""
        if not obj.conversation_history:
            return "История диалога пуста"
        
        preview = []
        for i, msg in enumerate(obj.conversation_history[-5:], 1):  # последние 5 сообщений
            role_icons = {
                'system': '⚙️',
                'user': '👤',
                'assistant': '🤖',
                'function': '🔧',
                'tool': '🛠️'
            }
            icon = role_icons.get(msg.get('role', ''), '❓')
            content = msg.get('content', '')[:100]
            if len(msg.get('content', '')) > 100:
                content += "..."
            
            preview.append(f"{icon} <strong>{msg.get('role', 'unknown')}:</strong> {content}")
        
        return format_html("<br>".join(preview))
    conversation_preview.short_description = 'Последние сообщения'
    
    def session_stats(self, obj):
        """Статистика сессии"""
        stats = [
            f"<strong>Всего сообщений:</strong> {obj.total_messages}",
            f"<strong>Входные токены:</strong> {obj.total_input_tokens}",
            f"<strong>Выходные токены:</strong> {obj.total_output_tokens}",
            f"<strong>Всего токенов:</strong> {obj.total_tokens}",
        ]
        return format_html("<br>".join(stats))
    session_stats.short_description = 'Статистика'
    
    def model_info(self, obj):
        """Информация о модели"""
        if not obj.checkpoint:
            return "Модель не указана"
        
        return format_html(
            """
            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px;">
                <strong>Модель:</strong> {}<br>
                <strong>Тип:</strong> {}<br>
                <strong>Параметров:</strong> {:,}<br>
                <strong>Контекст:</strong> {} токенов
            </div>
            """,
            obj.checkpoint.display_name,
            obj.checkpoint.get_model_type_display(),
            obj.checkpoint.parameter_count,
            obj.checkpoint.context_length
        )
    model_info.short_description = 'Информация о модели'
    
    def generation_info(self, obj):
        """Информация о конфигурации генерации"""
        if not obj.generation_config:
            return "Конфигурация по умолчанию"
        
        config = obj.generation_config
        info = []
        for key, value in config.items():
            info.append(f"<strong>{key}:</strong> {value}")
        
        return format_html("<br>".join(info))
    generation_info.short_description = 'Конфигурация генерации'
    
    @admin.action(description='Закрыть выбранные сессии')
    def close_sessions(self, request, queryset):
        active_sessions = queryset.filter(is_active=True)
        count = active_sessions.update(is_active=False, updated_at=datetime.now())
        self.message_user(request, f'Закрыто {count} сессий')
    
    @admin.action(description='Очистить неактивные сессии')
    def clear_inactive_sessions(self, request, queryset):
        month_ago = datetime.now() - timedelta(days=30)
        old_sessions = queryset.filter(is_active=False, last_activity__lte=month_ago)
        count = old_sessions.count()
        old_sessions.delete()
        self.message_user(request, f'Удалено {count} неактивных сессий')
    
    @admin.action(description='Экспортировать диалоги')
    def export_conversations(self, request, queryset):
        # Заглушка для экспорта
        self.message_user(
            request, 
            f'Экспорт {queryset.count()} диалогов в фоновом режиме',
            messages.SUCCESS
        )
    
    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        return queryset.select_related('checkpoint')


class ChatMessageAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'session_link', 'role_badge', 'content_preview', 
        'generation_info_badge', 'created_at'
    )
    list_filter = ('role', 'created_at', 'session__checkpoint__model_type')
    search_fields = ('content', 'session__session_id', 'generation_metadata')
    readonly_fields = ('created_at', 'generation_metadata_display')
    raw_id_fields = ('session', 'parent')
    
    fieldsets = (
        ('Основная информация', {
            'fields': ('session', 'role', 'parent')
        }),
        ('Содержимое', {
            'fields': ('content',),
            'classes': ('wide',)
        }),
        ('Метаданные генерации', {
            'fields': ('generation_metadata_display',),
            'classes': ('collapse',)
        }),
        ('Системная информация', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    def session_link(self, obj):
        if obj.session:
            url = reverse('admin:app_chatsession_change', args=[obj.session.id])
            return format_html(
                '<a href="{}" title="Перейти к сессии">{}...</a>',
                url, obj.session.session_id[:20]
            )
        return "-"
    session_link.short_description = 'Сессия'
    
    def role_badge(self, obj):
        """Бейдж для роли"""
        colors = {
            'system': 'gray',
            'user': 'blue',
            'assistant': 'green',
            'function': 'orange',
            'tool': 'purple'
        }
        color = colors.get(obj.role, 'black')
        return format_html(
            '<span style="padding: 2px 6px; background: {}; color: white; '
            'border-radius: 3px; font-size: 11px;">{}</span>',
            color, obj.get_role_display()
        )
    role_badge.short_description = 'Роль'
    
    def content_preview(self, obj):
        """Предпросмотр содержимого"""
        preview = obj.content[:150] + "..." if len(obj.content) > 150 else obj.content
        return format_html(
            '<span title="{}" style="cursor: help;">{}</span>',
            obj.content.replace('"', '&quot;'),
            preview
        )
    content_preview.short_description = 'Содержимое'
    
    def generation_info_badge(self, obj):
        """Бейдж с информацией о генерации"""
        if not obj.generation_metadata:
            return format_html('<span style="color: gray;">—</span>')
        
        metadata = obj.generation_metadata
        info = []
        
        if metadata.get('tokens'):
            info.append(f"токены: {metadata['tokens']}")
        if metadata.get('time'):
            info.append(f"{metadata['time']}с")
        
        return format_html(
            '<span style="font-size: 11px; color: #666;">{}</span>',
            ", ".join(info)
        )
    generation_info_badge.short_description = 'Генерация'
    
    def generation_metadata_display(self, obj):
        """Красивое отображение метаданных"""
        if not obj.generation_metadata:
            return "Нет метаданных"
        
        return format_html(
            '<pre style="background: #f8f9fa; padding: 10px; '
            'border-radius: 5px; overflow: auto;">{}</pre>',
            json.dumps(obj.generation_metadata, indent=2, ensure_ascii=False)
        )
    generation_metadata_display.short_description = 'Метаданные генерации'


class ModelComparisonAdminForm(forms.ModelForm):
    """Кастомная форма для сравнений"""
    class Meta:
        model = ModelComparison
        fields = '__all__'
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'prompts': forms.Textarea(attrs={'rows': 10, 'style': 'font-family: monospace'}),
            'results': forms.Textarea(attrs={'rows': 15, 'style': 'font-family: monospace'}),
            'metrics': forms.Textarea(attrs={'rows': 8, 'style': 'font-family: monospace'}),
        }


class ModelComparisonAdmin(admin.ModelAdmin):
    form = ModelComparisonAdminForm
    list_display = (
        'name', 'prompts_count', 'models_count', 'is_complete_badge',
        'created_at_display', 'updated_at_display'
    )
    list_filter = ('is_complete', 'created_at')
    search_fields = ('name', 'description')
    readonly_fields = (
        'created_at', 'updated_at', 'prompts_count', 'models_count',
        'results_preview', 'metrics_display', 'comparison_stats'
    )
    actions = ['mark_as_complete', 'mark_as_incomplete', 'calculate_metrics']
    
    fieldsets = (
        ('Основная информация', {
            'fields': ('name', 'description', 'is_complete')
        }),
        ('Промпты для сравнения', {
            'fields': ('prompts', 'prompts_count'),
            'classes': ('wide',)
        }),
        ('Результаты', {
            'fields': ('results_preview',),
            'classes': ('collapse',)
        }),
        ('Метрики', {
            'fields': ('metrics_display',),
            'classes': ('collapse',)
        }),
        ('Статистика', {
            'fields': ('comparison_stats',),
            'classes': ('collapse',)
        }),
        ('Системная информация', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def prompts_count(self, obj):
        return len(obj.prompts) if obj.prompts else 0
    prompts_count.short_description = 'Промптов'
    
    def models_count(self, obj):
        return len(obj.results) if obj.results else 0
    models_count.short_description = 'Моделей'
    
    def created_at_display(self, obj):
        return obj.created_at.strftime("%d.%m.%Y %H:%M")
    created_at_display.short_description = 'Создано'
    
    def updated_at_display(self, obj):
        return obj.updated_at.strftime("%d.%m.%Y %H:%M")
    updated_at_display.short_description = 'Обновлено'
    
    def is_complete_badge(self, obj):
        color = 'green' if obj.is_complete else 'orange'
        text = 'Завершено' if obj.is_complete else 'В процессе'
        return format_html(
            '<span style="padding: 2px 8px; background: {}; color: white; '
            'border-radius: 10px; font-size: 12px;">{}</span>',
            color, text
        )
    is_complete_badge.short_description = 'Статус'
    
    def results_preview(self, obj):
        """Предпросмотр результатов"""
        if not obj.results:
            return "Нет результатов"
        
        preview = []
        for model_id, model_data in list(obj.results.items())[:3]:  # первые 3 модели
            model_name = next(iter(model_data.values())).get('model_name', 'Unknown') if model_data else 'Unknown'
            responses_count = len(model_data)
            preview.append(f"<strong>{model_name}:</strong> {responses_count} ответов")
        
        return format_html("<br>".join(preview))
    results_preview.short_description = 'Предпросмотр результатов'
    
    def metrics_display(self, obj):
        """Отображение метрик"""
        if not obj.metrics:
            return "Метрики не вычислены"
        
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
    metrics_display.short_description = 'Метрики сравнения'
    
    def comparison_stats(self, obj):
        """Статистика сравнения"""
        stats = [
            f"<strong>Промптов:</strong> {len(obj.prompts) if obj.prompts else 0}",
            f"<strong>Моделей:</strong> {len(obj.results) if obj.results else 0}",
            f"<strong>Всего ответов:</strong> {sum(len(responses) for responses in obj.results.values()) if obj.results else 0}",
            f"<strong>Статус:</strong> {'Завершено' if obj.is_complete else 'В процессе'}",
        ]
        return format_html("<br>".join(stats))
    comparison_stats.short_description = 'Статистика сравнения'
    
    @admin.action(description='Пометить как завершенные')
    def mark_as_complete(self, request, queryset):
        updated = queryset.update(is_complete=True)
        self.message_user(request, f'Помечено как завершенные: {updated} сравнений')
    
    @admin.action(description='Пометить как незавершенные')
    def mark_as_incomplete(self, request, queryset):
        updated = queryset.update(is_complete=False)
        self.message_user(request, f'Помечено как незавершенные: {updated} сравнений')
    
    @admin.action(description='Пересчитать метрики')
    def calculate_metrics(self, request, queryset):
        for comparison in queryset:
            comparison.calculate_metrics()
        self.message_user(request, f'Метрики пересчитаны для {queryset.count()} сравнений')


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
        ('Основная информация', {
            'fields': ('name', 'description', 'base_model')
        }),
        ('Данные для обучения', {
            'fields': ('data_info', 'training_data_path', 'validation_data_path', 'data_config'),
            'classes': ('collapse',)
        }),
        ('Гиперпараметры', {
            'fields': ('hyperparameters_display',),
            'classes': ('collapse',)
        }),
        ('Прогресс и статус', {
            'fields': ('status', 'progress_bar', 'current_step', 'error_message')
        }),
        ('Результаты', {
            'fields': ('output_model', 'training_metrics_display'),
            'classes': ('collapse',)
        }),
        ('Логи выполнения', {
            'fields': ('logs_preview',),
            'classes': ('collapse',)
        }),
        ('Временные метки', {
            'fields': ('created_at', 'started_at', 'completed_at'),
            'classes': ('collapse',)
        }),
        ('Системная информация', {
            'fields': ('status_history', 'job_stats'),
            'classes': ('collapse',)
        }),
    )
    
    def base_model_link(self, obj):
        if obj.base_model:
            url = reverse('admin:app_modelcheckpoint_change', args=[obj.base_model.id])
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
            return format_html("<br>".join(info))
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
        
        return format_html("<br>".join(history))
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
        
        return format_html("<br>".join(stats))
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


# ============================================================================
# 2. RAG-МОДЕЛИ (новые модели из RAG системы)
# ============================================================================

class DocumentCollectionAdmin(ModelAdmin):
    """Админка для коллекций документов"""
    list_display = ('name', 'owner_id', 'is_public', 'is_active', 
                    'total_documents_display', 'total_size_display', 'created_at')
    list_filter = ('is_public', 'is_active', 'created_at', 'chunking_strategy')
    search_fields = ('name', 'description', 'owner_id')
    readonly_fields = ('created_at', 'updated_at', 'total_documents', 
                      'total_chunks', 'total_size_mb', 'total_documents_display',
                      'total_chunks_display', 'total_size_display')  # Добавил методы в readonly_fields
    fieldsets = (
        ('Основная информация', {
            'fields': ('name', 'description', 'owner_id', 'is_public', 'is_active')
        }),
        ('Настройки обработки', {
            'fields': ('chunking_strategy', 'chunk_size', 'chunk_overlap', 
                      'auto_process')
        }),
        ('Фильтры и метаданные', {
            'fields': ('metadata_filters',)
        }),
        ('Статистика', {
            'fields': ('total_documents_display', 'total_chunks_display', 
                      'total_size_display'),
            'classes': ('collapse',)
        }),
        ('Системная информация', {
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
    """Инлайн-админка для чанков документа"""
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
        ('Основная информация', {
            'fields': ('title', 'original_filename', 'collection', 'description', 'category')
        }),
        ('Тип и содержимое', {
            'fields': ('document_type', 'content_hash', 'raw_content_preview', 'processed_content')
        }),
        ('Файл', {
            'fields': ('file_path', 'file_size_display', 'file_encoding')
        }),
        ('Статус и обработка', {
            'fields': ('status', 'processing_progress', 'error_message', 'quality_score')
        }),
        ('Метаданные', {
            'fields': ('metadata', 'tags'),
            'classes': ('collapse',)
        }),
        ('Даты', {
            'fields': ('uploaded_at', 'processed_at', 'indexed_at'),
            'classes': ('collapse',)
        }),
        ('Системные поля', {
            'fields': ('is_active', 'requires_reindexing', 'chunk_count_display'),
            'classes': ('collapse',)
        })
    )
    
    def title_display(self, obj):
        """Отображение заголовка"""
        return obj.title or obj.original_filename
    title_display.short_description = 'Название'
    
    def collection_display(self, obj):
        """Отображение коллекции"""
        if obj.collection:
            return obj.collection.name
        return "Без коллекции"
    collection_display.short_description = 'Коллекция'
    
    def uploaded_at_short(self, obj):
        """Короткий формат даты загрузки"""
        return obj.uploaded_at.strftime('%d.%m.%Y %H:%M') if obj.uploaded_at else ''
    uploaded_at_short.short_description = 'Загружен'
    
    def file_size_display(self, obj):
        """Отображение размера файла"""
        if obj.file_size_mb:
            return f"{obj.file_size_mb:.2f} MB"
        return "Неизвестно"
    file_size_display.short_description = 'Размер файла'
    
    def raw_content_preview(self, obj):
        """Предпросмотр содержимого"""
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
        """Отображение количества чанков"""
        return obj.chunk_count
    chunk_count_display.short_description = 'Чанков'
    
    def extension_display(self, obj):
        """Отображение расширения файла"""
        try:
            return getattr(obj, 'extension', '')
        except:
            return ''
    extension_display.short_description = 'Расширение'
    
    def get_queryset(self, request):
        """Оптимизация запроса"""
        return super().get_queryset(request).select_related('collection')
    
    def save_model(self, request, obj, form, change):
        """Обработка сохранения модели в админке"""
        if not obj.title and obj.original_filename:
            obj.title = os.path.splitext(obj.original_filename)[0]
        super().save_model(request, obj, form, change)


class DocumentChunkAdmin(ModelAdmin):
    """Админка для чанков документов"""
    list_display = ('document_title', 'chunk_index', 'token_count', 
                    'has_embedding', 'relevance_score', 'usage_count')
    list_filter = ('has_embedding', 'chunk_strategy', 'is_active', 
                   'embedding_model')
    search_fields = ('content', 'document__title', 'document__original_filename')
    readonly_fields = ('created_at', 'updated_at', 'preview_display', 
                      'document_title')
    fieldsets = (
        ('Связи', {
            'fields': ('document', 'chunk_index')
        }),
        ('Содержимое', {
            'fields': ('content', 'token_count', 'preview_display')
        }),
        ('Позиция в документе', {
            'fields': ('start_char', 'end_char', 'start_line', 'end_line')
        }),
        ('Эмбеддинги', {
            'fields': ('has_embedding', 'embedding_model', 'embedding_version', 
                      'embedding_dim')
        }),
        ('Качество', {
            'fields': ('relevance_score', 'is_key_chunk')
        }),
        ('Использование', {
            'fields': ('usage_count', 'last_used')
        }),
        ('Метаданные', {
            'fields': ('metadata', 'chunk_strategy')
        }),
        ('Системная информация', {
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
    """Админка для моделей эмбеддингов"""
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
        ('Основная информация', {
            'fields': ('session_id', 'conversation_id', 'user_id', 'status')
        }),
        ('Запрос и контекст', {
            'fields': ('original_query', 'system_prompt', 'augmented_prompt')
        }),
        ('Модели', {
            'fields': ('embedding_model', 'llm_model', 'search_query')
        }),
        ('Ответ', {
            'fields': ('generated_answer', 'generation_metadata', 
                      'confidence_score')
        }),
        ('Производительность', {
            'fields': ('total_time_display', 'search_time_ms', 
                      'generation_time_ms')
        }),
        ('Ресурсы и стоимость', {
            'fields': ('input_tokens', 'output_tokens', 'total_tokens_display', 
                      'estimated_cost')
        }),
        ('Обратная связь', {
            'fields': ('has_feedback', 'feedback_score', 'feedback_comment', 
                      'feedback_at')
        }),
        ('Временные метки', {
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


# Регистрируем все модели через admin.site.register()
admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(ModelCheckpoint, ModelCheckpointAdmin)
admin.site.register(ModelMergeJob, ModelMergeJobAdmin)
admin.site.register(InferenceSession, InferenceSessionAdmin)
admin.site.register(ModelEvaluation, ModelEvaluationAdmin)
admin.site.register(ChatSession, ChatSessionAdmin)
admin.site.register(ChatMessage, ChatMessageAdmin)
admin.site.register(ModelComparison, ModelComparisonAdmin)
admin.site.register(FineTuningJob, FineTuningJobAdmin)
admin.site.register(DocumentCollection, DocumentCollectionAdmin)
admin.site.register(Document, DocumentAdmin)
admin.site.register(DocumentChunk, DocumentChunkAdmin)
admin.site.register(EmbeddingModel, EmbeddingModelAdmin)
admin.site.register(RAGSession, RAGSessionAdmin)
admin.site.register(SearchQuery, SearchQueryAdmin)
admin.site.register(SearchResult, SearchResultAdmin)
admin.site.register(SearchQualityFeedback, SearchQualityFeedbackAdmin)
admin.site.register(RAGEvaluation, RAGEvaluationAdmin)
admin.site.register(ProcessingPipeline, ProcessingPipelineAdmin)
admin.site.register(DocumentProcessor, DocumentProcessorAdmin)
admin.site.register(RetrievedChunk)
admin.site.register(DeletedDocumentLog)