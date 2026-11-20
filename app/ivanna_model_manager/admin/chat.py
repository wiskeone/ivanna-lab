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
        (_('Основная информация'), {
            'fields': ('session_id', 'user_id', 'checkpoint', 'is_active')
        }),
        (_('Конфигурация'), {
            'fields': ('system_prompt', 'model_config', 'generation_config'),
            'classes': ('collapse',)
        }),
        (_('История диалога'), {
            'fields': ('conversation_preview',),
            'classes': ('collapse',)
        }),
        (_('Статистика'), {
            'fields': ('session_stats', 'total_messages', 'total_input_tokens', 'total_output_tokens'),
            'classes': ('collapse',)
        }),
        (_('Системная информация'), {
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
        (_('Основная информация'), {
            'fields': ('session', 'role', 'parent')
        }),
        (_('Содержимое'), {
            'fields': ('content',),
            'classes': ('wide',)
        }),
        (_('Метаданные генерации'), {
            'fields': ('generation_metadata_display',),
            'classes': ('collapse',)
        }),
        (_('Системная информация'), {
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
        (_('Основная информация'), {
            'fields': ('name', 'description', 'is_complete')
        }),
        (_('Промпты для сравнения'), {
            'fields': ('prompts', 'prompts_count'),
            'classes': ('wide',)
        }),
        (_('Результаты'), {
            'fields': ('results_preview',),
            'classes': ('collapse',)
        }),
        (_('Метрики'), {
            'fields': ('metrics_display',),
            'classes': ('collapse',)
        }),
        (_('Статистика'), {
            'fields': ('comparison_stats',),
            'classes': ('collapse',)
        }),
        (_('Системная информация'), {
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
