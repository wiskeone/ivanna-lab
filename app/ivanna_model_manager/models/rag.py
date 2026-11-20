import os
import re
from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.db.models import Count, Avg, Max, Min, Q, Sum
from django.core.validators import MinValueValidator, MaxValueValidator
from django.core.exceptions import ValidationError
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from django.utils import timezone
import json
import random
import hashlib
import fnmatch
from pathlib import Path
from django.db.models import Q
import cryptography
import traceback

import logging

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)



class DocumentManager(models.Manager):
    """Кастомный менеджер для документов"""
    
    def get_queryset(self):
        """Возвращает только активные документы по умолчанию"""
        # Безопасная проверка наличия поля is_active
        try:
            # Проверяем, есть ли поле is_active в модели
            if hasattr(self.model, '_meta') and 'is_active' in [f.name for f in self.model._meta.get_fields()]:
                return super().get_queryset().filter(is_active=True)
        except Exception:
            pass
        return super().get_queryset()
    
    def all_documents(self):
        """Получить ВСЕ документы (включая неактивные)"""
        return super().get_queryset()
    
    def find_similar_by_content(self, content, limit=5, min_similarity=0.7):
        """
        Найти документы с похожим содержимым.
        В реальной системе здесь будет поиск по векторным эмбеддингам.
        """
        # Пока используем простой текстовый поиск
        documents = []
        content_words = set(content.lower().split())
        
        for doc in self.filter(status='INDEXED'):  # get_queryset уже фильтрует is_active=True
            if doc.processed_content:
                doc_words = set(doc.processed_content.lower().split())
                common_words = content_words.intersection(doc_words)
                
                if common_words:
                    similarity = len(common_words) / max(len(content_words), 1)
                    if similarity >= min_similarity:
                        documents.append({
                            'document': doc,
                            'similarity': similarity
                        })
        
        # Сортировка по схожести
        documents.sort(key=lambda x: x['similarity'], reverse=True)
        return [doc['document'] for doc in documents[:limit]]
    
    def get_by_collection_and_type(self, collection_id=None, doc_type=None):
        """Получить документы по коллекции и типу"""
        queryset = self.all()  # Используем self.all() чтобы получить активные
        
        if collection_id:
            queryset = queryset.filter(collection_id=collection_id)
        
        if doc_type:
            queryset = queryset.filter(document_type=doc_type)
        
        return queryset
    
    def get_statistics(self):
        """Получить статистику по документам"""
        from django.db.models import Count, Avg, Sum
        
        stats = self.all().aggregate(  # Используем self.all() чтобы получить активные
            total=Count('id'),
            by_type=Count('document_type'),
            avg_size=Avg('file_size_mb'),
            total_size=Sum('file_size_mb'),
            indexed=Count('id', filter=Q(status='INDEXED')),
            processing=Count('id', filter=Q(status='PROCESSING')),
            error=Count('id', filter=Q(status='ERROR')),
        )
        
        return stats


class DocumentCollection(models.Model):
    """
    Коллекция документов для RAG.
    Группирует связанные документы (например, по проекту, пользователю, теме).
    """
    name = models.CharField(max_length=255, help_text=_("Название коллекции"))
    description = models.TextField(blank=True, help_text=_("Описание коллекции"))
    
    # Владелец и доступ
    owner_id = models.CharField(max_length=100, blank=True, help_text=_("ID владельца (пользователя/организации)"))
    is_public = models.BooleanField(default=False, help_text=_("Доступна ли коллекция другим пользователям"))
    
    # Модель эмбеддингов
    # v5.2: реранкер коллекции (cross-encoder); null → авто по языку
    reranker_model = models.ForeignKey(
        'RerankerModel', null=True, blank=True,
        on_delete=models.SET_NULL, related_name='collections',
        help_text=_('Реранкер для гибридного поиска (пусто = авто по языку запроса)'))
    embedding_model = models.ForeignKey(
        'EmbeddingModel',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='collections',
        help_text=_("Модель эмбеддингов для этой коллекции")
    )
    
    # Статистика векторного индекса
    vector_index_created = models.BooleanField(default=False, help_text=_("Создан ли векторный индекс"))
    total_vectors = models.IntegerField(default=0, help_text=_("Количество векторов в индексе"))
    vector_dimension = models.IntegerField(null=True, blank=True, help_text=_("Размерность векторов"))
    index_metadata = models.JSONField(default=dict, blank=True, help_text=_("Метаданные векторного индекса"))
    
    # Настройки обработки
    chunking_strategy = models.CharField(
        max_length=50,
        default='recursive',
        choices=[
            ('recursive', _('Рекурсивное разделение')),
            ('semantic', _('Семантическое разделение')),
            ('fixed', _('Фиксированный размер')),
            ('paragraph', _('По параграфам')),
        ],
        help_text=_("Стратегия разделения на чанки")
    )
    
    chunk_size = models.IntegerField(default=1000, help_text=_("Размер чанка в токенах/символах"))
    chunk_overlap = models.IntegerField(default=200, help_text=_("Перекрытие между чанками"))
    
    # Метаданные фильтрации
    metadata_filters = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Фильтры по метаданным для этой коллекции {'doc_type': 'legal', 'year': 2023}")
    )
    
    # Статистика
    total_documents = models.IntegerField(default=0, help_text=_("Общее количество документов"))
    total_chunks = models.IntegerField(default=0, help_text=_("Общее количество чанков"))
    total_size_mb = models.FloatField(default=0.0, help_text=_("Общий размер документов в МБ"))
    
    # Системные поля
    is_active = models.BooleanField(default=True)
    auto_process = models.BooleanField(default=True, help_text=_("Автоматически обрабатывать новые документы"))
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['owner_id', 'is_active']),
            models.Index(fields=['is_public', 'is_active']),
            models.Index(fields=['embedding_model', 'is_active']),
            models.Index(fields=['vector_index_created']),
        ]
        verbose_name = 'Коллекция документов'
        verbose_name_plural = 'Коллекции документов'
    
    def __str__(self):
        return f"{self.name} ({self.total_documents} док.)"
    
    def clean(self):
        """Валидация коллекции"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValidationError("Перекрытие чанков должно быть меньше их размера")
    
    def update_statistics(self):
        """Обновить статистику коллекции"""
        from django.db.models import Count, Sum
        
        docs = self.documents.filter(is_active=True)
        self.total_documents = docs.count()
        
        # Считаем общее количество чанков и размер
        chunks_info = docs.aggregate(
            total_chunks=Count('chunks', filter=Q(chunks__is_active=True)),
            total_size=Sum('file_size_mb')
        )
        
        self.total_chunks = chunks_info['total_chunks'] or 0
        self.total_size_mb = chunks_info['total_size'] or 0.0
        
        # Считаем количество векторов (чанков с эмбеддингами)
        chunks_with_embeddings = DocumentChunk.objects.filter(
            document__collection=self,
            has_embedding=True,
            is_active=True
        ).count()
        
        self.total_vectors = chunks_with_embeddings
        
        # Если есть векторы, устанавливаем размерность
        if chunks_with_embeddings > 0 and not self.vector_dimension:
            chunk_with_embedding = DocumentChunk.objects.filter(
                document__collection=self,
                has_embedding=True,
                is_active=True
            ).first()
            if chunk_with_embedding and chunk_with_embedding.embedding_model:
                self.vector_dimension = chunk_with_embedding.embedding_model.embedding_dim
        
        self.save(update_fields=['total_documents', 'total_chunks', 'total_size_mb', 
                                'total_vectors', 'vector_dimension', 'updated_at'])
    
    def get_embedding_progress(self):
        """Получить прогресс создания эмбеддингов"""
        chunks_with_embeddings = DocumentChunk.objects.filter(
            document__collection=self,
            has_embedding=True,
            is_active=True
        ).count()
        
        total_chunks = DocumentChunk.objects.filter(
            document__collection=self,
            is_active=True
        ).count()
        
        documents_with_embeddings = Document.objects.filter(
            collection=self,
            is_active=True,
            chunks__has_embedding=True
        ).distinct().count()
        
        total_documents = self.documents.filter(is_active=True).count()
        
        return {
            'chunks': {
                'with_embeddings': chunks_with_embeddings,
                'total': total_chunks,
                'percent': (chunks_with_embeddings / total_chunks * 100) if total_chunks > 0 else 0
            },
            'documents': {
                'with_embeddings': documents_with_embeddings,
                'total': total_documents,
                'percent': (documents_with_embeddings / total_documents * 100) if total_documents > 0 else 0
            }
        }

    def build_vector_index(self, index_type='ivf'):
        """Запустить создание векторного индекса"""
        try:
            from .tasks import build_faiss_index_task
            task = build_faiss_index_task.delay(self.id, index_type)
            return task.id
        except Exception as e:
            logger.error(f"Error starting index build for collection {self.id}: {str(e)}")
            return None

    def search_in_index(self, query_vector: list, k: int = 10):
        """Поиск в векторном индексе"""
        try:
            from .tasks import search_faiss_index_task
            result = search_faiss_index_task.delay(self.id, query_vector, k)
            return result
        except Exception as e:
            logger.error(f"Error starting search for collection {self.id}: {str(e)}")
            return None

    def update_vector_index(self):
        """Обновить векторный индекс"""
        try:
            from .tasks import update_faiss_index_task
            task = update_faiss_index_task.delay(self.id)
            return task.id
        except Exception as e:
            logger.error(f"Error starting index update for collection {self.id}: {str(e)}")
            return None

    def get_index_stats(self):
        """Получить статистику векторного индекса"""
        try:
            from .services.faiss_service import FAISSService
            faiss_service = FAISSService()
            return faiss_service.get_index_stats(self.id)
        except Exception as e:
            logger.error(f"Error getting index stats for collection {self.id}: {str(e)}")
            return {'error': str(e)}

    @property
    def owner(self):
        """Получить объект владельца"""
        from django.contrib.auth.models import User
        try:
            if self.owner_id:
                return User.objects.get(id=int(self.owner_id))
        except (ValueError, User.DoesNotExist):
            return None
    
    @property
    def owner_username(self):
        """Имя владельца"""
        owner = self.owner
        return owner.username if owner else self.owner_id or "Система"
    
    def get_owner_display(self):
        """Отображаемое имя владельца"""
        if self.owner:
            return f"{self.owner.username} ({self.owner.get_full_name() or self.owner.email})"
        return self.owner_id or "Система"

    def get_effective_embedding_model(self):
        """
        Возвращает модель эмбеддингов, которая будет использоваться для этой коллекции.
        Если у коллекции явно не задана модель, возвращается глобальная модель по умолчанию.
        """
        if self.embedding_model:
            return self.embedding_model
        return EmbeddingModel.objects.filter(is_default=True).first()


class Document(models.Model):
    """
    Основная модель документа для RAG.
    """
    DOCUMENT_STATUS_CHOICES = [
        ('UPLOADED', 'Загружен'),
        ('PROCESSING', 'В обработке'),
        ('CHUNKED', 'Разделен на чанки'),
        ('EMBEDDED', 'Эмбеддинги созданы'),
        ('INDEXED', 'Проиндексирован'),
        ('ERROR', 'Ошибка'),
        ('ARCHIVED', 'В архиве'),
    ]
    
    DOCUMENT_TYPE_CHOICES = [
        ('TEXT', 'Текст'),
        ('PDF', 'PDF'),
        ('DOCX', 'Word'),
        ('HTML', 'HTML'),
        ('MARKDOWN', 'Markdown'),
        ('CODE', 'Исходный код'),
        ('CSV', 'Таблица CSV'),
        ('EXCEL', 'Excel'),
        ('RTF', 'RTF'),
        ('PPTX', 'Презентация'),
        ('EPUB', 'EPUB'),
        ('UNKNOWN', 'Неизвестный'),
    ]
    
    # Основная информация
    original_filename = models.CharField(max_length=500, help_text=_("Оригинальное имя файла"))
    title = models.CharField(max_length=500, blank=True, help_text=_("Название документа"))
    description = models.TextField(blank=True, help_text=_("Описание документа"))
    
    # Тип и содержимое
    document_type = models.CharField(max_length=20, choices=DOCUMENT_TYPE_CHOICES, default='UNKNOWN')
    content_hash = models.CharField(max_length=64, help_text=_("SHA256 хеш содержимого для дедупликации"))
    raw_content = models.TextField(blank=True, help_text=_("Исходное текстовое содержимое"))
    processed_content = models.TextField(blank=True, help_text=_("Очищенное содержимое после обработки"))
    
    # Пути и размеры
    file_path = models.CharField(max_length=1000, blank=True, help_text=_("Путь к исходному файлу"))
    file_size_mb = models.FloatField(default=0.0, help_text=_("Размер файла в МБ"))
    file_encoding = models.CharField(max_length=50, default='utf-8', help_text=_("Кодировка файла"))
    
    # Метаданные
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Произвольные метаданные документа")
    )
    
    # Теги и категории
    tags = models.JSONField(default=list, blank=True, help_text=_("Теги документа"))
    category = models.CharField(max_length=100, blank=True, help_text=_("Категория документа"))
    
    # Статус и обработка
    status = models.CharField(max_length=20, choices=DOCUMENT_STATUS_CHOICES, default='UPLOADED')
    processing_progress = models.FloatField(default=0.0, help_text=_("Прогресс обработки (0-100)"))
    error_message = models.TextField(blank=True, help_text=_("Сообщение об ошибке обработки"))
    
    # Качество и релевантность
    quality_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Оценка качества документа (0-1)")
    )
    
    # Связи
    collection = models.ForeignKey(
        DocumentCollection,
        on_delete=models.CASCADE,
        related_name='documents',
        null=True,
        blank=True
    )
    
    # Временные метки
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    indexed_at = models.DateTimeField(null=True, blank=True)
    
    # Системные поля
    is_active = models.BooleanField(default=True)
    requires_reindexing = models.BooleanField(default=False, help_text=_("Требуется переиндексация"))
    objects = DocumentManager()

    class Meta:
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['content_hash']),  # Для дедупликации
            models.Index(fields=['status', 'uploaded_at']),
            models.Index(fields=['collection', 'status']),
            models.Index(fields=['document_type', 'is_active']),
            models.Index(fields=['quality_score']),
            models.Index(fields=['requires_reindexing']),  # Для поиска документов для переиндексации
        ]
        verbose_name = 'Документ'
        verbose_name_plural = 'Документы'
    
    def __str__(self):
        return f"{self.title or self.original_filename} ({self.get_status_display()})"
    
    def clean(self):
        """Валидация документа"""
        # Автоматическое определение типа по расширению
        if not self.document_type or self.document_type == 'UNKNOWN':
            ext = os.path.splitext(self.original_filename)[1].lower()
            ext_to_type = {
                '.txt': 'TEXT',
                '.pdf': 'PDF',
                '.docx': 'DOCX',
                '.doc': 'DOCX',
                '.html': 'HTML',
                '.htm': 'HTML',
                '.md': 'MARKDOWN',
                '.py': 'CODE',
                '.js': 'CODE',
                '.java': 'CODE',
                '.cpp': 'CODE',
                '.csv': 'CSV',
                '.xlsx': 'EXCEL',
                '.xls': 'EXCEL',
            }
            self.document_type = ext_to_type.get(ext, 'UNKNOWN')
    
    def save(self, *args, **kwargs):
        """Переопределение save с автоматическим вычислением хеша"""
        # Вычисляем хеш содержимого если есть raw_content
        if self.raw_content and not self.content_hash:
            self.content_hash = hashlib.sha256(self.raw_content.encode()).hexdigest()
        
        # Автоматически задаем заголовок если не задан
        if not self.title and self.original_filename:
            self.title = os.path.splitext(self.original_filename)[0]
        
        super().save(*args, **kwargs)
        
        # Обновляем статистику коллекции
        if self.collection:
            self.collection.update_statistics()
    
    def file_url(self):
        """URL для доступа к файлу через медиа"""
        if self.file_path:
            # Убираем дублирование 'documents/' если оно есть
            if self.file_path.startswith('documents/'):
                # Уже содержит папку documents - используем как есть
                return f"{settings.MEDIA_URL}{self.file_path}"
            else:
                # Добавляем папку documents
                return f"{settings.MEDIA_URL}documents/{self.file_path}"
        return None

    @property
    def full_file_path(self):
        """Полный путь к файлу на диске"""
        if not self.file_path:
            return None
        
        # Логирование для отладки
        import logging
        logger = logging.getLogger(__name__)
        
        # Если путь уже абсолютный
        if os.path.isabs(self.file_path):
            logger.debug(f"Absolute path for doc {self.id}: {self.file_path}")
            return self.file_path
        
        # Если начинается с 'documents/'
        if self.file_path.startswith('documents/'):
            # Убираем возможное дублирование 'media/' если есть
            path = self.file_path
            if path.startswith('media/') and path.count('media/') > 1:
                path = 'media/' + path.split('media/')[-1]
            
            # Строим путь относительно MEDIA_ROOT
            full_path = os.path.join(settings.MEDIA_ROOT, path)
            logger.debug(f"Path with 'documents/' for doc {self.id}: {full_path}")
            return full_path
        
        # Просто имя файла - добавляем 'documents/'
        full_path = os.path.join(settings.MEDIA_ROOT, 'documents', self.file_path)
        logger.debug(f"Path with added 'documents/' for doc {self.id}: {full_path}")
        return full_path
    
    def file_exists(self):
        """Проверяет существование файла"""
        full_path = self.full_file_path
        return full_path and os.path.exists(full_path)
    
    def get_file_response(self):
        """Возвращает FileResponse для скачивания"""
        full_path = self.full_file_path
        if not full_path or not os.path.exists(full_path):
            # Попробуем найти файл в других возможных местах
            possible_paths = [
                full_path,
                os.path.join(settings.MEDIA_ROOT, self.file_path) if self.file_path else None,
                os.path.join(settings.MEDIA_ROOT, 'documents', str(self.collection.id), self.original_filename) if self.collection else None,
                os.path.join(settings.MEDIA_ROOT, 'documents', self.original_filename),
            ]
            
            for path in possible_paths:
                if path and os.path.exists(path):
                    full_path = path
                    break
            
            if not full_path or not os.path.exists(full_path):
                return None
        
        try:
            from django.http import FileResponse
            return FileResponse(
                open(full_path, 'rb'),
                as_attachment=True,
                filename=self.original_filename
            )
        except Exception as e:
            logger.error(f"Error creating file response: {str(e)}")
            return None
            
    @property
    def extension(self):
        """Расширение файла"""
        return os.path.splitext(self.original_filename)[1].lower()
    
    @property
    def has_embeddings(self):
        """Есть ли у документа эмбеддинги"""
        return self.chunks.filter(has_embedding=True).exists()
    
    @property
    def chunk_count(self):
        """Количество чанков документа"""
        return self.chunks.filter(is_active=True).count()
    
    def get_similar_documents(self, limit=5):
        """Найти похожие документы (по эмбеддингам чанков)"""
        # Находим эмбеддинги этого документа
        chunk_embeddings = list(self.chunks.filter(has_embedding=True).values_list('embedding_vector', flat=True))
        
        if not chunk_embeddings:
            return Document.objects.none()
        
        # Здесь будет логика поиска похожих документов через векторную БД
        # Пока возвращаем заглушку
        return Document.objects.filter(
            collection=self.collection,
            is_active=True
        ).exclude(id=self.id)[:limit]
    
    def start_processing(self):
        """Запустить обработку документа"""
        try:
            from .tasks import process_document_task
            return process_document_task.delay(self.id)
        except ImportError as e:
            logger.warning(f"Tasks module not available for start_processing: {e}")
            return None

    @property
    def chunks_with_embeddings(self):
        """Количество чанков с эмбеддингами"""
        return self.chunks.filter(has_embedding=True, is_active=True).count()
    
    @property
    def chunks_with_embeddings_from_collection_model(self):
        """Количество чанков с эмбеддингами от модели коллекции"""
        if not self.collection or not self.collection.embedding_model:
            return 0
        return self.chunks.filter(
            has_embedding=True,
            embedding_model=self.collection.embedding_model,
            is_active=True
        ).count()
    
    @property
    def embedding_model_used(self):
        """Модель эмбеддингов, используемая для этого документа."""
        # Проверяем, есть ли чанки с эмбеддингами
        chunk_with_emb = self.chunks.filter(
            has_embedding=True,
            is_active=True
        ).first()
        
        if chunk_with_emb and chunk_with_emb.embedding_model:
            return chunk_with_emb.embedding_model
        
        # Если чанков с эмбеддингами нет, возвращаем модель коллекции
        return self.collection.embedding_model if self.collection else None
    
    @property
    def embedding_dimension(self):
        """Размерность эмбеддингов документа"""
        model = self.embedding_model_used
        return model.embedding_dim if model else None
    
    @property
    def has_embeddings(self):
        """Есть ли у документа эмбеддинги (любые)"""
        return self.chunks.filter(has_embedding=True, is_active=True).exists()
    
    @property
    def has_complete_embeddings(self):
        """Все ли чанки имеют эмбеддинги"""
        total_chunks = self.chunks.filter(is_active=True).count()
        if total_chunks == 0:
            return False
        return self.chunks_with_embeddings == total_chunks


class DocumentChunk(models.Model):
    """
    Фрагмент (чанк) документа для векторного поиска.
    """
    CHUNK_STRATEGY_CHOICES = [
        ('recursive', _('Рекурсивный')),
        ('semantic', _('Семантический')),
        ('fixed', _('Фиксированный')),
        ('paragraph', _('Абзацный')),
        ('table', _('Табличный')),
        ('code', _('Код')),
    ]
    
    # Связь с документом
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='chunks'
    )
    
    # Содержимое и позиция
    # v5.2: страница источника (для PDF-превью чанка)
    page_number = models.IntegerField(null=True, blank=True, help_text=_('Номер страницы источника'))
    chunk_index = models.IntegerField(help_text=_("Порядковый номер чанка в документе"))
    content = models.TextField(help_text=_("Текстовое содержимое чанка"))
    token_count = models.IntegerField(default=0, help_text=_("Количество токенов в чанке"))
    
    # Позиция в исходном документе
    start_char = models.IntegerField(help_text=_("Начальная позиция в исходном тексте"))
    end_char = models.IntegerField(help_text=_("Конечная позиция в исходном тексте"))
    start_line = models.IntegerField(null=True, blank=True, help_text=_("Начальная строка (если применимо)"))
    end_line = models.IntegerField(null=True, blank=True, help_text=_("Конечная строка (если применимо)"))
    
    # Метаданные чанка
    metadata = models.JSONField(default=dict, blank=True, help_text=_("Метаданные чанка"))
    chunk_strategy = models.CharField(
        max_length=20,
        choices=CHUNK_STRATEGY_CHOICES,
        default='recursive'
    )
    
    # Эмбеддинги
    has_embedding = models.BooleanField(default=False, help_text=_("Создан ли эмбеддинг для чанка"))
    embedding_model = models.ForeignKey(
        'EmbeddingModel',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='chunks'
    )
    embedding_version = models.CharField(max_length=50, blank=True, help_text=_("Версия модели эмбеддингов"))
    
    # Векторное представление (в реальной системе это может храниться в векторной БД)
    embedding_vector = models.JSONField(
        null=True,
        blank=True,
        help_text=_("Вектор эмбеддинга (JSON список чисел)")
    )
    embedding_dim = models.IntegerField(null=True, blank=True, help_text=_("Размерность эмбеддинга"))
    
    # Качество и релевантность
    relevance_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Оценка релевантности чанка")
    )
    is_key_chunk = models.BooleanField(default=False, help_text=_("Ключевой чанк документа"))
    
    # Статистика использования
    usage_count = models.IntegerField(default=0, help_text=_("Сколько раз чанк был использован в поиске"))
    last_used = models.DateTimeField(null=True, blank=True)
    
    # Системные поля
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['document', 'chunk_index']
        indexes = [
            models.Index(fields=['document', 'chunk_index']),
            models.Index(fields=['has_embedding', 'is_active']),
            models.Index(fields=['embedding_model', 'has_embedding']),
            models.Index(fields=['relevance_score']),
            models.Index(fields=['is_key_chunk']),
        ]
        verbose_name = 'Фрагмент документа'
        verbose_name_plural = 'Фрагменты документов'
    
    def __str__(self):
        return f"Чанк {self.chunk_index} документа '{self.document.title}'"
    
    @property
    def preview(self):
        """Короткий превью содержимого"""
        return self.content[:100] + ('...' if len(self.content) > 100 else '')
    
    @property
    def document_title(self):
        """Название родительского документа"""
        return self.document.title
    
    def mark_used(self):
        """Отметить использование чанка"""
        self.usage_count += 1
        self.last_used = timezone.now()
        self.save(update_fields=['usage_count', 'last_used', 'updated_at'])
    
    def get_similar_chunks(self, limit=5):
        """Найти похожие чанки (по эмбеддингам)"""
        if not self.has_embedding or not self.embedding_vector:
            return DocumentChunk.objects.none()
        
        # В реальной системе здесь будет поиск по векторной БД
        # Пока возвращаем чанки из того же документа
        return DocumentChunk.objects.filter(
            document=self.document,
            is_active=True,
            has_embedding=True
        ).exclude(id=self.id).order_by('chunk_index')[:limit]


class EmbeddingModel(models.Model):
    """
    Модель для создания эмбеддингов (векторных представлений текста).
    """
    MODEL_TYPE_CHOICES = [
        ('LOCAL', 'Локальная модель'),
        ('OPENAI', 'OpenAI API'),
        ('COHERE', 'Cohere API'),
        ('HUGGINGFACE', 'Hugging Face'),
        ('SENTENCE_TRANSFORMERS', 'Sentence Transformers'),
    ]
    
    name = models.CharField(max_length=255, help_text=_("Название модели"))
    display_name = models.CharField(max_length=255, blank=True, help_text=_("Человекочитаемое имя для интерфейса"))
    model_type = models.CharField(max_length=30, choices=MODEL_TYPE_CHOICES, default='HUGGINGFACE')
    
    # Идентификаторы моделей
    model_id = models.CharField(max_length=500, help_text=_("Идентификатор модели (на HF, в API и т.д.)"))
    model_version = models.CharField(max_length=100, default='latest', help_text=_("Версия модели"))
    
    # Технические характеристики
    embedding_dim = models.IntegerField(help_text=_("Размерность эмбеддингов"))
    max_tokens = models.IntegerField(default=512, help_text=_("Максимальное количество токенов на вход"))
    is_multilingual = models.BooleanField(default=False, help_text=_("Поддерживает ли несколько языков"))
    
    # Производительность
    speed_ms_per_token = models.FloatField(null=True, blank=True, help_text=_("Скорость обработки (мс/токен)"))
    accuracy_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Оценка точности на тестовых данных")
    )
    
    # Конфигурация
    config = models.JSONField(default=dict, blank=True, help_text=_("Конфигурация модели"))
    
    # Лицензия и стоимость
    license_type = models.CharField(max_length=100, blank=True, help_text=_("Тип лицензии"))
    cost_per_1k_tokens = models.FloatField(null=True, blank=True, help_text=_("Стоимость за 1000 токенов (если API)"))
    
    # Статистика использования
    total_tokens_processed = models.BigIntegerField(default=0, help_text=_("Всего обработано токенов"))
    total_requests = models.IntegerField(default=0, help_text=_("Всего запросов к модели"))
    
    # Статус
    is_active = models.BooleanField(default=True, help_text=_("Доступна для использования"))
    is_default = models.BooleanField(default=False, help_text=_("Модель по умолчанию для новых коллекций"))
    is_downloaded = models.BooleanField(default=False, help_text=_("Скачана ли модель локально"))
    
    # Информация о загрузке
    download_progress = models.IntegerField(default=0, help_text=_("Прогресс загрузки (0-100)"))
    download_status = models.CharField(
        max_length=20,
        default='not_started',
        choices=[
            ('not_started', _('Не начата')),
            ('downloading', _('Загружается')),
            ('completed', _('Завершена')),
            ('failed', _('Ошибка')),
        ]
    )
    download_error = models.TextField(blank=True, help_text=_("Ошибка загрузки, если была"))
    
    # Временные метки
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-is_default', '-is_active', 'name']
        indexes = [
            models.Index(fields=['model_type', 'is_active']),
            models.Index(fields=['is_default']),
            models.Index(fields=['is_downloaded']),
        ]
        verbose_name = 'Модель эмбеддингов'
        verbose_name_plural = 'Модели эмбеддингов'
    
    def __str__(self):
        return f"{self.display_name or self.name} ({self.embedding_dim}d)"
    
    def clean(self):
        """Валидация модели"""
        if self.is_default and self.pk:
            # Если делаем эту модель моделью по умолчанию, снимаем флаг с других
            EmbeddingModel.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)
    
    def save(self, *args, **kwargs):
        """Переопределение save для обработки is_default"""
        # Если это первая модель и is_default не задан, устанавливаем его
        if not self.pk and not EmbeddingModel.objects.filter(is_default=True).exists():
            self.is_default = True
        
        super().save(*args, **kwargs)
        
        # Если установлен is_default, снимаем его с других моделей
        if self.is_default:
            EmbeddingModel.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)
