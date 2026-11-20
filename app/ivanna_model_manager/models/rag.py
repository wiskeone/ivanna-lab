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
    """Custom manager for documents."""

    def get_queryset(self):
        """Return only active documents by default."""
        # Safe check for the is_active field
        try:
            # Check whether the model has the is_active field
            if hasattr(self.model, '_meta') and 'is_active' in [f.name for f in self.model._meta.get_fields()]:
                return super().get_queryset().filter(is_active=True)
        except Exception:
            pass
        return super().get_queryset()
    
    def all_documents(self):
        """Get ALL documents (including inactive ones)."""
        return super().get_queryset()

    def find_similar_by_content(self, content, limit=5, min_similarity=0.7):
        """
        Find documents with similar content.
        In a real system this would use vector embedding search.
        """
        # For now a simple text search is used
        documents = []
        content_words = set(content.lower().split())

        for doc in self.filter(status='INDEXED'):  # get_queryset already filters is_active=True
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
        
        # Sort by similarity
        documents.sort(key=lambda x: x['similarity'], reverse=True)
        return [doc['document'] for doc in documents[:limit]]

    def get_by_collection_and_type(self, collection_id=None, doc_type=None):
        """Get documents by collection and type."""
        queryset = self.all()  # self.all() returns active documents
        
        if collection_id:
            queryset = queryset.filter(collection_id=collection_id)
        
        if doc_type:
            queryset = queryset.filter(document_type=doc_type)
        
        return queryset
    
    def get_statistics(self):
        """Get document statistics."""
        from django.db.models import Count, Avg, Sum

        stats = self.all().aggregate(  # self.all() returns active documents
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
    Document collection for RAG.
    Groups related documents (e.g. by project, user, or topic).
    """
    name = models.CharField(max_length=255, help_text=_("Название коллекции"))
    description = models.TextField(blank=True, help_text=_("Описание коллекции"))
    
    # Owner and access
    owner_id = models.CharField(max_length=100, blank=True, help_text=_("ID владельца (пользователя/организации)"))
    is_public = models.BooleanField(default=False, help_text=_("Доступна ли коллекция другим пользователям"))

    # Embedding model
    # v5.2: collection reranker (cross-encoder); null → auto by language
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
    
    # Vector index stats
    vector_index_created = models.BooleanField(default=False, help_text=_("Создан ли векторный индекс"))
    total_vectors = models.IntegerField(default=0, help_text=_("Количество векторов в индексе"))
    vector_dimension = models.IntegerField(null=True, blank=True, help_text=_("Размерность векторов"))
    index_metadata = models.JSONField(default=dict, blank=True, help_text=_("Метаданные векторного индекса"))
    
    # Processing settings
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
    
    # Metadata filters
    metadata_filters = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Фильтры по метаданным для этой коллекции {'doc_type': 'legal', 'year': 2023}")
    )
    
    # Statistics
    total_documents = models.IntegerField(default=0, help_text=_("Общее количество документов"))
    total_chunks = models.IntegerField(default=0, help_text=_("Общее количество чанков"))
    total_size_mb = models.FloatField(default=0.0, help_text=_("Общий размер документов в МБ"))
    
    # System fields
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
        """Validate the collection."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValidationError("Перекрытие чанков должно быть меньше их размера")

    def update_statistics(self):
        """Update collection statistics."""
        from django.db.models import Count, Sum
        
        docs = self.documents.filter(is_active=True)
        self.total_documents = docs.count()
        
        # Count total chunks and size
        chunks_info = docs.aggregate(
            total_chunks=Count('chunks', filter=Q(chunks__is_active=True)),
            total_size=Sum('file_size_mb')
        )
        
        self.total_chunks = chunks_info['total_chunks'] or 0
        self.total_size_mb = chunks_info['total_size'] or 0.0
        
        # Count vectors (chunks with embeddings)
        chunks_with_embeddings = DocumentChunk.objects.filter(
            document__collection=self,
            has_embedding=True,
            is_active=True
        ).count()
        
        self.total_vectors = chunks_with_embeddings
        
        # Set the vector dimension if vectors exist
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
        """Get embedding creation progress."""
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
        """Start building the vector index."""
        try:
            from ..tasks import build_faiss_index_task
            task = build_faiss_index_task.delay(self.id, index_type)
            return task.id
        except Exception as e:
            logger.error(f"Error starting index build for collection {self.id}: {str(e)}")
            return None

    def search_in_index(self, query_vector: list, k: int = 10):
        """Search in the vector index."""
        try:
            from ..tasks import search_faiss_index_task
            result = search_faiss_index_task.delay(self.id, query_vector, k)
            return result
        except Exception as e:
            logger.error(f"Error starting search for collection {self.id}: {str(e)}")
            return None

    def update_vector_index(self):
        """Update the vector index."""
        try:
            from ..tasks import update_faiss_index_task
            task = update_faiss_index_task.delay(self.id)
            return task.id
        except Exception as e:
            logger.error(f"Error starting index update for collection {self.id}: {str(e)}")
            return None

    def get_index_stats(self):
        """Get vector index statistics."""
        try:
            from ..services.faiss_service import FAISSService
            faiss_service = FAISSService()
            return faiss_service.get_index_stats(self.id)
        except Exception as e:
            logger.error(f"Error getting index stats for collection {self.id}: {str(e)}")
            return {'error': str(e)}

    @property
    def owner(self):
        """Get the owner object."""
        from django.contrib.auth.models import User
        try:
            if self.owner_id:
                return User.objects.get(id=int(self.owner_id))
        except (ValueError, User.DoesNotExist):
            return None
    
    @property
    def owner_username(self):
        """Owner username."""
        owner = self.owner
        return owner.username if owner else self.owner_id or "Система"
    
    def get_owner_display(self):
        """Owner display name."""
        if self.owner:
            return f"{self.owner.username} ({self.owner.get_full_name() or self.owner.email})"
        return self.owner_id or "Система"

    def get_effective_embedding_model(self):
        """
        Return the embedding model used for this collection.
        If the collection has no explicit model, the global default is used.
        """
        if self.embedding_model:
            return self.embedding_model
        return EmbeddingModel.objects.filter(is_default=True).first()


class Document(models.Model):
    """
    Main document model for RAG.
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
    
    # Basic info
    original_filename = models.CharField(max_length=500, help_text=_("Оригинальное имя файла"))
    title = models.CharField(max_length=500, blank=True, help_text=_("Название документа"))
    description = models.TextField(blank=True, help_text=_("Описание документа"))
    
    # Type and content
    document_type = models.CharField(max_length=20, choices=DOCUMENT_TYPE_CHOICES, default='UNKNOWN')
    content_hash = models.CharField(max_length=64, help_text=_("SHA256 хеш содержимого для дедупликации"))
    raw_content = models.TextField(blank=True, help_text=_("Исходное текстовое содержимое"))
    processed_content = models.TextField(blank=True, help_text=_("Очищенное содержимое после обработки"))
    
    # Paths and sizes
    file_path = models.CharField(max_length=1000, blank=True, help_text=_("Путь к исходному файлу"))
    file_size_mb = models.FloatField(default=0.0, help_text=_("Размер файла в МБ"))
    file_encoding = models.CharField(max_length=50, default='utf-8', help_text=_("Кодировка файла"))
    
    # Metadata
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text=_("Произвольные метаданные документа")
    )

    # Tags and categories
    tags = models.JSONField(default=list, blank=True, help_text=_("Теги документа"))
    category = models.CharField(max_length=100, blank=True, help_text=_("Категория документа"))
    
    # Status and processing
    status = models.CharField(max_length=20, choices=DOCUMENT_STATUS_CHOICES, default='UPLOADED')
    processing_progress = models.FloatField(default=0.0, help_text=_("Прогресс обработки (0-100)"))
    error_message = models.TextField(blank=True, help_text=_("Сообщение об ошибке обработки"))
    
    # Quality and relevance
    quality_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Оценка качества документа (0-1)")
    )
    
    # Relations
    collection = models.ForeignKey(
        DocumentCollection,
        on_delete=models.CASCADE,
        related_name='documents',
        null=True,
        blank=True
    )
    
    # Timestamps
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    indexed_at = models.DateTimeField(null=True, blank=True)
    
    # System fields
    is_active = models.BooleanField(default=True)
    requires_reindexing = models.BooleanField(default=False, help_text=_("Требуется переиндексация"))
    objects = DocumentManager()

    class Meta:
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['content_hash']),  # For deduplication
            models.Index(fields=['status', 'uploaded_at']),
            models.Index(fields=['collection', 'status']),
            models.Index(fields=['document_type', 'is_active']),
            models.Index(fields=['quality_score']),
            models.Index(fields=['requires_reindexing']),  # For finding documents to reindex
        ]
        verbose_name = 'Документ'
        verbose_name_plural = 'Документы'
    
    def __str__(self):
        return f"{self.title or self.original_filename} ({self.get_status_display()})"
    
    def clean(self):
        """Validate the document."""
        # Detect the type automatically from the extension
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
        """save() override with automatic hash computation."""
        # Compute the content hash if raw_content is present
        if self.raw_content and not self.content_hash:
            self.content_hash = hashlib.sha256(self.raw_content.encode()).hexdigest()

        # Set the title automatically if missing
        if not self.title and self.original_filename:
            self.title = os.path.splitext(self.original_filename)[0]

        super().save(*args, **kwargs)

        # Update collection statistics
        if self.collection:
            self.collection.update_statistics()

    def file_url(self):
        """Media URL for accessing the file."""
        if self.file_path:
            # Remove duplicate 'documents/' if present
            if self.file_path.startswith('documents/'):
                # Already contains the documents folder — use as is
                return f"{settings.MEDIA_URL}{self.file_path}"
            else:
                # Add the documents folder
                return f"{settings.MEDIA_URL}documents/{self.file_path}"
        return None

    @property
    def full_file_path(self):
        """Full file path on disk."""
        if not self.file_path:
            return None

        # Debug logging
        import logging
        logger = logging.getLogger(__name__)

        # Path is already absolute
        if os.path.isabs(self.file_path):
            logger.debug(f"Absolute path for doc {self.id}: {self.file_path}")
            return self.file_path

        # Starts with 'documents/'
        if self.file_path.startswith('documents/'):
            # Remove possible duplicated 'media/'
            path = self.file_path
            if path.startswith('media/') and path.count('media/') > 1:
                path = 'media/' + path.split('media/')[-1]

            # Build the path relative to MEDIA_ROOT
            full_path = os.path.join(settings.MEDIA_ROOT, path)
            logger.debug(f"Path with 'documents/' for doc {self.id}: {full_path}")
            return full_path

        # Bare filename — prepend 'documents/'
        full_path = os.path.join(settings.MEDIA_ROOT, 'documents', self.file_path)
        logger.debug(f"Path with added 'documents/' for doc {self.id}: {full_path}")
        return full_path

    def file_exists(self):
        """Check whether the file exists."""
        full_path = self.full_file_path
        return full_path and os.path.exists(full_path)
    
    def get_file_response(self):
        """Return a FileResponse for download."""
        full_path = self.full_file_path
        if not full_path or not os.path.exists(full_path):
            # Try to find the file in other possible locations
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
        """File extension."""
        return os.path.splitext(self.original_filename)[1].lower()

    @property
    def has_embeddings(self):
        """Whether the document has embeddings."""
        return self.chunks.filter(has_embedding=True).exists()

    @property
    def chunk_count(self):
        """Number of document chunks."""
        return self.chunks.filter(is_active=True).count()

    def get_similar_documents(self, limit=5):
        """Find similar documents (via chunk embeddings)."""
        # Collect this document's embeddings
        chunk_embeddings = list(self.chunks.filter(has_embedding=True).values_list('embedding_vector', flat=True))

        if not chunk_embeddings:
            return Document.objects.none()

        # Real vector-DB similarity search would go here
        # Placeholder for now
        return Document.objects.filter(
            collection=self.collection,
            is_active=True
        ).exclude(id=self.id)[:limit]
    
    def start_processing(self):
        """Start document processing."""
        try:
            from ..tasks import process_document_task
            return process_document_task.delay(self.id)
        except ImportError as e:
            logger.warning(f"Tasks module not available for start_processing: {e}")
            return None

    @property
    def chunks_with_embeddings(self):
        """Number of chunks with embeddings."""
        return self.chunks.filter(has_embedding=True, is_active=True).count()

    @property
    def chunks_with_embeddings_from_collection_model(self):
        """Number of chunks with embeddings from the collection model."""
        if not self.collection or not self.collection.embedding_model:
            return 0
        return self.chunks.filter(
            has_embedding=True,
            embedding_model=self.collection.embedding_model,
            is_active=True
        ).count()
    
    @property
    def embedding_model_used(self):
        """Embedding model used for this document."""
        # Check whether there are chunks with embeddings
        chunk_with_emb = self.chunks.filter(
            has_embedding=True,
            is_active=True
        ).first()
        
        if chunk_with_emb and chunk_with_emb.embedding_model:
            return chunk_with_emb.embedding_model
        
        # No chunks with embeddings — fall back to the collection model
        return self.collection.embedding_model if self.collection else None

    @property
    def embedding_dimension(self):
        """Embedding dimension of the document."""
        model = self.embedding_model_used
        return model.embedding_dim if model else None
    
    @property
    def has_embeddings(self):
        """Whether the document has any embeddings."""
        return self.chunks.filter(has_embedding=True, is_active=True).exists()

    @property
    def has_complete_embeddings(self):
        """Whether all chunks have embeddings."""
        total_chunks = self.chunks.filter(is_active=True).count()
        if total_chunks == 0:
            return False
        return self.chunks_with_embeddings == total_chunks


class DocumentChunk(models.Model):
    """
    Document fragment (chunk) for vector search.
    """
    CHUNK_STRATEGY_CHOICES = [
        ('recursive', _('Рекурсивный')),
        ('semantic', _('Семантический')),
        ('fixed', _('Фиксированный')),
        ('paragraph', _('Абзацный')),
        ('table', _('Табличный')),
        ('code', _('Код')),
    ]
    
    # Relation to the document
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='chunks'
    )
    
    # Content and position
    # v5.2: source page (for chunk PDF preview)
    page_number = models.IntegerField(null=True, blank=True, help_text=_('Номер страницы источника'))
    chunk_index = models.IntegerField(help_text=_("Порядковый номер чанка в документе"))
    content = models.TextField(help_text=_("Текстовое содержимое чанка"))
    token_count = models.IntegerField(default=0, help_text=_("Количество токенов в чанке"))
    
    # Position in the source document
    start_char = models.IntegerField(help_text=_("Начальная позиция в исходном тексте"))
    end_char = models.IntegerField(help_text=_("Конечная позиция в исходном тексте"))
    start_line = models.IntegerField(null=True, blank=True, help_text=_("Начальная строка (если применимо)"))
    end_line = models.IntegerField(null=True, blank=True, help_text=_("Конечная строка (если применимо)"))
    
    # Chunk metadata
    metadata = models.JSONField(default=dict, blank=True, help_text=_("Метаданные чанка"))
    chunk_strategy = models.CharField(
        max_length=20,
        choices=CHUNK_STRATEGY_CHOICES,
        default='recursive'
    )
    
    # Embeddings
    has_embedding = models.BooleanField(default=False, help_text=_("Создан ли эмбеддинг для чанка"))
    embedding_model = models.ForeignKey(
        'EmbeddingModel',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='chunks'
    )
    embedding_version = models.CharField(max_length=50, blank=True, help_text=_("Версия модели эмбеддингов"))
    
    # Vector representation (in a real system this may live in a vector DB)
    embedding_vector = models.JSONField(
        null=True,
        blank=True,
        help_text=_("Вектор эмбеддинга (JSON список чисел)")
    )
    embedding_dim = models.IntegerField(null=True, blank=True, help_text=_("Размерность эмбеддинга"))
    
    # Quality and relevance
    relevance_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Оценка релевантности чанка")
    )
    is_key_chunk = models.BooleanField(default=False, help_text=_("Ключевой чанк документа"))
    
    # Usage stats
    usage_count = models.IntegerField(default=0, help_text=_("Сколько раз чанк был использован в поиске"))
    last_used = models.DateTimeField(null=True, blank=True)

    # System fields
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
        """Short content preview."""
        return self.content[:100] + ('...' if len(self.content) > 100 else '')

    @property
    def document_title(self):
        """Parent document title."""
        return self.document.title

    def mark_used(self):
        """Mark the chunk as used."""
        self.usage_count += 1
        self.last_used = timezone.now()
        self.save(update_fields=['usage_count', 'last_used', 'updated_at'])
    
    def get_similar_chunks(self, limit=5):
        """Find similar chunks (via embeddings)."""
        if not self.has_embedding or not self.embedding_vector:
            return DocumentChunk.objects.none()

        # Real vector-DB search would go here
        # Chunks from the same document are returned for now
        return DocumentChunk.objects.filter(
            document=self.document,
            is_active=True,
            has_embedding=True
        ).exclude(id=self.id).order_by('chunk_index')[:limit]


class EmbeddingModel(models.Model):
    """
    Model for creating embeddings (vector representations of text).
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
    
    # Model identifiers
    model_id = models.CharField(max_length=500, help_text=_("Идентификатор модели (на HF, в API и т.д.)"))
    model_version = models.CharField(max_length=100, default='latest', help_text=_("Версия модели"))
    
    # Technical specs
    embedding_dim = models.IntegerField(help_text=_("Размерность эмбеддингов"))
    max_tokens = models.IntegerField(default=512, help_text=_("Максимальное количество токенов на вход"))
    is_multilingual = models.BooleanField(default=False, help_text=_("Поддерживает ли несколько языков"))
    
    # Performance
    speed_ms_per_token = models.FloatField(null=True, blank=True, help_text=_("Скорость обработки (мс/токен)"))
    accuracy_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=_("Оценка точности на тестовых данных")
    )
    
    # Configuration
    config = models.JSONField(default=dict, blank=True, help_text=_("Конфигурация модели"))

    # License and cost
    license_type = models.CharField(max_length=100, blank=True, help_text=_("Тип лицензии"))
    cost_per_1k_tokens = models.FloatField(null=True, blank=True, help_text=_("Стоимость за 1000 токенов (если API)"))
    
    # Usage stats
    total_tokens_processed = models.BigIntegerField(default=0, help_text=_("Всего обработано токенов"))
    total_requests = models.IntegerField(default=0, help_text=_("Всего запросов к модели"))
    
    # Status
    is_active = models.BooleanField(default=True, help_text=_("Доступна для использования"))
    is_default = models.BooleanField(default=False, help_text=_("Модель по умолчанию для новых коллекций"))
    is_downloaded = models.BooleanField(default=False, help_text=_("Скачана ли модель локально"))
    
    # Download info
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
    
    # Timestamps
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
        """Validate the model."""
        if self.is_default and self.pk:
            # When this model becomes the default, unset the flag on the others
            EmbeddingModel.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)

    def save(self, *args, **kwargs):
        """save() override handling is_default."""
        # First model ever and no default set — make it the default
        if not self.pk and not EmbeddingModel.objects.filter(is_default=True).exists():
            self.is_default = True

        super().save(*args, **kwargs)

        # Unset is_default on the other models
        if self.is_default:
            EmbeddingModel.objects.filter(is_default=True).exclude(pk=self.pk).update(is_default=False)
