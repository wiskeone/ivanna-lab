import os
import logging
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

import nltk
import numpy as np
from django.utils import timezone
from django.conf import settings
from django.db import transaction
from nltk.tokenize import sent_tokenize

from .models import (
    Document, DocumentChunk, DocumentCollection,
    DocumentProcessor, EmbeddingModel
)
from .services.embedding_service import embedding_service
from .services.vector_store_manager import get_vector_store_manager
from .services.rag_service import EnhancedSearchService
from .services.redis_model_service import redis_model_manager

from celery import shared_task

logger = logging.getLogger(__name__)

# Скачиваем необходимые данные nltk при импорте
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


# --------------------------------------------------------------------------
# Вспомогательные функции для извлечения текста
# --------------------------------------------------------------------------
def extract_text_from_pdf(file_path: str) -> str:
    """Извлечение текста из PDF (pdfplumber + fallback PyMuPDF)."""
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if text.strip():
                return text
    except Exception as e:
        logger.debug(f"pdfplumber failed: {e}")

    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        logger.debug(f"PyMuPDF failed: {e}")
    return ""


def extract_text_from_docx(file_path: str) -> str:
    """Извлечение текста из DOCX."""
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""


def extract_text_from_txt(file_path: str) -> str:
    """Извлечение текста из TXT с автоопределением кодировки."""
    import chardet
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
            encoding = chardet.detect(raw)['encoding'] or 'utf-8'
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            return f.read()
    except Exception as e:
        logger.error(f"TXT extraction error: {e}")
        return ""


def extract_text_from_html(file_path: str) -> str:
    """Извлечение текста из HTML."""
    try:
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator="\n")
    except Exception as e:
        logger.error(f"HTML extraction error: {e}")
        return ""


def extract_text_from_markdown(file_path: str) -> str:
    """Извлечение текста из Markdown."""
    try:
        import markdown
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html = markdown.markdown(f.read())
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
    except Exception as e:
        logger.error(f"Markdown extraction error: {e}")
        return ""


def get_file_hash(file_path: str) -> str:
    """SHA256 хеш файла."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# --------------------------------------------------------------------------
# Создание чанков (улучшенное с учётом предложений)
# --------------------------------------------------------------------------
def create_chunks(
    document: Document,
    content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[DocumentChunk]:
    """
    Разделить текст на чанки с учётом границ предложений.
    """
    sentences = sent_tokenize(content)
    chunks = []
    current_chunk = []
    current_len = 0
    idx = 0
    start_char = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len <= chunk_size:
            current_chunk.append(sent)
            current_len += sent_len
        else:
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                end_char = start_char + len(chunk_text)
                chunk = DocumentChunk.objects.create(
                    document=document,
                    chunk_index=idx,
                    content=chunk_text,
                    token_count=len(chunk_text.split()),
                    start_char=start_char,
                    end_char=end_char,
                    chunk_strategy='semantic',
                    metadata={
                        'chunk_size': chunk_size,
                        'chunk_overlap': chunk_overlap,
                        'char_length': len(chunk_text),
                        'word_count': len(chunk_text.split()),
                        'num_sentences': len(current_chunk)
                    }
                )
                chunks.append(chunk)
                idx += 1
                # Новый чанк начинаем с перекрытием (последние предложения)
                overlap_sentences = 0
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= chunk_overlap:
                        overlap_len += len(s)
                        overlap_sentences += 1
                    else:
                        break
                current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_len = sum(len(s) for s in current_chunk)
                start_char = end_char - current_len
                # Добавляем текущее предложение
                current_chunk.append(sent)
                current_len += sent_len
            else:
                # Если предложение длиннее chunk_size, разбиваем по словам
                words = sent.split()
                part = []
                part_len = 0
                for w in words:
                    w_len = len(w) + 1
                    if part_len + w_len <= chunk_size:
                        part.append(w)
                        part_len += w_len
                    else:
                        if part:
                            part_text = ' '.join(part)
                            end_char = start_char + len(part_text)
                            chunk = DocumentChunk.objects.create(
                                document=document,
                                chunk_index=idx,
                                content=part_text,
                                token_count=len(part),
                                start_char=start_char,
                                end_char=end_char,
                                chunk_strategy='word_split',
                                metadata={
                                    'chunk_size': chunk_size,
                                    'chunk_overlap': chunk_overlap,
                                    'char_length': len(part_text),
                                    'word_count': len(part)
                                }
                            )
                            chunks.append(chunk)
                            idx += 1
                            # Перекрытие для следующего куска
                            overlap_words = 0
                            overlap_len = 0
                            for w2 in reversed(part):
                                if overlap_len + len(w2) + 1 <= chunk_overlap:
                                    overlap_len += len(w2) + 1
                                    overlap_words += 1
                                else:
                                    break
                            part = part[-overlap_words:] if overlap_words > 0 else []
                            part_len = sum(len(w) + 1 for w in part)
                            start_char = end_char - part_len
                        part.append(w)
                        part_len += w_len
                if part:
                    part_text = ' '.join(part)
                    end_char = start_char + len(part_text)
                    chunk = DocumentChunk.objects.create(
                        document=document,
                        chunk_index=idx,
                        content=part_text,
                        token_count=len(part),
                        start_char=start_char,
                        end_char=end_char,
                        chunk_strategy='word_split',
                        metadata={
                            'chunk_size': chunk_size,
                            'chunk_overlap': chunk_overlap,
                            'char_length': len(part_text),
                            'word_count': len(part)
                        }
                    )
                    chunks.append(chunk)
                    idx += 1
                # после обработки длинного предложения сбрасываем current_chunk
                current_chunk = []
                current_len = 0
                start_char = end_char

    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        end_char = start_char + len(chunk_text)
        chunk = DocumentChunk.objects.create(
            document=document,
            chunk_index=idx,
            content=chunk_text,
            token_count=len(chunk_text.split()),
            start_char=start_char,
            end_char=end_char,
            chunk_strategy='semantic',
            metadata={
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'char_length': len(chunk_text),
                'word_count': len(chunk_text.split()),
                'num_sentences': len(current_chunk)
            }
        )
        chunks.append(chunk)

    return chunks


# --------------------------------------------------------------------------
# Реализация process_document_task
# --------------------------------------------------------------------------
def _process_document_impl(self, document_id: int) -> Dict[str, Any]:
    """
    Полная реализация обработки документа.
    Вызывается из process_document_task.
    """
    logger.info(f"🚀 Processing document {document_id}")

    # Блокировка Redis для предотвращения параллельной обработки
    lock_key = f"doc_process:{document_id}"
    if redis_model_manager.redis.get(lock_key):
        logger.warning(f"Document {document_id} is already being processed, skipping")
        return {'success': False, 'error': 'Already processing'}
    redis_model_manager.redis.setex(lock_key, 3600, '1')

    try:
        with transaction.atomic():
            document = Document.objects.select_for_update().get(id=document_id)
            if document.status == 'PROCESSING':
                logger.warning(f"Document {document_id} already processing")
                redis_model_manager.redis.delete(lock_key)
                return {'success': False, 'error': 'Already processing'}
            document.status = 'PROCESSING'
            document.processing_progress = 10
            document.save()

        # Удаляем старые чанки (мягкое удаление)
        old_chunks = document.chunks.filter(is_active=True)
        old_count = old_chunks.count()
        if old_count:
            old_chunks.update(is_active=False)
            logger.info(f"Deactivated {old_count} old chunks")

        # Создаём лог обработки
        processor = DocumentProcessor.objects.create(
            document=document,
            status='STARTED',
            progress=10
        )

        # 1. Получаем путь к файлу и извлекаем текст
        file_path = document.full_file_path
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {document.original_filename}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            raw_text = extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            raw_text = extract_text_from_docx(file_path)
        elif ext in ['.html', '.htm']:
            raw_text = extract_text_from_html(file_path)
        elif ext == '.md':
            raw_text = extract_text_from_markdown(file_path)
        else:
            raw_text = extract_text_from_txt(file_path)

        if not raw_text.strip():
            raise ValueError("No extractable text found")

        document.raw_content = raw_text
        document.content_hash = get_file_hash(file_path)
        document.save()

        processor.status = 'EXTRACTING'
        processor.progress = 30
        processor.save()

        # 2. Создаём чанки
        chunk_size = document.metadata.get('chunk_size', 1000)
        chunk_overlap = document.metadata.get('chunk_overlap', 200)
        chunks = create_chunks(document, raw_text, chunk_size, chunk_overlap)

        if not chunks:
            raise ValueError("No chunks created")

        document.processing_progress = 50
        document.save()

        processor.status = 'CHUNKING'
        processor.progress = 50
        processor.chunks_created = len(chunks)
        processor.save()

        # 3. Создаём эмбеддинги, используя эффективную модель коллекции
        embeddings_created = 0
        if document.collection:
            embedding_model = document.collection.get_effective_embedding_model()
            if embedding_model:
                processor.status = 'EMBEDDING'
                processor.progress = 70
                processor.save()

                try:
                    batch_size = 32
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        texts = [ch.content for ch in batch]
                        embeddings = embedding_service.create_embeddings_batch(
                            texts=texts,
                            model_id=embedding_model.id,
                            normalize=True
                        )
                        for j, chunk in enumerate(batch):
                            chunk.embedding_vector = embeddings[j].tolist()
                            chunk.embedding_model = embedding_model
                            chunk.has_embedding = True
                            chunk.embedding_dim = len(embeddings[j])
                            chunk.save()
                        embeddings_created += len(batch)

                        progress = 70 + int((i + len(batch)) / len(chunks) * 20)
                        self.update_state(
                            state='PROGRESS',
                            meta={'progress': progress, 'current_step': f'Embedding {i+len(batch)}/{len(chunks)}'}
                        )

                    logger.info(f"Created {embeddings_created} embeddings for document {document_id}")

                    # Индексируем в FAISS
                    if document.collection.id:
                        vstore = get_vector_store_manager()
                        added = vstore.index_chunks(chunks, document.collection.id)
                        logger.info(f"Indexed {added} vectors for collection {document.collection.id}")

                except Exception as e:
                    logger.error(f"Embedding error: {e}")
            else:
                logger.warning(f"No embedding model available for collection {document.collection.id}, skipping embeddings")
        else:
            logger.warning(f"Document {document_id} has no collection, skipping embeddings")

        # 4. Завершение
        with transaction.atomic():
            document = Document.objects.get(id=document_id)
            document.status = 'INDEXED' if embeddings_created > 0 else 'CHUNKED'
            document.processing_progress = 100
            document.processed_at = timezone.now()
            if embeddings_created > 0:
                document.indexed_at = timezone.now()
            document.save()

            if document.collection:
                document.collection.update_statistics()

        processor.status = 'COMPLETED'
        processor.progress = 100
        processor.embeddings_created = embeddings_created
        processor.processing_time_ms = (timezone.now() - processor.started_at).total_seconds() * 1000
        processor.save()

        # Инвалидируем кэш BM25 для коллекции
        if document.collection:
            search_service = EnhancedSearchService()
            search_service.invalidate_bm25_cache(document.collection.id)

        logger.info(f"✅ Document {document_id} processed: {len(chunks)} chunks, {embeddings_created} embeddings")
        return {
            'success': True,
            'document_id': document_id,
            'chunks_created': len(chunks),
            'embeddings_created': embeddings_created,
            'status': document.status
        }

    except Exception as e:
        logger.error(f"❌ Error processing document {document_id}: {e}", exc_info=True)
        try:
            doc = Document.objects.get(id=document_id)
            doc.status = 'ERROR'
            doc.error_message = str(e)
            doc.save()
        except:
            pass
        try:
            processor.status = 'FAILED'
            processor.error_message = str(e)
            processor.save()
        except:
            pass
        return {'success': False, 'error': str(e), 'document_id': document_id}

    finally:
        redis_model_manager.redis.delete(lock_key)


# --------------------------------------------------------------------------
# Создание эмбеддингов для коллекции
# --------------------------------------------------------------------------
def _create_embeddings_for_collection_impl(self, collection_id: int) -> Dict[str, Any]:
    """
    Создание эмбеддингов для всех чанков коллекции, у которых их нет.
    Использует эффективную модель коллекции.
    """
    from .models import DocumentChunk, EmbeddingModel

    logger.info(f"Creating embeddings for collection {collection_id}")

    try:
        collection = DocumentCollection.objects.get(id=collection_id)
        embedding_model = collection.get_effective_embedding_model()
        if not embedding_model:
            return {'success': False, 'error': 'No embedding model available for this collection'}

        # Чанки без эмбеддингов
        chunks = DocumentChunk.objects.filter(
            document__collection=collection,
            has_embedding=False,
            is_active=True
        ).exclude(content='')

        total = chunks.count()
        if total == 0:
            if DocumentChunk.objects.filter(document__collection=collection).count() == 0:
                return {'success': False, 'error': 'No chunks in collection', 'needs_processing': True}
            return {'success': True, 'message': 'All chunks already have embeddings', 'processed': 0}

        processed = 0
        batch_size = 32
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            texts = [ch.content for ch in batch]
            try:
                embeddings = embedding_service.create_embeddings_batch(
                    texts=texts,
                    model_id=embedding_model.id,
                    normalize=True
                )
                for j, chunk in enumerate(batch):
                    chunk.embedding_vector = embeddings[j].tolist()
                    chunk.embedding_model = embedding_model
                    chunk.has_embedding = True
                    chunk.embedding_dim = len(embeddings[j])
                    chunk.save()
                processed += len(batch)
                self.update_state(
                    state='PROGRESS',
                    meta={'current': processed, 'total': total, 'percent': int(processed / total * 100)}
                )
            except Exception as e:
                logger.error(f"Batch embedding error: {e}")

        # Индексируем в FAISS
        if processed > 0:
            vstore = get_vector_store_manager()
            vstore.update_index(collection_id)

        logger.info(f"Created {processed} embeddings for collection {collection_id}")
        return {'success': True, 'processed': processed, 'total': total}

    except DocumentCollection.DoesNotExist:
        return {'success': False, 'error': f'Collection {collection_id} not found'}
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        return {'success': False, 'error': str(e)}


# --------------------------------------------------------------------------
# Полная перестройка эмбеддингов и FAISS индекса для коллекции
# --------------------------------------------------------------------------
def _rebuild_embeddings_and_index_impl(self, collection_id: int) -> Dict[str, Any]:
    """
    Удаляет все существующие эмбеддинги чанков коллекции,
    создаёт их заново (используя эффективную модель коллекции)
    и перестраивает FAISS индекс.
    """
    logger.info(f"🔄 Rebuilding embeddings and index for collection {collection_id}")

    try:
        collection = DocumentCollection.objects.get(id=collection_id)
        embedding_model = collection.get_effective_embedding_model()
        if not embedding_model:
            return {'success': False, 'error': 'No embedding model available for this collection'}

        # Получаем все чанки коллекции (активные)
        chunks = DocumentChunk.objects.filter(
            document__collection=collection,
            is_active=True
        ).exclude(content='')

        total = chunks.count()
        if total == 0:
            return {'success': False, 'error': 'No chunks in collection'}

        # Сбрасываем эмбеддинги
        chunks.update(
            has_embedding=False,
            embedding_vector=None,
            embedding_model=None,
            embedding_dim=None
        )
        logger.info(f"Reset embeddings for {total} chunks")

        # Создаём эмбеддинги батчами
        processed = 0
        batch_size = 32
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            texts = [ch.content for ch in batch]
            try:
                embeddings = embedding_service.create_embeddings_batch(
                    texts=texts,
                    model_id=embedding_model.id,
                    normalize=True
                )
                for j, chunk in enumerate(batch):
                    chunk.embedding_vector = embeddings[j].tolist()
                    chunk.embedding_model = embedding_model
                    chunk.has_embedding = True
                    chunk.embedding_dim = len(embeddings[j])
                    chunk.save()
                processed += len(batch)
                self.update_state(
                    state='PROGRESS',
                    meta={'current': processed, 'total': total, 'percent': int(processed / total * 100)}
                )
            except Exception as e:
                logger.error(f"Batch embedding error: {e}")
                # Продолжаем с остальными, не прерываем

        logger.info(f"Created embeddings for {processed} chunks")

        # Перестраиваем FAISS индекс
        if processed > 0:
            vstore = get_vector_store_manager()
            success = vstore.build_index(collection_id, force=True)
            if not success:
                logger.error("Failed to rebuild FAISS index")
        else:
            logger.warning("No embeddings created, skipping FAISS index rebuild")

        # Обновляем статистику коллекции
        collection.update_statistics()

        logger.info(f"✅ Rebuilt embeddings and index for collection {collection_id}")
        return {'success': True, 'collection_id': collection_id, 'processed': processed, 'total': total}

    except DocumentCollection.DoesNotExist:
        return {'success': False, 'error': f'Collection {collection_id} not found'}
    except Exception as e:
        logger.error(f"Error rebuilding embeddings: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


# --------------------------------------------------------------------------
# Построение FAISS индекса для коллекции
# --------------------------------------------------------------------------
def _build_faiss_index_impl(collection_id: int, index_type: str = 'flat') -> Dict[str, Any]:
    """
    Полное перестроение FAISS индекса для коллекции.
    """
    logger.info(f"Building FAISS index for collection {collection_id}, type={index_type}")

    try:
        vstore = get_vector_store_manager()
        success = vstore.build_index(collection_id, index_type=index_type, force=True)
        if success:
            return {'success': True, 'collection_id': collection_id, 'index_type': index_type}
        else:
            return {'success': False, 'error': 'Failed to build index'}
    except Exception as e:
        logger.error(f"Error building index: {e}")
        return {'success': False, 'error': str(e)}


# --------------------------------------------------------------------------
# Обновление FAISS индекса (добавление новых чанков)
# --------------------------------------------------------------------------
def _update_faiss_index_impl(collection_id: int) -> Dict[str, Any]:
    """
    Обновление FAISS индекса: добавление новых чанков.
    """
    logger.info(f"Updating FAISS index for collection {collection_id}")
    try:
        vstore = get_vector_store_manager()
        added = vstore.update_index(collection_id)
        return {'success': True, 'collection_id': collection_id, 'added': added}
    except Exception as e:
        logger.error(f"Error updating index: {e}")
        return {'success': False, 'error': str(e)}


# --------------------------------------------------------------------------
# Пересоздание чанков документа
# --------------------------------------------------------------------------
def _regenerate_document_chunks_impl(self, document_id: int) -> Dict[str, Any]:
    """
    Пересоздание чанков документа с новыми параметрами.
    """
    logger.info(f"Regenerating chunks for document {document_id}")

    try:
        document = Document.objects.get(id=document_id, is_active=True)
        old_chunks = list(document.chunks.filter(is_active=True))

        # Получаем параметры из метаданных
        chunk_size = document.metadata.get('chunk_size', 1000)
        chunk_overlap = document.metadata.get('chunk_overlap', 200)

        # Деактивируем старые чанки
        document.chunks.update(is_active=False)

        # Создаём новые чанки
        content = document.processed_content or document.raw_content
        if not content:
            raise ValueError("Document has no content")

        new_chunks = create_chunks(document, content, chunk_size, chunk_overlap)

        # Создаём эмбеддинги, используя эффективную модель коллекции
        embeddings_created = 0
        if document.collection:
            embedding_model = document.collection.get_effective_embedding_model()
            if embedding_model:
                texts = [ch.content for ch in new_chunks]
                try:
                    embeddings = embedding_service.create_embeddings_batch(
                        texts=texts,
                        model_id=embedding_model.id,
                        normalize=True
                    )
                    for i, chunk in enumerate(new_chunks):
                        chunk.embedding_vector = embeddings[i].tolist()
                        chunk.embedding_model = embedding_model
                        chunk.has_embedding = True
                        chunk.embedding_dim = len(embeddings[i])
                        chunk.save()
                    embeddings_created = len(new_chunks)
                except Exception as e:
                    logger.error(f"Embedding error: {e}")

        # Индексируем в FAISS
        if embeddings_created > 0 and document.collection:
            vstore = get_vector_store_manager()
            vstore.index_chunks(new_chunks, document.collection.id)

        # Обновляем статус
        document.status = 'INDEXED' if embeddings_created > 0 else 'CHUNKED'
        document.requires_reindexing = False
        document.save()

        # Инвалидируем кэш BM25
        if document.collection:
            search_service = EnhancedSearchService()
            search_service.invalidate_bm25_cache(document.collection.id)

        logger.info(f"Regenerated {len(new_chunks)} chunks for document {document_id}")
        return {
            'success': True,
            'document_id': document_id,
            'new_chunks': len(new_chunks),
            'embeddings_created': embeddings_created
        }

    except Document.DoesNotExist:
        return {'success': False, 'error': 'Document not found'}
    except Exception as e:
        logger.error(f"Error regenerating chunks: {e}")
        return {'success': False, 'error': str(e)}


# --------------------------------------------------------------------------
# Celery-задачи (обёртки)
# --------------------------------------------------------------------------
@shared_task(bind=True, queue='gpu_tasks', max_retries=3)
def process_document_task(self, document_id):
    return _process_document_impl(self, document_id)


@shared_task(bind=True, queue='gpu_tasks', max_retries=3)
def create_embeddings_for_collection_task(self, collection_id):
    return _create_embeddings_for_collection_impl(self, collection_id)


@shared_task(bind=True, queue='gpu_tasks')
def rebuild_embeddings_and_index_task(self, collection_id):
    return _rebuild_embeddings_and_index_impl(self, collection_id)


@shared_task(queue='cpu_tasks')
def build_faiss_index_task(collection_id, index_type='flat'):
    return _build_faiss_index_impl(collection_id, index_type)


@shared_task(queue='cpu_tasks')
def update_faiss_index_task(collection_id):
    return _update_faiss_index_impl(collection_id)


@shared_task(bind=True, queue='cpu_tasks')
def regenerate_document_chunks_task(self, document_id):
    return _regenerate_document_chunks_impl(self, document_id)