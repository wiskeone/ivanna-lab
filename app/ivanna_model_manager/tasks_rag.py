import os
import logging
import hashlib
import re
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

# Download the required nltk data at import time
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


# --------------------------------------------------------------------------
# Text extraction helpers
# --------------------------------------------------------------------------
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF (pdfplumber + PyMuPDF fallback)."""
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
    """Extract text from DOCX."""
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT with encoding auto-detection."""
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
    """Extract text from HTML."""
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
    """Extract text from Markdown."""
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



# v5.2: page-by-page extraction and extended chunking ------------------------
def extract_pdf_pages(file_path: str):
    """PDF text page by page (pdfplumber → PyMuPDF fallback)."""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
        if any(t.strip() for t in pages):
            return pages
    except Exception as e:
        logger.debug(f"pdfplumber pages failed: {e}")
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        pages = [p.get_text() for p in doc]
        doc.close()
        return pages
    except Exception as e:
        logger.debug(f"PyMuPDF pages failed: {e}")
    return []


def _page_of(offset: int, page_map):
    for start, end, no in page_map:
        if start <= offset < end:
            return no
    return page_map[-1][2] if page_map else None


def _mk_extra_chunks(document, extras, start_idx) -> list:
    """Persist ready-made extra chunks (tables/images/CSV blocks).

    Audit fix: create_chunks_v2 accepted extra_chunks but never used it —
    table/image chunks extracted by v5.2 extractors were dropped (and XLSX,
    which has no plain text, always failed with "No chunks created").
    """
    created = []
    idx = start_idx
    for ec in extras or []:
        text = (ec.get('content') or '').strip()
        if not text:
            continue
        md = ec.get('metadata') or {}
        created.append(DocumentChunk.objects.create(
            document=document,
            chunk_index=idx,
            content=text,
            token_count=len(text.split()),
            start_char=0,   # no offset in the source text — synthetic chunk
            end_char=len(text),
            chunk_strategy=ec.get('strategy') or 'table',
            page_number=md.get('page'),
            metadata={'extra_chunk': True, **md},
        ))
        idx += 1
    return created


def create_chunks_v2(document, content: str, chunk_size: int = 1000,
                     chunk_overlap: int = 200, strategy: str = 'semantic',
                     page_map=None, extra_chunks=None) -> list:
    """v5.2: chunking with paragraph, page and pre-made chunk support (tables/images).

    strategy: 'paragraph' — blocks split on empty lines, small ones merged up to chunk_size;
              'semantic'/'fixed'/'recursive' — by sentences (old create_chunks logic).
    page_map: [(start_char, end_char, page_no)] — maps chunks to PDF pages.
    extra_chunks: pre-made chunks (tables/images) — saved after the main text.
    """
    page_map = page_map or []
    if strategy == 'paragraph':
        chunks = []

        def _mk(text, start, idx):
            c = DocumentChunk.objects.create(
                document=document, chunk_index=idx, content=text,
                token_count=len(text.split()),
                start_char=start, end_char=start + len(text),
                chunk_strategy='paragraph',
                page_number=_page_of(start, page_map) if page_map else None,
                metadata={'chunk_size': chunk_size, 'chunk_overlap': 0,
                          'char_length': len(text), 'word_count': len(text.split())})
            return c

        idx = 0

        def _flush(text, start):
            nonlocal idx
            if text.strip():
                chunks.append(_mk(text, start, idx))
                idx += 1

        buf, buf_len, buf_start, pos = [], 0, None, 0
        for p in (p.strip() for p in re.split(r'\n\s*\n', content)):
            if not p:
                continue
            start = content.find(p, pos)
            if start < 0:
                start = pos
            pos = start + len(p)

            # v5.2: a "paragraph" larger than chunk_size (PDF text without
            # empty lines glues dozens of pages together) — split into
            # sub-chunks by sentences
            if len(p) > chunk_size:
                _flush('\n\n'.join(buf), buf_start)
                buf, buf_len, buf_start = [], 0, None
                try:
                    sents = sent_tokenize(p)
                except Exception:
                    sents = [p[i:i + chunk_size] for i in range(0, len(p), chunk_size)]
                cur, clen, cstart, i = [], 0, 0, 0
                for sent in sents:
                    off = p.find(sent, i)
                    if off < 0:
                        off = i
                    i = off + len(sent)
                    if cur and clen + len(sent) > chunk_size:
                        _flush(' '.join(cur), start + cstart)
                        cstart, cur, clen = off, [sent], len(sent)
                    else:
                        if not cur:
                            cstart = off
                        cur.append(sent)
                        clen += len(sent)
                _flush(' '.join(cur), start + cstart)
                continue

            if not buf or buf_len + len(p) + 2 <= chunk_size:
                if buf_start is None:
                    buf_start = start
                buf.append(p)
                buf_len += len(p) + 2
            else:
                _flush('\n\n'.join(buf), buf_start)
                buf, buf_len, buf_start = [p], len(p) + 2, start
        if buf:
            _flush('\n\n'.join(buf), buf_start)
        chunks.extend(_mk_extra_chunks(document, extra_chunks, idx))
        return chunks

    # semantic/fixed/recursive — proven create_chunks logic + pages
    chunks = create_chunks(document, content, chunk_size, chunk_overlap)
    if page_map:
        for c in chunks:
            c.page_number = _page_of(c.start_char, page_map)
            c.save(update_fields=['page_number'])
    chunks.extend(_mk_extra_chunks(document, extra_chunks, len(chunks)))
    return chunks


def _pdf_extra_chunks(file_path: str, document) -> list:
    """PDF tables (pdfplumber) and images (PyMuPDF) as separate chunks."""
    extra = []
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            tno = 0
            for i, page in enumerate(pdf.pages, start=1):
                for tb in (page.extract_tables() or [])[:10]:
                    rows = ['\t'.join(str(c or '') for c in row) for row in tb]
                    t = '\n'.join(r for r in rows if r.strip())
                    if len(t) >= 30:
                        extra.append({'content': f'[Таблица, стр. {i}]\n{t}',
                                      'strategy': 'table',
                                      'metadata': {'is_table': True, 'page': i}})
                        tno += 1
                if tno >= 50:
                    break
    except Exception as e:
        logger.debug(f"PDF tables: {e}")
    try:
        import fitz
        doc = fitz.open(file_path)
        img_rel = f'documents/{document.id}/images'
        img_dir = os.path.join(settings.MEDIA_ROOT, img_rel)
        os.makedirs(img_dir, exist_ok=True)
        n_img = 0
        for i, page in enumerate(doc, start=1):
            for j, img in enumerate(page.get_images(full=True)[:5]):
                try:
                    pix = fitz.Pixmap(doc, img[0])
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    fname = f'page-{i}-{j}.png'
                    pix.save(os.path.join(img_dir, fname))
                    extra.append({'content': f'[Изображение, стр. {i}] сохранено: {img_rel}/{fname}',
                                  'strategy': 'table',
                                  'metadata': {'is_image': True, 'page': i,
                                               'image_path': f'{img_rel}/{fname}'}})
                    n_img += 1
                except Exception as e:
                    logger.debug(f"image {i}-{j}: {e}")
            if n_img >= 20:
                break
        doc.close()
    except Exception as e:
        logger.debug(f"PDF images: {e}")
    return extra


def _tabular_chunks_csv(file_path: str) -> list:
    """CSV → table chunks (header repeated in each chunk)."""
    import csv as _csv
    extra = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
            rows = list(_csv.reader(f, delimiter=';'))
        if not rows:
            return extra
        header = rows[0]
        step = 50
        for i in range(1, len(rows), step):
            block = rows[i:i + step]
            t = '\n'.join('\t'.join(str(c or '') for c in r) for r in block)
            extra.append({'content': f'[CSV, строки {i}–{i + len(block) - 1}]\n'
                                     f'\t'.join(str(h) for h in header) + '\n' + t,
                          'strategy': 'table', 'metadata': {'is_table': True, 'rows': len(block)}})
    except Exception as e:
        logger.debug(f"CSV: {e}")
    return extra


def _tabular_chunks_xlsx(file_path: str) -> list:
    """XLSX → table chunks per sheet."""
    extra = []
    try:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            rows = [[c.value for c in row] for row in ws.iter_rows(max_row=500)]
            rows = [r for r in rows if any(v is not None for v in r)]
            if not rows:
                continue
            header = rows[0]
            step = 50
            for i in range(1, len(rows), step):
                block = rows[i:i + step]
                t = '\n'.join('\t'.join('' if v is None else str(v) for v in r) for r in block)
                extra.append({'content': f'[XLSX «{ws.title}», строки {i + 1}–{i + len(block)}]\n'
                                         f'\t'.join(str(h) for h in header) + '\n' + t,
                              'strategy': 'table', 'metadata': {'is_table': True, 'sheet': ws.title}})
        wb.close()
    except Exception as e:
        logger.debug(f"XLSX: {e}")
    return extra



# v5.2: DOCX (tables/images)/RTF/PPTX/EPUB formats -------------------------
def extract_text_from_rtf(file_path: str) -> str:
    """RTF → text (striprtf)."""
    try:
        try:  # striprtf 0.0.x — function in striprtf.striprtf; older builds — striprtf.main
            from striprtf.striprtf import rtf_to_text
        except ImportError:
            from striprtf.main import rtf_to_text
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return rtf_to_text(f.read()) or ""
    except Exception as e:
        logger.error(f"RTF extraction error: {e}")
        return ""


def extract_text_from_doc_legacy(file_path: str) -> str:
    """Binary .doc: legacy converters, if present on the system."""
    import subprocess
    for cmd in (['antiword', file_path], ['catdoc', file_path]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return ""


def extract_pptx_pages(file_path: str):
    """PPTX → list of pages (slide text + notes). Tables — handled separately."""
    pages = []
    try:
        from pptx import Presentation
        prs = Presentation(file_path)
        for slide in prs.slides:
            parts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    t = shape.text_frame.text.strip()
                    if t:
                        parts.append(t)
            if getattr(slide, 'has_notes_slide', False) and slide.notes_slide:
                nt = slide.notes_slide.notes_text_frame.text.strip()
                if nt:
                    parts.append(f"[Заметки]\n{nt}")
            pages.append("\n\n".join(parts))
    except Exception as e:
        logger.error(f"PPTX extraction error: {e}")
    return pages


def extract_epub_chapters(file_path: str):
    """EPUB → chapters as "pages" (zipfile + BeautifulSoup, no external dependencies)."""
    pages = []
    try:
        import zipfile
        from bs4 import BeautifulSoup
        with zipfile.ZipFile(file_path) as z:
            names = sorted(n for n in z.namelist()
                           if n.lower().endswith(('.xhtml', '.html', '.htm')))
            for n in names:
                soup = BeautifulSoup(z.read(n), 'html.parser')
                for tag in soup(["script", "style"]):
                    tag.decompose()
                t = soup.get_text(separator="\n").strip()
                if t:
                    pages.append(t)
    except Exception as e:
        logger.error(f"EPUB extraction error: {e}")
    return pages


def _html_tables_chunks(file_path: str) -> list:
    """HTML tables as table chunks."""
    extra = []
    try:
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        for i, tb in enumerate(soup.find_all('table')[:50], start=1):
            rows = []
            for tr in tb.find_all('tr')[:200]:
                cells = [c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'])]
                if cells:
                    rows.append('\t'.join(cells))
            if len(rows) >= 2:
                extra.append({'content': f'[HTML-таблица #{i}]\n' + '\n'.join(rows),
                              'strategy': 'table', 'metadata': {'is_table': True, 'source': 'html'}})
    except Exception as e:
        logger.debug(f"HTML tables: {e}")
    return extra


def _zip_media_chunks(file_path: str, document, media_dir_in_zip: str, kind: str) -> list:
    """Images from office containers (word/media, ppt/media) into media + placeholder chunks."""
    extra = []
    try:
        import zipfile
        img_rel = f'documents/{document.id}/images'
        img_dir = os.path.join(settings.MEDIA_ROOT, img_rel)
        os.makedirs(img_dir, exist_ok=True)
        n = 0
        with zipfile.ZipFile(file_path) as z:
            for name in z.namelist():
                if not name.startswith(media_dir_in_zip):
                    continue
                if not name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    continue
                fname = name.rsplit('/', 1)[-1]
                with open(os.path.join(img_dir, fname), 'wb') as out:
                    out.write(z.read(name))
                extra.append({'content': f'[{kind}: {fname}] сохранено: {img_rel}/{fname}',
                              'strategy': 'table',
                              'metadata': {'is_image': True,
                                           'image_path': f'{img_rel}/{fname}'}})
                n += 1
                if n >= 20:
                    break
    except Exception as e:
        logger.debug(f"{kind} media: {e}")
    return extra


def _docx_tables_chunks(file_path: str) -> list:
    """DOCX tables as table chunks."""
    extra = []
    try:
        from docx import Document as DocxDocument
        doc = DocxDocument(file_path)
        for i, tb in enumerate(doc.tables[:50], start=1):
            rows = []
            for tr in tb.rows[:200]:
                cells = [c.text.strip() for c in tr.cells]
                if any(cells):
                    rows.append('\t'.join(cells))
            if len(rows) >= 2:
                extra.append({'content': f'[DOCX-таблица #{i}]\n' + '\n'.join(rows),
                              'strategy': 'table', 'metadata': {'is_table': True, 'source': 'docx'}})
    except Exception as e:
        logger.debug(f"DOCX tables: {e}")
    return extra


def _pptx_tables_chunks(file_path: str) -> list:
    """PPTX tables as table chunks (with slide number)."""
    extra = []
    try:
        from pptx import Presentation
        prs = Presentation(file_path)
        for i, slide in enumerate(prs.slides, start=1):
            for shape in slide.shapes:
                if getattr(shape, 'has_table', False):
                    rows = []
                    for tr in shape.table.rows:  # rows does not support slicing
                        cells = [c.text.strip() for c in tr.cells]
                        if any(cells):
                            rows.append('\t'.join(cells))
                        if len(rows) >= 200:
                            break
                    if len(rows) >= 2:
                        extra.append({'content': f'[PPTX-таблица, слайд {i}]\n' + '\n'.join(rows),
                                      'strategy': 'table',
                                      'metadata': {'is_table': True, 'source': 'pptx', 'page': i}})
    except Exception as e:
        logger.debug(f"PPTX tables: {e}")
    return extra


def get_file_hash(file_path: str) -> str:
    """SHA256 hash of the file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# --------------------------------------------------------------------------
# Chunk creation (improved, sentence-aware)
# --------------------------------------------------------------------------
def create_chunks(
    document: Document,
    content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[DocumentChunk]:
    """
    Split text into chunks respecting sentence boundaries.
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
                # Start the new chunk with overlap (last sentences)
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
                # Append the current sentence
                current_chunk.append(sent)
                current_len += sent_len
            else:
                # If a sentence is longer than chunk_size, split by words
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
                            # Overlap for the next piece
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
                # after processing a long sentence, reset current_chunk
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
# process_document_task implementation
# --------------------------------------------------------------------------
def _process_document_impl(self, document_id: int) -> Dict[str, Any]:
    """
    Full document processing implementation.
    Called from process_document_task.
    """
    logger.info(f"🚀 Processing document {document_id}")

    # Redis lock to prevent parallel processing
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

        # Remove old chunks (soft delete)
        old_chunks = document.chunks.filter(is_active=True)
        old_count = old_chunks.count()
        stale_chunk_ids = []
        if old_count:
            stale_chunk_ids = list(old_chunks.values_list('id', flat=True))
            old_chunks.update(is_active=False)
            logger.info(f"Deactivated {old_count} old chunks")

        # Create the processing log
        processor = DocumentProcessor.objects.create(
            document=document,
            status='STARTED',
            progress=10
        )

        # 1. Get the file path and extract text
        file_path = document.full_file_path
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {document.original_filename}")

        ext = os.path.splitext(file_path)[1].lower()
        page_map = []       # v5.2: [(start, end, page_no)] to map chunks to pages
        extra_chunks = []   # v5.2: tables/images/CSV blocks as separate chunks

        if ext == '.pdf':
            pages = extract_pdf_pages(file_path)
            raw_text = "\n".join(pages)
            offset = 0
            for i, pt in enumerate(pages, start=1):
                if offset:
                    offset += 1  # \n separator
                page_map.append((offset, offset + len(pt), i))
                offset += len(pt)
            extra_chunks = _pdf_extra_chunks(file_path, document)
        elif ext == '.docx':
            raw_text = extract_text_from_docx(file_path)
            extra_chunks = _docx_tables_chunks(file_path) + \
                _zip_media_chunks(file_path, document, 'word/media/', 'DOCX-изображение')
        elif ext == '.doc':
            raw_text = extract_text_from_doc_legacy(file_path)
            if not raw_text.strip():
                raise ValueError(
                    "Бинарный .doc не распознан (в образе нет antiword/catdoc/soffice). "
                    "Пересохраните документ как DOCX или PDF — пайплайн поддерживает их полностью.")
        elif ext == '.rtf':
            raw_text = extract_text_from_rtf(file_path)
        elif ext == '.pptx':
            pages = extract_pptx_pages(file_path)
            raw_text = "\n\n".join(f'[Слайд {i}]\n{t}' for i, t in enumerate(pages, start=1) if t.strip())
            offset = 0
            for i, pt in enumerate(pages, start=1):
                if offset:
                    offset += 2  # \n\n
                page_map.append((offset, offset + len(pt), i))
                offset += len(pt)
            extra_chunks = _pptx_tables_chunks(file_path) + \
                _zip_media_chunks(file_path, document, 'ppt/media/', 'PPTX-изображение')
        elif ext == '.epub':
            pages = extract_epub_chapters(file_path)
            raw_text = "\n\n".join(pages)
            offset = 0
            for i, pt in enumerate(pages, start=1):
                if offset:
                    offset += 2
                page_map.append((offset, offset + len(pt), i))
                offset += len(pt)
        elif ext in ['.html', '.htm']:
            raw_text = extract_text_from_html(file_path)
            extra_chunks = _html_tables_chunks(file_path)
        elif ext == '.md':
            raw_text = extract_text_from_markdown(file_path)
        elif ext == '.csv':
            raw_text = extract_text_from_txt(file_path)
            extra_chunks = _tabular_chunks_csv(file_path)
        elif ext == '.xlsx':
            raw_text = ''
            extra_chunks = _tabular_chunks_xlsx(file_path)
        else:
            raw_text = extract_text_from_txt(file_path)

        for ec in extra_chunks:
            ec['content'] = ec['content'].replace('\x00', '')

        if not raw_text.strip() and not extra_chunks:
            raise ValueError("No extractable text found")

        raw_text = (raw_text or '').replace('\x00', '')  # strip NUL before any save
        document.raw_content = raw_text
        document.content_hash = get_file_hash(file_path)
        document.metadata['page_map'] = page_map
        document.save()

        processor.status = 'EXTRACTING'
        processor.progress = 30
        processor.save()

        # 2. Create chunks: v5.2 — settings come from the collection (document
        # metadata remains the override point)
        collection = document.collection
        chunk_size = document.metadata.get(
            'chunk_size', getattr(collection, 'chunk_size', 1000) or 1000)
        chunk_overlap = document.metadata.get(
            'chunk_overlap', getattr(collection, 'chunk_overlap', 200) or 200)
        strategy = document.metadata.get(
            'chunking_strategy', getattr(collection, 'chunking_strategy', '') or 'semantic')
        chunks = create_chunks_v2(document, raw_text, chunk_size, chunk_overlap,
                                  strategy=strategy, page_map=page_map,
                                  extra_chunks=extra_chunks)

        if not chunks:
            raise ValueError("No chunks created")

        document.processing_progress = 50
        document.save()

        processor.status = 'CHUNKING'
        processor.progress = 50
        processor.chunks_created = len(chunks)
        processor.save()

        # 3. Create embeddings using the collection's effective model
        embeddings_created = 0
        faiss_ok = False
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

                except Exception as e:
                    logger.error(f"Embedding error: {e}")
            else:
                logger.warning(f"No embedding model available for collection {document.collection.id}, skipping embeddings")
        else:
            logger.warning(f"Document {document_id} has no collection, skipping embeddings")

        # Index into FAISS (separate try: an indexing failure must not turn
        # the document into INDEXED — audit fix, the status used to reflect
        # embeddings only, even if vectors never made it into the index)
        if document.collection and embeddings_created > 0:
            try:
                vstore = get_vector_store_manager()
                added = vstore.index_chunks(chunks, document.collection.id)
                faiss_ok = added > 0
                logger.info(f"Indexed {added} vectors for collection {document.collection.id}")
                # Audit fix: vectors of deactivated chunks stayed in the index
                # forever (duplicates in search results) — remove them here
                if stale_chunk_ids:
                    removed = vstore.remove_stale_vectors(
                        document.collection.id, stale_chunk_ids)
                    if removed:
                        logger.info(f"Removed {removed} stale vectors for collection {document.collection.id}")
            except Exception as e:
                logger.error(f"FAISS indexing error: {e}")
                faiss_ok = False

        # 4. Finish
        with transaction.atomic():
            document = Document.objects.get(id=document_id)
            document.status = 'INDEXED' if (embeddings_created > 0 and faiss_ok) else 'CHUNKED'
            # embeddings exist but never made it into the index — flag for reindex
            if embeddings_created > 0 and not faiss_ok:
                document.requires_reindexing = True
            document.processing_progress = 100
            document.processed_at = timezone.now()
            if embeddings_created > 0 and faiss_ok:
                document.indexed_at = timezone.now()
            document.save()

            if document.collection:
                document.collection.update_statistics()

        processor.status = 'COMPLETED'
        processor.progress = 100
        processor.embeddings_created = embeddings_created
        processor.processing_time_ms = (timezone.now() - processor.started_at).total_seconds() * 1000
        processor.save()

        # Invalidate the BM25 cache for the collection
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
# Embedding creation for a collection
# --------------------------------------------------------------------------
def _create_embeddings_for_collection_impl(self, collection_id: int) -> Dict[str, Any]:
    """
    Create embeddings for all collection chunks that do not have them yet.
    Uses the collection's effective model.
    """
    from .models import DocumentChunk, EmbeddingModel

    logger.info(f"Creating embeddings for collection {collection_id}")

    try:
        collection = DocumentCollection.objects.get(id=collection_id)
        embedding_model = collection.get_effective_embedding_model()
        if not embedding_model:
            return {'success': False, 'error': 'No embedding model available for this collection'}

        # Chunks without embeddings
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

        # Index into FAISS
        # Audit fix: update_index silently returned 0 when the collection had
        # no index yet — embeddings existed but nothing was indexed
        if processed > 0:
            vstore = get_vector_store_manager()
            if collection.vector_index_created:
                vstore.update_index(collection_id)
            else:
                vstore.build_index(collection_id)

        logger.info(f"Created {processed} embeddings for collection {collection_id}")
        return {'success': True, 'processed': processed, 'total': total}

    except DocumentCollection.DoesNotExist:
        return {'success': False, 'error': f'Collection {collection_id} not found'}
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        return {'success': False, 'error': str(e)}


# --------------------------------------------------------------------------
# Full rebuild of embeddings and the FAISS index for a collection
# --------------------------------------------------------------------------
def _rebuild_embeddings_and_index_impl(self, collection_id: int) -> Dict[str, Any]:
    """
    Deletes all existing embeddings of the collection's chunks,
    recreates them (using the collection's effective model),
    and rebuilds the FAISS index.
    """
    logger.info(f"🔄 Rebuilding embeddings and index for collection {collection_id}")

    try:
        collection = DocumentCollection.objects.get(id=collection_id)
        embedding_model = collection.get_effective_embedding_model()
        if not embedding_model:
            return {'success': False, 'error': 'No embedding model available for this collection'}

        # Fetch all chunks of the collection (active)
        chunks = DocumentChunk.objects.filter(
            document__collection=collection,
            is_active=True
        ).exclude(content='')

        total = chunks.count()
        if total == 0:
            return {'success': False, 'error': 'No chunks in collection'}

        # Reset embeddings
        chunks.update(
            has_embedding=False,
            embedding_vector=None,
            embedding_model=None,
            embedding_dim=None
        )
        logger.info(f"Reset embeddings for {total} chunks")

        # Create embeddings in batches
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
                # Continue with the rest, do not abort

        logger.info(f"Created embeddings for {processed} chunks")

        # Rebuild the FAISS index
        if processed > 0:
            vstore = get_vector_store_manager()
            success = vstore.build_index(collection_id, force=True)
            if not success:
                logger.error("Failed to rebuild FAISS index")
        else:
            logger.warning("No embeddings created, skipping FAISS index rebuild")

        # Update collection statistics
        collection.update_statistics()

        # v5.2: sync document statuses — a document whose chunks are fully
        # covered by embeddings is marked INDEXED (otherwise it stays "Uploaded"
        # after the rebuild even though vectors are ready)
        for doc in collection.documents.all():
            active = doc.chunks.filter(is_active=True)
            total_c = active.count()
            embedded = active.filter(has_embedding=True).count()
            if total_c > 0 and embedded == total_c and doc.status != 'INDEXED':
                doc.status = 'INDEXED'
                doc.save(update_fields=['status'])
                logger.info(f"📄 Document {doc.id} → INDEXED ({embedded} chunks embedded)")

        logger.info(f"✅ Rebuilt embeddings and index for collection {collection_id}")
        return {'success': True, 'collection_id': collection_id, 'processed': processed, 'total': total}

    except DocumentCollection.DoesNotExist:
        return {'success': False, 'error': f'Collection {collection_id} not found'}
    except Exception as e:
        logger.error(f"Error rebuilding embeddings: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


# --------------------------------------------------------------------------
# Building the FAISS index for a collection
# --------------------------------------------------------------------------
def _build_faiss_index_impl(collection_id: int, index_type: str = 'flat') -> Dict[str, Any]:
    """
    Full rebuild of the FAISS index for a collection.
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
# Updating the FAISS index (adding new chunks)
# --------------------------------------------------------------------------
def _update_faiss_index_impl(collection_id: int) -> Dict[str, Any]:
    """
    Update the FAISS index: add new chunks.
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
# Regenerating document chunks
# --------------------------------------------------------------------------
def _regenerate_document_chunks_impl(self, document_id: int) -> Dict[str, Any]:
    """
    Recreate document chunks with new parameters.
    """
    logger.info(f"Regenerating chunks for document {document_id}")

    try:
        document = Document.objects.get(id=document_id, is_active=True)
        old_chunks = list(document.chunks.filter(is_active=True))

        # Get parameters from metadata
        chunk_size = document.metadata.get('chunk_size', 1000)
        chunk_overlap = document.metadata.get('chunk_overlap', 200)

        # Deactivate old chunks
        document.chunks.update(is_active=False)

        # Create new chunks
        content = document.processed_content or document.raw_content
        if not content:
            raise ValueError("Document has no content")

        new_chunks = create_chunks(document, content, chunk_size, chunk_overlap)

        # Create embeddings using the collection's effective model
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

        # Index into FAISS
        if embeddings_created > 0 and document.collection:
            vstore = get_vector_store_manager()
            vstore.index_chunks(new_chunks, document.collection.id)

        # Update the status
        document.status = 'INDEXED' if embeddings_created > 0 else 'CHUNKED'
        document.requires_reindexing = False
        document.save()

        # Invalidate the BM25 cache
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
# Celery tasks (wrappers)
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