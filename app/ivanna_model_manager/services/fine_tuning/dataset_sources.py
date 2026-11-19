"""Collect fine-tuning data from every source the create-form offers.

Audit fix (F1): the create view stores the selected dataset ids in
job.training_config (distillation_datasets / manual_datasets / hf_datasets /
document_collections), but the trainers only read the M2M relations — which
nothing populates — so every UI-created job failed with
"No instruction examples found". The M2M (EvaluationDataset) is still read
for backward compatibility with ORM-created jobs.
"""
import json
import logging
import os

from ...models import (
    DistillationDataset,
    DocumentChunk,
    DocumentCollection,
    HuggingFaceDataset,
    ManualInstructionSet,
)

logger = logging.getLogger(__name__)


def _pairs_from_questions(questions) -> list:
    """(instruction, answer) pairs from EvaluationDataset.questions format."""
    pairs = []
    for q in questions or []:
        if not isinstance(q, dict):
            continue
        if 'answer' in q:
            inst = q.get('text', '')
            ans = q['answer']
        elif 'output' in q:
            inst = q.get('instruction', q.get('text', ''))
            ans = q['output']
        else:
            continue
        inst, ans = (inst or '').strip(), (ans or '').strip()
        if inst and ans:
            pairs.append((inst, ans))
    return pairs


def _pairs_from_hf_jsonl(file_path) -> list:
    """(instruction, answer) pairs from a HuggingFaceDataset JSONL file."""
    pairs = []
    if not file_path or not os.path.exists(file_path):
        logger.warning(f"HF dataset file missing: {file_path}")
        return pairs
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            inst = (item.get('instruction') or item.get('text')
                    or item.get('question') or '')
            ans = (item.get('output') or item.get('answer')
                   or item.get('response') or '')
            inst, ans = inst.strip(), ans.strip()
            if inst and ans:
                pairs.append((inst, ans))
    return pairs


def collect_instruction_pairs(job) -> list:
    """All (instruction, answer) pairs selected on the create form."""
    cfg = job.training_config or {}
    pairs = []

    # 1. Legacy path: M2M EvaluationDatasets (ORM-created jobs)
    for ds in job.instruction_datasets.all():
        pairs.extend(_pairs_from_questions(ds.questions))

    # 2. Distillation datasets (items: question_text / answer_text)
    for ds in DistillationDataset.objects.filter(
            id__in=cfg.get('distillation_datasets') or []):
        for it in ds.items or []:
            if not isinstance(it, dict):
                continue
            q = (it.get('question_text') or '').strip()
            a = (it.get('answer_text') or '').strip()
            if q and a:
                pairs.append((q, a))

    # 3. Manual instruction sets
    for ms in ManualInstructionSet.objects.filter(
            id__in=cfg.get('manual_datasets') or []):
        for mi in ms.instructions.all():
            q, a = (mi.question or '').strip(), (mi.answer or '').strip()
            if q and a:
                pairs.append((q, a))

    # 4. HF datasets (JSONL written by import_huggingface_dataset_task)
    for hf in HuggingFaceDataset.objects.filter(
            id__in=cfg.get('hf_datasets') or []):
        pairs.extend(_pairs_from_hf_jsonl(hf.file_path))

    return pairs


def collect_document_collections(job) -> list:
    """Union of document collections selected on the form (training_config)
    and the legacy M2M relation."""
    cfg = job.training_config or {}
    ids = list(cfg.get('document_collections') or [])
    collections = list(DocumentCollection.objects.filter(
        id__in=ids, is_active=True))
    seen = {c.id for c in collections}
    for c in job.document_collections.all():
        if c.id not in seen:
            collections.append(c)
            seen.add(c.id)
    return collections


def collect_document_texts(job, limit_per_collection: int = 10000) -> list:
    """Chunk texts from all selected document collections."""
    texts = []
    for collection in collect_document_collections(job):
        chunks = DocumentChunk.objects.filter(
            document__collection=collection,
            is_active=True
        ).values_list('content', flat=True)[:limit_per_collection]
        texts.extend(chunks)
    return texts
