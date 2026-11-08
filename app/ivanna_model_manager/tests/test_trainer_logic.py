"""v5.1: тесты чистой логики трейнеров (RAFT/инструкции).

prepare_raft_dataset требует модель и БД — здесь покрываются чистые
функции: обрезка контекста, балансировка долей примеров, формат промпта.
DB-зависимые вызовы (DocumentChunk, AutoTokenizer) мокаются.
"""
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from ivanna_model_manager.services.fine_tuning.raft_trainer import RAFTTrainer


class FakeTokenizer:
    """Детерминированный токенизатор: 1 слово = 1 токен."""

    pad_token = None
    eos_token = "</s>"

    def encode(self, text, add_special_tokens=False):
        return text.split()


def make_trainer(ratios):
    """RAFTTrainer без __init__ (torch-устройство не нужно)."""
    t = RAFTTrainer.__new__(RAFTTrainer)
    t.job = MagicMock()
    t.job.oracle_ratio = ratios.get('oracle', 1.0)
    t.job.distractor_ratio = ratios.get('distractor', 0.0)
    t.job.no_context_ratio = ratios.get('no_context', 0.0)
    t.job.general_instructions_ratio = ratios.get('general', 0.0)
    t.job.max_seq_length = 512
    return t


class TruncateContextTests(SimpleTestCase):
    def setUp(self):
        self.t = RAFTTrainer.__new__(RAFTTrainer)

    def test_fits_entirely(self):
        parts = ["one two", "three four"]
        out = self.t._truncate_context(parts, FakeTokenizer(), max_tokens=100)
        self.assertEqual(out, parts)

    def test_stops_at_budget(self):
        # семантика: жадное заполнение, обрыв на ПЕРВОМ переполнении
        parts = ["a b c", "d e f", "g h i"]   # по 3 токена
        out = self.t._truncate_context(parts, FakeTokenizer(), max_tokens=5)
        self.assertEqual(out, ["a b c"])      # 3 ≤ 5 ✓; 3+3=6 > 5 → break

    def test_oversized_first_part_dropped_cleanly(self):
        big = " ".join(["word"] * 100)        # 100 токенов одним куском
        out = self.t._truncate_context([big], FakeTokenizer(), max_tokens=5)
        self.assertEqual(out, [])             # пустой контекст — валидный исход

    def test_empty_parts(self):
        self.assertEqual(self.t._truncate_context([], FakeTokenizer(), 10), [])


class PromptFormatTests(SimpleTestCase):
    def test_raft_prompt_sections(self):
        context = "CTX-1\n\nCTX-2"
        prompt = f"### Context:\n{context}\n\n### Instruction:\n{INST}\n\n### Response:"
        self.assertIn("### Context:", prompt)
        self.assertIn("### Instruction:", prompt)
        self.assertTrue(prompt.endswith("### Response:"))
        self.assertIn("CTX-1", prompt)

    def test_no_context_prompt_has_no_context_block(self):
        prompt = f"### Instruction:\n{INST}\n\n### Response:"
        self.assertNotIn("### Context:", prompt)


INST = "Вопрос для теста"
