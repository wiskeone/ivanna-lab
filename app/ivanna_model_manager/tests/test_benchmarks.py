"""v5.3: benchmarks tests — pure logic, no GPU and no lm-eval import needed.

Covers: runner command building, summary normalization, best-score picking,
curated catalog sync (install state preserved).
"""
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase

from ivanna_model_manager.models import Benchmark, BenchmarkRun, ModelCheckpoint
from ivanna_model_manager.tasks_benchmarks import (
    _build_lm_eval_command, _json_safe, _prepare_quantized_model,)
from ivanna_model_manager.services.benchmarks.catalog import curated_entries, CURATED_BENCHMARKS


class JsonSafeTests(SimpleTestCase):
    def test_nan_and_inf_become_none(self):
        data = {'a': float('nan'), 'b': [1.0, float('inf')], 'c': 'x'}
        out = _json_safe(data)
        self.assertIsNone(out['a'])
        self.assertIsNone(out['b'][1])
        self.assertEqual(out['b'][0], 1.0)
        self.assertEqual(out['c'], 'x')


class CommandBuildingTests(SimpleTestCase):
    def _run(self, quantized=False, batch_size='auto', num_fewshot=None, limit=None):
        run = SimpleNamespace(
            model=SimpleNamespace(local_path='/app/media/model_cache/qwen'),
            quantized=quantized, batch_size=batch_size,
            num_fewshot=num_fewshot, limit=limit)
        return run


    def test_basic_command(self):
        run = self._run()
        cmd = _build_lm_eval_command(run, ['gsm8k'], '/tmp/out')
        self.assertIn('-m', cmd)
        idx = cmd.index('lm_eval')
        self.assertEqual(cmd[idx], 'lm_eval')
        self.assertIn('--model', cmd)
        self.assertEqual(cmd[cmd.index('--tasks') + 1], 'gsm8k')
        model_args = cmd[cmd.index('--model_args') + 1]
        self.assertIn('pretrained=/app/media/model_cache/qwen', model_args)
        self.assertIn('trust_remote_code=True', model_args)
        self.assertNotIn('load_in_4bit', model_args)

    def test_quantized_command_uses_injected_path(self):
        # quantization moved out of CLI args: the runner gets a model dir
        # whose config.json carries the bnb config (_prepare_quantized_model)
        run = self._run(quantized=True, batch_size='8', num_fewshot=5, limit=100)
        cmd = _build_lm_eval_command(run, ['mmlu', 'gsm8k'], '/tmp/out',
                                     model_path='/tmp/out/model_quantized')
        model_args = cmd[cmd.index('--model_args') + 1]
        self.assertIn('pretrained=/tmp/out/model_quantized', model_args)
        self.assertNotIn('load_in_4bit', model_args)
        self.assertNotIn('quantization=', model_args)
        self.assertEqual(cmd[cmd.index('--tasks') + 1], 'mmlu,gsm8k')
        self.assertEqual(cmd[cmd.index('--num_fewshot') + 1], '5')
        self.assertEqual(cmd[cmd.index('--limit') + 1], '100')

    def test_prepare_quantized_model_injects_config(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            with open(os.path.join(src, 'config.json'), 'w') as fh:
                json.dump({'model_type': 'qwen3_5', 'vocab_size': 100}, fh)
            with open(os.path.join(src, 'model.safetensors'), 'wb') as fh:
                fh.write(b'x')
            dst = _prepare_quantized_model(src, out)
            cfg = json.load(open(os.path.join(dst, 'config.json')))
            self.assertEqual(cfg['quantization_config']['quant_method'], 'bitsandbytes')
            self.assertEqual(cfg['quantization_config']['bnb_4bit_quant_type'], 'nf4')
            # weights are symlinked, not copied
            self.assertTrue(os.path.islink(os.path.join(dst, 'model.safetensors')))


class SummaryNormalizationTests(TestCase):
    def _make_run(self, summary):
        import uuid

        from django.contrib.auth.models import User

        user = User.objects.filter(username='benchuser').first() or \
            User.objects.create_user(username='benchuser', password='x')
        model = ModelCheckpoint.objects.create(
            name=f'bench-model-{uuid.uuid4().hex[:8]}', model_type='CUSTOM')
        bench = Benchmark.objects.create(name='gsm8k', source='curated')
        run = BenchmarkRun.objects.create(model=model, created_by=user.username)
        run.benchmarks.set([bench])
        run.summary = summary
        return run

    def test_best_scores_prefers_acc_norm(self):
        run = self._make_run({'hellaswag': {'acc,none': 0.5, 'acc_norm,none': 0.72}})
        self.assertEqual(run.best_scores(), {'hellaswag': 0.72})

    def test_best_scores_fallback_exact_match(self):
        run = self._make_run({'gsm8k': {'exact_match,strict-match': 0.41}})
        self.assertEqual(run.best_scores(), {'gsm8k': 0.41})

    def test_best_scores_excludes_sample_len(self):
        # humaneval summary: sample_len is bookkeeping, must not win as the score
        run = self._make_run({'humaneval': {'sample_len': 164, 'pass@1,create_test': 0.2256}})
        self.assertEqual(run.best_scores(), {'humaneval': 0.2256})

    def test_best_scores_ignores_non_numeric(self):
        run = self._make_run({'ifeval': {'inst_level_loose_acc': 0.3, 'note': 'text'}})
        self.assertEqual(run.best_scores(), {'ifeval': 0.3})

    def test_empty_summary(self):
        run = self._make_run({})
        self.assertEqual(run.best_scores(), {})


class CuratedCatalogSyncTests(TestCase):
    def test_sync_creates_and_preserves_install_state(self):
        # emulate one sync
        first = {k: v for k, v in CURATED_BENCHMARKS[0].items() if k != 'popular'}
        bench, created = Benchmark.objects.get_or_create(
            name=first['name'], defaults={**first, 'source': 'curated'})
        bench.install_status = 'completed'
        bench.save(update_fields=['install_status'])

        from ivanna_model_manager.tasks_benchmarks import _sync_curated
        count = _sync_curated()
        self.assertGreaterEqual(count, len(CURATED_BENCHMARKS))
        bench.refresh_from_db()
        # re-sync must NOT reset the install state
        self.assertEqual(bench.install_status, 'completed')

    def test_curated_entries_have_unique_names(self):
        names = [e['name'] for e in curated_entries()]
        self.assertEqual(len(names), len(set(names)))
