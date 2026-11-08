"""v5.0: тесты валидатора/нормализатора пресетов генерации (чистая логика).

Запускаются локально со стабами тяжёлых модулей и в CI внутри CPU-образа.
"""
from django.test import SimpleTestCase

from ivanna_model_manager.services.generation_presets import (
    apply_preset_to_config, normalize_preset_config,
)


class NormalizeConfigTests(SimpleTestCase):
    def test_valid_config_passes(self):
        cfg, errors = normalize_preset_config({
            'temperature': 0.3, 'max_tokens': 1024, 'top_p': 0.9,
        })
        self.assertEqual(errors, [])
        self.assertEqual(cfg['temperature'], 0.3)
        self.assertEqual(cfg['max_tokens'], 1024)
        self.assertEqual(cfg['top_p'], 0.9)

    def test_unknown_keys_dropped(self):
        cfg, errors = normalize_preset_config({'temperature': 0.5, 'quantization': 'int4'})
        self.assertNotIn('quantization', cfg)
        self.assertEqual(len(errors), 1)
        self.assertIn('unknown parameter', errors[0])

    def test_out_of_range_clamped(self):
        cfg, errors = normalize_preset_config({'temperature': 5.0, 'max_tokens': -10})
        self.assertEqual(cfg['temperature'], 2.0)   # clamp к верхней границе
        self.assertEqual(cfg['max_tokens'], 1)      # clamp к нижней
        self.assertEqual(len(errors), 2)

    def test_non_numeric_dropped(self):
        cfg, errors = normalize_preset_config({'temperature': 'hot'})
        # нечисловое отброшено → подставлен дефолт
        self.assertEqual(cfg['temperature'], 0.7)
        self.assertTrue(any('not a number' in e for e in errors))

    def test_non_dict_returns_defaults(self):
        cfg, errors = normalize_preset_config('surprise')
        self.assertEqual(errors, ['config must be an object'])
        self.assertIn('temperature', cfg)

    def test_defaults_filled(self):
        cfg, _ = normalize_preset_config({'top_p': 0.5})
        self.assertEqual(cfg['temperature'], 0.7)   # дефолт
        self.assertEqual(cfg['max_tokens'], 512)
        self.assertEqual(cfg['top_p'], 0.5)         # заданное не перебивается


class ApplyPresetTests(SimpleTestCase):
    def test_preset_overrides_generation_config(self):
        session = {'generation_config': {'temperature': 0.9, 'top_k': 40},
                   'model_id': 5}
        out = apply_preset_to_config(session, {'temperature': 0.2, 'top_p': 0.6})
        gen = out['generation_config']
        self.assertEqual(gen['temperature'], 0.2)   # пресет приоритетен
        self.assertEqual(gen['top_p'], 0.6)
        self.assertEqual(gen['top_k'], 40)          # не упомянут в пресете — сохранён
        # исходная сессия не мутирует
        self.assertEqual(session['generation_config']['temperature'], 0.9)

    def test_apply_to_empty_session(self):
        out = apply_preset_to_config(None, {'temperature': 0.1})
        self.assertEqual(out['generation_config']['temperature'], 0.1)

    def test_other_session_keys_preserved(self):
        out = apply_preset_to_config({'rag_config': {'enabled': True}, 'x': 1},
                                     {'temperature': 0.4})
        self.assertEqual(out['rag_config'], {'enabled': True})
        self.assertEqual(out['x'], 1)
        self.assertIn('generation_config', out)
