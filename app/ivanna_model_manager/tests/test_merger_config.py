"""v4: тесты генератора mergekit-конфигурации (чистая логика, без GPU).

Регресс-щит для YAML, который скармливается mergekit-yaml: неверный
конфиг = часы впустую на фоне.
"""
from django.test import SimpleTestCase
import yaml
from ivanna_model_manager.services.merger import ModelMerger


class MergeConfigTests(SimpleTestCase):
    def setUp(self):
        self.m = ModelMerger()

    def _parse(self, **kw):
        return yaml.safe_load(self.m._build_merge_config(**kw))

    def test_linear_has_two_models_with_weights(self):
        cfg = self._parse(base_path='/m/base', merge_path='/m/other',
                          method='linear', ratio=0.3, density=0.7)
        self.assertEqual(cfg['merge_method'], 'linear')
        self.assertEqual(cfg['base_model'], '/m/base')
        weights = [m['parameters']['weight'] for m in cfg['models']]
        self.assertAlmostEqual(weights[0], 0.7)   # 1 - ratio
        self.assertAlmostEqual(weights[1], 0.3)   # ratio

    def test_ties_adds_density_and_mechanism(self):
        cfg = self._parse(base_path='/m/base', merge_path='/m/other',
                          method='ties', ratio=0.5, density=0.8)
        self.assertEqual(cfg['parameters']['density'], 0.8)
        self.assertTrue(cfg['parameters']['normalize'])
        self.assertEqual(cfg['parameters']['merge_mechanism'], 'ties')

    def test_dare_has_density_without_ties_mechanism(self):
        cfg = self._parse(base_path='/m/base', merge_path='/m/other',
                          method='dare', ratio=0.5, density=0.6)
        self.assertEqual(cfg['parameters']['density'], 0.6)
        self.assertNotIn('merge_mechanism', cfg['parameters'])

    def test_kwargs_are_merged_without_shadowing_core_keys(self):
        cfg = self._parse(base_path='/m/base', merge_path='/m/other',
                          method='linear', ratio=0.5, density=0.7,
                          dtype='float16', custom_flag=True)
        self.assertEqual(cfg['dtype'], 'float16')  # kwargs не перебили dtype
        self.assertTrue(cfg['custom_flag'])

    def test_validate_requires_core_fields(self):
        self.assertFalse(self.m.validate_merge_config({'base_model': 'x'}))
        self.assertTrue(self.m.validate_merge_config(
            {'base_model': 'b', 'merge_model': 'm', 'output_path': 'o'}))

    def test_resource_estimate_scales_with_size(self):
        r = self.m.estimate_merge_resources(10)
        self.assertAlmostEqual(r['estimated_memory_gb'], 25.0)
        self.assertAlmostEqual(r['estimated_disk_gb'], 12.0)
