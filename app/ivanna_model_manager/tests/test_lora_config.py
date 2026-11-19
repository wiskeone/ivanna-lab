"""v5.3 Tier-1: build_lora_config / resolve_target_modules / apply_lora_plus tests."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from django.test import SimpleTestCase

from ivanna_model_manager.services.fine_tuning.lora import (
    build_lora_config, resolve_target_modules, TARGET_PRESETS)


def _job(model_path='/app/media/model_cache/qwen3_5-0_6b-pilot', **kw):
    base = dict(lora_r=16, lora_alpha=32, lora_dropout=0.05,
                use_dora=False, use_rslora=False,
                target_modules=[], target_preset='hybrid-full')
    base.update(kw)
    return SimpleNamespace(model_path=model_path, **base)


class ResolveTargetsTests(SimpleTestCase):
    def test_explicit_target_modules_win(self):
        job = _job(target_modules=['q_proj'])
        self.assertEqual(resolve_target_modules(job, '/x'), ['q_proj'])

    def test_hybrid_preset(self):
        targets = resolve_target_modules(_job(), '/x')
        self.assertIn('in_proj_qkvz', targets)
        self.assertIn('down_proj', targets)
        self.assertNotIn('conv1d', targets)  # LoRA-Conv1d unsupported

    def test_attention_preset(self):
        targets = resolve_target_modules(_job(target_preset='attention'), '/x')
        self.assertEqual(targets, ['q_proj', 'k_proj', 'v_proj', 'o_proj'])

    def test_non_hybrid_fallback(self):
        import os, tempfile, json
        with tempfile.TemporaryDirectory() as d:
            json.dump({'model_type': 'llama'}, open(os.path.join(d, 'config.json'), 'w'))
            targets = resolve_target_modules(_job(model_path=d, target_preset=''), d)
        self.assertEqual(targets, TARGET_PRESETS['attention+mlp'])


class BuildLoraConfigTests(SimpleTestCase):
    def test_flags_propagate(self):
        # peft import guard: the CPU test container may lack peft — skip
        try:
            from peft import LoraConfig  # noqa
        except ImportError:
            self.skipTest('peft not installed')
        cfg = build_lora_config(_job(use_dora=True, use_rslora=True), '/x')
        self.assertTrue(cfg.use_dora)
        self.assertTrue(cfg.use_rslora)
        self.assertEqual(cfg.r, 16)


class LoraPlusTests(SimpleTestCase):
    def test_apply_lora_plus_builds_param_groups(self):
        import torch
        from ivanna_model_manager.services.fine_tuning.lora import apply_lora_plus

        a1 = torch.nn.Parameter(torch.zeros(2, 2))
        b1 = torch.nn.Parameter(torch.zeros(2, 2))
        model = MagicMock()
        model.named_parameters.return_value = [
            ('base.layers.0.self_attn.q_proj.lora_A.default.weight', a1),
            ('base.layers.0.self_attn.q_proj.lora_B.default.weight', b1),
            ('base.layers.0.embed_tokens.weight', torch.nn.Parameter(torch.zeros(1))),
        ]
        trainer = MagicMock()
        trainer.model = model
        trainer.args = SimpleNamespace(learning_rate=1e-4, adam_beta1=0.9,
                                       adam_beta2=0.999, weight_decay=0.0)
        opt = apply_lora_plus(trainer, 16)
        groups = opt.param_groups
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]['lr'], 1e-4)         # A group
        self.assertEqual(groups[1]['lr'], 1.6e-3)        # B group = lr*16
        self.assertIs(trainer.optimizer, opt)

    def test_ratio_leq_one_is_noop(self):
        from ivanna_model_manager.services.fine_tuning.lora import apply_lora_plus
        trainer = MagicMock()
        trainer.args = SimpleNamespace(learning_rate=1e-4)
        self.assertIsNone(apply_lora_plus(trainer, 1.0))
