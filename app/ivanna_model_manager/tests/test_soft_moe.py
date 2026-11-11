"""v5.2: тесты мягкого MoE (порт weight_lab, чистая логика — без GPU).

Математика патча проверяется на миниатюрных safetensors-файлах: неверный
патч = испорченная модель, поэтому формулы steer/amplify закреплены тестами.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from django.test import SimpleTestCase

try:
    import torch  # нужен только для PatchMathTests
    # hasattr: заглушки-стабы (локальный запуск без GPU-стека) не считаем настоящим torch
    TORCH_OK = hasattr(torch, "randn") and hasattr(torch, "outer")
except ImportError:
    TORCH_OK = False

from ivanna_model_manager.services.soft_moe import patching as moe_patching
from ivanna_model_manager.services.soft_moe import prompts as moe_prompts
from ivanna_model_manager.services.soft_moe import router as moe_router
from ivanna_model_manager.services.soft_moe import scan as moe_scan


class AucTests(SimpleTestCase):
    def test_perfect_separation(self):
        self.assertEqual(moe_scan.auc(np.array([1., 2., 3.]), np.array([-1., 0., 0.5])), 1.0)

    def test_inverted(self):
        self.assertEqual(moe_scan.auc(np.array([-1., 0.]), np.array([1., 2.])), 0.0)

    def test_identical_is_half(self):
        self.assertEqual(moe_scan.auc(np.array([1., 1.]), np.array([1., 1.])), 0.5)


class HoldoutTests(SimpleTestCase):
    def test_sizes_and_determinism(self):
        a = moe_scan.split_holdout(30, 30, seed=7, holdout_fraction=0.33)
        b = moe_scan.split_holdout(30, 30, seed=7, holdout_fraction=0.33)
        # сид → воспроизводимость (как в лабе); сравнение поэлементно (np-массивы)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(a, b)))
        tr_d, tr_c, te_d, te_c = a
        self.assertEqual(len(tr_d) + len(te_d), 30)
        self.assertEqual(len(tr_c) + len(te_c), 30)
        self.assertEqual(len(set(tr_d.tolist()) & set(te_d.tolist())), 0)  # hold-out без пересечений

    def test_tiny_sets_keep_one_test_sample(self):
        _, _, te_d, te_c = moe_scan.split_holdout(3, 3, seed=1, holdout_fraction=0.33)
        self.assertGreaterEqual(len(te_d), 1)
        self.assertGreaterEqual(len(te_c), 1)


class AutoLayersTests(SimpleTestCase):
    def _report(self, writers):
        return {"writers": {f"L{w['layer']:02d}:{w['module']}": w for w in writers}}

    def test_threshold_rule(self):
        # топ-8 медиана = 2.0, порог 0.6·2 = 1.2 → слои с суммой |proj| ≥ 1.2
        writers = [{"layer": l, "module": "mlp.down_proj", "proj": p, "norm": abs(p)}
                   for l, p in [(23, 4.0), (22, 2.0), (21, 2.0), (19, 2.0), (20, 2.0),
                                (18, 2.0), (17, 2.0), (16, 2.0), (10, 1.0), (5, 0.1)]]
        layers = moe_scan.auto_layers_from_writers(self._report(writers))
        self.assertIn(23, layers)
        self.assertNotIn(10, layers)
        self.assertNotIn(5, layers)

    def test_empty_report(self):
        self.assertEqual(moe_scan.auto_layers_from_writers({"writers": {}}), [])


class MatchOutTensorTests(SimpleTestCase):
    def test_llama_style_names(self):
        self.assertEqual(moe_patching.match_out_tensor("model.layers.23.mlp.down_proj.weight"),
                         (23, "mlp.down_proj"))
        self.assertEqual(moe_patching.match_out_tensor("model.layers.5.self_attn.o_proj.weight"),
                         (5, "self_attn.o_proj"))

    def test_non_out_tensors_ignored(self):
        self.assertIsNone(moe_patching.match_out_tensor("model.layers.3.self_attn.q_proj.weight"))
        self.assertIsNone(moe_patching.match_out_tensor("model.embed_tokens.weight"))
        self.assertIsNone(moe_patching.match_out_tensor("lm_head.weight"))


@unittest.skipUnless(TORCH_OK, "torch недоступен — полный прогон в CPU-образе CI")
class PatchMathTests(SimpleTestCase):
    """Формулы steer/amplify на миниатюрных safetensors-файлах."""

    HID, FFN = 8, 16
    LAYER0 = "model.layers.0.mlp.down_proj.weight"
    OTHER = "model.layers.2.mlp.gate_proj.weight"

    def _make_fixture(self, tmp):
        import torch
        from safetensors.torch import save_file

        src = os.path.join(tmp, "src")
        scan_dir = os.path.join(tmp, "scan")
        os.makedirs(src, exist_ok=True)
        os.makedirs(scan_dir, exist_ok=True)

        tensors = {
            self.LAYER0: torch.randn(self.FFN, self.HID),
            "model.layers.1.self_attn.o_proj.weight": torch.randn(self.HID, self.HID),
            self.OTHER: torch.randn(self.HID, self.FFN),
        }
        save_file(tensors, os.path.join(src, "model.safetensors"),
                  metadata={"format": "pt"})
        with open(os.path.join(src, "config.json"), "w") as f:
            json.dump({"model_type": "test"}, f)

        rng = np.random.default_rng(0)
        d_last = np.zeros((4, self.HID), dtype=np.float32)
        d_last[1] = rng.normal(size=self.HID)          # d̂ для слоя 0 (конвенция L+1)
        i_bar = rng.normal(size=self.FFN).astype(np.float32)  # mean-вход слоя 0
        npz = os.path.join(scan_dir, "scan_dirs.npz")
        np.savez_compressed(npz, d_last=d_last, n_layers=3, **{"mi|0|mlp.down_proj": i_bar})

        report = {"writers": {"L00:mlp.down_proj":
                              {"layer": 0, "module": "mlp.down_proj",
                               "proj": 3.0, "norm": 10.0}}}
        return src, npz, report, tensors, d_last, i_bar

    def _run(self, mode):
        import torch
        from safetensors.torch import load_file

        tmp = tempfile.mkdtemp()
        src, npz, report, tensors, d_last, i_bar = self._make_fixture(tmp)
        out = os.path.join(tmp, "out")
        moe_patching.patch_model_dir(src, out, npz, report, mode=mode, alpha=0.2,
                                     layers="auto")
        with open(os.path.join(out, "config.json")) as f:
            self.assertEqual(json.load(f), {"model_type": "test"})
        patched = load_file(os.path.join(out, "model.safetensors"))
        d = torch.from_numpy(d_last[1] / np.linalg.norm(d_last[1]))
        return tensors, patched, d, i_bar

    def test_steer_formula(self):
        import torch
        tensors, patched, d, i_bar = self._run("steer")
        w0 = tensors[self.LAYER0]
        expected = (w0.to(torch.float32)
                    + 0.2 * torch.outer(d, torch.from_numpy(i_bar)) / float(np.linalg.norm(i_bar) ** 2))
        torch.testing.assert_close(patched[self.LAYER0], expected.to(w0.dtype),
                                   rtol=1e-4, atol=1e-4)
        # out-матрица слоя 1 не входит в auto-выбор (писатель только L0) → не изменена
        self.assertTrue(torch.equal(
            patched["model.layers.1.self_attn.o_proj.weight"],
            tensors["model.layers.1.self_attn.o_proj.weight"]))
        # не-out-тензор скопирован байт-в-байт
        self.assertTrue(torch.equal(patched[self.OTHER], tensors[self.OTHER]))

    def test_amplify_formula(self):
        import torch
        tensors, patched, d, _ = self._run("amplify")
        w0 = tensors[self.LAYER0]
        expected = w0.to(torch.float32) + 0.2 * torch.outer(d, d @ w0.to(torch.float32))
        torch.testing.assert_close(patched[self.LAYER0], expected.to(w0.dtype),
                                   rtol=1e-4, atol=1e-4)

    def test_dtype_preserved(self):
        import torch
        from safetensors.torch import load_file, save_file

        tmp = tempfile.mkdtemp()
        src, npz, report, _, _, _ = self._make_fixture(tmp)
        # перезаписываем слой 0 в float16 — патч должен сохранить dtype
        st = os.path.join(src, "model.safetensors")
        t = load_file(st)
        t[self.LAYER0] = t[self.LAYER0].to(torch.float16)
        save_file(t, st, metadata={"format": "pt"})
        out = os.path.join(tmp, "out")
        moe_patching.patch_model_dir(src, out, npz, report, mode="steer", alpha=0.2,
                                     layers="auto")
        patched = load_file(os.path.join(out, "model.safetensors"))
        self.assertEqual(patched[self.LAYER0].dtype, torch.float16)


class PromptSetTests(SimpleTestCase):
    def test_clean_dedup_and_length_filter(self):
        texts = ["Жёсткая посадка ВС из-за обледенения, экипаж действовал по РЛЭ.",
                 "жёсткая посадка вс из-за обледенения, экипаж действовал по рлэ.",  # дедуп по 80 символам
                 "коротко",                                    # < min_len
                 "Корректный промпт про инструмент и смазку в ангаре."]
        out = moe_prompts.clean_prompt_set(texts, min_len=10, max_len=200)
        self.assertEqual(len(out), 2)
        self.assertIn("Корректный промпт", out[1])

    def test_html_cleaning(self):
        raw = ["<p>Описание&nbsp;нарушения</p> с деталями инцидента на борту."]
        out = moe_prompts.clean_prompt_set(raw, min_len=5)
        self.assertEqual(out, ["Описание нарушения с деталями инцидента на борту."])

    def test_save_load_roundtrip(self):
        import tempfile
        texts = ["промпт раз", "промпт два"]
        path = os.path.join(tempfile.mkdtemp(), "set.txt")
        n = moe_prompts.save_prompt_set(path, texts)
        self.assertEqual(n, 2)
        self.assertEqual(moe_prompts.load_texts(path), texts)
        self.assertEqual(moe_prompts.load_texts("/nonexistent/xx.txt"), [])

    def test_balance_warning(self):
        w = moe_prompts.balance_warning({"n": 30, "median": 400},
                                        {"n": 30, "median": 50})
        self.assertIn("Медианы длин расходятся", w)
        self.assertEqual(moe_prompts.balance_warning({"n": 30, "median": 400},
                                                     {"n": 30, "median": 380}), "")


class RouterRuleTests(SimpleTestCase):
    """Правило margin на подменённом embedding-сервисе (без загрузки моделей)."""

    class _FakeEmbedder:
        def __init__(self, dim=4):
            self.dim = dim

        def _emb(self, text):
            v = np.zeros(self.dim, dtype=np.float32)
            v[0 if "домен" in text else 1] = 1.0
            return v

        def create_embedding(self, text, model_id=None, normalize=True):
            return self._emb(text)

        def create_embeddings_batch(self, texts, model_id=None, normalize=True, batch_size=32):
            return np.stack([self._emb(t) for t in texts])

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patcher = patch.object(moe_router, "embedding_service",
                                    self._FakeEmbedder())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_build_and_route(self):
        domain = [f"домен промпт {i}" for i in range(5)]
        contrast = [f"быт промпт {i}" for i in range(5)]
        moe_router.build_centroids(self.tmp, domain, contrast)
        C_d, C_c = moe_router.load_centroids(self.tmp)
        self.assertAlmostEqual(float(np.linalg.norm(C_d)), 1.0, places=5)

        r = moe_router.route_text("домен вопрос", C_d, C_c, threshold=0.08)
        self.assertEqual(r["route"], "expert")
        self.assertGreater(r["margin"], 0.08)
        r2 = moe_router.route_text("быт вопрос", C_d, C_c, threshold=0.08)
        self.assertEqual(r2["route"], "general")

    def test_test_router_accuracy_gate(self):
        domain = [f"домен {i}" for i in range(6)]
        contrast = [f"быт {i}" for i in range(6)]
        moe_router.build_centroids(self.tmp, domain, contrast)
        report = moe_router.test_router(
            self.tmp, [f"домен eval {i}" for i in range(4)],
            [f"быт eval {i}" for i in range(4)], threshold=0.08,
            out_json=os.path.join(self.tmp, "router_test.json"))
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["gate"], "PASS")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "router_test.json")))
