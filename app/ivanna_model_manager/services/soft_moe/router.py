"""Роутер тем (порт weight_lab/router.py, Этап 4).

Центроиды тем = mean embedding scan-наборов (НЕ eval — честная проверка).
Правило: cos(тема) − cos(контраст) > threshold → эксперт темы, иначе general.
Порог по умолчанию 0.08 (в лабе 20/20 = 100%). Гейт теста: точность ≥ 90%.
"""
import json
from pathlib import Path

import numpy as np

from ..embedding_service import embedding_service

ROUTER_TEST_GATE = 0.90


def _centroid(model_id, texts) -> np.ndarray:
    E = embedding_service.create_embeddings_batch(list(texts), model_id=model_id,
                                                  normalize=True)
    C = E.mean(0)
    n = np.linalg.norm(C)
    return C / n if n > 0 else C


def build_centroids(theme_dir_path, domain_texts, contrast_texts,
                    embedding_model_id=None) -> dict:
    """Центроиды темы → centroids_domain.npy / centroids_contrast.npy + отчёт."""
    d = Path(theme_dir_path)
    d.mkdir(parents=True, exist_ok=True)
    C_d = _centroid(embedding_model_id, domain_texts)
    C_c = _centroid(embedding_model_id, contrast_texts)
    np.save(d / "centroids_domain.npy", C_d)
    np.save(d / "centroids_contrast.npy", C_c)
    return {
        "centroids_domain": str(d / "centroids_domain.npy"),
        "centroids_contrast": str(d / "centroids_contrast.npy"),
        "embedding_model_id": embedding_model_id,
        "n_domain": len(domain_texts), "n_contrast": len(contrast_texts),
    }


def load_centroids(theme_dir_path):
    d = Path(theme_dir_path)
    C_d = np.load(d / "centroids_domain.npy")
    C_c = np.load(d / "centroids_contrast.npy")
    return C_d, C_c


def route_text(text, centroids_domain, centroids_contrast,
               threshold: float = 0.08, embedding_model_id=None) -> dict:
    """Маршрут одного запроса: expert|general + косинусы + margin."""
    e = embedding_service.create_embedding(text, model_id=embedding_model_id,
                                           normalize=True)
    cos_d = float(e @ centroids_domain)
    cos_c = float(e @ centroids_contrast)
    margin = cos_d - cos_c
    return {
        "route": "expert" if margin > threshold else "general",
        "cos_domain": round(cos_d, 3),
        "cos_contrast": round(cos_c, 3),
        "margin": round(margin, 3),
        "threshold": threshold,
    }


def test_router(theme_dir_path, eval_domain_texts, eval_contrast_texts,
                threshold: float = 0.08, embedding_model_id=None,
                out_json=None) -> dict:
    """Точность роутера на свежих eval-наборах (гейт ≥ 90%)."""
    C_d, C_c = load_centroids(theme_dir_path)
    cases = [(t, "expert") for t in eval_domain_texts] + \
            [(t, "general") for t in eval_contrast_texts]
    ok, rows = 0, []
    for t, want in cases:
        r = route_text(t, C_d, C_c, threshold, embedding_model_id)
        got = r["route"]
        ok += (got == want)
        rows.append({"want": want, "got": got, "margin": r["margin"],
                     "text": t[:80]})
    accuracy = ok / len(cases) if cases else 0.0
    report = {
        "accuracy": round(accuracy, 3),
        "n_cases": len(cases),
        "gate": "PASS" if accuracy >= ROUTER_TEST_GATE else "FAIL",
        "gate_required": ROUTER_TEST_GATE,
        "threshold": threshold,
        "misses": [r for r in rows if r["got"] != r["want"]],
    }
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"accuracy": report["accuracy"], "threshold": threshold,
                       "rows": rows}, f, ensure_ascii=False, indent=1)
    return report
