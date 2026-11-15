"""Prompt set quality gate (port of weight_lab/themes/check_bge.py, Stage 0).

Validation via embeddings (bge-m3 in the lab; here — any EmbeddingModel from
the platform registry):
  1) cross cos(domain, contrast) mean < 0.5 — sets do not overlap lexically;
  2) intra cos(domain) > intra cos(contrast) — domain is denser than contrast;
  3) length balance (warning, non-blocking).
"""
import numpy as np

from ..embedding_service import embedding_service
from . import prompts

CROSS_GATE = 0.5


def validate_prompt_sets(domain_texts, contrast_texts, embedding_model_id=None) -> dict:
    """Prompt set validation report: cosines, closest pair, lengths, gate.

    Returns dict:
      n_domain/n_contrast, cross_mean, cross_max, intra_domain, intra_contrast,
      closest_pair {domain, contrast, cos}, length {domain, contrast}, warning,
      gate: 'PASS'|'FAIL', gate_reason
    """
    if len(domain_texts) < 3 or len(contrast_texts) < 3:
        return {"gate": "FAIL",
                "gate_reason": "Not enough prompts: need ≥3 in each set (25–30 recommended).",
                "n_domain": len(domain_texts), "n_contrast": len(contrast_texts)}

    E_d = embedding_service.create_embeddings_batch(
        list(domain_texts), model_id=embedding_model_id, normalize=True)
    E_c = embedding_service.create_embeddings_batch(
        list(contrast_texts), model_id=embedding_model_id, normalize=True)

    C = E_d @ E_c.T  # [n_d, n_c]
    cross_mean = float(C.mean())
    cross_max = float(C.max())
    intra_d = float((E_d @ E_d.T).mean())
    intra_c = float((E_c @ E_c.T).mean())

    wi, wj = np.unravel_index(C.argmax(), C.shape)
    closest_pair = {
        "cos": round(cross_max, 3),
        "domain": domain_texts[int(wi)][:120],
        "contrast": contrast_texts[int(wj)][:120],
    }

    d_stats = prompts.length_stats(domain_texts)
    c_stats = prompts.length_stats(contrast_texts)
    warning = prompts.balance_warning(d_stats, c_stats)

    gate_ok = cross_mean < CROSS_GATE and intra_d > intra_c
    reason = ""
    if not gate_ok:
        if cross_mean >= CROSS_GATE:
            reason = f"Sets overlap: cross cos = {cross_mean:.3f} ≥ {CROSS_GATE}."
        else:
            reason = ("Contrast is denser than domain (intra_contrast > intra_domain) "
                      "— domain is not monolithic.")

    return {
        "n_domain": len(domain_texts),
        "n_contrast": len(contrast_texts),
        "cross_mean": round(cross_mean, 3),
        "cross_max": round(cross_max, 3),
        "intra_domain": round(intra_d, 3),
        "intra_contrast": round(intra_c, 3),
        "closest_pair": closest_pair,
        "length": {"domain": d_stats, "contrast": c_stats},
        "warning": warning,
        "gate": "PASS" if gate_ok else "FAIL",
        "gate_reason": reason,
    }
