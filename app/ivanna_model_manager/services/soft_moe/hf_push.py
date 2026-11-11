"""Публикация мягкого MoE-эксперта на Hugging Face.

upload_folder чекпойнта + авто-README-карточка (RU/EN): метод, тема, параметры
патча, гейты скана/роутера. Токен — UserProfile.huggingface_token.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_model_card(job, theme) -> str:
    """README.md карточки эксперта (RU + EN кратко)."""
    scan = job.scan_report or {}
    patch = job.patch_report or {}
    base = job.base_checkpoint
    base_id = base.huggingface_id or base.name
    layers = patch.get("layers", [])
    lines = [
        "---",
        f"base_model: {base_id}",
        "tags:",
        "  - soft-moe",
        "  - weight-steering",
        "  - domain-expert",
        "  - no-training",
        "library_name: transformers",
        "---",
        "",
        f"# {base.display_name or base.name} — Soft MoE expert «{theme.name}»",
        "",
        "## RU",
        "",
        "Мягкий MoE-эксперт, полученный из плотной модели **без дообучения**",
        "(weight-space steering): направление домена d̂ выделено активационным сканом",
        "(разность средних hidden states доменных vs контрастных промптов) и вшито",
        "steer-патчем в выходные матрицы (down_proj/o_proj/out_proj) выбранных слоёв:",
        "",
        f"- базовая модель: `{base_id}`",
        f"- тема (домен): **{theme.name}**" + (f" — {theme.description}" if theme.description else ""),
        f"- режим: `{patch.get('mode', 'steer')}`, α = {patch.get('alpha', 'n/a')}",
        f"- слои: {layers if layers else 'auto (по писателям)'}",
        f"- гейт скана: AUC = {scan.get('gate_auc', 'n/a')} ({scan.get('gate', 'n/a')})",
        "",
        "Роутер (эмбеддинги + margin) выбирает эту сборку вместо базовой для запросов темы.",
        "",
        "### Метод",
        "",
        "Вдохновлено Weight-Space Programming (Takeyama et al., MIT CSAIL, 2026):",
        "правки весов без градиентов. Расширение лабы Ivanna: обратная операция —",
        "не абляция поведения, а steer-усиление доменного направления, вшитое в веса",
        "(работает там, где инференс-хуки недоступны).",
        "",
        "## EN (short)",
        "",
        "A *soft MoE* domain expert derived from a dense model **without any training**:",
        "the domain direction d̂ is extracted via activation scans (mean hidden-state",
        "difference between domain and contrast prompts) and baked into the weights with",
        "a rank-one steer patch on output matrices of selected layers. An embedding-space",
        "router (cosine margin) picks this variant instead of the base model.",
        f"Patch mode: `{patch.get('mode', 'steer')}`, alpha = {patch.get('alpha', 'n/a')},",
        f"layers: {layers if layers else 'auto'}. Scan gate AUC = {scan.get('gate_auc', 'n/a')}.",
        "",
        "Created with [Ivanna Lab](https://github.com/wiskeone/ivanna-lab) —",
        "self-hosted LLM model lab.",
    ]
    return "\n".join(lines) + "\n"


def push_expert(job, theme, repo_id: str, token: str, private: bool = True,
                notes: str = "") -> dict:
    """Пуш чекпойнта эксперта на HF. Возвращает {repo_id, url, files}."""
    from huggingface_hub import HfApi

    ckpt = job.output_model
    if ckpt is None or not ckpt.local_path or not Path(ckpt.local_path).exists():
        raise ValueError("Чекпойнт эксперта не найден локально — джоба завершена?")

    card = build_model_card(job, theme)
    if notes:
        card += "\n## Notes\n\n" + notes.strip() + "\n"
    (Path(ckpt.local_path) / "README.md").write_text(card, encoding="utf-8")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    logger.info(f"🚀 pushing MoE expert to {repo_id} (private={private})")
    res = api.upload_folder(
        folder_path=ckpt.local_path, repo_id=repo_id, repo_type="model",
        commit_message=f"Soft MoE expert '{theme.name}' (steer patch, no training)")
    url = f"https://huggingface.co/{repo_id}"
    return {"repo_id": repo_id, "url": url, "commit_url": getattr(res, "commit_url", url)}
