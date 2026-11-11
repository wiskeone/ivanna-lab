"""Наборы промптов темы (порт weight_lab/themes, Этап 0).

Формат файла — txt, один промпт на строку. В лабе проверено: длины доменных
и контрастных наборов должны быть выровнены (медианы в одном порядке), иначе
длина протекает в направление d̂.
"""
import html
import re
import statistics
from pathlib import Path

from django.conf import settings

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def theme_dir(slug: str) -> Path:
    """Каталог артефактов темы: MEDIA_ROOT/moe_themes/<slug>/."""
    d = Path(settings.MEDIA_ROOT) / "moe_themes" / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def clean_text(text: str) -> str:
    """Чистка HTML (как в extract_prompts лабы): unescape → strip-тегов → whitespace."""
    text = html.unescape(text or "")
    text = TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ")
    return WS_RE.sub(" ", text).strip()


def clean_prompt_set(texts, min_len: int = 20, max_len: int = 2000) -> list:
    """Чистка + фильтр длины + дедуп по первым 80 символам (пул годных промптов)."""
    seen, out = set(), []
    for t in texts:
        t = clean_text(t)
        if not (min_len <= len(t) <= max_len):
            continue
        key = t[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def parse_prompts(raw: str) -> list:
    """Textarea/файл → список промптов (строки, пустые выкидываются)."""
    return [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]


def save_prompt_set(path: Path, texts) -> int:
    """Сохранить набор (одна строка = один промпт). Возвращает количество."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [t.replace("\n", " ").strip() for t in texts if t.strip()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def load_texts(path) -> list:
    """Файл → список промптов. Пустой/отсутствующий файл → пустой список."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def length_stats(texts) -> dict:
    """Статистика длин набора — для контроля баланса домен/контраст."""
    lens = [len(t) for t in texts]
    if not lens:
        return {"n": 0}
    lens.sort()
    return {
        "n": len(lens),
        "median": round(statistics.median(lens)),
        "min": lens[0],
        "max": lens[-1],
    }


def balance_warning(domain_stats: dict, contrast_stats: dict, factor: float = 2.0) -> str:
    """Предупреждение, если медианы длин расходятся более чем в factor раз
    (в лабе длины выравнивали осознанно: иначе длина протечёт в направление)."""
    if domain_stats.get("n", 0) < 3 or contrast_stats.get("n", 0) < 3:
        return ""
    d, c = domain_stats["median"], contrast_stats["median"]
    if d and c and max(d, c) / max(1, min(d, c)) > factor:
        return (
            f"Медианы длин расходятся: домен {d} vs контраст {c}. "
            f"Выровняйте наборы по длине, иначе длина протечёт в направление d̂."
        )
    return ""
