"""Theme prompt sets (port of weight_lab/themes, Stage 0).

File format — txt, one prompt per line. Verified in the lab: lengths of the domain
and contrast sets must be aligned (medians of the same order), otherwise
length leaks into the direction d̂.
"""
import html
import re
import statistics
from pathlib import Path

from django.conf import settings

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def theme_dir(slug: str) -> Path:
    """Theme artifact directory: MEDIA_ROOT/moe_themes/<slug>/."""
    d = Path(settings.MEDIA_ROOT) / "moe_themes" / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def clean_text(text: str) -> str:
    """HTML cleaning (as in the lab's extract_prompts): unescape → strip tags → whitespace."""
    text = html.unescape(text or "")
    text = TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ")
    return WS_RE.sub(" ", text).strip()


def clean_prompt_set(texts, min_len: int = 20, max_len: int = 2000) -> list:
    """Clean + length filter + dedup by first 80 chars (pool of usable prompts)."""
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
    """Textarea/file → list of prompts (lines, empty ones dropped)."""
    return [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]


def save_prompt_set(path: Path, texts) -> int:
    """Save a set (one line = one prompt). Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [t.replace("\n", " ").strip() for t in texts if t.strip()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def load_texts(path) -> list:
    """File → list of prompts. Empty/missing file → empty list."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def length_stats(texts) -> dict:
    """Length stats of a set — for domain/contrast balance control."""
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
    """Warning if median lengths diverge by more than factor
    (in the lab lengths were aligned deliberately: otherwise length leaks into the direction)."""
    if domain_stats.get("n", 0) < 3 or contrast_stats.get("n", 0) < 3:
        return ""
    d, c = domain_stats["median"], contrast_stats["median"]
    if d and c and max(d, c) / max(1, min(d, c)) > factor:
        return (
            f"Median lengths diverge: domain {d} vs contrast {c}. "
            f"Align the sets by length, otherwise length leaks into the direction d̂."
        )
    return ""
