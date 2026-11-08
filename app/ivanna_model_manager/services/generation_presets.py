"""
v5.0: пресеты параметров генерации — валидация и нормализация.

Чистая логика без Django-зависимостей (кроме логгера) — покрыта
юнит-тестами. Whitelist ключей: всё, что не входит, отбрасывается,
значения приводятся к допустимым диапазонам.
"""
import logging

logger = logging.getLogger(__name__)

# Допустимые ключи → (тип, минимум, максимум)
PARAM_SPECS = {
    'temperature': (float, 0.0, 2.0),
    'max_tokens': (int, 1, 32768),
    'top_p': (float, 0.0, 1.0),
    'top_k': (int, 1, 200),
    'repetition_penalty': (float, 1.0, 2.0),
}

DEFAULTS = {
    'temperature': 0.7,
    'max_tokens': 512,
    'top_p': 0.95,
}


def normalize_preset_config(raw: dict) -> tuple[dict, list]:
    """
    Приводит конфиг пресета к валидному виду.

    Возвращает (config, errors): config — только whitelisted-ключи с
    приведёнными типами и значениями в диапазоне (вне диапазона — clamp
    к границе), errors — человекочитаемые предупреждения о том, что было
    отброшено/скорректировано.
    """
    config, errors = {}, []
    if not isinstance(raw, dict):
        return dict(DEFAULTS), ['config must be an object']
    for key, value in raw.items():
        if key not in PARAM_SPECS:
            errors.append(f'unknown parameter "{key}" — dropped')
            continue
        cast, lo, hi = PARAM_SPECS[key]
        try:
            if cast is int:
                v = int(round(float(value)))
            else:
                v = float(value)
        except (TypeError, ValueError):
            errors.append(f'"{key}": value "{value}" is not a number — dropped')
            continue
        clamped = min(max(v, lo), hi)
        if clamped != v:
            errors.append(f'"{key}"={v} out of range [{lo}, {hi}] — clamped to {clamped}')
        config[key] = clamped
    # дефолты для базовых параметров, если не заданы
    for k, v in DEFAULTS.items():
        config.setdefault(k, v)
    return config, errors


def apply_preset_to_config(session_config: dict, preset_config: dict) -> dict:
    """
    Мержит пресет в generation_config сессии (пресет имеет приоритет).
    Возвращает обновлённый session_config (мутирует копию).
    """
    merged = dict(session_config or {})
    gen = dict(merged.get('generation_config') or {})
    gen.update(preset_config)
    merged['generation_config'] = gen
    return merged
