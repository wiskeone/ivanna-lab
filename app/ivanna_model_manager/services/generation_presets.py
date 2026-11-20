"""
v5.0: generation parameter presets — validation and normalization.

Pure logic with no Django dependencies (except the logger) — covered
by unit tests. Key whitelist: anything not on it is dropped; values
are clamped to the allowed ranges.
"""
import logging

logger = logging.getLogger(__name__)

# Allowed keys -> (type, min, max)
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
    Normalize a preset config into a valid form.

    Returns (config, errors): config contains only whitelisted keys with
    coerced types and in-range values (out-of-range values are clamped
    to the boundary); errors holds human-readable warnings about what was
    dropped/corrected.
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
    # defaults for base parameters, if not set
    for k, v in DEFAULTS.items():
        config.setdefault(k, v)
    return config, errors


def apply_preset_to_config(session_config: dict, preset_config: dict) -> dict:
    """
    Merge a preset into the session generation_config (the preset wins).
    Returns the updated session_config (mutates a copy).
    """
    merged = dict(session_config or {})
    gen = dict(merged.get('generation_config') or {})
    gen.update(preset_config)
    merged['generation_config'] = gen
    return merged
