"""Shared LoRA configuration for all trainers (v5.3 Tier-1 upgrade).

One source of truth instead of three hardcoded LoraConfig sites:
- arch-aware target-module presets (Qwen3.5 hybrid: gated delta net modules)
- DoRA / rsLoRA flags
- LoRA+ (separate learning rate for the lora_B matrices)
"""
import json
import logging

import torch
from peft import LoraConfig

logger = logging.getLogger(__name__)

# conv1d is intentionally absent: peft has no LoRA-Conv1d, and conv weights are 3D
ATTENTION_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
ATTENTION_MLP_MODULES = ATTENTION_MODULES + ['gate_proj', 'up_proj', 'down_proj']
HYBRID_LINEAR_ATTN_MODULES = ['in_proj_qkvz', 'in_proj_ba', 'out_proj']

TARGET_PRESETS = {
    'attention': ATTENTION_MODULES,
    'attention+mlp': ATTENTION_MLP_MODULES,
    'hybrid-full': ATTENTION_MLP_MODULES + HYBRID_LINEAR_ATTN_MODULES,
}


def _base_model_type(model_path):
    """model_type from the checkpoint config.json (e.g. 'qwen3_5' = hybrid arch)."""
    import os
    try:
        with open(os.path.join(model_path, 'config.json'), encoding='utf-8') as fh:
            return json.load(fh).get('model_type', '')
    except Exception:
        return ''


def resolve_target_modules(job, model_path):
    """Target module list for the job: explicit selection, or an arch-aware preset."""
    if job.target_modules:
        return list(job.target_modules)
    preset = getattr(job, 'target_preset', '') or 'hybrid-full'
    if preset in TARGET_PRESETS:
        return list(TARGET_PRESETS[preset])
    # non-hybrid fallback: attention+mlp covers classic dense archs
    model_type = _base_model_type(model_path)
    if 'qwen3_5' in model_type or 'hybrid' in model_type:
        return list(TARGET_PRESETS['hybrid-full'])
    return list(TARGET_PRESETS['attention+mlp'])


def build_lora_config(job, model_path):
    """LoraConfig from job fields: r/alpha/dropout + DoRA + rsLoRA + target preset."""
    return LoraConfig(
        r=job.lora_r,
        lora_alpha=job.lora_alpha,
        target_modules=resolve_target_modules(job, model_path),
        lora_dropout=job.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
        use_dora=job.use_dora,
        use_rslora=job.use_rslora,
    )


def apply_lora_plus(trainer, ratio):
    """LoRA+: separate learning rate for the lora_B matrices (and DoRA magnitudes).

    Builds the trainer optimizer with two parameter groups: lora_A at the base
    learning_rate, lora_B at learning_rate * ratio. Call after Trainer creation,
    before train() — Trainer skips its own optimizer when one is attached.
    """
    if not ratio or ratio <= 1.0:
        return None
    base_lr = trainer.args.learning_rate
    group_a, group_b = [], []
    for name, param in trainer.model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_B' in name or 'lora_magnitude' in name:
            group_b.append(param)
        elif 'lora_A' in name:
            group_a.append(param)
    if not group_b:
        logger.warning('LoRA+: no lora_B parameters found — ratio ignored')
        return None
    optimizer = torch.optim.AdamW(
        [{'params': group_a, 'lr': base_lr},
         {'params': group_b, 'lr': base_lr * ratio}],
        lr=base_lr,
        betas=(trainer.args.adam_beta1, trainer.args.adam_beta2),
        weight_decay=trainer.args.weight_decay,
    )
    logger.info(f'LoRA+: {len(group_a)} A params @ {base_lr}, '
                f'{len(group_b)} B params @ {base_lr * ratio}')
    trainer.optimizer = optimizer
    return optimizer
