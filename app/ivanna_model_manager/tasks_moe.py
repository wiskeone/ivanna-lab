"""Задачи мягкого MoE (v5.2): валидация темы, скан+патч эксперта, роутер, HF-push.

Пайплайн джобы (Этапы лабы):
  VALIDATING → SCANNING (d̂, AUC-гейт, писатели) → PATCHING (steer/amplify)
  → [EVALUATING] → [ROUTING] → COMPLETED (новый ModelCheckpoint-эксперт).

Гейт AUC не пройден → джоба останавливается со статусом FAILED и отчётом скана
(патч не делается — принцип «каждый патч через гейт» из лабы).
"""
import logging
import os
import traceback

from celery import shared_task
from django.utils import timezone

from .models import ModelCheckpoint, MoEExpertJob, MoETheme
from .services.soft_moe import evalgate as moe_evalgate
from .services.soft_moe import hf_push as moe_hf_push
from .services.soft_moe import patching as moe_patching
from .services.soft_moe import prompts as moe_prompts
from .services.soft_moe import router as moe_router
from .services.soft_moe import scan as moe_scan
from .services.soft_moe import validation as moe_validation

logger = logging.getLogger(__name__)


# ─────────────────────────── Этап 0: валидация темы ───────────────────────────

@shared_task(bind=True, queue='cpu_tasks', name='ivanna_model_manager.tasks_moe.moe_validate_theme_task')
def moe_validate_theme_task(self, theme_id):
    """Гейт наборов промптов: bge-косинусы cross/intra + баланс длин (CPU)."""
    theme = MoETheme.objects.get(pk=theme_id)
    theme.validation_status = 'PENDING'
    theme.save(update_fields=['validation_status'])
    try:
        domain = moe_prompts.load_texts(theme.domain_prompts_path())
        contrast = moe_prompts.load_texts(theme.contrast_prompts_path())
        report = moe_validation.validate_prompt_sets(
            domain, contrast, embedding_model_id=theme.get_embedding_model_id())
        theme.validation_report = report
        theme.validation_status = 'PASSED' if report.get('gate') == 'PASS' else 'FAILED'
        theme.save(update_fields=['validation_report', 'validation_status'])
        return report
    except Exception as e:
        theme.validation_report = {'gate': 'FAIL', 'error': str(e)}
        theme.validation_status = 'FAILED'
        theme.save(update_fields=['validation_report', 'validation_status'])
        raise


# ─────────────────────────── Этапы 1–4: сборка эксперта ───────────────────────────

@shared_task(bind=True, queue='gpu_tasks', time_limit=86400, soft_time_limit=None,
             name='ivanna_model_manager.tasks_moe.moe_build_expert_task')
def moe_build_expert_task(self, job_id):
    """Скан активаций → гейт AUC → steer/amplify патч → eval-гейт → роутер."""
    job = MoEExpertJob.objects.get(pk=job_id)
    theme = job.theme
    try:
        job.status = 'VALIDATING'
        job.started_at = job.started_at or timezone.now()
        job.celery_task_id = self.request.id
        job.save(update_fields=['status', 'started_at', 'celery_task_id'])
        job.add_log("🚀 Soft MoE expert build started")

        base = job.base_checkpoint
        src = base.local_path
        if not src or not os.path.exists(src):
            raise FileNotFoundError(f"Базовая модель не найдена локально: {src}")

        domain = moe_prompts.load_texts(theme.domain_prompts_path())
        contrast = moe_prompts.load_texts(theme.contrast_prompts_path())
        if len(domain) < 3 or len(contrast) < 3:
            raise ValueError("Наборы промптов темы пусты (нужно ≥3+3).")
        job.add_log(f"Prompts: domain {len(domain)}, contrast {len(contrast)}")

        job_dir = job.job_dir()
        os.makedirs(job_dir, exist_ok=True)

        # GGUF-вход → конвертация в safetensors (dequant)
        if (os.path.isdir(src) and not any(f.endswith('.safetensors') for f in os.listdir(src))) \
                or src.endswith('.gguf'):
            job.add_log("GGUF input: converting to safetensors (dequant)")
            job.update_progress(2, "GGUF → safetensors")
            converted = moe_patching.convert_gguf_to_safetensors(
                src, os.path.join(job_dir, 'base_converted'),
                progress_cb=lambda d, t, s: job.update_progress(2 + 8 * d / max(1, t), f"GGUF: {s}"))
            src = converted

        # ── Этап 1: скан (прогресс 10–55%) ──
        job.status = 'SCANNING'
        job.save(update_fields=['status'])
        npz_path = os.path.join(job_dir, 'scan_dirs.npz')

        def scan_progress(done, total, step):
            job.update_progress(10 + 45 * done / max(1, total), f"Скан: {step}")

        job.update_progress(10, "Скан: загрузка модели")
        scan_report = moe_scan.scan_theme(
            src, domain, contrast, npz_path, load_mode=job.load_mode,
            seed=job.holdout_seed, progress_cb=scan_progress)
        job.scan_report = scan_report
        job.save(update_fields=['scan_report'])
        job.add_log(f"Scan: AUC gate = {scan_report['gate_auc']} ({scan_report['gate']}), "
                    f"layers {scan_report['n_layers']}, {scan_report['sec']}s")

        if scan_report['gate'] != 'PASS':
            raise ValueError(
                f"Scan gate FAILED (AUC={scan_report['gate_auc']} <= 0.85): "
                f"theme direction not extracted. Check prompt sets (theme "
                f"validation) and lengths — no patch applied.")

        # ── Режим анализа: только скан и отчёты, без патча (основа решения) ──
        if not job.do_patch:
            job.add_log("📊 Режим анализа: скан и отчёты готовы, патч не выполнялся (do_patch=False). "
                        "Смотрите визуализацию на странице задачи.")
            job.status = 'COMPLETED'
            job.progress = 100.0
            job.completed_at = timezone.now()
            job.save()
            return {'success': True, 'analysis_only': True, 'scan_gate': scan_report['gate']}

        # ── Этап 2: патч (прогресс 55–75%) ──
        job.status = 'PATCHING'
        job.save(update_fields=['status'])
        out_dir = os.path.join(job_dir, 'expert')
        job.update_progress(55, "Патч весов")

        # v5.2: цель операции. ablate = WSP-абляция (Эксп.1): направление темы
        # ВЫЧИТАЕТСЯ из весов — (I−α·d̂d̂ᵀ)W, реализуется amplify с α<0.
        eff_mode, eff_alpha = job.mode, job.alpha
        if job.goal == 'ablate':
            eff_mode, eff_alpha = 'amplify', -abs(job.alpha)
            job.add_log(f"⚖ WSP ablation: theme direction '{theme.name}' is suppressed "
                        f"((I−α·d̂d̂ᵀ)W, α={eff_alpha}); theme domain prompts = behavior to remove")

        def patch_progress(done, total, step):
            job.update_progress(55 + 20 * done / max(1, total), f"Патч: {step}")

        patch_report = moe_patching.patch_model_dir(
            src, out_dir, npz_path, scan_report,
            mode=eff_mode, alpha=eff_alpha, layers=job.layers, position=job.position,
            progress_cb=patch_progress)
        job.patch_report = patch_report
        job.save(update_fields=['patch_report'])
        job.add_log(f"Patch: {patch_report['patched_count']} tensors, "
                    f"layers {patch_report['layers']}, α={job.alpha}")

        # ── регистрация выходного чекпойнта (паттерн merge_models_task) ──
        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        out_name = f"moe_{theme.slug}_{base.name}_{timestamp}"
        output_model = ModelCheckpoint.objects.create(
            name=out_name[:250],
            display_name=f"{base.display_name or base.name} — MoE эксперт «{theme.name}»",
            model_type=base.model_type,
            parameter_count=base.parameter_count,
            status='DOWNLOADED',
            local_path=out_dir,
            relative_path=f"moe_experts/{job.pk}/expert",
            is_active=True,
            parent=base,
            tags=(base.tags or []) + ['moe-expert'],
            license_type=base.license_type,
            context_length=base.context_length,
            vocab_size=base.vocab_size,
        )

        # ── Этап 3: eval-гейт (прогресс 75–92%) ──
        if job.do_eval:
            eval_d = moe_prompts.load_texts(theme.eval_domain_path())
            eval_g = moe_prompts.load_texts(theme.eval_contrast_path())
            if len(eval_d) >= 2 and len(eval_g) >= 2:
                job.status = 'EVALUATING'
                job.save(update_fields=['status'])
                job.add_log(f"Eval gate: {len(eval_d)} domain + {len(eval_g)} general")

                def eval_progress(done, total, step):
                    job.update_progress(75 + 17 * done / max(1, total), f"Eval: {step}")

                # v5.2: большие модели на малых GPU — nf4-генерация в eval.
                # ИЗВЕСТНОЕ ОГРАНИЧЕНИЕ: bitsandbytes 0.49.2 + transformers 6.x падают
                # на повторной nf4-загрузке (Params4bit/_is_hf_initialized) — для
                # моделей ≥2.5B eval временно пропускается, эксперт регистрируется
                param_b = base.parameter_count or 0
                quant = param_b >= 2_500_000_000
                max_tokens_eval = 192 if quant else moe_evalgate.DEFAULT_MAX_NEW_TOKENS
                if quant:
                    # KNOWN ISSUE (v5.2, verified on Qwen3.5-4B / GTX 1650 4GB):
                    # eval generation for >=2.5B is not viable through any path —
                    #   GPU nf4 subprocess: accelerate 1.6.0 → Params4bit kwarg bug;
                    #     accelerate 1.15.0 → CPU-bound degenerate loop (~0.03 tok/s);
                    #   CPU fallback: hours per prompt set (same cause);
                    #   fla/causal-conv1d kernels: triton crashes on CPU-offloaded
                    #     tensors; causal-conv1d needs nvcc (not in runtime image).
                    # Expert is chat-ready (int4). Fix path: GPU image with nvcc
                    # to build fast-path kernels, or library version pinning.
                    eval_report = {}
                    job.add_log("Eval gate skipped for >=2.5B models (documented "
                                "hardware/library limitation — see evalgate.py)")
                else:
                    eval_report = moe_evalgate.run_eval_gate(
                        base.local_path, out_dir, eval_d, eval_g,
                        os.path.join(job_dir, 'eval'), max_new_tokens=max_tokens_eval,
                        quantized=False, progress_cb=eval_progress)
                job.eval_report = eval_report
                job.save(update_fields=['eval_report'])
                if eval_report:
                    job.add_log(f"Eval: domain Δ{eval_report['summary']['domain_len']['delta_pct']}%, "
                                f"general Δ{eval_report['summary']['general_len']['delta_pct']}%, "
                                f"think leaks {eval_report['summary']['think_leaks']}")
            else:
                job.add_log("Eval gate skipped: theme eval sets empty (need >=2+2)")

        # ── Этап 4: роутер (прогресс 92–100%) ──
        if job.do_router:
            job.status = 'ROUTING'
            job.save(update_fields=['status'])
            job.update_progress(92, "Роутер: центроиды")
            emb_id = theme.get_embedding_model_id()
            moe_router.build_centroids(theme.theme_dir(), domain, contrast,
                                       embedding_model_id=emb_id)
            eval_d = moe_prompts.load_texts(theme.eval_domain_path())
            eval_g = moe_prompts.load_texts(theme.eval_contrast_path())
            if len(eval_d) >= 2 and len(eval_g) >= 2:
                router_report = moe_router.test_router(
                    theme.theme_dir(), eval_d, eval_g,
                    threshold=theme.router_threshold, embedding_model_id=emb_id,
                    out_json=os.path.join(theme.theme_dir(), 'router_test.json'))
                theme.router_report = router_report
                theme.router_built_at = timezone.now()
                theme.save(update_fields=['router_report', 'router_built_at'])
                job.router_report = router_report
                job.save(update_fields=['router_report'])
                job.add_log(f"Router: accuracy {router_report['accuracy']:.0%} "
                            f"({router_report['gate']})")
            else:
                theme.router_built_at = timezone.now()
                theme.save(update_fields=['router_built_at'])
                job.add_log("Router: centroids built (test skipped — no eval sets)")

        job.complete(output_model)
        logger.info(f"✅ moe_build_expert_task {job_id} completed → checkpoint {output_model.id}")
        return {'success': True, 'output_model_id': output_model.id}

    except Exception as e:
        logger.exception("❌ MoE job %s failed", job_id)
        try:
            job = MoEExpertJob.objects.get(pk=job_id)
            job.fail(str(e), {'error': str(e), 'traceback': traceback.format_exc()})
        except Exception:  # noqa: BLE001, S110 — фолбэк отчёта не должен маскировать исходную ошибку
            pass
        return {'success': False, 'error': str(e)}


# ─────────────────────────── Роутер отдельно ───────────────────────────

@shared_task(bind=True, queue='cpu_tasks', name='ivanna_model_manager.tasks_moe.moe_router_build_task')
def moe_router_build_task(self, theme_id):
    """Центроиды + тест точности роутера для темы (по её наборам)."""
    theme = MoETheme.objects.get(pk=theme_id)
    domain = moe_prompts.load_texts(theme.domain_prompts_path())
    contrast = moe_prompts.load_texts(theme.contrast_prompts_path())
    emb_id = theme.get_embedding_model_id()
    moe_router.build_centroids(theme.theme_dir(), domain, contrast,
                               embedding_model_id=emb_id)
    eval_d = moe_prompts.load_texts(theme.eval_domain_path())
    eval_g = moe_prompts.load_texts(theme.eval_contrast_path())
    if len(eval_d) >= 2 and len(eval_g) >= 2:
        report = moe_router.test_router(
            theme.theme_dir(), eval_d, eval_g,
            threshold=theme.router_threshold, embedding_model_id=emb_id,
            out_json=os.path.join(theme.theme_dir(), 'router_test.json'))
    else:
        report = {"accuracy": None, "gate": "SKIP", "note": "нет eval-наборов"}
    theme.router_report = report
    theme.router_built_at = timezone.now()
    theme.save(update_fields=['router_report', 'router_built_at'])
    return report


# ─────────────────────────── HF-публикация ───────────────────────────

@shared_task(bind=True, queue='cpu_tasks', name='ivanna_model_manager.tasks_moe.moe_push_to_hf_task')
def moe_push_to_hf_task(self, job_id, repo_id, private=True, notes=""):
    """Публикация эксперта на Hugging Face (токен из профиля пользователя)."""
    from .models import UserProfile

    job = MoEExpertJob.objects.select_related('theme', 'base_checkpoint',
                                              'output_model').get(pk=job_id)
    token = ""
    if job.created_by:
        profile = UserProfile.objects.filter(user__username=job.created_by).first()
        token = profile.huggingface_token if profile else ""
    if not token:
        # fallback: общий HF_TOKEN из окружения (.env) — валиден для аккаунта,
        # под которым настроена платформа
        token = os.environ.get('HF_TOKEN', '')
    if not token:
        raise ValueError("HF-токен не найден ни в профиле, ни в HF_TOKEN окружения.")

    result = moe_hf_push.push_expert(job, job.theme, repo_id, token,
                                     private=private, notes=notes)
    job.hf_repo_id = repo_id
    job.save(update_fields=['hf_repo_id'])
    job.add_log(f"🤗 Опубликовано: {result['url']}")
    return result


# ─────────────────── v5.2: LLM-интерпретация прогона ───────────────────
@shared_task(bind=True, queue='gpu_tasks', time_limit=3600, soft_time_limit=None,
             name='ivanna_model_manager.tasks_moe.moe_interpret_task')
def moe_interpret_task(self, job_id):
    """Кайллер-фича: модель джобы сама читает дайджест отчётов и пишет
    интерпретацию «что произошло и что делать» на странице задачи."""
    import torch
    from .services.soft_moe.evalgate import _load_for_generation

    job = MoEExpertJob.objects.get(pk=job_id)
    theme, patch = job.theme, job.patch_report or {}
    scan = job.scan_report or {}
    er = (job.eval_report or {}).get('summary') or {}
    rr = job.router_report or {}
    top = [w for w in (scan.get('top_writers') or [])[:4]]

    digest = (
        f"EXPERIMENT REPORT (soft MoE weight-steering, no training)\n"
        f"base model: {job.base_checkpoint.name} ({job.base_checkpoint.parameter_count or '?'} params)\n"
        f"theme/domain: {theme.name}\n"
        f"scan: hold-out AUC gate = {scan.get('gate_auc')} ({scan.get('gate')}), "
        f"layers={scan.get('n_layers')}\n"
        f"top writers: {[(w['layer'], w['proj']) for w in top]}\n"
        f"patch: mode={patch.get('mode')}, alpha={patch.get('alpha')}, "
        f"layers={patch.get('layers')}, tensors patched={patch.get('patched_count')}\n"
        f"eval summary: {er if er else 'skipped'}\n"
        f"router accuracy: {rr.get('accuracy')} ({rr.get('gate')})\n"
    )

    prompt = (
        "You are an interpretability assistant inside a model lab. A user ran a "
        "weight-space steering experiment (a domain direction extracted from "
        "activations was baked into the model with a rank-one patch). "
        "Below is the machine report. Write a SHORT interpretation for the user "
        "(6-10 bullet points, plain language): what each number means, whether the "
        "experiment succeeded, and concrete next actions (tune alpha, narrow layers, "
        "improve prompt sets, evaluate vs base). Do not invent numbers.\n\n"
        f"{digest}"
    )

    try:
        param_b = job.base_checkpoint.parameter_count or 0
        model, tok = _load_for_generation(job.base_checkpoint.local_path,
                                          quantized=param_b >= 2_500_000_000)
        try:
            text = tok.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                           add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=500, do_sample=False,
                                 pad_token_id=tok.eos_token_id or 0)
        interpretation = tok.decode(out[0][ids['input_ids'].shape[1]:],
                                    skip_special_tokens=True).strip()
        job.interpretation = interpretation
        job.save(update_fields=['interpretation'])
        job.add_log("🧠 LLM-интерпретация готова (см. карточку на странице)")
        del model, tok
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {'success': True, 'length': len(interpretation)}
    except Exception as e:
        import traceback
        job.add_log(f"❌ Интерпретация не удалась: {e}")
        return {'success': False, 'error': str(e), 'traceback': traceback.format_exc()}
