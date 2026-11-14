"""Лёгкий eval-гейт патча (порт weight_lab/eval_theme.py, Этап 3).

Генерация base vs expert на свежих eval-промптах (не пересекаться со скановыми!),
прод-параметры лабы: temp 0.7 / top_p 0.9. Метрики: длина, think-утечки,
повторные пары слов. Тексты сохраняются для ручной выборочной проверки
(в лабе — 5/5 ответов патча глазами).

Параметры генерации и детали — в docs/soft_moe.md.
"""
import json
import sys
import re
import statistics
import time
from pathlib import Path

import torch

THINK_PAT = re.compile(r"<think>|</think>")

DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9


def answer_metrics(answer: str) -> dict:
    """Метрики одного ответа (think-утечки, длина, повторы пар слов)."""
    a = answer or ""
    words = a.lower().split()
    pairs = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
    return {
        "len_chars": len(a),
        "think_leak": bool(THINK_PAT.search(a)),
        "repeat_pairs": len(pairs) - len(set(pairs)),
    }


def _load_for_generation(model_path: str, quantized: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = {"trust_remote_code": True}
    dtype = torch.bfloat16 if (torch.cuda.is_available() and
                               torch.cuda.is_bf16_supported()) else torch.float16
    if quantized:
        # малые GPU: nf4-генерация (bf16-офлоад 4B ловит device-side assert)
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map="auto", **kwargs)
        model.eval()
        tok = AutoTokenizer.from_pretrained(model_path, **kwargs)
        return model, tok
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto", **kwargs)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_path, **kwargs)
    return model, tok


def _generate(model, tok, prompt: str, max_new_tokens: int) -> str:
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False)
        ids = tok(text, return_tensors="pt").to(model.device)
    else:
        ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
            pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def _run_set(model, tok, prompts_list, outdir: Path, tag: str,
             max_new_tokens: int) -> dict:
    rows = []
    for i, p in enumerate(prompts_list):
        t0 = time.time()
        ans = _generate(model, tok, p, max_new_tokens)
        m = answer_metrics(ans)
        m["sec"] = round(time.time() - t0, 1)
        m["prompt"] = p
        rows.append(m)
        (outdir / f"{tag}_{i:02d}.txt").write_text(
            f"ПРОМПТ:\n{p}\n\nОТВЕТ:\n{ans}\n", encoding="utf-8")
    agg = {
        "n": len(rows),
        "mean_len": round(statistics.mean(r["len_chars"] for r in rows), 0) if rows else 0,
        "think_leaks": sum(r["think_leak"] for r in rows),
        "max_repeat_pairs": max((r["repeat_pairs"] for r in rows), default=0),
    }
    return {"agg": agg, "rows": rows}


RUNNER_CODE = """
# Изолированный генератор: свежий процесс = чистый bnb-контекст.
# Фон: второй quantized load в одном процессе падает в accelerate
# (Params4bit/_is_hf_initialized, bnb 0.49.2 + transformers 6.x).
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path, out_path, max_new_tokens, quantized = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == '1'
tasks = json.load(open(sys.argv[5]))

kwargs = {'trust_remote_code': True}
if quantized:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True,
                             llm_int8_enable_fp32_cpu_offload=True)
    kwargs.update(quantization_config=bnb, device_map='auto')

dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
if not quantized:
    kwargs['torch_dtype'] = dtype
model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
model.eval()
tok = AutoTokenizer.from_pretrained(model_path, **kwargs)

results = []
for item in tasks:
    prompt, max_new = item['prompt'], item.get('max_new_tokens', max_new_tokens)
    if getattr(tok, 'chat_template', None):
        msgs = [{'role': 'user', 'content': prompt}]
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors='pt').to(model.device)
    else:
        ids = tok(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                             temperature=0.7, top_p=0.9,
                             pad_token_id=tok.eos_token_id or 0)
    results.append(tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True))

json.dump(results, open(out_path, 'w'), ensure_ascii=False)
"""


def _generate_batch_subprocess(model_path: str, prompts: list, max_new_tokens: int,
                               quantized: bool, out_path: str) -> list:
    """Генерация батча промптов в изолированном подпроцессе."""
    import json
    import subprocess
    tasks = [{'prompt': p, 'max_new_tokens': max_new_tokens} for p in prompts]
    json.dump(tasks, open(out_path + '.tasks.json', 'w'), ensure_ascii=False)
    r = subprocess.run([sys.executable, '-c', RUNNER_CODE, model_path,
                        out_path + '.answers.json', str(max_new_tokens),
                        '1' if quantized else '0', out_path + '.tasks.json'],
                       capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        raise RuntimeError(f'generation subprocess failed: {r.stderr[-600:]}')
    return json.load(open(out_path + '.answers.json'))


def run_eval_gate(base_path: str, expert_path: str, eval_domain_texts,
                  eval_contrast_texts, out_dir: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                  quantized: bool = False, progress_cb=None) -> dict:
    """Прогон двух сборок на двух наборах (v5.2: генерация в подпроцессах).

    Смысл гейта (лаба): доменные ответы эксперта не хуже/полне базовых,
    общие не деградировали. Числа — прокси (длина); финальное решение —
    ручная выборочная проверка сохранённых текстов.
    """
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {"base": base_path, "expert": expert_path, "max_new_tokens": max_new_tokens,
              "quantized": quantized, "sets": {}}
    for model_tag, path in (("base", base_path), ("expert", expert_path)):
        if progress_cb:
            progress_cb(0, 100, f"загрузка {model_tag}")
        for set_tag, texts in (("eval_domain", eval_domain_texts),
                               ("eval_general", eval_contrast_texts)):
            if progress_cb:
                progress_cb(0, len(texts), f"{model_tag}/{set_tag}")
            answers = _generate_batch_subprocess(
                path, texts, max_new_tokens, quantized,
                str(outdir / f'{model_tag}_{set_tag}'))
            rows = []
            for i, (p, ans) in enumerate(zip(texts, answers)):
                t0 = time.time()
                m = answer_metrics(ans)
                m['sec'] = round(time.time() - t0, 1)
                m['prompt'] = p
                rows.append(m)
                (outdir / f'{model_tag}_{set_tag}_{i:02d}.txt').write_text(
                    f'ПРОМПТ:\n{p}\n\nОТВЕТ:\n{ans}\n', encoding='utf-8')
            agg = {
                'n': len(rows),
                'mean_len': round(statistics.mean(r['len_chars'] for r in rows), 0) if rows else 0,
                'think_leaks': sum(r['think_leak'] for r in rows),
                'max_repeat_pairs': max((r['repeat_pairs'] for r in rows), default=0),
            }
            report['sets'][f'{model_tag}/{set_tag}'] = {'agg': agg, 'rows': rows}
            if progress_cb:
                progress_cb(len(texts), len(texts), f'{model_tag}/{set_tag}')

    # сводка для гейта: направление изменений
    b_d = report["sets"]["base/eval_domain"]["agg"]["mean_len"]
    e_d = report["sets"]["expert/eval_domain"]["agg"]["mean_len"]
    b_g = report["sets"]["base/eval_general"]["agg"]["mean_len"]
    e_g = report["sets"]["expert/eval_general"]["agg"]["mean_len"]
    report["summary"] = {
        "domain_len": {"base": b_d, "expert": e_d,
                       "delta_pct": round(100 * (e_d - b_d) / max(1, b_d), 1)},
        "general_len": {"base": b_g, "expert": e_g,
                        "delta_pct": round(100 * (e_g - b_g) / max(1, b_g), 1)},
        "think_leaks": sum(v["agg"]["think_leaks"] for v in report["sets"].values()),
    }
    with open(outdir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return report
