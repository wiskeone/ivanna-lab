"""Активационный скан темы (порт weight_lab/theme_scan.py, Этап 1).

d̂_theme[l] = mean_h(domain) − mean_h(contrast) — направления в residual stream
по слоям, две позиции: last (последний токен ввода) и mean (среднее по токенам).

Валидация (гейты лабы):
  1) AUC проекции h на d̂: domain vs contrast > 0.85 (hold-out, Mann-Whitney без sklearn);
  2) каузальные писатели: (mean_domain − mean_contrast) выходов out-модулей,
     проекция на d̂ — выбор слоёв для патча Этапа 2;
  3) mean-входы ī out-модулей на доменных промптах — для steer-патча.

Обобщения относительно лабы:
  - суффикс-матчинг out-модулей (down_proj/o_proj/out_proj) — любая архитектура;
  - загрузка safetensors-чекпойнта или GGUF (через transformers gguf_file);
  - режимы load_mode: auto (bf16 + device_map=auto, CPU-offload при нехватке VRAM),
    int4 (bitsandbytes nf4 — скан 4B на 4GB VRAM), cpu (полностью на CPU).
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

# Суффиксы out-модулей (выход пишет в residual stream) — как в лабе
OUT_MODULE_SUFFIXES = ("down_proj", "o_proj", "out_proj")

AUC_GATE = 0.85


# ─────────────────────────── загрузка модели ───────────────────────────

def _pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _gguf_in_dir(path: Path):
    """GGUF-файл в каталоге (или сам путь — gguf-файл)?"""
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path.parent, path.name
    if path.is_dir():
        ggufs = sorted(path.glob("*.gguf"))
        if ggufs:
            return path, ggufs[0].name
    return None, None


def load_model_for_scan(model_path: str, load_mode: str = "auto"):
    """Загрузить модель для форвардов. Возвращает (model, tokenizer, info).

    load_mode: auto | int4 | cpu (см. модуль docstring).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_path)
    gguf_dir, gguf_name = _gguf_in_dir(path)
    source = f"{gguf_dir} (gguf: {gguf_name})" if gguf_name else str(path)
    dtype = _pick_dtype()
    kwargs = {"trust_remote_code": True}

    if gguf_name:
        # GGUF-вход: dequant в bf16 (как в лабе), без device_map — потом при необходимости
        model = AutoModelForCausalLM.from_pretrained(
            str(gguf_dir), gguf_file=gguf_name, torch_dtype=dtype, **kwargs)
        tok = AutoTokenizer.from_pretrained(str(gguf_dir), gguf_file=gguf_name, **kwargs)
        model.to("cuda" if torch.cuda.is_available() and load_mode != "cpu" else "cpu")
    elif load_mode == "int4":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True)  # 4GB-карты: часть модулей уходит на CPU
        model = AutoModelForCausalLM.from_pretrained(
            str(path), quantization_config=bnb, device_map="auto", **kwargs)
        tok = AutoTokenizer.from_pretrained(str(path), **kwargs)
    elif load_mode == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=dtype, device_map="cpu", **kwargs)
        tok = AutoTokenizer.from_pretrained(str(path), **kwargs)
    else:  # auto
        model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=dtype, device_map="auto", **kwargs)
        tok = AutoTokenizer.from_pretrained(str(path), **kwargs)

    model.eval()
    info = {"source": source, "dtype": str(dtype).replace("torch.", ""),
            "load_mode": load_mode, "model_class": type(model).__name__}
    return model, tok, info


def get_decoder_layers(model):
    """Список decoder-слоёв с fallback по распространённым именам."""
    for attr in ("model.layers", "transformer.h", "transformer.layers",
                 "gpt_neox.layers", "blocks"):
        obj = model
        ok = True
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and obj is not None and len(obj) > 0:
            return list(obj)
    raise RuntimeError(
        "Не найдены decoder-слои (пробовал model.layers / transformer.h / "
        "transformer.layers / gpt_neox.layers / blocks) — архитектура не поддержана сканом.")


def discover_out_modules(model, suffixes=OUT_MODULE_SUFFIXES):
    """[(layer_idx, module_tail, module)] — модули, пишущие в residual stream.

    module_tail — имя модуля относительно слоя (последние сегменты), например
    'mlp.down_proj' / 'self_attn.o_proj'; совпадает с tail в safetensors-именах.
    """
    out = []
    for li, lyr in enumerate(get_decoder_layers(model)):
        for n, m in lyr.named_modules():
            if n and n.endswith(suffixes) and hasattr(m, "weight"):
                out.append((li, n, m))
    if not out:
        raise RuntimeError("Не найдены out-модули (down_proj/o_proj/out_proj) — "
                           "патчить нечего.")
    return out


# ─────────────────────────── математика валидации ───────────────────────────

def auc(scores_s, scores_c) -> float:
    """Mann-Whitney U AUC без sklearn (как в лабе), со средними рангами связок.

    В лабе связки не встречались (проекции непрерывны), но для честности
    метрики на дискретных/повторных данных — средние ранги (scipy 'average').
    """
    all_v = np.concatenate([scores_s, scores_c])
    n = len(all_v)
    order = all_v.argsort(kind="mergesort")
    sorted_v = all_v[order]
    # границы групп равных значений
    new_group = np.r_[True, sorted_v[1:] != sorted_v[:-1]]
    group_idx = new_group.cumsum() - 1                      # dense-ранг каждого элемента
    bounds = np.r_[np.nonzero(new_group)[0], n]             #start/end групп в sort-порядке
    avg_rank_per_group = (bounds[:-1] + bounds[1:] + 1) / 2  # средний 1-базный ранг группы
    ranks = np.empty(n, dtype=float)
    ranks[order] = avg_rank_per_group[group_idx]
    n_s, n_c = len(scores_s), len(scores_c)
    r_s = ranks[:n_s].sum()
    return float((r_s - n_s * (n_s + 1) / 2) / (n_s * n_c))


def split_holdout(n_domain: int, n_contrast: int, seed: int, holdout_fraction: float = 0.33):
    """Пермутация с сидом → (train_domain_idx, train_contrast_idx, test_domain_idx, test_contrast_idx)."""
    rng = np.random.default_rng(seed)
    d_idx = rng.permutation(n_domain)
    c_idx = rng.permutation(n_contrast)
    n_d_te = max(1, round(n_domain * holdout_fraction))
    n_c_te = max(1, round(n_contrast * holdout_fraction))
    return (d_idx[n_d_te:], c_idx[n_c_te:], d_idx[:n_d_te], c_idx[:n_c_te])


# ─────────────────────────── сам скан ───────────────────────────

class _ScanState:
    """Приёмник хуков: активации последней позиции по (setname, idx)."""

    def __init__(self):
        self.current_key = None          # ("domain"|"contrast", idx)
        self.h_last = {}                 # key -> [L+1, hid]
        self.h_mean = {}
        self.mod_out = {}                # (setname, idx, li, tail) -> vec@last
        self.mod_in = {}                 # (setname, idx, li, tail) -> vec@last

    def attach(self, out_modules):
        handles = []

        def mk_hook(li, tail):
            def hook(mod, inp, out):
                if self.current_key is None:
                    return
                t = out[0] if isinstance(out, tuple) else out
                self.mod_out[(*self.current_key, li, tail)] = \
                    t[0, -1, :].detach().to(torch.float32).cpu()
            return hook

        def mk_pre_hook(li, tail):
            def pre_hook(mod, args):
                if self.current_key is None:
                    return
                x = args[0]
                self.mod_in[(*self.current_key, li, tail)] = \
                    x[0, -1, :].detach().to(torch.float32).cpu()
            return pre_hook

        for li, tail, m in out_modules:
            handles.append(m.register_forward_hook(mk_hook(li, tail)))
            handles.append(m.register_forward_pre_hook(mk_pre_hook(li, tail)))
        return handles


def _build_inputs(tok, texts):
    """Промпт → input_ids. Chat-рендер как в проде; enable_thinking — только Qwen."""
    inputs = []
    for txt in texts:
        if getattr(tok, "chat_template", None):
            msgs = [{"role": "user", "content": txt}]
            try:
                text = tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False,
                    enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                               tokenize=False)
            ids = tok(text, return_tensors="pt").input_ids
        else:
            ids = tok(txt, return_tensors="pt").input_ids
        inputs.append(ids)
    return inputs


def scan_theme(model_path, domain_texts, contrast_texts, out_npz, load_mode="auto",
               seed: int = 7, holdout_fraction: float = 0.33, progress_cb=None) -> dict:
    """Полный скан темы. Пишет npz (d_last/d_mean/mean-входы) и возвращает отчёт.

    progress_cb(done, total, step) — опционально, для прогресса задачи.
    """
    t0 = time.time()
    n_total = len(domain_texts) + len(contrast_texts)
    done = 0

    model, tok, info = load_model_for_scan(model_path, load_mode)
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    out_modules = discover_out_modules(model)
    state = _ScanState()
    handles = state.attach(out_modules)

    def report_progress(step):
        nonlocal done
        if progress_cb:
            progress_cb(done, n_total, step)

    def scan_set(name, texts):
        nonlocal done
        inputs = _build_inputs(tok, texts)
        for i, ids in enumerate(inputs):
            state.current_key = (name, i)
            with torch.no_grad():
                out = model(ids.to(model.device), output_hidden_states=True)
            seq = torch.stack([h[0] for h in out.hidden_states], dim=0)  # [L+1, seq, hid]
            state.h_last[(name, i)] = seq[:, -1, :].to(torch.float32).cpu().numpy()
            state.h_mean[(name, i)] = seq.mean(dim=1).to(torch.float32).cpu().numpy()
            done += 1
            report_progress(f"forward {name}[{i + 1}/{len(texts)}]")

    scan_set("domain", list(domain_texts))
    scan_set("contrast", list(contrast_texts))
    for h in handles:
        h.remove()
    state.current_key = None

    # ── направления по train-половине (hold-out честности) ──
    tr_d, tr_c, te_d, te_c = split_holdout(
        len(domain_texts), len(contrast_texts), seed, holdout_fraction)

    def mean_over(store, keys, n_layers_):
        return np.stack([np.stack([store[k] for k in keys]).mean(0)[l]
                         for l in range(n_layers_ + 1)])

    domain_keys_tr = [("domain", i) for i in tr_d]
    contrast_keys_tr = [("contrast", i) for i in tr_c]

    variants = {}
    for pos, store in (("last", state.h_last), ("mean", state.h_mean)):
        D = (mean_over(store, domain_keys_tr, n_layers)
             - mean_over(store, contrast_keys_tr, n_layers))  # [L+1, hid]
        variants[pos] = D

    # сохраняем всё скановое сразу (грабли лабы: артефакты — до секций анализа)
    mean_inputs = {}
    for (sname, i, li, tail), v in state.mod_in.items():
        if sname == "domain":
            mean_inputs.setdefault((li, tail), []).append(v)
    np.savez_compressed(
        out_npz,
        d_last=variants["last"], d_mean=variants["mean"],
        n_layers=n_layers,
        **{f"mi|{li}|{tail}": torch.stack(vs).mean(0).numpy()
           for (li, tail), vs in mean_inputs.items()})

    # ── AUC по слоям (всегда на last-позиции — «состояние готовности ответить») ──
    domain_keys_te = [("domain", i) for i in te_d]
    contrast_keys_te = [("contrast", i) for i in te_c]
    auc_report = {}
    best_auc = 0.0
    for pos, D in variants.items():
        a_tr, a_te = [], []
        for l in range(n_layers + 1):
            d = D[l]
            d = d / (np.linalg.norm(d) + 1e-9)
            s_tr = np.array([state.h_last[k][l] @ d for k in domain_keys_tr])
            c_tr = np.array([state.h_last[k][l] @ d for k in contrast_keys_tr])
            s_te = np.array([state.h_last[k][l] @ d for k in domain_keys_te])
            c_te = np.array([state.h_last[k][l] @ d for k in contrast_keys_te])
            a_tr.append(auc(s_tr, c_tr))
            a_te.append(auc(s_te, c_te))
        auc_report[pos] = {"train": [round(float(v), 3) for v in a_tr],
                           "test": [round(float(v), 3) for v in a_te]}
        best_auc = max(best_auc, float(np.max(a_te)))

    # ── писатели: (mean_domain − mean_contrast) выходов out-модулей @ last ──
    D_last = variants["last"]
    writers = {}
    for (li, tail) in {(li, tail) for (_, _, li, tail) in state.mod_out}:
        v_d = np.stack([state.mod_out[("domain", i, li, tail)]
                        for i in range(len(domain_texts))])
        v_c = np.stack([state.mod_out[("contrast", i, li, tail)]
                        for i in range(len(contrast_texts))])
        diff = v_d.mean(0) - v_c.mean(0)
        d = D_last[min(li + 1, n_layers)]
        d = d / (np.linalg.norm(d) + 1e-9)
        writers[f"L{li:02d}:{tail}"] = {
            "layer": li, "module": tail,
            "proj": round(float(diff @ d), 2),
            "norm": round(float(np.linalg.norm(diff)), 2),
        }
    top_writers = sorted(writers.values(), key=lambda w: -abs(w["proj"]))[:16]

    report = {
        **info,
        "n_domain": len(domain_texts), "n_contrast": len(contrast_texts),
        "n_layers": n_layers, "n_out_modules": len(out_modules),
        "out_module_tails": sorted({tail for _, tail, _ in out_modules}),
        "holdout": {"seed": seed, "fraction": holdout_fraction,
                    "train": [len(tr_d), len(tr_c)], "test": [len(te_d), len(te_c)]},
        "auc": auc_report,
        "gate_auc": round(best_auc, 3),
        "gate": "PASS" if best_auc > AUC_GATE else "FAIL",
        "writers": writers,
        "top_writers": top_writers,
        "sec": round(time.time() - t0, 1),
    }
    with open(Path(out_npz).with_name("scan_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    # освобождение GPU до патча (del в замыкании ломает ruff F821 — обнуляем)
    model = tok = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def auto_layers_from_writers(scan_report: dict, thr_factor: float = 0.6):
    """Слои для патча по писателям: сумма |proj| модулей слоя,
    порог = 0.6 · медианы топ-8 (как в лабе)."""
    per_layer = {}
    for w in scan_report.get("writers", {}).values():
        per_layer[w["layer"]] = per_layer.get(w["layer"], 0.0) + abs(w["proj"])
    if not per_layer:
        return []
    ranked = sorted(per_layer.items(), key=lambda kv: -kv[1])
    thr = float(np.median([v for _, v in ranked[:8]])) * thr_factor
    return sorted(l for l, v in per_layer.items() if v >= thr)
