"""Steer/amplify патч safetensors-чекпойнта (порт weight_lab/patch_theme.py, Этап 2).

Режимы (как в лабе):
  steer   W' = W + α·outer(d̂, ī)/||ī||²  — при типичном входе (x≈ī) сдвиг выхода ≈ α·||d̂_nat||;
          ī — mean вход модуля на доменных промптах (привязка усиления к типичному входу)
  amplify W' = (I + α·d̂d̂ᵀ)W              — зеркало абляции Эксп. 1, усиливает существующие
                                          записи модуля вдоль d̂ (запасной режим)

d̂ берётся D_last[l+1] — модуль слоя l пишет в residual ПОСЛЕ слоя l (конвенция лабы).
Не трогаем embed/lm_head/norm — патчатся только out-матрицы выбранных слоёв
(суффиксы down_proj/o_proj/out_proj по умолчанию).

safetensors-версия: правка в fp32 → каст в исходный dtype тензора, нетронутые
тензоры копируются как есть. GGUF-вход: конвертация через transformers
(dequant bf16) → стандартный safetensors-чекпойнт.
"""
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from .scan import OUT_MODULE_SUFFIXES, auto_layers_from_writers

# tail тензора в safetensors: <prefix>.<layers-key>.<li>.<tail>.weight
# (Llama/Qwen/Mistral: model.layers.23.mlp.down_proj.weight; GPT-2: h.0.attn...)
TENSOR_RE = re.compile(
    r"^(?P<pre>.*?)(?P<lk>layers|h|blocks|layer)\.(?P<li>\d+)\.(?P<tail>.+)\.weight$")


def match_out_tensor(name, suffixes=OUT_MODULE_SUFFIXES):
    """'model.layers.23.mlp.down_proj.weight' → (23, 'mlp.down_proj') | None."""
    m = TENSOR_RE.match(name)
    if not m:
        return None
    tail = m.group("tail")
    if not tail.endswith(suffixes):
        return None
    return int(m.group("li")), tail


def auto_layers(scan_report, thr_factor: float = 0.6):
    return auto_layers_from_writers(scan_report, thr_factor)


def resolve_layers(layers_arg, scan_report, thr_factor: float = 0.6):
    """'auto' | '19,20,21' → отсортированный список слоёв."""
    if str(layers_arg).strip().lower() == "auto":
        return auto_layers(scan_report, thr_factor)
    return sorted(int(x) for x in str(layers_arg).split(",") if x.strip())


def convert_gguf_to_safetensors(model_path: str, out_dir: str, progress_cb=None) -> str:
    """GGUF (файл или каталог с .gguf) → HF-чекпойнт safetensors (dequant bf16/fp16).

    Требует RAM ≈ размер модели (как скан в лабе: ~5.6 ГБ RSS для 2B Q4).
    Возвращает путь к выходному каталогу.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .scan import _gguf_in_dir, _pick_dtype

    path = Path(model_path)
    gguf_dir, gguf_name = _gguf_in_dir(path)
    if not gguf_name:
        raise ValueError(f"GGUF не найден: {model_path}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(10, 100, "загрузка GGUF (dequant)")
    dtype = _pick_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        str(gguf_dir), gguf_file=gguf_name, torch_dtype=dtype, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(str(gguf_dir), gguf_file=gguf_name,
                                        trust_remote_code=True)
    if progress_cb:
        progress_cb(60, 100, "save_pretrained (safetensors)")
    model.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    del model, tok
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if progress_cb:
        progress_cb(100, 100, "GGUF → safetensors готово")
    return str(out)


def patch_model_dir(src_dir, out_dir, dirs_npz, scan_report, mode="steer",
                    alpha=0.2, layers="auto", suffixes=OUT_MODULE_SUFFIXES,
                    position="last", progress_cb=None) -> dict:
    """Патч чекпойнта: читаем safetensors лениво, правим out-матрицы выбранных слоёв.

    Несовпадающие тензоры копируются байт-в-байт (новый файл safetensors),
    прочие файлы (config.json, tokenizer.*, *.json) копируются как есть.
    Память: пик ≈ размер одного safetensors-файла.
    position: 'last' | 'mean' — какую пару направлений из скана использовать
    (mean-входы ī в steer всегда с last-позиции — в лабе проверен только last).
    Возвращает отчёт (mode/alpha/layers/tensors/delta_l1).
    """
    src, out = Path(src_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Z = np.load(dirs_npz)
    D = (Z["d_mean"] if position == "mean" else Z["d_last"]).astype(np.float32)  # [L+1, hid]
    hid = D.shape[1]

    chosen = resolve_layers(layers, scan_report)
    if not chosen:
        raise ValueError("Не выбран ни один слой для патча (layers=auto по writers "
                         "дал пустой список — проверьте scan_report).")

    # d̂ по слоям (единичные)
    dirs = {}
    for l in chosen:
        d = D[min(l + 1, len(D) - 1)]
        n = np.linalg.norm(d)
        if n > 1e-6:
            dirs[l] = (d / n).astype(np.float32)
    if not dirs:
        raise ValueError("Все направления d̂ нулевые — скан не выделил тему.")

    # mean-входы ī: ключи npz 'mi|{li}|{tail}'
    mean_in = {}
    for key in Z.files:
        if key.startswith("mi|"):
            _, li, tail = key.split("|", 2)
            mean_in[(int(li), tail)] = Z[key].astype(np.float32)

    st_files = sorted(src.glob("*.safetensors"))
    if not st_files:
        raise ValueError(f"В {src} нет *.safetensors — сначала конвертируйте GGUF "
                         f"(convert_gguf_to_safetensors).")

    report = {"mode": mode, "alpha": alpha, "layers": chosen, "suffixes": list(suffixes),
              "hid": int(hid), "tensors": [], "patched_count": 0}
    total = len(st_files)

    for fi, st in enumerate(st_files):
        from safetensors.torch import load_file, save_file
        tensors = load_file(str(st))  # {name: torch tensor}
        n_patched_file = 0
        for name in list(tensors):
            hit = match_out_tensor(name, suffixes)
            li, tail = hit if hit else (None, None)
            if li is None or li not in dirs:
                continue
            W = tensors[name]
            if W.dim() != 2 or W.shape[0] != hid:
                continue  # не out-матрица (например, transposed-конвенция) — пропускаем
            orig_dtype = W.dtype
            W32 = W.to(torch.float32)
            d = torch.from_numpy(dirs[li])
            if mode == "amplify":
                W2 = W32 + alpha * torch.outer(d, d @ W32)
            else:  # steer
                i_bar = mean_in.get((li, tail))
                if i_bar is None:
                    report["tensors"].append({"name": name, "skipped": "нет mean-входа"})
                    continue
                i_t = torch.from_numpy(np.ascontiguousarray(i_bar))
                n_i = float(torch.norm(i_t))
                if n_i < 1e-9:
                    report["tensors"].append({"name": name, "skipped": "нулевой mean-вход"})
                    continue
                W2 = W32 + alpha * torch.outer(d, i_t) / (n_i ** 2)
            delta_l1 = float((W2 - W32).abs().sum())
            tensors[name] = W2.to(orig_dtype)
            report["tensors"].append({"name": name, "delta_l1": delta_l1,
                                      "orig_l1": float(W32.abs().sum())})
            n_patched_file += 1
        save_file(tensors, str(out / st.name), metadata={"format": "pt"})
        report["patched_count"] += n_patched_file
        del tensors
        if progress_cb:
            progress_cb(fi + 1, total, f"{st.name}: патчено {n_patched_file}")

    # прочие файлы (config, tokenizer, index) — как есть
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, out / f.name)

    with open(out / "patch_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    return report
