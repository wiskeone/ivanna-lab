"""Steer/amplify patch of a safetensors checkpoint (port of weight_lab/patch_theme.py, Stage 2).

Modes (as in the lab):
  steer   W' = W + α·outer(d̂, ī)/||ī||²  — at a typical input (x≈ī) the output shift ≈ α·||d̂_nat||;
          ī — mean input of the module on domain prompts (gain tied to the typical input)
  amplify W' = (I + α·d̂d̂ᵀ)W              — mirror of ablation Exp. 1, amplifies the module's
                                          existing entries along d̂ (fallback mode)

d̂ is taken from D_last[l+1] — a layer-l module writes into the residual AFTER layer l (lab convention).
embed/lm_head/norm are untouched — only out-matrices of selected layers are patched
(down_proj/o_proj/out_proj suffixes by default).

safetensors version: edit in fp32 → cast back to the tensor's original dtype; untouched
tensors are copied as is. GGUF input: conversion via transformers
(dequant bf16) → a standard safetensors checkpoint.
"""
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from .scan import OUT_MODULE_SUFFIXES, auto_layers_from_writers

# tensor tail in safetensors: <prefix>.<layers-key>.<li>.<tail>.weight
# (Llama/Qwen/Mistral: model.layers.23.mlp.down_proj.weight; GPT-2: h.0.attn...)
TENSOR_RE = re.compile(
    r"^(?P<pre>.*?)(?P<lk>layers|h|blocks|layer)\.(?P<li>\d+)\.(?P<tail>.+)\.weight$")


def bucket_matrix(arr, n=24):
    """|ΔW| [out, in] → per-bucket means [n, n] — compact heatmap."""
    a = np.abs(np.asarray(arr, dtype=np.float32))
    oh, ih = a.shape
    out = np.zeros((min(n, oh) or 1, min(n, ih) or 1), dtype=np.float32)
    ob = np.array_split(np.arange(oh), out.shape[0])
    ib = np.array_split(np.arange(ih), out.shape[1])
    for i, rows in enumerate(ob):
        for j, cols in enumerate(ib):
            out[i, j] = float(a[np.ix_(rows, cols)].mean())
    return [[round(float(v), 6) for v in row] for row in out]


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
    """'auto' | '19,20,21' → sorted list of layers."""
    if str(layers_arg).strip().lower() == "auto":
        return auto_layers(scan_report, thr_factor)
    return sorted(int(x) for x in str(layers_arg).split(",") if x.strip())


def convert_gguf_to_safetensors(model_path: str, out_dir: str, progress_cb=None) -> str:
    """GGUF (file or directory with .gguf) → HF safetensors checkpoint (dequant bf16/fp16).

    Requires RAM ≈ model size (like the lab scan: ~5.6 GB RSS for a 2B Q4).
    Returns the output directory path.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .scan import _gguf_in_dir, _pick_dtype

    path = Path(model_path)
    gguf_dir, gguf_name = _gguf_in_dir(path)
    if not gguf_name:
        raise ValueError(f"GGUF not found: {model_path}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(10, 100, "loading GGUF (dequant)")
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
        progress_cb(100, 100, "GGUF → safetensors done")
    return str(out)


def build_col_mask(tensor_name, n_cols, col_masks):
    """Column mask for a tensor from the click-tool selection: 1.0 inside kept
    [start, end] ranges, 0.0 outside. None when no selection applies."""
    if not col_masks:
        return None
    import torch as _torch
    mask = None
    for suffix, ranges in (col_masks or {}).items():
        if tensor_name.endswith(suffix):
            mask = _torch.zeros(n_cols, dtype=torch.float32)
            for start, end in ranges:
                mask[start:end] = 1.0
            break
    return mask


def patch_model_dir(src_dir, out_dir, dirs_npz, scan_report, mode="steer",
                    alpha=0.2, layers="auto", suffixes=OUT_MODULE_SUFFIXES,
                    position="last", progress_cb=None, col_masks=None) -> dict:
    """Patch a checkpoint: lazy safetensors reading, edit out-matrices of selected layers.

    Non-matching tensors are copied byte-for-byte (into the new safetensors file),
    other files (config.json, tokenizer.*, *.json) are copied as is.
    Memory: peak ≈ size of one safetensors file.
    position: 'last' | 'mean' — which pair of scan directions to use
    (mean inputs ī in steer always come from the last position — the lab verified only last).
    col_masks: optional {suffix: [[start, end], ...]} — keep only these INPUT-column
    ranges of the rank-one update on matching tensors (click-tool group selection);
    columns outside the ranges get zero delta.
    Returns a report (mode/alpha/layers/tensors/delta_l1).
    """
    src, out = Path(src_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Z = np.load(dirs_npz)
    D = (Z["d_mean"] if position == "mean" else Z["d_last"]).astype(np.float32)  # [L+1, hid]
    hid = D.shape[1]

    chosen = resolve_layers(layers, scan_report)
    if not chosen:
        raise ValueError("No layer selected for patching (layers=auto over writers "
                         "gave an empty list — check scan_report).")

    # d̂ per layer (unit-norm)
    dirs = {}
    for l in chosen:
        d = D[min(l + 1, len(D) - 1)]
        n = np.linalg.norm(d)
        if n > 1e-6:
            dirs[l] = (d / n).astype(np.float32)
    if not dirs:
        raise ValueError("All d̂ directions are zero — the scan did not extract the theme.")

    # mean inputs ī: npz keys 'mi|{li}|{tail}'
    mean_in = {}
    for key in Z.files:
        if key.startswith("mi|"):
            _, li, tail = key.split("|", 2)
            mean_in[(int(li), tail)] = Z[key].astype(np.float32)

    st_files = sorted(src.glob("*.safetensors"))
    if not st_files:
        raise ValueError(f"No *.safetensors in {src} — convert the GGUF first "
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
                continue  # not an out-matrix (e.g. a transposed convention) — skip
            orig_dtype = W.dtype
            W32 = W.to(torch.float32)
            d = torch.from_numpy(dirs[li])
            col_mask = build_col_mask(name, W.shape[1], col_masks)
            if mode == "amplify":
                delta = alpha * torch.outer(d, d @ W32)
                if col_mask is not None:
                    delta = delta * col_mask
                W2 = W32 + delta
            else:  # steer
                i_bar = mean_in.get((li, tail))
                if i_bar is None:
                    report["tensors"].append({"name": name, "skipped": "no mean-input"})
                    continue
                i_t = torch.from_numpy(np.ascontiguousarray(i_bar))
                n_i = float(torch.norm(i_t))
                if n_i < 1e-9:
                    report["tensors"].append({"name": name, "skipped": "zero mean-input"})
                    continue
                delta = alpha * torch.outer(d, i_t) / (n_i ** 2)
                if col_mask is not None:
                    delta = delta * col_mask
                W2 = W32 + delta
            delta = (W2 - W32)
            delta_l1 = float(delta.abs().sum())
            tensors[name] = W2.to(orig_dtype)
            report["tensors"].append({"name": name, "delta_l1": delta_l1,
                                      "orig_l1": float(W32.abs().sum()),
                                      "ih": int(delta.shape[1]), "oh": int(delta.shape[0]),
                                      "heatmap": bucket_matrix(delta.numpy())})
            n_patched_file += 1
        save_file(tensors, str(out / st.name), metadata={"format": "pt"})
        report["patched_count"] += n_patched_file
        del tensors
        if progress_cb:
            progress_cb(fi + 1, total, f"{st.name}: patched {n_patched_file}")

    # other files (config, tokenizer, index) — as is
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, out / f.name)

    with open(out / "patch_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    return report
