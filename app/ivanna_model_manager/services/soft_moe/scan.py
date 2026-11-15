"""Theme activation scan (port of weight_lab/theme_scan.py, Stage 1).

d̂_theme[l] = mean_h(domain) − mean_h(contrast) — directions in the residual stream
per layer, two positions: last (last input token) and mean (average over tokens).

Validation (lab gates):
  1) AUC of the h projection onto d̂: domain vs contrast > 0.85
     (hold-out, Mann-Whitney without sklearn);
  2) causal writers: (mean_domain − mean_contrast) of out-module outputs,
     projection onto d̂ — layer selection for the Stage 2 patch;
  3) mean inputs ī of out-modules on domain prompts — for the steer patch.

Generalizations vs the lab:
  - suffix matching of out-modules (down_proj/o_proj/out_proj) — any architecture;
  - loading a safetensors checkpoint or GGUF (via transformers gguf_file);
  - load_mode: auto (bf16 + device_map=auto, CPU-offload when VRAM is short),
    int4 (bitsandbytes nf4 — 4B scan on 4GB VRAM), cpu (entirely on CPU).
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

# out-module suffixes (their output goes into the residual stream) — as in the lab
OUT_MODULE_SUFFIXES = ("down_proj", "o_proj", "out_proj")

AUC_GATE = 0.85


# ─────────────────────────── model loading ───────────────────────────

def _pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _gguf_in_dir(path: Path):
    """GGUF file in the directory (or the path itself — a gguf file)?"""
    if path.is_file() and path.suffix.lower() == ".gguf":
        return path.parent, path.name
    if path.is_dir():
        ggufs = sorted(path.glob("*.gguf"))
        if ggufs:
            return path, ggufs[0].name
    return None, None


def load_model_for_scan(model_path: str, load_mode: str = "auto", full_gpu: bool = False):
    """Load a model for forward passes. Returns (model, tokenizer, info).

    load_mode: auto | int4 | cpu (see the module docstring).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_path)
    gguf_dir, gguf_name = _gguf_in_dir(path)
    source = f"{gguf_dir} (gguf: {gguf_name})" if gguf_name else str(path)
    dtype = _pick_dtype()
    kwargs = {"trust_remote_code": True}

    if gguf_name:
        # GGUF input: dequant to bf16 (as in the lab), no device_map — move later if needed
        model = AutoModelForCausalLM.from_pretrained(
            str(gguf_dir), gguf_file=gguf_name, torch_dtype=dtype, **kwargs)
        tok = AutoTokenizer.from_pretrained(str(gguf_dir), gguf_file=gguf_name, **kwargs)
        model.to("cuda" if torch.cuda.is_available() and load_mode != "cpu" else "cpu")
    elif load_mode == "int4":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            # v5.2: offload breaks causal_conv1d kernels (conv modules on CPU);
            # for models fitting in VRAM — full_gpu puts everything on the GPU
            llm_int8_enable_fp32_cpu_offload=not full_gpu)
        device_map = {'': 'cuda:0'} if full_gpu else 'auto'
        model = AutoModelForCausalLM.from_pretrained(
            str(path), quantization_config=bnb, device_map=device_map, **kwargs)
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
    """Decoder layer list with fallback over common names."""
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
        "Decoder layers not found (tried model.layers / transformer.h / "
        "transformer.layers / gpt_neox.layers / blocks) — architecture not "
        "supported by the scan.")


def discover_out_modules(model, suffixes=OUT_MODULE_SUFFIXES):
    """[(layer_idx, module_tail, module)] — modules writing into the residual stream.

    module_tail — module name relative to the layer (last segments), e.g.
    'mlp.down_proj' / 'self_attn.o_proj'; matches the tail in safetensors names.
    """
    out = []
    for li, lyr in enumerate(get_decoder_layers(model)):
        for n, m in lyr.named_modules():
            if n and n.endswith(suffixes) and hasattr(m, "weight"):
                out.append((li, n, m))
    if not out:
        raise RuntimeError("No out-modules found (down_proj/o_proj/out_proj) — "
                           "nothing to patch.")
    return out


# ─────────────────────────── validation math ───────────────────────────

def auc(scores_s, scores_c) -> float:
    """Mann-Whitney U AUC without sklearn (as in the lab), with average ranks for ties.

    The lab had no ties (projections are continuous), but for metric honesty
    on discrete/repeated data — average ranks (scipy 'average').
    """
    all_v = np.concatenate([scores_s, scores_c])
    n = len(all_v)
    order = all_v.argsort(kind="mergesort")
    sorted_v = all_v[order]
    # boundaries of equal-value groups
    new_group = np.r_[True, sorted_v[1:] != sorted_v[:-1]]
    group_idx = new_group.cumsum() - 1                      # dense rank of each element
    bounds = np.r_[np.nonzero(new_group)[0], n]             # group start/end in sort order
    avg_rank_per_group = (bounds[:-1] + bounds[1:] + 1) / 2  # mean 1-based rank of a group
    ranks = np.empty(n, dtype=float)
    ranks[order] = avg_rank_per_group[group_idx]
    n_s, n_c = len(scores_s), len(scores_c)
    r_s = ranks[:n_s].sum()
    return float((r_s - n_s * (n_s + 1) / 2) / (n_s * n_c))


def split_holdout(n_domain: int, n_contrast: int, seed: int, holdout_fraction: float = 0.33):
    """Seeded permutation → (train_domain_idx, train_contrast_idx, test_domain_idx, test_contrast_idx)."""
    rng = np.random.default_rng(seed)
    d_idx = rng.permutation(n_domain)
    c_idx = rng.permutation(n_contrast)
    n_d_te = max(1, round(n_domain * holdout_fraction))
    n_c_te = max(1, round(n_contrast * holdout_fraction))
    return (d_idx[n_d_te:], c_idx[n_c_te:], d_idx[:n_d_te], c_idx[:n_c_te])


# ─────────────────────────── the scan itself ───────────────────────────

class _ScanState:
    """Hook sink: last-position activations keyed by (setname, idx)."""

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
                    torch.nan_to_num(t[0, -1, :].detach().to(torch.float32).cpu())
            return hook

        def mk_pre_hook(li, tail):
            def pre_hook(mod, args):
                if self.current_key is None:
                    return
                x = args[0]
                self.mod_in[(*self.current_key, li, tail)] = \
                    torch.nan_to_num(x[0, -1, :].detach().to(torch.float32).cpu())
            return pre_hook

        for li, tail, m in out_modules:
            handles.append(m.register_forward_hook(mk_hook(li, tail)))
            handles.append(m.register_forward_pre_hook(mk_pre_hook(li, tail)))
        return handles


def _build_inputs(tok, texts):
    """Prompt → input_ids. Chat rendering as in prod; enable_thinking — Qwen only."""
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
               seed: int = 7, holdout_fraction: float = 0.33, full_gpu: bool = False,
               progress_cb=None) -> dict:
    """Full theme scan. Writes npz (d_last/d_mean/mean inputs) and returns a report.

    progress_cb(done, total, step) — optional, for task progress.
    """
    t0 = time.time()
    n_total = len(domain_texts) + len(contrast_texts)
    done = 0

    model, tok, info = load_model_for_scan(model_path, load_mode, full_gpu=full_gpu)
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
            seq = torch.stack([torch.nan_to_num(h[0]) for h in out.hidden_states], dim=0)  # [L+1, seq, hid]
            state.h_last[(name, i)] = seq[:, -1, :].to(torch.float32).cpu().numpy()
            state.h_mean[(name, i)] = seq.mean(dim=1).to(torch.float32).cpu().numpy()
            done += 1
            report_progress(f"forward {name}[{i + 1}/{len(texts)}]")

    scan_set("domain", list(domain_texts))
    scan_set("contrast", list(contrast_texts))
    for h in handles:
        h.remove()
    state.current_key = None

    # ── directions over the train half (hold-out honesty) ──
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

    # save all scan data at once (lab pitfall: artifacts — before the analysis sections)
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

    # ── AUC per layer (always at the last position — the "readiness to answer" state) ──
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

    # ── writers: (mean_domain − mean_contrast) of out-module outputs @ last ──
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
        proj = float(diff @ d)
        nrm = float(np.linalg.norm(diff))
        # bf16 outliers on synthetic formats give inf/nan — skip them honestly
        if not np.isfinite(proj) or not np.isfinite(nrm):
            writers[f"L{li:02d}:{tail}"] = {"layer": li, "module": tail,
                                            "proj": None, "norm": None, "nan": True}
            continue
        writers[f"L{li:02d}:{tail}"] = {
            "layer": li, "module": tail,
            "proj": round(proj, 2),
            "norm": round(nrm, 2),
        }
    top_writers = sorted((w for w in writers.values() if w["proj"] is not None),
                         key=lambda w: -abs(w["proj"]))[:16]

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

    # free GPU before patching (del in a closure trips ruff F821 — null out instead)
    model = tok = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def auto_layers_from_writers(scan_report: dict, thr_factor: float = 0.6):
    """Layers to patch based on writers: sum of |proj| of a layer's modules,
    threshold = 0.6 · median of the top-8 (as in the lab)."""
    per_layer = {}
    for w in scan_report.get("writers", {}).values():
        if w.get("proj") is None or w.get("nan"):
            continue  # nan projections (bf16 outliers) are excluded from layer selection
        per_layer[w["layer"]] = per_layer.get(w["layer"], 0.0) + abs(w["proj"])
    if not per_layer:
        return []
    ranked = sorted(per_layer.items(), key=lambda kv: -kv[1])
    thr = float(np.median([v for _, v in ranked[:8]])) * thr_factor
    picked = sorted(l for l, v in per_layer.items() if v >= thr)
    if len(picked) < 3:
        # synthetic formats (tool-call etc.) give a flat profile —
        # take the top-3 layers instead of an empty list
        picked = sorted(l for l, v in ranked[:3])
    return picked
