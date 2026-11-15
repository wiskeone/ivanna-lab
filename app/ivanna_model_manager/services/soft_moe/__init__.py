"""Soft MoE: thematic experts without fine-tuning.

Port of the technique from weight_lab (Weight-Space Programming, Phase A "soft MoE"):
a domain direction d̂ is extracted from a dense model via an activation scan
(difference of mean hidden states on domain vs contrast prompts), baked into
late-layer output matrices with a steer patch; an external embedding router
picks the variant per request.

Modules:
  prompts    — theme prompt sets (txt files, cleaning, filters, dedup)
  validation — prompt set gate: bge cosines cross/intra + length balance
  scan       — activation scan: d̂, hold-out AUC, writers, mean inputs ī
  patching   — steer/amplify safetensors patch (+ GGUF→safetensors conversion)
  router     — theme centroids + margin rule + accuracy test
  evalgate   — light eval gate: base vs expert on fresh prompts
  hf_push    — expert publishing to Hugging Face with an auto model card
"""
