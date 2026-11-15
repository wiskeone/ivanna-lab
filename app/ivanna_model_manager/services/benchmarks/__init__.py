"""Benchmark integration (v5.3).

Catalog of standard LLM benchmarks + lm-eval-harness task registry sync.
The engine itself lives in an isolated venv (/opt/lmeval-venv) inside the
GPU image and runs as a subprocess (see tasks_benchmarks.py).
"""
