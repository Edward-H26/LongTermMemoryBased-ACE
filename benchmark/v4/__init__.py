"""CL-bench v4 benchmark modules.

V4 adds dual memory (local + global), context-level parallel execution,
capped-output retry, per-task step scoring, and dual preflight validation.

Modules
-------
infer_baseline      – GPT-5.1 baseline with capped-output retry.
infer_ace           – ACE inference with dual memory and step scoring.
complete_pipeline   – Post-inference orchestrator with parallel eval.
preflight           – Pre-run static + smoke validation.
run                 – Top-level orchestrator with preflight gating.
"""
