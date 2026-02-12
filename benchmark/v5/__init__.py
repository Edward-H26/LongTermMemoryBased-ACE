"""CL-bench v5 benchmark modules.

V5 adds durable progress journaling, crash-safe resume, fail-soft
memory operations, full-pipeline metered cost reporting, OpenAI
billed-cost reconciliation, and post-run JSONL sanitization.

Modules
-------
infer_baseline      – GPT-5.1 baseline (v5 naming, stratified sampling).
infer_ace           – ACE inference with progress journal and fail-soft memory.
complete_pipeline   – Post-inference orchestrator with retry and parity checks.
preflight           – Pre-run validation with unsanitized-JSONL detection.
backfill_cost       – Retroactive cost-report regeneration for v4/v5 artifacts.
run                 – Top-level orchestrator with run metadata and sanitization.
"""
