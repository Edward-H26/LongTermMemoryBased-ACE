"""CL-bench v6 benchmark modules.

V6 isolates artifacts from v5, enforces Gemini-first ACE internals,
restores learning defaults, and adds per-task diagnostics.

Modules
-------
infer_baseline      - GPT-5.1 baseline with v6 artifact naming.
infer_ace           - ACE inference with strict routing, v6 quality gate tuning.
complete_pipeline   - Post-inference orchestrator with diagnostics before sanitize.
preflight           - Pre-run validation, estimation, and smoke orchestration.
policy_replay       - Planner policy replay from graded outcomes.
task_diagnostics    - Per-task baseline vs ACE failure and transition analysis.
run                 - Top-level orchestrator with preflight and report pipeline.
"""
