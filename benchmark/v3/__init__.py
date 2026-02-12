"""CL-bench v3 benchmark modules.

V3 adds deterministic seeded sampling (via ``benchmark.sampling``)
and the quality-gated online update mechanism.

Modules
-------
infer_baseline      – GPT-5.1 baseline with deterministic sampling.
infer_ace           – ACE inference with quality-gated learning.
complete_pipeline   – Post-inference orchestrator (eval → error → compare).
run                 – Top-level orchestrator.
"""
