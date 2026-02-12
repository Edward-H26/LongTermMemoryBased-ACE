# Refactoring and Cleanup Changelog

This document records all structural changes, security cleanups, and code reorganization applied to the repository.

---

## Phase 1: Security and Risk Cleanup

### 1.1 Secret Pattern Removal

**Problem:** `docs/SETUP.md` contained placeholder strings that resembled real API keys and database credentials. GitHub secret scanning flagged these as potential credential leaks across multiple commits.

**Resolution:** Rebased the entire commit history from root to replace all placeholder credential patterns with clearly-fake examples:

| Field | Before | After |
|-------|--------|-------|
| `NEO4J_URI` | Realistic-looking URI | `neo4j+s://abcdef12.databases.neo4j.io` |
| `NEO4J_PASSWORD` | Realistic password string | `your-generated-password` |
| API keys | Realistic key formats | `your-gemini-api-key-here`, `your-openai-api-key-here` |

All commits were rewritten to ensure no commit in the history contains flagged patterns.

### 1.2 JSONL Sanitization

**Problem:** Raw inference JSONL files in `benchmark/results/` contained full model outputs, messages, rubrics, and grading rationales — data that should not be published.

**Resolution:**
- Ran `benchmark/sanitize.py` (formerly `benchmark/sanitize_results.py`) over all result versions in-place.
- Sanitized files retain only: `task_id`, `metadata`, `metrics`, `score`, `requirement_status`, and `error_classification.task_error_types`.
- Removed: `messages`, `model_output`, `rubrics`, `grading_rationale`, `error_classification.per_rubric_classification`, `metrics.step_scoring.steps`, `metrics.memory_error`.

### 1.3 `publish_safe/` Directory Removal

**Problem:** The `benchmark/publish_safe/` directory and `benchmark/publish_safe_artifacts.py` script were an earlier attempt at creating sanitized copies for publishing. With in-place sanitization implemented, these were redundant and added confusion.

**Resolution:**
- Deleted `benchmark/publish_safe/` directory and all contents.
- Deleted `benchmark/publish_safe_artifacts.py`.
- Removed from `.gitignore` and all documentation references.

### 1.4 `raw_results_backup/` Removal

**Problem:** `benchmark/raw_results_backup/` contained unsanitized copies of inference results created by the `maybe_backup_file()` mechanism in `sanitize_results.py`. These raw files contained the same sensitive data that sanitization was meant to remove.

**Resolution:**
- Deleted `benchmark/raw_results_backup/` directory and all contents from disk and git.
- Removed `maybe_backup_file()` function from `sanitize.py`.
- Removed `--raw-backup-dir` CLI argument from `sanitize.py`.
- Removed `shutil` import that was only used for backups.
- Removed all backup-related references from documentation (`README.md`, `docs/SETUP.md`, `docs/RESULTS_STRUCTURE.md`).
- Removed backup paths from `.gitignore`.

---

## Phase 2: Benchmark Module Restructuring

### 2.1 Problem Statement

The `benchmark/` directory contained 27 Python files in a flat structure. Version-specific files used filename suffixes (`_v3`, `_v4`, `_v5`) for disambiguation, making it difficult to:

1. Identify which files belong to which benchmark version.
2. Navigate the codebase when working on a specific version.
3. Maintain shared logic that was copy-pasted across versions.

Approximately 600–800 lines of utility code were duplicated across version-specific files:

| Function | Duplicated In | Count |
|----------|--------------|-------|
| `load_jsonl` / `append_jsonl` | All inference + eval + error + compare files | 11× |
| `get_task_id` | All inference + eval files | 12× |
| `call_api` | All inference files (2 variants) | 7× |
| `utc_now_iso` | Orchestrators + inference files | 6× |
| `META_STRATEGY_SEEDS` | ACE inference v3–v5 | 4× |
| `format_guidance` / `inject_guidance` | ACE inference v3–v5 | 4× |
| `clear_neo4j_all` | ACE inference + orchestrators | 5× |
| `infer_with_retry` | Baseline inference v3–v5 | 4× |
| `_safe_env_int` / `_safe_env_float` | ACE inference v4–v5 + orchestrators | 4× |

### 2.2 New Directory Structure

#### Before

```
benchmark/
├── infer_baseline.py
├── infer_ace.py
├── infer_baseline_v2.py
├── infer_ace_direct_v2.py
├── infer_baseline_v3.py
├── infer_ace_direct_v3.py
├── infer_baseline_v4.py
├── infer_ace_direct_v4.py
├── infer_baseline_v5.py
├── infer_ace_direct_v5.py
├── run_v3.py
├── run_v4.py
├── run_v5.py
├── preflight_v4.py
├── preflight_v5.py
├── complete_v3_pipeline.py
├── complete_v4_pipeline.py
├── complete_v5_pipeline.py
├── backfill_cost_v5.py
├── sanitize_results.py
├── eval.py
├── error_analysis.py
├── compare.py
├── costing.py
├── metrics.py
├── sampling.py
└── monitor_v3.sh
```

#### After

```
benchmark/
├── common/                  # Shared utilities extracted from version files
│   ├── __init__.py          # Re-exports all shared functions
│   ├── io.py                # load_jsonl, append_jsonl, load_json, write_json
│   ├── identifiers.py       # get_task_id, get_context_id, get_context_category
│   ├── api.py               # call_api (capped-output retry), infer_with_retry
│   ├── env.py               # utc_now_iso, safe_env_int, safe_env_float, safe_env_bool
│   ├── neo4j_utils.py       # clear_neo4j_all
│   ├── ace_shared.py        # META_STRATEGY_SEEDS, format/inject guidance, seed memory
│   ├── pipeline.py          # Orchestrator helpers (subprocess env, run metadata)
│   └── llm_utils.py         # parse_response_text, extract_usage
├── v1/
│   ├── __init__.py
│   ├── infer_baseline.py
│   └── infer_ace.py
├── v2/
│   ├── __init__.py
│   ├── infer_baseline.py
│   └── infer_ace.py
├── v3/
│   ├── __init__.py
│   ├── infer_baseline.py
│   ├── infer_ace.py
│   ├── run.py
│   └── complete_pipeline.py
├── v4/
│   ├── __init__.py
│   ├── infer_baseline.py
│   ├── infer_ace.py
│   ├── run.py
│   ├── preflight.py
│   └── complete_pipeline.py
├── v5/
│   ├── __init__.py
│   ├── infer_baseline.py
│   ├── infer_ace.py
│   ├── run.py
│   ├── preflight.py
│   ├── complete_pipeline.py
│   └── backfill_cost.py
├── eval.py                  # Version-agnostic rubric evaluation
├── error_analysis.py        # Version-agnostic error classification
├── compare.py               # Version-agnostic comparison reports
├── costing.py               # Token-cost and billed reconciliation
├── metrics.py               # Token/latency collection helpers
├── sampling.py              # Seeded subset sampling + manifest reuse
├── sanitize.py              # JSONL sanitization for publishing
└── results/                 # Output artifacts (unchanged)
```

### 2.3 File Move Mapping

All moves were performed with `git mv` to preserve history.

| Original Path | New Path |
|--------------|----------|
| `benchmark/infer_baseline.py` | `benchmark/v1/infer_baseline.py` |
| `benchmark/infer_ace.py` | `benchmark/v1/infer_ace.py` |
| `benchmark/infer_baseline_v2.py` | `benchmark/v2/infer_baseline.py` |
| `benchmark/infer_ace_direct_v2.py` | `benchmark/v2/infer_ace.py` |
| `benchmark/infer_baseline_v3.py` | `benchmark/v3/infer_baseline.py` |
| `benchmark/infer_ace_direct_v3.py` | `benchmark/v3/infer_ace.py` |
| `benchmark/run_v3.py` | `benchmark/v3/run.py` |
| `benchmark/complete_v3_pipeline.py` | `benchmark/v3/complete_pipeline.py` |
| `benchmark/infer_baseline_v4.py` | `benchmark/v4/infer_baseline.py` |
| `benchmark/infer_ace_direct_v4.py` | `benchmark/v4/infer_ace.py` |
| `benchmark/run_v4.py` | `benchmark/v4/run.py` |
| `benchmark/preflight_v4.py` | `benchmark/v4/preflight.py` |
| `benchmark/complete_v4_pipeline.py` | `benchmark/v4/complete_pipeline.py` |
| `benchmark/infer_baseline_v5.py` | `benchmark/v5/infer_baseline.py` |
| `benchmark/infer_ace_direct_v5.py` | `benchmark/v5/infer_ace.py` |
| `benchmark/run_v5.py` | `benchmark/v5/run.py` |
| `benchmark/preflight_v5.py` | `benchmark/v5/preflight.py` |
| `benchmark/complete_v5_pipeline.py` | `benchmark/v5/complete_pipeline.py` |
| `benchmark/backfill_cost_v5.py` | `benchmark/v5/backfill_cost.py` |
| `benchmark/sanitize_results.py` | `benchmark/sanitize.py` |

### 2.4 Module Path Migration

All `python -m` commands, subprocess calls, and import references were updated:

| Old Module Path | New Module Path |
|----------------|-----------------|
| `benchmark.infer_baseline` | `benchmark.v1.infer_baseline` |
| `benchmark.infer_ace` | `benchmark.v1.infer_ace` |
| `benchmark.infer_baseline_v2` | `benchmark.v2.infer_baseline` |
| `benchmark.infer_ace_direct_v2` | `benchmark.v2.infer_ace` |
| `benchmark.infer_baseline_v3` | `benchmark.v3.infer_baseline` |
| `benchmark.infer_ace_direct_v3` | `benchmark.v3.infer_ace` |
| `benchmark.run_v3` | `benchmark.v3.run` |
| `benchmark.complete_v3_pipeline` | `benchmark.v3.complete_pipeline` |
| `benchmark.infer_baseline_v4` | `benchmark.v4.infer_baseline` |
| `benchmark.infer_ace_direct_v4` | `benchmark.v4.infer_ace` |
| `benchmark.run_v4` | `benchmark.v4.run` |
| `benchmark.preflight_v4` | `benchmark.v4.preflight` |
| `benchmark.complete_v4_pipeline` | `benchmark.v4.complete_pipeline` |
| `benchmark.infer_baseline_v5` | `benchmark.v5.infer_baseline` |
| `benchmark.infer_ace_direct_v5` | `benchmark.v5.infer_ace` |
| `benchmark.run_v5` | `benchmark.v5.run` |
| `benchmark.preflight_v5` | `benchmark.v5.preflight` |
| `benchmark.complete_v5_pipeline` | `benchmark.v5.complete_pipeline` |
| `benchmark.backfill_cost_v5` | `benchmark.v5.backfill_cost` |
| `benchmark.sanitize_results` | `benchmark.sanitize` |

### 2.5 Common Module Contents

The `benchmark/common/` package provides canonical implementations of shared utility functions. Each submodule focuses on one concern:

| Module | Exports | Source |
|--------|---------|--------|
| `io.py` | `load_jsonl`, `append_jsonl`, `load_json`, `write_json` | Extracted from `infer_baseline_v5.py` |
| `identifiers.py` | `get_task_id`, `get_context_id`, `get_context_category` | Extracted from `infer_ace_direct_v5.py` |
| `api.py` | `call_api`, `infer_with_retry` | Extracted from `infer_baseline_v5.py` (v4+ variant with `max_completion_tokens`) |
| `env.py` | `utc_now_iso`, `safe_env_int`, `safe_env_float`, `safe_env_bool` | Extracted from `run_v5.py` |
| `neo4j_utils.py` | `clear_neo4j_all` | Extracted from `infer_ace_direct_v5.py` |
| `ace_shared.py` | `META_STRATEGY_SEEDS`, `format_guidance`, `inject_guidance`, `seed_memory_if_empty`, `filter_transferable_lessons`, `merge_retrieved_bullets`, `count_seed_and_learned` | Extracted from `infer_ace_direct_v5.py` |
| `pipeline.py` | `count_lines`, `is_writable_dir`, `build_subprocess_env`, `init_run_meta`, `mark_phase_start`, `mark_phase_end` | Extracted from `run_v5.py` |
| `llm_utils.py` | `parse_response_text`, `extract_usage` | Extracted from `infer_ace_direct_v5.py` |

All exports are also available from `benchmark.common` directly via `__init__.py` re-exports.

### 2.6 Subprocess and Import Updates

The following files had internal subprocess calls and imports updated to use new module paths:

| File | Changes |
|------|---------|
| `benchmark/v3/run.py` | Docstring + 3 subprocess module paths |
| `benchmark/v4/run.py` | 1 import + 3 subprocess module paths |
| `benchmark/v5/run.py` | 1 import + 3 subprocess + 1 sanitize module path |
| `benchmark/v4/preflight.py` | 3 `REQUIRED_MODULES` entries + subprocess in `run_smoke` |
| `benchmark/v5/preflight.py` | 3 `REQUIRED_MODULES` entries + subprocess in `run_smoke` |

### 2.7 Documentation Updates

All documentation files were updated to reflect the new structure:

| File | Changes |
|------|---------|
| `README.md` | Project structure tree replaced; all `python -m` command paths updated; documentation section updated |
| `docs/ALGORITHM.md` | §1 stream listing, §6.1–6.7 section headings, all code block commands, prose module references |
| `docs/SETUP.md` | All command examples across V5/V4/V3 sections, file path references, troubleshooting section |
| `docs/REFACTORING.md` | This file (new) |

### 2.8 Version-Agnostic Files

These files remained at the `benchmark/` top level because they are shared across all versions:

- `eval.py` — Rubric-based evaluation (version-agnostic)
- `error_analysis.py` — Error type classification (version-agnostic)
- `compare.py` — Side-by-side comparison report generation
- `costing.py` — Token-cost and billed reconciliation utilities
- `metrics.py` — Token/latency collection helpers
- `sampling.py` — Seeded subset sampling and manifest reuse
- `sanitize.py` — JSONL sanitization for publishing (renamed from `sanitize_results.py`)
- `monitor_v3.sh` — Shell monitoring script

---

## Future Work

### Incremental Import Migration

Version-specific files (`v1/`–`v5/`) currently retain local copies of shared functions for stability. A future pass can replace these with imports from `benchmark.common`:

```python
# Before (local copy in each file):
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

# After (import from common):
from benchmark.common import load_jsonl
```

This migration should be done incrementally with testing to verify each version's pipeline still produces identical results.
