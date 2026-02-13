# Benchmark Results Structure (Canonical)

This document defines the canonical artifact layout and schema conventions for versioned benchmark outputs.

## 1. Canonical directory tree

All benchmark outputs are version-scoped:

- `benchmark/results/v1`
- `benchmark/results/v2`
- `benchmark/results/v3`
- `benchmark/results/v4`
- `benchmark/results/v5`

Do not write new v4 and v5 artifacts into unversioned `benchmark/results/`.

## 2. Core artifact families

### 2.1 Inference JSONL

- `baseline_v*.jsonl`
- `ace_v*.jsonl`

Each row contains:

- `task_id`
- `messages`
- `model_output`
- `rubrics`
- `metadata` including `task_id`, `context_id`, `context_category`
- `metrics` with latency and token fields plus version-specific diagnostics

Sanitized publish schema for inference JSONL:

- Keep: `task_id`, `metadata`, `metrics`
- Remove: `messages`, `model_output`, `rubrics`
- Nested metrics removals:
  - `metrics.step_scoring.steps`
  - `metrics.memory_error`

### 2.2 ACE auxiliary metrics JSON

- `ace_v*_metrics.json`

Schema:

- `summary`:
  `total_calls`, `total_prompt_tokens`, `total_completion_tokens`, `total_tokens`,
  `avg_latency_ms`, `median_latency_ms`, `p95_latency_ms`
- `records`: per-call usage rows from ACE auxiliary model calls

### 2.3 Graded JSONL

- `baseline_v*_graded.jsonl`
- `ace_v*_graded.jsonl`

Adds grading fields:

- `grading_rationale`
- `requirement_status`
- `score` (`0` or `1`)

Sanitized publish schema for graded JSONL:

- Keep: `task_id`, `metadata`, `metrics`, `score`, `requirement_status`
- Remove: `messages`, `model_output`, `rubrics`, `grading_rationale`

### 2.4 Error-analysis JSONL

- `baseline_v*_graded_errors.jsonl`
- `ace_v*_graded_errors.jsonl`

Adds:

- `error_classification.task_error_types`
- `error_classification.per_rubric_classification`

Error types are non-exclusive.

Sanitized publish schema for graded error JSONL:

- Keep: `task_id`, `metadata`, `metrics`, `score`, `requirement_status`
- Keep: `error_classification.task_error_types`
- Remove:
  - `messages`
  - `model_output`
  - `rubrics`
  - `grading_rationale`
  - `error_classification.per_rubric_classification`

### 2.5 Eval and error phase metrics JSON (v5)

- `baseline_v5_graded_eval_metrics.json`
- `ace_v5_graded_eval_metrics.json`
- `baseline_v5_graded_errors_error_metrics.json`
- `ace_v5_graded_errors_error_metrics.json`

Eval metrics schema:

- `model`
- `total_calls`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `api_fail_count`
- `json_fail_count`
- `graded_ok_count`
- `no_output_count`
- `started_at`
- `ended_at`
- `wall_seconds`

Error metrics schema:

- `model`
- `total_calls`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `classified_count`
- `fallback_count`
- `api_fail_count`
- `started_at`
- `ended_at`
- `wall_seconds`

### 2.6 Comparison report markdown and JSON

- `comparison_report_v*.md`
- `comparison_report_v*.json`

Markdown tables:

1. solving rates
2. error distribution
3. inference token/latency/estimated cost
4. per-category token/latency
5. runtime diagnostics
5B. planner policy diagnostics
6. full-pipeline actual metered cost
7. OpenAI billed reconciliation

JSON summary keeps existing keys and adds:

- `full_pipeline_cost_metered`
- `cost_breakdown_by_phase`
- `openai_billed_reconciliation`
- `cost_reconciliation_status`
- `baseline_planner_diagnostics`
- `ace_planner_diagnostics`

### 2.7 Deterministic subset manifests

- `subset_manifest_v*_seed<seed>_n<samples>.json`

Schema:

- `dataset`
- `split`
- `seed`
- `max_samples`
- `sampling_strategy`
- `selected_count`
- `created_at`
- `task_ids` (ordered)

### 2.8 Preflight reports

- `preflight_v4*.json`
- `preflight_v5*.json`

Top-level fields:

- `generated_at`
- `preflight_mode`
- `status`
- `blocking_issues`
- `warnings`
- `config`
- `static_checks`
- `estimate`
- `smoke`

### 2.9 Smoke output directories

Examples:

- `benchmark/results/v4/smoke_v4_seed42_n5/`
- `benchmark/results/v5/smoke_v5_seed42_n5/`

Smoke directories mirror full pipeline output for a 3 to 5 task subset.

## 3. V5 durability and run metadata artifacts

### 3.1 ACE progress journal

- `<ace_output>.progress.jsonl`

Per-row fields:

- `task_id`
- `index` (manifest order index)
- `row` (full inference result row)

### 3.2 ACE completion marker

- `<ace_output>.complete.json`

Fields:

- `generated_at`
- `output_path`
- `progress_path`
- `selected_count`
- `completed_count`
- `is_complete`

### 3.3 Run metadata

- `run_v5_meta.json`

Fields:

- `run_id`
- `started_at`
- `ended_at`
- `manifest`
- `seed`
- `max_samples`
- `sampling_strategy`
- `memory_scope`
- `output_dir`
- `phases`

`phases` uses start and end timestamps for:

- `inference`
- `evaluation`
- `error_analysis`
- `compare`
- optional orchestration phases such as `post_pipeline`

### 3.4 Planner replay and policy state artifacts

- `policy_replay_v5.json`
- `planner_policy_baseline_v5.json`
- `planner_policy_ace_v5.json`
- optional runtime planner state path if configured with `ACE_PLANNER_STATE_PATH`

`policy_replay_v5.json` summarizes:

- replay mode (`both`, `baseline`, or `ace`)
- rows processed per stream
- update counts per stream
- mean replay reward per stream
- action-level replay update counts

## 4. Cost backfill artifacts

Backfill utility outputs:

- `cost_backfill_v4.md`
- `cost_backfill_v4.json`
- `cost_backfill_v5.md`
- `cost_backfill_v5.json`

Backfill files are supplemental and non-destructive by default.

## 5. Publish-safe allowlist

Only these artifact name patterns are publish-safe:

- `comparison_report_*.md`
- `comparison_report_*.json`
- `subset_manifest_*.json`
- `preflight_*.json`
- `ace_*_metrics.json`
- `policy_replay_v5.json`
- `run_*_meta.json`
- `cost_backfill_*.json`
- `cost_backfill_*.md`
- `*.complete.json`

Raw JSONL and progress journals should remain local by default.
