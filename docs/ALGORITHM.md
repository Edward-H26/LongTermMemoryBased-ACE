# LTMBSE ACE Algorithm (V5 Accurate Specification)

## 1. System Overview

The project combines a LangGraph reasoning agent with ACE memory and a CL-bench benchmarking pipeline.

Core runtime graph:

```text
START -> router -> planner -> solver -> critic -> ace_learning -> END
```

Core benchmark streams:

- Baseline v3: `benchmark/v3/infer_baseline.py`
- ACE direct v3: `benchmark/v3/infer_ace.py`
- One-command orchestration: `benchmark/v3/run.py`
- Baseline v4: `benchmark/v4/infer_baseline.py`
- ACE direct v4: `benchmark/v4/infer_ace.py`
- One-command orchestration (v4): `benchmark/v4/run.py`
- Baseline v5: `benchmark/v5/infer_baseline.py`
- ACE direct v5: `benchmark/v5/infer_ace.py`
- Policy replay v5: `benchmark/v5/policy_replay.py`
- One-command orchestration (v5): `benchmark/v5/run.py`

## 2. Memory Model

ACE memory stores structured bullets in Neo4j via `Neo4jMemoryStore`.
Each bullet tracks content and usage signals, including semantic, episodic, and procedural strengths.

Important operational properties:

1. Context isolation:
   bullets are retrieved by `context_scope_id` to avoid cross-context contamination.
2. Grow-and-refine updates:
   delta application supports new, update, and remove operations.
3. Decay-aware retrieval:
   ranking blends relevance, strength, and memory type priority.

Primary code paths:

- `src/ace_memory.py`
- `src/storage.py`

## 3. Retrieval Scoring Defaults

Current code defaults are loaded from environment in `src/ace_memory.py`:

- `ACE_MEMORY_BASE_STRENGTH = 100.0`
- `ACE_WEIGHT_RELEVANCE = 0.60`
- `ACE_WEIGHT_STRENGTH = 0.20`
- `ACE_WEIGHT_TYPE = 0.20`
- `ACE_MIN_LEARNED_BULLETS = 2`
- `ACE_SEED_BULLET_PENALTY = 0.25`
- `ACE_LEARNED_BULLET_BONUS = 0.08`

These are framework defaults. CL-bench experiments may tune relevance and strength weights separately.

## 4. Quality Gate (Current Defaults)

Quality gate logic is implemented in `src/ace_components.py` and reused by runtime plus benchmark v5.

## 4.1 Config and Defaults

`QualityGateConfig` defaults:

- `gate_score_min = 0.60`
- `lesson_score_min = 0.55`
- `overlap_min = 0.05`
- `confidence_min = 0.70`
- `max_accepted_lessons = 4`

These can be overridden by:

- `ACE_QG_GATE_SCORE_MIN`
- `ACE_QG_LESSON_SCORE_MIN`
- `ACE_QG_OVERLAP_MIN`
- `ACE_QG_CONFIDENCE_MIN`
- `ACE_QG_MAX_ACCEPTED_LESSONS`

## 4.2 Scoring Components

For each extracted lesson:

1. Relevance score:

```text
lexical_jaccard = |tokens(question) ∩ tokens(lesson)| / |tokens(question) ∪ tokens(lesson)|
precision = |tokens(question) ∩ tokens(lesson)| / |tokens(lesson)|
recall = |tokens(question) ∩ tokens(lesson)| / |tokens(question)|
f1_overlap = 2 * precision * recall / (precision + recall)
coverage = |tokens(question) ∩ tokens(lesson)| / min(|tokens(question)|, |tokens(lesson)|)
relevance_score = 0.50 * lexical_jaccard + 0.30 * f1_overlap + 0.20 * coverage
```

2. Lesson quality score:

```text
token_score = min(token_count / 20, 1.0) * 0.6
tags_score = 0.2 if tags exist else 0.0
type_score = 0.2 if type in {success, failure, domain, tool} else 0.0
lesson_score = min(token_score + tags_score + type_score, 1.0)
```

3. Lesson confidence score:

```text
verifier_score = step_summary.overall_confidence if available
                 else mean(lesson.confidence) when present
                 else 0.5 * lesson_score + 0.5 * relevance_score
confidence_score = 0.45 * lesson_score + 0.40 * relevance_score + 0.15 * verifier_score
```

4. Lesson acceptance filter:

```text
accept if relevance_score >= overlap_min
      and lesson_score >= lesson_score_min
      and confidence_score >= confidence_min
      and content is non-empty
```

5. Top-k cap:

Accepted candidates are sorted by `(confidence_score, lesson_score, relevance_score)` descending, then truncated to `max_accepted_lessons`.

## 4.3 Task-Level Gate

After accepted lessons are selected:

```text
output_score = 1.0 if model output is non-empty else 0.0
accepted_quality_avg = mean(accepted lesson scores)
accepted_confidence_avg = mean(accepted lesson confidence scores)
gate_score = 0.35 * output_score + 0.35 * accepted_quality_avg + 0.30 * accepted_confidence_avg
should_apply_update = accepted_lessons non-empty and gate_score >= gate_score_min
```

Only if `should_apply_update` is true are accepted lessons curated into memory deltas.

## 4.4 Diagnostics Schema

Diagnostics returned by `apply_quality_gate` and attached to outputs include:

- `config`
- `output_valid`
- `output_score`
- `accepted_quality_avg`
- `accepted_confidence_avg`
- `accepted_relevance_avg`
- `step_confidence`
- `gate_score`
- `should_apply_update`
- `num_lessons_input`
- `num_lessons_accepted`
- `num_lessons_rejected`
- `rejection_counts`
- `rejected_examples`

In ACE v3 benchmark rows, this appears at:

- `metrics.quality_gate`

## 5. Deterministic Sampling and Manifest Semantics (V3)

Implemented in `benchmark/sampling.py`.

## 5.1 Manifest Purpose

Manifest guarantees baseline and ACE process the same ordered subset.
This is required for fair comparison.

## 5.2 Manifest Schema

Manifest fields:

- `dataset`
- `split`
- `seed`
- `max_samples`
- `sampling_strategy`
- `selected_count`
- `created_at`
- `task_ids` (ordered)

## 5.3 Selection Logic

If manifest file exists:

- load manifest,
- reconstruct subset by `task_ids` order.

If manifest does not exist:

- select subset using strategy:
  `task_random` uses deterministic index sampling,
  `context_dense` prioritizes contexts with at least two tasks then fills to target size,
- sort indices,
- build subset in sorted-index order,
- persist manifest when a path is provided.

## 6. Benchmark V3 Pipelines

## 6.1 Baseline v3 (`benchmark/v3/infer_baseline.py`)

Flow:

1. Load dataset.
2. Resolve subset with seed and optional manifest.
3. Resume by existing output `task_id` set when not clearing results.
4. Call OpenAI model per task.
5. Write one row per task:
   - `task_id`, `messages`, `model_output`, `rubrics`, `metadata`, `metrics`.

## 6.2 ACE direct v3 (`benchmark/v3/infer_ace.py`)

Flow:

1. Load deterministic subset through same sampling module.
2. Group pending tasks by `context_id`.
3. Retrieve context-scoped memory bullets.
4. Inject guidance before the last user message.
5. Call solver model.
6. Reflect lessons with Gemini reflector.
7. Apply quality gate.
8. Apply accepted lessons to memory only when gate passes.
9. Write row with quality-gate diagnostics and ACE delta summary.

Row-level metrics include:

- token and latency fields,
- `reflector_tokens`,
- `num_bullets_retrieved`,
- `num_lessons_extracted`,
- `num_lessons_accepted`,
- `ace_delta`,
- `quality_gate`.

## 6.3 One-command v3 orchestration (`benchmark/v3/run.py`)

Default command:

```bash
python -m benchmark.v3.run \
  --manifest benchmark/results/v3/subset_manifest_v3_seed42_n200.json \
  --max-samples 200 \
  --seed 42
```

Default behavior:

1. Clear v3 outputs (unless `--no-clear-results`).
2. Clear Neo4j (unless `--no-clear-db`).
3. Run baseline v3 and ACE v3 inference in parallel.
4. Enforce completion check against requested sample count.
5. If `--with-report` (default true), call `benchmark.v3.complete_pipeline --skip-wait`.

`benchmark/v3/complete_pipeline.py` currently uses fixed `TARGET = 200`.
For non-200 subset runs, use `--no-with-report` and run eval/error/compare manually.

## 6.4 Benchmark v4 (`benchmark/v4/infer_baseline.py`, `benchmark/v4/infer_ace.py`)

V4 keeps deterministic sampling and adds:

1. Capped-output retry:
   if a response is empty and completion tokens hit cap, retry with a higher cap.
2. Memory scope modes:
   `local`, `global`, `hybrid`.
3. Dual-memory updates:
   local updates follow quality gate pass;
   global updates require gate pass plus `gate_score >= ACE_GLOBAL_GATE_SCORE_MIN`.
4. Context-level parallel execution:
   tasks are processed per-context concurrently, while final write order remains manifest order.
5. Step scoring diagnostics:
   each output records process-level step scoring summary.

New v4 metrics include:

- `num_local_bullets_retrieved`
- `num_global_bullets_retrieved`
- `num_seed_bullets_retrieved`
- `num_learned_bullets_retrieved`
- `memory_scope_mode`
- `completion_capped`
- `empty_output_retry_count`
- `step_scoring`

## 6.5 One-command v4 orchestration (`benchmark/v4/run.py`)

Default command:

```bash
python -m benchmark.v4.run \
  --manifest benchmark/results/v4/subset_manifest_v4_seed42_n200.json \
  --max-samples 200 \
  --seed 42 \
  --memory-scope hybrid
```

Default behavior:

1. Clear v4 outputs (unless `--no-clear-results`).
2. Clear Neo4j (unless `--no-clear-db`).
3. Run baseline v4 and ACE v4 inference in parallel.
4. Enforce completion check against requested sample count.
5. If `--with-report` (default true), call `benchmark.v4.complete_pipeline --output-dir ... --skip-wait --max-samples ...`.

### 6.5.1 Dual preflight support (`benchmark/v4/preflight.py`)

`run_v4` supports a preflight branch controlled by:

- `--preflight-mode {off,static,smoke,both}` (default `off`)
- `--smoke-samples {3,4,5}` (default `5`)
- `--smoke-output-dir` (default `benchmark/results/v4/smoke_v4`)
- `--estimate-source {auto,v4,v3,heuristic}` (default `auto`)
- `--preflight-report` (default `benchmark/results/v4/preflight_v4.json`)

Preflight flow:

1. `static`: validate env, module availability, manifest/output path usability, and build low/base/high cost-time estimate without benchmark API calls.
2. `smoke`: execute full mini pipeline on `3` to `5` tasks (inference, eval, error analysis, compare) in smoke output dir.
3. `both`: run static first, then smoke if static returns no blocking issues.
4. Write unified JSON report with checks, assumptions, estimates, measured smoke metrics, and scaled full-run projection.

### 6.5.2 Sample-size safety guard

`benchmark/sampling.py` now rejects `max_samples <= 0` with a clear error message:

- valid: `max_samples` is `None` or any positive integer
- invalid: `max_samples <= 0`

This prevents accidental full-dataset execution from invalid command arguments.

## 6.6 Benchmark v5 reliability and planner extensions (`benchmark/v5/infer_baseline.py`, `benchmark/v5/infer_ace.py`)

V5 adds fault tolerance, resumability, and a max-quality hybrid planner loop:

1. Storage reliability:
   `src/storage.py` retries Neo4j load/save and handles both `Neo4jError` and `DriverError` paths.
2. Session recovery:
   on `SessionExpired`, driver reset/reconnect is triggered when enabled.
3. Fail-soft memory persistence:
   `src/ace_memory.py` records persist status (`ok`, retries, error) instead of forcing fatal exceptions by default.
4. Durable per-task journaling:
   ACE writes each task result into `<output>.progress.jsonl` as soon as the task completes.
5. Resume from progress:
   startup merges completed IDs from final output and progress journal.
6. Deterministic finalization:
   output can be rebuilt in manifest order from output + progress sources (`--finalize-order`).
7. Completion marker:
   `<output>.complete.json` records selected and completed row counts.
8. Policy-driven planning:
   both baseline and ACE v5 use `PlannerPolicy` (`src/planner_policy.py`) with epsilon plus UCB action selection.
9. Recursive test-time reasoning:
   both streams run `run_recursive_openai_reasoning` (`src/reasoning_loop.py`) with action-specific round and candidate budgets.
10. Confidence-aware gate and learned-memory retrieval:
   ACE gate consumes step-confidence signals and retrieval enforces learned-bullet quota with seed downweighting.

New v5 runtime metrics include:

- `progress_written`
- `resume_source`
- `memory_write_retries`
- `memory_write_failed`
- `memory_error`
- `recursion`
- `planner`
- `step_scoring.overall_confidence`

## 6.7 One-command v5 orchestration (`benchmark/v5/run.py`)

Default command:

```bash
python -m benchmark.v5.run \
  --manifest benchmark/results/v5/subset_manifest_v5_seed42_n200.json \
  --max-samples 200 \
  --seed 42 \
  --memory-scope hybrid
```

Default behavior:

1. Clear v5 outputs (unless `--no-clear-results`).
2. Clear Neo4j (unless `--no-clear-db`).
3. Run baseline v5 and ACE v5 inference in parallel.
4. Write and maintain `benchmark/results/v5/run_v5_meta.json` with phase start and end timestamps.
5. Enforce completion check against requested sample count.
6. If `--with-report` (default true), call `benchmark.v5.complete_pipeline --output-dir ... --skip-wait --max-samples ...` and pass dual-source cost flags and run metadata path.

### 6.7.1 V5 preflight (`benchmark/v5/preflight.py`)

`run_v5` preflight options:

- `--preflight-mode {off,static,smoke,both}`
- `--smoke-samples {3,4,5}`
- `--smoke-output-dir` default `benchmark/results/v5/smoke_v5`
- `--estimate-source {auto,v5,v4,v3,heuristic}`
- `--preflight-report` default `benchmark/results/v5/preflight_v5.json`

Static mode now validates progress-journal path writability in addition to manifest and output paths.
In `auto` estimate mode, source priority is `v4` report data first, then `v5`, then `v3`, then heuristic fallback.
Step-scoring token and latency assumptions use empirical auxiliary-call profiles from `ace_v*_metrics.json` when available, with explicit fallback defaults when missing.
Smoke measured-cost scaling prefers full-pipeline measured cost from `comparison_report_v5.json.full_pipeline_cost_metered.combined_total_cost_usd`, with inference-only fallback when absent.

### 6.7.2 V5 post-inference pipeline (`benchmark/v5/complete_pipeline.py`)

V5 post pipeline runs:

1. baseline and ACE eval in parallel;
2. retry failed side only (bounded by `--stage-retry-max`);
3. write eval usage metrics into:
   `baseline_v5_graded_eval_metrics.json`, `ace_v5_graded_eval_metrics.json`;
4. replay terminal graded rewards into planner state via `benchmark/v5/policy_replay.py`;
5. write replay output into `policy_replay_v5.json`;
6. baseline and ACE error analysis in parallel;
7. retry failed side only;
8. write error-analysis usage metrics into:
   `baseline_v5_graded_errors_error_metrics.json`, `ace_v5_graded_errors_error_metrics.json`;
9. task-id parity check before compare;
10. compare with cost mode and billing policy flags;
11. strict billed reconciliation when `cost_mode=dual_source` and `billing_policy=strict`.

## 7. Runtime Parity With Benchmark Gate

Runtime graph learning uses the same gate function and defaults:

- `src/agent.py` calls `pipeline.process_execution(...)` in `ace_learning_node`.
- `src/ace_components.py` `ACEPipeline.process_execution` calls `apply_quality_gate(...)` with `QualityGateConfig.from_env()`.
- `src/agent.py` planner node uses `PlannerPolicy` from `src/planner_policy.py`.
- `benchmark/v5/infer_baseline.py` and `benchmark/v5/infer_ace.py` use the same `PlannerPolicy` module for action selection and online updates.
- `src/agent.py` uses `run_recursive_text_refinement` and benchmark v5 uses `run_recursive_openai_reasoning` from `src/reasoning_loop.py`.

This means online runtime updates and benchmark v5 updates share gate logic and thresholds by default.

## 8. Comparison and Reporting Logic

`benchmark.compare.py` computes:

- Table 1 solving rates by category.
- Table 2 error distributions.
- Table 3 token, latency, and estimated cost.
- Table 4 per-category token and latency.
- Table 5 runtime diagnostics when fields are present:
  carryover coverage, learned retrieval rate, capped-output rate, mean step score, quality-gate apply rate.
- Table 5B planner diagnostics when planner fields are present:
  dominant action, action distribution, explore rate, recursion success rate, mean planner reward proxy, policy update counts and rates.
- Table 6 full-pipeline actual metered cost by phase:
  baseline inference, ACE inference, ACE auxiliary, baseline eval, ACE eval, baseline error analysis, ACE error analysis, and totals.
- Table 7 OpenAI billed reconciliation:
  project-scoped billed cost for run window, metered-vs-billed delta, and reconciliation status.
- V5 diagnostics extensions:
  resume recovery rate, memory write failure rate, progress checkpoint count.

Cost mode behavior:

1. `legacy`:
   keep inference-only comparability fields and skip billed reconciliation.
2. `dual_source`:
   aggregate metered tokens from graded rows, ACE auxiliary metrics, eval metrics, and error metrics.
3. strict billing policy:
   compare fails if admin credentials, project scope, run window, or billed query response is missing or unusable.

Additive JSON fields in `comparison_report_v5.json`:

- `full_pipeline_cost_metered`
- `cost_breakdown_by_phase`
- `openai_billed_reconciliation`
- `cost_reconciliation_status`
- `baseline_planner_diagnostics`
- `ace_planner_diagnostics`

It also enforces baseline versus ACE task-id set equality before report generation.

Detailed formulas are documented in:

- `docs/COMPARISON_REPORT_V2_CALCULATION_DETAILS.md`

## 9. Related Documentation

- `docs/SETUP.md`: environment and end-to-end commands.
- `docs/CL_BENCH_CALCULATION_DETAILS.md`: upstream CL-bench computation details.
- `docs/RESULTS_STRUCTURE.md`: canonical versioned output tree and schemas.
- `README.md`: operational quick start and artifact expectations.
