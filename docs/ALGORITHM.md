# LTMBSE ACE Algorithm (V3 Accurate Specification)

## 1. System Overview

The project combines a LangGraph reasoning agent with ACE memory and a CL-bench benchmarking pipeline.

Core runtime graph:

```text
START -> router -> planner -> solver -> critic -> ace_learning -> END
```

Core benchmark streams:

- Baseline v3: `benchmark/infer_baseline_v3.py`
- ACE direct v3: `benchmark/infer_ace_direct_v3.py`
- One-command orchestration: `benchmark/run_v3.py`

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
- `ACE_WEIGHT_RELEVANCE = 0.25`
- `ACE_WEIGHT_STRENGTH = 0.55`
- `ACE_WEIGHT_TYPE = 0.20`

These are framework defaults. CL-bench experiments may tune relevance and strength weights separately.

## 4. V3 Quality Gate (Implemented)

Quality gate logic is implemented in `src/ace_components.py`.

## 4.1 Config and Defaults

`QualityGateConfig` defaults:

- `gate_score_min = 0.65`
- `lesson_score_min = 0.60`
- `overlap_min = 0.10`
- `max_accepted_lessons = 2`

These can be overridden by:

- `ACE_QG_GATE_SCORE_MIN`
- `ACE_QG_LESSON_SCORE_MIN`
- `ACE_QG_OVERLAP_MIN`
- `ACE_QG_MAX_ACCEPTED_LESSONS`

## 4.2 Scoring Components

For each extracted lesson:

1. Overlap score:

```text
overlap_score = |tokens(question) ∩ tokens(lesson)| / |tokens(question) ∪ tokens(lesson)|
```

2. Lesson quality score:

```text
token_score = min(token_count / 20, 1.0) * 0.6
tags_score = 0.2 if tags exist else 0.0
type_score = 0.2 if type in {success, failure, domain, tool} else 0.0
lesson_score = min(token_score + tags_score + type_score, 1.0)
```

3. Lesson acceptance filter:

```text
accept if overlap_score >= overlap_min
      and lesson_score >= lesson_score_min
      and content is non-empty
```

4. Top-k cap:

Accepted candidates are sorted by `(lesson_score, overlap_score)` descending, then truncated to `max_accepted_lessons`.

## 4.3 Task-Level Gate

After accepted lessons are selected:

```text
output_score = 1.0 if model output is non-empty else 0.0
accepted_quality_avg = mean(accepted lesson scores)
gate_score = 0.5 * output_score + 0.5 * accepted_quality_avg
should_apply_update = accepted_lessons non-empty and gate_score >= gate_score_min
```

Only if `should_apply_update` is true are accepted lessons curated into memory deltas.

## 4.4 Diagnostics Schema

Diagnostics returned by `apply_quality_gate` and attached to outputs include:

- `config`
- `output_valid`
- `output_score`
- `accepted_quality_avg`
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
- `selected_count`
- `created_at`
- `task_ids` (ordered)

## 5.3 Selection Logic

If manifest file exists:

- load manifest,
- reconstruct subset by `task_ids` order.

If manifest does not exist:

- sample deterministic indices using `random.Random(seed).sample(...)`,
- sort indices,
- build subset in sorted-index order,
- persist manifest when a path is provided.

## 6. Benchmark V3 Pipelines

## 6.1 Baseline v3 (`benchmark/infer_baseline_v3.py`)

Flow:

1. Load dataset.
2. Resolve subset with seed and optional manifest.
3. Resume by existing output `task_id` set when not clearing results.
4. Call OpenAI model per task.
5. Write one row per task:
   - `task_id`, `messages`, `model_output`, `rubrics`, `metadata`, `metrics`.

## 6.2 ACE direct v3 (`benchmark/infer_ace_direct_v3.py`)

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

## 6.3 One-command v3 orchestration (`benchmark/run_v3.py`)

Default command:

```bash
python -m benchmark.run_v3 \
  --manifest benchmark/results/subset_manifest_v3_seed42_n200.json \
  --max-samples 200 \
  --seed 42
```

Default behavior:

1. Clear v3 outputs (unless `--no-clear-results`).
2. Clear Neo4j (unless `--no-clear-db`).
3. Run baseline v3 and ACE v3 inference in parallel.
4. Enforce completion check against requested sample count.
5. If `--with-report` (default true), call `benchmark.complete_v3_pipeline --skip-wait`.

`benchmark.complete_v3_pipeline.py` currently uses fixed `TARGET = 200`.
For non-200 subset runs, use `--no-with-report` and run eval/error/compare manually.

## 7. Runtime Parity With Benchmark Gate

Runtime graph learning uses the same gate function and defaults:

- `src/agent.py` calls `pipeline.process_execution(...)` in `ace_learning_node`.
- `src/ace_components.py` `ACEPipeline.process_execution` calls `apply_quality_gate(...)` with `QualityGateConfig.from_env()`.

This means online runtime updates and benchmark v3 updates share gate logic and thresholds by default.

## 8. Comparison and Reporting Logic

`benchmark.compare.py` computes:

- Table 1 solving rates by category.
- Table 2 error distributions.
- Table 3 token, latency, and estimated cost.
- Table 4 per-category token and latency.

It also enforces baseline versus ACE task-id set equality before report generation.

Detailed formulas are documented in:

- `docs/COMPARISON_REPORT_V2_CALCULATION_DETAILS.md`

## 9. Related Documentation

- `docs/SETUP.md`: environment and end-to-end commands.
- `docs/CL_BENCH_CALCULATION_DETAILS.md`: upstream CL-bench computation details.
- `README.md`: operational quick start and artifact expectations.
