# Comparison Report V2 Calculation Details

## 1. Scope

This document explains how `/benchmark/results/v2/comparison_report_v2.md` and `/benchmark/results/v2/comparison_report_v2.json` are computed by:

- `/benchmark/compare.py`

It covers all four report tables, formulas, denominators, rounding, and integrity checks.

## 2. Inputs and Outputs

## 2.1 Required Inputs

- Baseline graded JSONL:
  - `/benchmark/results/v2/baseline_v2_graded.jsonl`
- ACE graded JSONL:
  - `/benchmark/results/v2/ace_v2_graded.jsonl`

## 2.2 Optional but Normally Used Inputs

- Baseline error classifications:
  - `/benchmark/results/v2/baseline_v2_graded_errors.jsonl`
- ACE error classifications:
  - `/benchmark/results/v2/ace_v2_graded_errors.jsonl`

## 2.3 Outputs

- Markdown report:
  - `/benchmark/results/v2/comparison_report_v2.md`
- JSON summary:
  - `/benchmark/results/v2/comparison_report_v2.json`

## 3. Integrity Constraints Before Metric Calculation

`compare.py` enforces task-id set parity between baseline and ACE graded files.

Rule:

```text
set(task_id_baseline) must equal set(task_id_ace)
```

If sets differ, execution raises `ValueError` and no report is produced.

Reference:

- `/benchmark/compare.py:275-283`

## 4. Table 1 Calculation: Solving Rate by Category

For each model stream:

```text
total = len(graded_rows)
score_1 = count(row.score == 1)
overall_percent = (score_1 / total) * 100
```

Per category (`metadata.context_category`):

```text
category_percent = (category_score_1 / category_total) * 100
```

Also recorded:

- `category_counts` for display as `n=...` in the header.

References:

- `/benchmark/compare.py:53-79`
- `/benchmark/compare.py:177-213`

## 5. Table 2 Calculation: Error Analysis Distribution

Input rows come from `_graded_errors.jsonl` files where each failed task has:

```json
"error_classification": {
  "task_error_types": ["CONTEXT_IGNORED", "FORMAT_ERROR", ...]
}
```

For each error type:

```text
count = number of rows where type appears in task_error_types
percent = (count / total_tasks_in_graded_file) * 100
```

Important denominator detail:

- denominator is total tasks (all graded tasks), not only failed tasks.

Error types are multi-label, so row totals can exceed 100 percent.

References:

- `/benchmark/compare.py:82-93`
- `/benchmark/error_analysis.py:139-159`

## 6. Table 3 Calculation: Token Usage, Latency, and Cost

For each stream (baseline or ACE), `compute_metrics` iterates over `row.metrics`.

## 6.1 Token Aggregates

From per-row `metrics`:

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

Computed:

```text
total_prompt_tokens = sum(prompt_tokens where total_tokens is present)
total_completion_tokens = sum(completion_tokens where total_tokens is present)
total_tokens = sum(total_tokens where total_tokens is present)
avg_tokens = total_tokens / len(data)
avg_prompt_tokens = total_prompt_tokens / len(data)
avg_completion_tokens = total_completion_tokens / len(data)
```

## 6.2 Latency Aggregates

Latency uses rows with non-zero `latency_ms`.

Computed:

```text
avg_latency_ms = sum(latencies) / len(latencies)
p50_latency_ms = percentile(sorted_latencies, 50)
p95_latency_ms = percentile(sorted_latencies, 95)
```

Percentile implementation:

```text
idx = int(len(sorted_values) * p / 100)
value = sorted_values[min(idx, len(sorted_values) - 1)]
```

## 6.3 Cost Formula

Current pricing constants in `compare.py`:

- `GPT51_INPUT_PRICE = 1.25` (USD per 1M input tokens)
- `GPT51_OUTPUT_PRICE = 10.00` (USD per 1M output tokens)

Estimated cost:

```text
cost_usd = (total_prompt_tokens / 1_000_000) * 1.25
         + (total_completion_tokens / 1_000_000) * 10.00
```

References:

- `/benchmark/compare.py:22-25`
- `/benchmark/compare.py:96-147`
- `/benchmark/compare.py:234-245`

## 7. Table 4 Calculation: Per-Category Token and Latency

Rows are grouped by `metadata.context_category`.
For each group, the same `compute_metrics` logic is applied.

Displayed fields:

- `avg_tokens`
- `avg_latency_ms`

References:

- `/benchmark/compare.py:150-160`
- `/benchmark/compare.py:246-257`

## 8. Rounding and Presentation Rules

Markdown report formatting:

- solving rates and error rates: 1 decimal place
- average tokens and latency: 0 decimals
- estimated cost: 2 decimals
- signed deltas include `+` or `-`

JSON summary keeps full floating-point precision from computations.

References:

- `/benchmark/compare.py:197-245`
- `/benchmark/compare.py:321-334`

## 9. Worked Examples from Current V2 Artifacts

Using `/benchmark/results/v2/comparison_report_v2.json`:

## 9.1 Baseline Overall Solve Rate

```text
score_1 = 36
total = 200
overall = (36 / 200) * 100 = 18.0%
```

## 9.2 ACE Overall Solve Rate

```text
score_1 = 33
total = 200
overall = (33 / 200) * 100 = 16.5%
```

## 9.3 Baseline Estimated Cost

```text
total_prompt_tokens = 3,332,980
total_completion_tokens = 290,892
cost = (3,332,980 / 1,000,000) * 1.25
     + (290,892 / 1,000,000) * 10.00
     = 7.075145 -> shown as $7.08
```

## 9.4 ACE Estimated Cost

```text
total_prompt_tokens = 3,371,008
total_completion_tokens = 271,267
cost = (3,371,008 / 1,000,000) * 1.25
     + (271,267 / 1,000,000) * 10.00
     = 6.92643 -> shown as $6.93
```

## 10. Command Used to Generate V2 Report

```bash
python -m benchmark.compare \
  --baseline benchmark/results/v2/baseline_v2_graded.jsonl \
  --ace benchmark/results/v2/ace_v2_graded.jsonl \
  --baseline-errors benchmark/results/v2/baseline_v2_graded_errors.jsonl \
  --ace-errors benchmark/results/v2/ace_v2_graded_errors.jsonl \
  --output benchmark/results/v2/comparison_report_v2.md \
  --title-label V2
```
