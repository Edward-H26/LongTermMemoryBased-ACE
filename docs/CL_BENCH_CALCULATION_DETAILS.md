# CL-bench Calculation Details (Upstream Reference)

## 1. Scope and Source Snapshot

This document explains how the original CL-bench pipeline computes outputs and metrics in the upstream repository.

- Upstream repository: `Tencent-Hunyuan/CL-bench`
- Commit used for analysis: `f34052c893e932c90c5caa417990c756d069e26b`
- Core scripts:
  - `infer.py`
  - `eval.py`
- Paper sources used for metric interpretation:
  - `sections/bench.tex`
  - `sections/experiment.tex`
  - `tables/main_table.tex`
  - `tables/error_analysis.tex`

## 2. Upstream Data Schemas

### 2.1 Dataset Row (input to `infer.py`)

Each row in `CL-bench.jsonl` contains:

```json
{
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "rubrics": ["...", "..."],
  "metadata": {
    "task_id": "uuid",
    "context_id": "uuid",
    "context_category": "Rule System Application",
    "sub_category": "..."
  }
}
```

### 2.2 Inference Output Row (written by `infer.py`)

`infer.py` writes JSONL rows with:

```json
{
  "idx": "task_id",
  "messages": [...],
  "model_output": "...",
  "rubrics": [...],
  "metadata": {...}
}
```

`idx` is set to `metadata.task_id` for stable resume behavior.

## 3. Inference Calculation Flow (`infer.py`)

## 3.1 Sample Selection

`--max-samples` uses a simple prefix slice:

```text
if args.max_samples:
    data = data[:args.max_samples]
```

This is not seeded random sampling. It is deterministic by input order.

## 3.2 Resume Logic

If output already exists, completed samples are read and their `idx` values are collected into `completed_indices`.
Pending set is then:

```text
tasks = [item for item in data if task_id(item) not in completed_indices]
```

## 3.3 Per-Task Inference

For each pending row:

1. Validate `messages` exists.
2. Call OpenAI-compatible chat completion API with retry.
3. Store response text as `model_output`.
4. Append one JSONL row immediately.

Failures are logged and skipped, but they are not written as partial result rows.

## 4. Evaluation Calculation Flow (`eval.py`)

## 4.1 Rubric Formatting

Rubrics are converted into a numbered checklist string. Dictionary rubrics use `rubric_criteria`; string rubrics use raw text.

## 4.2 Judge Prompt Contract

Judge prompt enforces strict binary grading:

- Score `1`: all rubric requirements satisfied.
- Score `0`: any requirement missing, malformed, or violated.

The script requires JSON response with:

- `Grading Rationale`
- `List of Requirement Satisfaction Status`
- `Overall Score`

## 4.3 Failure Fallback Rules

If model output is empty, score is forced to `0`.
If API call fails after retries, score is forced to `0`.
If judge JSON parsing fails after retries, score is forced to `0`.

This makes final score computation conservative and total-row stable.

## 4.4 Solve Rate Formulas in Script

Given graded rows:

```text
total = number of rows
score_1 = count(score == 1)
score_0 = count(score == 0)
solving_rate = score_1 / total
```

Per category (`metadata.context_category`):

```text
category_rate = category_score_1 / category_total
```

## 5. Error Type Interpretation in Paper

Upstream public scripts (`infer.py`, `eval.py`) do not include an error-classification script.
The CL-bench paper reports error categories in a separate analysis process:

- `Context Ignored`
- `Context Misused`
- `Format Error`
- `Refusal`

Rows can exceed 100 percent because error types are multi-label per failed solution.

## 6. Paper-Reported vs Script-Computed Metrics

## 6.1 Script-Computed

`eval.py` produces single-run solving rates from one graded JSONL file.

## 6.2 Paper-Reported

Paper main table reports `mean ± std` percentages across three runs for each model and category.

This means leaderboard values are aggregation statistics over repeated trials, not direct output from one call to `eval.py`.

## 7. Source Mapping

## 7.1 Upstream Repo Files

- Inference flow and output schema:
  - [`infer.py` lines 103-127](https://github.com/Tencent-Hunyuan/CL-bench/blob/f34052c893e932c90c5caa417990c756d069e26b/infer.py#L103-L127)
  - [`infer.py` lines 170-173](https://github.com/Tencent-Hunyuan/CL-bench/blob/f34052c893e932c90c5caa417990c756d069e26b/infer.py#L170-L173)
  - [`infer.py` lines 174-187](https://github.com/Tencent-Hunyuan/CL-bench/blob/f34052c893e932c90c5caa417990c756d069e26b/infer.py#L174-L187)
- Evaluation binary prompt and parse behavior:
  - [`eval.py` lines 93-126](https://github.com/Tencent-Hunyuan/CL-bench/blob/f34052c893e932c90c5caa417990c756d069e26b/eval.py#L93-L126)
  - [`eval.py` lines 175-247](https://github.com/Tencent-Hunyuan/CL-bench/blob/f34052c893e932c90c5caa417990c756d069e26b/eval.py#L175-L247)
- Solving-rate statistics:
  - [`eval.py` lines 373-414](https://github.com/Tencent-Hunyuan/CL-bench/blob/f34052c893e932c90c5caa417990c756d069e26b/eval.py#L373-L414)

## 7.2 CL-bench Paper Source (arXiv Source Bundle)

The following files were inspected from `arXiv:2602.03587` source package:

- `sections/bench.tex` lines 141-170: rubric design, strict task pass rule, verifier reliability notes.
- `sections/experiment.tex` lines 5-10 and 29-33: three trials and solving-rate framing.
- `tables/main_table.tex` lines 12-14: mean ± std across three runs.
- `tables/error_analysis.tex` lines 12-15 and 23-26: error-type table semantics.

ArXiv links:

- Paper abstract page: [https://arxiv.org/abs/2602.03587](https://arxiv.org/abs/2602.03587)
- Source package: [https://arxiv.org/src/2602.03587](https://arxiv.org/src/2602.03587)
