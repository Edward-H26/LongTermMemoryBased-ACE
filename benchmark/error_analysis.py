"""
Error Analysis Classification for CL-bench Table 2.

Performs a secondary LLM classification pass on failed tasks (score 0)
to categorize failures into: Context Ignored, Context Misused, Format Error, Refusal.

Per the CL-bench paper (Section 5, Table 3), error types are not mutually exclusive.
"""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


ERROR_CLASSIFICATION_PROMPT = """You are an error analysis classifier for a context learning benchmark.

A language model was given a context (containing novel knowledge) and a task. The model's output failed to satisfy one or more evaluation rubrics. Your job is to classify WHY each rubric failed.

For the model output and each failed rubric below, classify the failure into ONE of these error types:

1. CONTEXT_IGNORED: The model failed because it did not reference or use information that was explicitly provided in the context. The model either overlooked relevant content or relied on pre-trained knowledge instead.

2. CONTEXT_MISUSED: The model referenced contextual information but applied it incorrectly, misinterpreted it, made errors in reasoning about it, or drew wrong conclusions from it.

3. FORMAT_ERROR: The model violated formatting, structure, or output requirements specified in the system prompt or context. The content may be correct but the presentation is wrong.

4. REFUSAL: The model refused to answer, claimed insufficient information, or produced an empty or irrelevant response.

A task can have MULTIPLE error types. Classify based on the dominant failure mode of each rubric.

## Model Output
{model_output}

## Failed Rubrics
{failed_rubrics}

## Grading Rationale
{grading_rationale}

Output ONLY valid JSON:
{{
    "task_error_types": ["CONTEXT_IGNORED", "FORMAT_ERROR"],
    "per_rubric_classification": [
        {{"rubric_index": 0, "error_type": "CONTEXT_IGNORED", "reasoning": "brief explanation"}},
        ...
    ]
}}"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_metrics_output_path(output_path: str) -> str:
    base_name, _ = os.path.splitext(output_path)
    return f"{base_name}_error_metrics.json"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding = "utf-8") as f:
        for line in f:
            raw = line.strip()
            if raw:
                data.append(json.loads(raw))
    return data


def append_jsonl(item: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok = True)
    with open(path, "a", encoding = "utf-8") as f:
        f.write(json.dumps(item, ensure_ascii = False) + "\n")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(payload, f, indent = 2, ensure_ascii = False)


def get_task_id(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        task_id = metadata.get("task_id", item.get("task_id", ""))
        return task_id if isinstance(task_id, str) else ""
    task_id = item.get("task_id", "")
    return task_id if isinstance(task_id, str) else ""


def parse_response_text(result_text: str) -> str:
    cleaned = result_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def extract_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def classify_errors(
    client: OpenAI,
    model: str,
    item: Dict[str, Any],
    max_retries: int = 3,
) -> Tuple[Dict[str, Any], bool, Dict[str, int], int, bool]:
    model_output = str(item.get("model_output", ""))
    rubrics = item.get("rubrics", [])
    requirement_status = item.get("requirement_status", [])
    grading_rationale = str(item.get("grading_rationale", ""))

    if not model_output.strip():
        return {"task_error_types": ["REFUSAL"], "per_rubric_classification": []}, False, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0, False

    failed_rubrics: List[str] = []
    rubrics_list = rubrics if isinstance(rubrics, list) else []
    status_list = requirement_status if isinstance(requirement_status, list) else []
    for idx, rubric in enumerate(rubrics_list):
        if isinstance(rubric, str):
            rubric_text = rubric
        elif isinstance(rubric, dict):
            rubric_text = str(rubric.get("rubric_criteria", str(rubric)))
        else:
            rubric_text = str(rubric)
        status = status_list[idx] if idx < len(status_list) else "no"
        status_no = (isinstance(status, str) and status.lower() == "no") or (isinstance(status, bool) and not status)
        if status_no:
            failed_rubrics.append(f"Rubric {idx + 1} (FAILED): {rubric_text}")

    if not failed_rubrics:
        return {"task_error_types": [], "per_rubric_classification": []}, False, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0, False

    failed_text = "\n".join(failed_rubrics)
    prompt = ERROR_CLASSIFICATION_PROMPT.format(
        model_output = model_output[:3000],
        failed_rubrics = failed_text,
        grading_rationale = grading_rationale[:2000],
    )

    api_attempts = 0
    for attempt in range(max_retries):
        try:
            api_attempts += 1
            response = client.chat.completions.create(
                model = model,
                messages = [{"role": "user", "content": prompt}],
                max_completion_tokens = 4096,
            )
            usage = extract_usage(response)
            result_text = parse_response_text(response.choices[0].message.content or "")
            parsed = json.loads(result_text)
            if "task_error_types" in parsed:
                return parsed, False, usage, api_attempts, False
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)

    fallback = {"task_error_types": ["CONTEXT_MISUSED"], "per_rubric_classification": []}
    return fallback, True, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, api_attempts, True


def compute_error_distribution(output_path: str, total_tasks: int) -> Dict[str, float]:
    data = load_jsonl(output_path)
    error_counts = {
        "CONTEXT_IGNORED": 0,
        "CONTEXT_MISUSED": 0,
        "FORMAT_ERROR": 0,
        "REFUSAL": 0,
    }

    for item in data:
        classification = item.get("error_classification", {})
        error_types = classification.get("task_error_types", []) if isinstance(classification, dict) else []
        if not isinstance(error_types, list):
            continue
        for error_type in error_types:
            if error_type in error_counts:
                error_counts[error_type] += 1

    print(f"\nError Analysis Distribution (out of {total_tasks} total tasks):")
    distribution: Dict[str, float] = {}
    for error_type, count in error_counts.items():
        pct = (count / total_tasks * 100.0) if total_tasks > 0 else 0.0
        distribution[error_type] = pct
        print(f"  {error_type}: {count} ({pct:.1f}%)")
    print("  Note: error types are not mutually exclusive; a task can have multiple types.")
    return distribution


def build_metrics_payload(
    judge_model: str,
    counters: Counter,
    usage_totals: Dict[str, int],
    total_calls: int,
    started_at: str,
    ended_at: str,
    wall_seconds: float,
    total_input_rows: int,
    pending_rows: int,
    failed_rows: int,
    existing_completed_rows: int,
) -> Dict[str, Any]:
    prompt_tokens = int(usage_totals.get("prompt_tokens", 0))
    completion_tokens = int(usage_totals.get("completion_tokens", 0))
    total_tokens = int(usage_totals.get("total_tokens", prompt_tokens + completion_tokens))
    return {
        "model": judge_model,
        "total_calls": int(total_calls),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "classified_count": int(counters.get("classified", 0)),
        "fallback_count": int(counters.get("fallback", 0)),
        "api_fail_count": int(counters.get("api_fail", 0)),
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_seconds": float(wall_seconds),
        "total_input_rows": int(total_input_rows),
        "failed_rows": int(failed_rows),
        "pending_rows": int(pending_rows),
        "existing_completed_rows": int(existing_completed_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description = "CL-bench Error Analysis")
    parser.add_argument("--input", type = str, required = True, help = "Graded JSONL file")
    parser.add_argument("--output", type = str, default = None)
    parser.add_argument("--metrics-output", type = str, default = None)
    parser.add_argument("--judge-model", type = str, default = "gpt-5.1")
    parser.add_argument("--api-key", type = str, default = None)
    parser.add_argument("--base-url", type = str, default = None)
    parser.add_argument("--max-retries", type = int, default = 3)
    args = parser.parse_args()

    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(os.path.dirname(args.input), f"{base_name}_errors.jsonl")
    if args.metrics_output is None:
        args.metrics_output = default_metrics_output_path(args.output)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    data = load_jsonl(args.input)
    total_tasks = len(data)
    failed_tasks = [item for item in data if item.get("score") == 0]
    print(f"Total tasks: {total_tasks}, Failed tasks: {len(failed_tasks)}")
    print(f"Output: {args.output}")
    print(f"Metrics: {args.metrics_output}")

    completed_ids = set()
    existing_rows = 0
    if os.path.exists(args.output):
        existing = load_jsonl(args.output)
        existing_rows = len(existing)
        completed_ids = {get_task_id(item) for item in existing if get_task_id(item)}

    pending = [item for item in failed_tasks if get_task_id(item) not in completed_ids]
    started_at = utc_now_iso()
    started_perf = time.perf_counter()

    counters: Counter = Counter()
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_calls = 0

    if not pending:
        ended_at = utc_now_iso()
        metrics_payload = build_metrics_payload(
            judge_model = args.judge_model,
            counters = counters,
            usage_totals = usage_totals,
            total_calls = total_calls,
            started_at = started_at,
            ended_at = ended_at,
            wall_seconds = time.perf_counter() - started_perf,
            total_input_rows = total_tasks,
            pending_rows = 0,
            failed_rows = len(failed_tasks),
            existing_completed_rows = existing_rows,
        )
        write_json(args.metrics_output, metrics_payload)
        print("All failed tasks already classified")
        compute_error_distribution(args.output, total_tasks)
        return

    print(f"Classifying errors for {len(pending)} failed tasks...")
    for item in tqdm(pending, desc = "Error Analysis"):
        classification, used_fallback, usage, api_attempts, api_failed = classify_errors(
            client,
            args.judge_model,
            item,
            max_retries = args.max_retries,
        )
        total_calls += int(api_attempts)
        usage_totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        usage_totals["completion_tokens"] += int(usage.get("completion_tokens", 0))
        usage_totals["total_tokens"] += int(usage.get("total_tokens", 0))
        if used_fallback:
            counters["fallback"] += 1
        if api_failed:
            counters["api_fail"] += 1
        counters["classified"] += 1

        result = {**item, "error_classification": classification}
        append_jsonl(result, args.output)

    ended_at = utc_now_iso()
    metrics_payload = build_metrics_payload(
        judge_model = args.judge_model,
        counters = counters,
        usage_totals = usage_totals,
        total_calls = total_calls,
        started_at = started_at,
        ended_at = ended_at,
        wall_seconds = time.perf_counter() - started_perf,
        total_input_rows = total_tasks,
        pending_rows = len(pending),
        failed_rows = len(failed_tasks),
        existing_completed_rows = existing_rows,
    )
    write_json(args.metrics_output, metrics_payload)

    compute_error_distribution(args.output, total_tasks)
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
