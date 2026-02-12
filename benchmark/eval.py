"""
CL-bench Evaluation: GPT-5.1 as judge.

Grades model outputs against rubrics using binary 0/1 scoring.
Adapted from https://github.com/Tencent-Hunyuan/CL-bench/blob/main/eval.py

Usage:
    python -m benchmark.eval --input benchmark/results/v1/baseline.jsonl --judge-model gpt-5.1
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_metrics_output_path(output_path: str) -> str:
    base_name, _ = os.path.splitext(output_path)
    return f"{base_name}_eval_metrics.json"


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


def build_rubrics_text(rubrics: List[Any]) -> str:
    if not rubrics:
        return "No specific rubrics provided."
    lines: List[str] = []
    for idx, rubric in enumerate(rubrics, 1):
        if isinstance(rubric, dict):
            criteria = str(rubric.get("rubric_criteria", "")).strip()
        else:
            criteria = str(rubric).strip()
        if criteria:
            lines.append(f"{idx}. {criteria}")
    return "\n".join(lines) if lines else "No specific rubrics provided."


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


def call_judge(
    client: OpenAI,
    model: str,
    rubrics_text: str,
    model_output: str,
    max_retries: int = 3,
) -> Tuple[Optional[str], Optional[str], Dict[str, int], int]:
    grading_prompt = (
        "Starting now, you are a rigorous instruction-following grading teacher. Your task is to accurately grade and score student answers based on the [Rubrics].\n\n"
        "Grading Criteria\n"
        "This is a strict, all-or-nothing grading system. The final score is binary.\n"
        "To receive a score of 1, the student's answer must perfectly satisfy every single requirement listed in the [Rubrics].\n"
        "If even one requirement is not fully met, the final score will be 0.\n"
        "Grading Process\n"
        "Please strictly follow the steps below for analysis:\n"
        "Step 1: Analyze the Standard Answer\n"
        "List all explicit requirements in the [Rubrics] item by item.\n"
        "Identify implicit requirements.\n"
        "Define specific evaluation criteria for each requirement.\n"
        "Step 2: Check Each Requirement Against the Student's Answer\n"
        "For every requirement in the [Rubrics], verify one by one whether the student's answer fully satisfies it.\n"
        "Step 3: Self-Reflection\n"
        "Before giving the final score, conduct: Completeness Check, Strictness Check, Consistency Check, Objectivity Check.\n"
        "Output Format Requirements\n"
        "Please strictly output ONLY the following JSON format (do not output any other content):\n"
        "{\n"
        "  \"Grading Rationale\": \"Your detailed grading rationale\",\n"
        "  \"List of Requirement Satisfaction Status\": [\"yes\", \"no\", ...],\n"
        "  \"Overall Score\": 0 or 1\n"
        "}\n\n"
        "Content to Be Graded\n"
        f"[Rubrics]:\n{rubrics_text}\n"
        f"[Student Response]:\n{model_output}\n"
    )

    messages = [{"role": "user", "content": grading_prompt}]
    api_attempts = 0

    for attempt in range(max_retries):
        try:
            api_attempts += 1
            try:
                response = client.chat.completions.create(
                    model = model,
                    messages = messages,
                    max_completion_tokens = 4096,
                    response_format = {"type": "json_object"},
                )
            except Exception:
                api_attempts += 1
                response = client.chat.completions.create(
                    model = model,
                    messages = messages,
                    max_completion_tokens = 4096,
                )
            usage = extract_usage(response)
            content = response.choices[0].message.content or ""
            return parse_response_text(content), None, usage, api_attempts
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None, "api_fail", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, api_attempts

    return None, "api_fail", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, api_attempts


def get_task_id(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        task_id = metadata.get("task_id", item.get("task_id", ""))
        return task_id if isinstance(task_id, str) else ""
    task_id = item.get("task_id", "")
    return task_id if isinstance(task_id, str) else ""


def calculate_statistics(output_path: str) -> None:
    if not os.path.exists(output_path):
        return
    data = load_jsonl(output_path)
    total = len(data)
    score_0 = sum(1 for item in data if item.get("score") == 0)
    score_1 = sum(1 for item in data if item.get("score") == 1)

    print("\nFinal Statistics:")
    print(f"  Total samples: {total}")
    print(f"  Score 0: {score_0}")
    print(f"  Score 1: {score_1}")

    if total > 0:
        solving_rate = score_1 / total
        print(f"\n  Solving Rate: {solving_rate:.4f} ({score_1}/{total})")

    category_stats: Dict[str, Dict[str, int]] = {}
    for item in data:
        metadata = item.get("metadata", {})
        category = metadata.get("context_category", "Unknown") if isinstance(metadata, dict) else "Unknown"
        stats = category_stats.setdefault(category, {"total": 0, "score_0": 0, "score_1": 0})
        stats["total"] += 1
        if item.get("score") == 1:
            stats["score_1"] += 1
        else:
            stats["score_0"] += 1

    if category_stats:
        print("\n  Scores by context_category:")
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            rate = stats["score_1"] / stats["total"] if stats["total"] else 0
            print(f"    {category}: total={stats['total']}, score_1={stats['score_1']}, rate={rate:.4f}")


def build_metrics_payload(
    judge_model: str,
    counters: Counter,
    usage_totals: Dict[str, int],
    total_calls: int,
    started_at: str,
    ended_at: str,
    wall_seconds: float,
    input_rows: int,
    pending_rows: int,
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
        "api_fail_count": int(counters.get("api_fail", 0)),
        "json_fail_count": int(counters.get("json_fail", 0)),
        "graded_ok_count": int(counters.get("graded_ok", 0)),
        "no_output_count": int(counters.get("no_output", 0)),
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_seconds": float(wall_seconds),
        "input_rows": int(input_rows),
        "pending_rows": int(pending_rows),
        "existing_completed_rows": int(existing_completed_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description = "CL-bench Evaluation")
    parser.add_argument("--input", type = str, required = True)
    parser.add_argument("--output", type = str, default = None)
    parser.add_argument("--metrics-output", type = str, default = None)
    parser.add_argument("--judge-model", type = str, default = "gpt-5.1")
    parser.add_argument("--api-key", type = str, default = None)
    parser.add_argument("--base-url", type = str, default = None)
    parser.add_argument("--max-retries", type = int, default = 3)
    args = parser.parse_args()

    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(os.path.dirname(args.input), f"{base_name}_graded.jsonl")
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

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Metrics: {args.metrics_output}")
    print(f"Judge: {args.judge_model}")

    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} samples")

    completed_ids = set()
    existing_rows = 0
    if os.path.exists(args.output):
        existing = load_jsonl(args.output)
        completed_ids = {get_task_id(item) for item in existing if get_task_id(item)}
        existing_rows = len(existing)
        print(f"Found {len(completed_ids)} completed, resuming remaining")

    pending = [item for item in data if get_task_id(item) not in completed_ids]
    started_at = utc_now_iso()
    started_perf = time.perf_counter()

    counters: Counter = Counter()
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_calls = 0
    success_count = 0
    fail_count = 0

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
            input_rows = len(data),
            pending_rows = 0,
            existing_completed_rows = existing_rows,
        )
        write_json(args.metrics_output, metrics_payload)
        print("All samples already evaluated")
        calculate_statistics(args.output)
        return

    print(f"Evaluating {len(pending)} samples...")
    for item in tqdm(pending, desc = "Evaluating"):
        model_output = str(item.get("model_output", ""))
        rubrics = item.get("rubrics", [])

        if not model_output.strip():
            result = {
                **item,
                "grading_rationale": "No model output (score 0)",
                "requirement_status": [],
                "score": 0,
            }
            append_jsonl(result, args.output)
            success_count += 1
            counters["no_output"] += 1
            continue

        rubrics_text = build_rubrics_text(rubrics if isinstance(rubrics, list) else [])
        grading_result, judge_error, usage, api_attempts = call_judge(
            client,
            args.judge_model,
            rubrics_text,
            model_output,
            args.max_retries,
        )
        total_calls += int(api_attempts)
        usage_totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        usage_totals["completion_tokens"] += int(usage.get("completion_tokens", 0))
        usage_totals["total_tokens"] += int(usage.get("total_tokens", 0))

        if not grading_result:
            result = {
                **item,
                "grading_rationale": "API call failed (score 0)",
                "requirement_status": [],
                "score": 0,
            }
            append_jsonl(result, args.output)
            fail_count += 1
            counters[judge_error or "api_fail"] += 1
            continue

        try:
            result_json = json.loads(grading_result)
            if "Overall Score" not in result_json:
                raise ValueError("Missing 'Overall Score'")
            result = {
                **item,
                "grading_rationale": result_json.get("Grading Rationale", ""),
                "requirement_status": result_json.get("List of Requirement Satisfaction Status", []),
                "score": result_json.get("Overall Score", 0),
            }
            append_jsonl(result, args.output)
            success_count += 1
            counters["graded_ok"] += 1
        except (json.JSONDecodeError, ValueError):
            result = {
                **item,
                "grading_rationale": f"JSON parse failed: {grading_result[:500]}",
                "requirement_status": [],
                "score": 0,
            }
            append_jsonl(result, args.output)
            fail_count += 1
            counters["json_fail"] += 1

    ended_at = utc_now_iso()
    metrics_payload = build_metrics_payload(
        judge_model = args.judge_model,
        counters = counters,
        usage_totals = usage_totals,
        total_calls = total_calls,
        started_at = started_at,
        ended_at = ended_at,
        wall_seconds = time.perf_counter() - started_perf,
        input_rows = len(data),
        pending_rows = len(pending),
        existing_completed_rows = existing_rows,
    )
    write_json(args.metrics_output, metrics_payload)

    print(f"\nDone: {success_count} success, {fail_count} failed")
    calculate_statistics(args.output)


if __name__ == "__main__":
    main()
