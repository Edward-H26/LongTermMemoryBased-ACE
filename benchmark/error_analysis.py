"""
Error Analysis Classification for CL-bench Table 2.

Performs a secondary LLM classification pass on failed tasks (score 0)
to categorize failures into: Context Ignored, Context Misused, Format Error, Refusal.

Per the CL-bench paper (Section 5, Table 3), error types are NOT mutually exclusive.
One task can exhibit multiple error types, so row totals exceed 100%.

Usage:
    python -m benchmark.error_analysis --input benchmark/results/baseline_graded.jsonl
"""

import json
import os
import time
import argparse

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from tqdm import tqdm


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


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_jsonl(item, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_task_id(item):
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get("task_id", item.get("task_id", ""))
    return item.get("task_id", "")


def classify_errors(client, model, item, max_retries=3):
    model_output = item.get("model_output", "")
    rubrics = item.get("rubrics", [])
    requirement_status = item.get("requirement_status", [])
    grading_rationale = item.get("grading_rationale", "")

    if not model_output or not model_output.strip():
        return {"task_error_types": ["REFUSAL"], "per_rubric_classification": []}

    failed_rubrics = []
    for i, rubric in enumerate(rubrics):
        rubric_text = rubric if isinstance(rubric, str) else rubric.get("rubric_criteria", str(rubric))
        status = requirement_status[i] if i < len(requirement_status) else "no"
        if isinstance(status, str) and status.lower() == "no":
            failed_rubrics.append(f"Rubric {i + 1} (FAILED): {rubric_text}")
        elif isinstance(status, bool) and not status:
            failed_rubrics.append(f"Rubric {i + 1} (FAILED): {rubric_text}")

    if not failed_rubrics:
        return {"task_error_types": [], "per_rubric_classification": []}

    failed_text = "\n".join(failed_rubrics)
    prompt = ERROR_CLASSIFICATION_PROMPT.format(
        model_output=model_output[:3000],
        failed_rubrics=failed_text,
        grading_rationale=grading_rationale[:2000],
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4096,
            )
            result_text = response.choices[0].message.content.strip()

            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            result = json.loads(result_text)
            if "task_error_types" in result:
                return result
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)

    return {"task_error_types": ["CONTEXT_MISUSED"], "per_rubric_classification": []}


def compute_error_distribution(output_path, total_tasks):
    data = load_jsonl(output_path)
    error_counts = {
        "CONTEXT_IGNORED": 0,
        "CONTEXT_MISUSED": 0,
        "FORMAT_ERROR": 0,
        "REFUSAL": 0,
    }

    for item in data:
        error_types = item.get("error_classification", {}).get("task_error_types", [])
        for et in error_types:
            if et in error_counts:
                error_counts[et] += 1

    print(f"\nError Analysis Distribution (out of {total_tasks} total tasks):")
    for error_type, count in error_counts.items():
        pct = (count / total_tasks * 100) if total_tasks > 0 else 0
        print(f"  {error_type}: {count} ({pct:.1f}%)")
    print("  Note: error types are NOT mutually exclusive; a task can have multiple types.")
    return error_counts


def main():
    parser = argparse.ArgumentParser(description="CL-bench Error Analysis")
    parser.add_argument("--input", type=str, required=True, help="Graded JSONL file")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default="gpt-5.1")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"benchmark/results/{base_name}_errors.jsonl"

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

    completed_ids = set()
    if os.path.exists(args.output):
        existing = load_jsonl(args.output)
        completed_ids = {get_task_id(item) for item in existing if get_task_id(item)}

    pending = [item for item in failed_tasks if get_task_id(item) not in completed_ids]
    if not pending:
        print("All failed tasks already classified")
        compute_error_distribution(args.output, total_tasks)
        return

    print(f"Classifying errors for {len(pending)} failed tasks...")
    for item in tqdm(pending, desc="Error Analysis"):
        classification = classify_errors(client, args.judge_model, item)
        result = {**item, "error_classification": classification}
        append_jsonl(result, args.output)

    compute_error_distribution(args.output, total_tasks)
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
