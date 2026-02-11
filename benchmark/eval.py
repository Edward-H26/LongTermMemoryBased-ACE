"""
CL-bench Evaluation: GPT-5.1 as judge.

Grades model outputs against rubrics using binary 0/1 scoring.
Adapted from https://github.com/Tencent-Hunyuan/CL-bench/blob/main/eval.py

Usage:
    python -m benchmark.eval --input benchmark/results/baseline.jsonl --judge-model gpt-5.1
"""

import json
import os
import time
import argparse
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from tqdm import tqdm


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


def build_rubrics_text(rubrics):
    if not rubrics:
        return "No specific rubrics provided."
    lines = []
    for i, rubric in enumerate(rubrics, 1):
        if isinstance(rubric, dict):
            criteria = rubric.get("rubric_criteria", "").strip()
        else:
            criteria = str(rubric).strip()
        if criteria:
            lines.append(f"{i}. {criteria}")
    return "\n".join(lines) if lines else "No specific rubrics provided."


def call_judge(client, model, rubrics_text, model_output, max_retries=3):
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
        '  "Grading Rationale": "Your detailed grading rationale",\n'
        '  "List of Requirement Satisfaction Status": ["yes", "no", ...],\n'
        '  "Overall Score": 0 or 1\n'
        "}\n\n"
        "Content to Be Graded\n"
        f"[Rubrics]:\n{rubrics_text}\n"
        f"[Student Response]:\n{model_output}\n"
    )

    messages = [{"role": "user", "content": grading_prompt}]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=model, messages=messages, max_completion_tokens=4096)
            result_text = response.choices[0].message.content.strip()

            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()

            return result_text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None

    return None


def get_task_id(item):
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get("task_id", item.get("task_id", ""))
    return item.get("task_id", "")


def calculate_statistics(output_path):
    if not os.path.exists(output_path):
        return
    data = load_jsonl(output_path)
    total = len(data)
    score_0 = sum(1 for item in data if item.get("score") == 0)
    score_1 = sum(1 for item in data if item.get("score") == 1)

    print(f"\nFinal Statistics:")
    print(f"  Total samples: {total}")
    print(f"  Score 0: {score_0}")
    print(f"  Score 1: {score_1}")

    if total > 0:
        solving_rate = score_1 / total
        print(f"\n  Solving Rate: {solving_rate:.4f} ({score_1}/{total})")

    category_stats = {}
    for item in data:
        metadata = item.get("metadata", {})
        if isinstance(metadata, dict):
            category = metadata.get("context_category", "Unknown")
        else:
            category = "Unknown"
        stats = category_stats.setdefault(category, {"total": 0, "score_0": 0, "score_1": 0})
        stats["total"] += 1
        if item.get("score") == 1:
            stats["score_1"] += 1
        else:
            stats["score_0"] += 1

    if category_stats:
        print(f"\n  Scores by context_category:")
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            rate = stats["score_1"] / stats["total"] if stats["total"] else 0
            print(f"    {category}: total={stats['total']}, score_1={stats['score_1']}, rate={rate:.4f}")


def main():
    parser = argparse.ArgumentParser(description="CL-bench Evaluation")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default="gpt-5.1")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"benchmark/results/{base_name}_graded.jsonl"

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
    print(f"Judge: {args.judge_model}")

    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} samples")

    completed_ids = set()
    if os.path.exists(args.output):
        existing = load_jsonl(args.output)
        completed_ids = {get_task_id(item) for item in existing if get_task_id(item)}
        print(f"Found {len(completed_ids)} completed, resuming remaining")

    pending = [item for item in data if get_task_id(item) not in completed_ids]
    if not pending:
        print("All samples already evaluated")
        calculate_statistics(args.output)
        return

    print(f"Evaluating {len(pending)} samples...")
    success_count = 0
    fail_count = 0

    for item in tqdm(pending, desc="Evaluating"):
        task_id = get_task_id(item)
        model_output = item.get("model_output", "")
        rubrics = item.get("rubrics", [])

        if not model_output or not model_output.strip():
            result = {**item, "grading_rationale": "No model output (score 0)", "requirement_status": [], "score": 0}
            append_jsonl(result, args.output)
            success_count += 1
            continue

        rubrics_text = build_rubrics_text(rubrics)
        grading_result = call_judge(client, args.judge_model, rubrics_text, model_output, args.max_retries)

        if not grading_result:
            result = {**item, "grading_rationale": "API call failed (score 0)", "requirement_status": [], "score": 0}
            append_jsonl(result, args.output)
            fail_count += 1
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
        except (json.JSONDecodeError, ValueError):
            result = {
                **item,
                "grading_rationale": f"JSON parse failed: {grading_result[:500]}",
                "requirement_status": [],
                "score": 0,
            }
            append_jsonl(result, args.output)
            fail_count += 1

    print(f"\nDone: {success_count} success, {fail_count} failed")
    calculate_statistics(args.output)


if __name__ == "__main__":
    main()
