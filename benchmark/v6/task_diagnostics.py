"""
Per-task diagnostics for baseline vs ACE outcomes in v6.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding = "utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    parent = os.path.dirname(path) if os.path.dirname(path) else "."
    os.makedirs(parent, exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii = False) + "\n")


def write_text(path: str, content: str) -> None:
    parent = os.path.dirname(path) if os.path.dirname(path) else "."
    os.makedirs(parent, exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        f.write(content)


def get_task_id(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        value = metadata.get("task_id", item.get("task_id", ""))
        if isinstance(value, str) and value:
            return value
    value = item.get("task_id", "")
    return value if isinstance(value, str) else ""


def get_category(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        value = metadata.get("context_category", "")
        if isinstance(value, str) and value:
            return value
    return "Unknown"


def parse_score(item: Dict[str, Any]) -> int:
    value = item.get("score", 0)
    try:
        return int(value)
    except Exception:
        return 0


def parse_failed_rubric_count(item: Dict[str, Any]) -> int:
    status = item.get("requirement_status", [])
    if not isinstance(status, list):
        return 0
    count = 0
    for entry in status:
        if isinstance(entry, bool):
            if not entry:
                count += 1
            continue
        if isinstance(entry, (int, float)):
            if float(entry) <= 0.0:
                count += 1
            continue
        text = str(entry).strip().lower()
        if text in {"no", "false", "0", "fail", "failed"}:
            count += 1
    return count


def parse_transition(baseline_score: int, ace_score: int) -> str:
    if baseline_score == 0 and ace_score == 0:
        return "both_fail"
    if baseline_score == 0 and ace_score == 1:
        return "ace_gain"
    if baseline_score == 1 and ace_score == 0:
        return "ace_regress"
    if baseline_score == 1 and ace_score == 1:
        return "both_pass"
    return "other"


def parse_evaluator_artifact(item: Dict[str, Any]) -> bool:
    score = parse_score(item)
    requirement_status = item.get("requirement_status", [])
    return bool(score == 0 and isinstance(requirement_status, list) and len(requirement_status) == 0)


def parse_error_types_by_task(path: str) -> Dict[str, List[str]]:
    rows = load_jsonl(path)
    output: Dict[str, List[str]] = {}
    for row in rows:
        task_id = get_task_id(row)
        if not task_id:
            continue
        classification = row.get("error_classification", {})
        if not isinstance(classification, dict):
            continue
        values = classification.get("task_error_types", [])
        if not isinstance(values, list):
            values = []
        clean = sorted(set(str(value).strip() for value in values if str(value).strip()))
        output[task_id] = clean
    return output


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def build_markdown(rows: List[Dict[str, Any]]) -> str:
    transition_counts = Counter(row.get("transition_type", "") for row in rows)
    error_counts: Counter = Counter()
    planner_counts: Counter = Counter()
    baseline_artifacts = 0
    ace_artifacts = 0

    for row in rows:
        if bool(row.get("baseline_evaluator_artifact", False)):
            baseline_artifacts += 1
        if bool(row.get("ace_evaluator_artifact", False)):
            ace_artifacts += 1
        planner_action = str(row.get("ace_planner_action", "")).strip()
        if planner_action:
            planner_counts[planner_action] += 1
        for error_type in row.get("ace_error_types", []):
            error_counts[error_type] += 1

    lines: List[str] = []
    lines.append("# V6 Task Diagnostics\n")
    lines.append("## Summary\n")
    both_fail = transition_counts.get("both_fail", 0)
    both_pass = transition_counts.get("both_pass", 0)
    ace_gain = transition_counts.get("ace_gain", 0)
    ace_regress = transition_counts.get("ace_regress", 0)
    lines.append(f"- Total paired tasks: {len(rows)}")
    lines.append(f"- both_fail: {both_fail}")
    lines.append(f"- both_pass: {both_pass}")
    lines.append(f"- ace_gain: {ace_gain}")
    lines.append(f"- ace_regress: {ace_regress}")
    lines.append(f"- Baseline evaluator artifacts: {baseline_artifacts}")
    lines.append(f"- ACE evaluator artifacts: {ace_artifacts}")
    lines.append("")

    if planner_counts:
        lines.append("## ACE Planner Actions\n")
        lines.append("| Action | Count |")
        lines.append("|---|---|")
        for action, count in sorted(planner_counts.items(), key = lambda row: (-row[1], row[0])):
            lines.append(f"| {action} | {count} |")
        lines.append("")

    if error_counts:
        lines.append("## ACE Error Types\n")
        lines.append("| Error Type | Count |")
        lines.append("|---|---|")
        for error_type, count in sorted(error_counts.items(), key = lambda row: (-row[1], row[0])):
            lines.append(f"| {error_type} | {count} |")
        lines.append("")

    lines.append("## Per-Task Table\n")
    lines.append(
        "| Task ID | Category | Baseline | ACE | Transition | Action | Failed Rubrics B/A | "
        "ACE Errors | Gate Score | Lessons In/Accepted | Seed/Learned Retrieval | ACE Evaluator Artifact | Routing |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    for row in rows:
        ace_error_types = ",".join(row.get("ace_error_types", []))
        routing = row.get("routing_snapshot", {})
        if isinstance(routing, dict):
            solver_model = str(routing.get("solver_model", ""))
            reflector_backend = str(routing.get("reflector_backend", ""))
            reflector_model = str(routing.get("reflector_model", ""))
            step_backend = str(routing.get("step_scorer_backend", ""))
            step_model = str(routing.get("step_scorer_model", ""))
            routing_text = (
                f"solver={solver_model};"
                f"reflector={reflector_backend}:{reflector_model};"
                f"step={step_backend}:{step_model}"
            )
        else:
            routing_text = ""
        task_id = str(row.get("task_id", ""))
        category = str(row.get("category", ""))
        baseline_score = int(row.get("baseline_score", 0))
        ace_score = int(row.get("ace_score", 0))
        transition_type = str(row.get("transition_type", ""))
        ace_planner_action = str(row.get("ace_planner_action", ""))
        baseline_failed_rubrics = int(row.get("baseline_failed_rubrics", 0))
        ace_failed_rubrics = int(row.get("ace_failed_rubrics", 0))
        ace_gate_score = float(row.get("ace_gate_score", 0.0))
        ace_lessons_extracted = int(row.get("ace_lessons_extracted", 0))
        ace_lessons_accepted = int(row.get("ace_lessons_accepted", 0))
        ace_seed_retrieval_count = int(row.get("ace_seed_retrieval_count", 0))
        ace_learned_retrieval_count = int(row.get("ace_learned_retrieval_count", 0))
        ace_evaluator_artifact = bool(row.get("ace_evaluator_artifact", False))
        lines.append(
            f"| {task_id} | {category} | {baseline_score} | "
            f"{ace_score} | {transition_type} | {ace_planner_action} | "
            f"{baseline_failed_rubrics}/{ace_failed_rubrics} | {ace_error_types} | "
            f"{ace_gate_score:.3f} | {ace_lessons_extracted}/{ace_lessons_accepted} | "
            f"{ace_seed_retrieval_count}/{ace_learned_retrieval_count} | "
            f"{ace_evaluator_artifact} | {routing_text} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description = "Generate v6 per-task diagnostics for baseline vs ACE")
    parser.add_argument("--baseline-graded", type = str, required = True)
    parser.add_argument("--ace-graded", type = str, required = True)
    parser.add_argument("--baseline-errors", type = str, default = "")
    parser.add_argument("--ace-errors", type = str, default = "")
    parser.add_argument("--output-jsonl", type = str, default = "benchmark/results/v6/task_diagnostics_v6.jsonl")
    parser.add_argument("--output-md", type = str, default = "benchmark/results/v6/task_diagnostics_v6.md")
    args = parser.parse_args()

    baseline_rows = load_jsonl(args.baseline_graded)
    ace_rows = load_jsonl(args.ace_graded)
    baseline_by_id = {get_task_id(row): row for row in baseline_rows if get_task_id(row)}
    ace_by_id = {get_task_id(row): row for row in ace_rows if get_task_id(row)}
    ace_error_types_by_id = parse_error_types_by_task(args.ace_errors)

    ordered_ids: List[str] = []
    seen = set()
    for row in baseline_rows:
        task_id = get_task_id(row)
        if task_id and task_id not in seen:
            seen.add(task_id)
            ordered_ids.append(task_id)
    for row in ace_rows:
        task_id = get_task_id(row)
        if task_id and task_id not in seen:
            seen.add(task_id)
            ordered_ids.append(task_id)

    diagnostics_rows: List[Dict[str, Any]] = []
    for task_id in ordered_ids:
        baseline_row = baseline_by_id.get(task_id, {})
        ace_row = ace_by_id.get(task_id, {})
        baseline_score = parse_score(baseline_row)
        ace_score = parse_score(ace_row)
        transition_type = parse_transition(baseline_score, ace_score)

        ace_metrics = ace_row.get("metrics", {})
        if not isinstance(ace_metrics, dict):
            ace_metrics = {}
        ace_planner = ace_metrics.get("planner", {})
        if not isinstance(ace_planner, dict):
            ace_planner = {}
        ace_recursion = ace_metrics.get("recursion", {})
        if not isinstance(ace_recursion, dict):
            ace_recursion = {}
        ace_quality_gate = ace_metrics.get("quality_gate", {})
        if not isinstance(ace_quality_gate, dict):
            ace_quality_gate = {}
        routing_snapshot = ace_metrics.get("model_routing", {})
        if not isinstance(routing_snapshot, dict):
            routing_snapshot = {}

        row = {
            "task_id": task_id,
            "category": get_category(ace_row or baseline_row),
            "baseline_score": baseline_score,
            "ace_score": ace_score,
            "transition_type": transition_type,
            "ace_planner_action": str(ace_planner.get("action_id", "")),
            "ace_planner_explore": bool(ace_planner.get("explore", False)),
            "ace_recursion_rounds_planned": safe_int(ace_recursion.get("rounds_planned"), 0),
            "ace_recursion_rounds_used": safe_int(ace_recursion.get("rounds_used"), 0),
            "ace_recursion_candidates_per_round": safe_int(ace_recursion.get("candidates_per_round"), 0),
            "ace_recursion_candidate_calls": safe_int(ace_recursion.get("candidate_calls"), 0),
            "ace_recursion_exit_reason": str(ace_recursion.get("exit_reason", "")),
            "ace_recursion_improved": bool(ace_recursion.get("improved", False)),
            "baseline_failed_rubrics": parse_failed_rubric_count(baseline_row),
            "ace_failed_rubrics": parse_failed_rubric_count(ace_row),
            "ace_error_types": ace_error_types_by_id.get(task_id, []),
            "ace_gate_score": safe_float(ace_quality_gate.get("gate_score"), 0.0),
            "ace_lessons_extracted": safe_int(ace_metrics.get("num_lessons_extracted"), 0),
            "ace_lessons_accepted": safe_int(ace_metrics.get("num_lessons_accepted"), 0),
            "ace_seed_retrieval_count": safe_int(ace_metrics.get("num_seed_bullets_retrieved"), 0),
            "ace_learned_retrieval_count": safe_int(ace_metrics.get("num_learned_bullets_retrieved"), 0),
            "baseline_evaluator_artifact": parse_evaluator_artifact(baseline_row),
            "ace_evaluator_artifact": parse_evaluator_artifact(ace_row),
            "routing_snapshot": routing_snapshot,
        }
        diagnostics_rows.append(row)

    write_jsonl(args.output_jsonl, diagnostics_rows)
    write_text(args.output_md, build_markdown(diagnostics_rows))
    print(f"Task diagnostics JSONL: {args.output_jsonl}")
    print(f"Task diagnostics Markdown: {args.output_md}")
    print(f"Paired task rows: {len(diagnostics_rows)}")


if __name__ == "__main__":
    main()
