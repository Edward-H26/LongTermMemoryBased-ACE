"""
Replay terminal benchmark rewards into V5 planner policy state.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from src.planner_policy import PlannerPolicy, compute_shaped_reward, default_planner_state_path


PLANNER_ACTIONS = ["direct", "explore", "refine", "deep_refine"]


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
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


def extract_task_id(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        task_id = metadata.get("task_id", item.get("task_id", ""))
        if isinstance(task_id, str) and task_id:
            return task_id
    task_id = item.get("task_id", "")
    return task_id if isinstance(task_id, str) else ""


def extract_action_id(item: Dict[str, Any]) -> str:
    metrics = item.get("metrics", {})
    if not isinstance(metrics, dict):
        return ""
    planner = metrics.get("planner", {})
    if not isinstance(planner, dict):
        return ""
    action_id = planner.get("action_id", "")
    return action_id if isinstance(action_id, str) else ""


def extract_reward_inputs(item: Dict[str, Any]) -> Dict[str, Any]:
    metrics = item.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    step_scoring = metrics.get("step_scoring", {})
    if not isinstance(step_scoring, dict):
        step_scoring = {}

    recursion = metrics.get("recursion", {})
    if not isinstance(recursion, dict):
        recursion = {}

    quality_gate = metrics.get("quality_gate", {})
    if not isinstance(quality_gate, dict):
        quality_gate = {}

    terminal_score = _safe_float(item.get("score", 0.0), 0.0)
    output_valid = bool(str(item.get("model_output", "")).strip())
    if not output_valid:
        output_valid = _safe_float(metrics.get("completion_tokens", 0.0), 0.0) > 0.0

    step_score = _safe_float(
        metrics.get("step_score_mean", step_scoring.get("mean_step_score", 0.0)),
        0.0,
    )
    step_confidence = _safe_float(step_scoring.get("overall_confidence", 0.7), 0.7)
    gate_applied = _safe_bool(quality_gate.get("should_apply_update", False), False)
    recursion_improved = _safe_bool(recursion.get("improved", False), False)

    return {
        "terminal_score": terminal_score,
        "output_valid": output_valid,
        "step_score": step_score,
        "step_confidence": step_confidence,
        "quality_gate_applied": gate_applied,
        "recursion_improved": recursion_improved,
    }


def replay_rows(
    rows: List[Dict[str, Any]],
    planner: PlannerPolicy,
    stream: str,
) -> Dict[str, Any]:
    updates = 0
    skipped = 0
    reward_sum = 0.0
    by_action: Dict[str, int] = {}
    unique_task_ids = set()

    for item in rows:
        task_id = extract_task_id(item)
        if task_id:
            unique_task_ids.add(task_id)
        action_id = extract_action_id(item)
        if not action_id:
            skipped += 1
            continue

        reward_inputs = extract_reward_inputs(item)
        reward_payload = compute_shaped_reward(
            step_score = reward_inputs["step_score"],
            output_valid = reward_inputs["output_valid"],
            quality_gate_applied = reward_inputs["quality_gate_applied"],
            recursion_improved = reward_inputs["recursion_improved"],
            terminal_score = reward_inputs["terminal_score"],
            step_confidence = reward_inputs["step_confidence"],
        )
        update_result = planner.update(
            action_id = action_id,
            reward = reward_payload.get("final_reward", 0.0),
            confidence = reward_payload.get("confidence", 0.7),
            metadata = {
                "stream": stream,
                "task_id": task_id,
                "phase": "policy_replay_v5",
            },
        )
        if bool(update_result.get("updated", False)):
            updates += 1
            reward_sum += float(reward_payload.get("final_reward", 0.0))
            by_action[action_id] = by_action.get(action_id, 0) + 1
        else:
            skipped += 1

    return {
        "stream": stream,
        "rows": len(rows),
        "unique_task_ids": len(unique_task_ids),
        "updates": updates,
        "skipped": skipped,
        "mean_terminal_reward": (reward_sum / updates) if updates > 0 else 0.0,
        "updates_by_action": by_action,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description = "Replay V5 graded rewards into planner policy state")
    parser.add_argument("--baseline-graded", type = str, required = True)
    parser.add_argument("--ace-graded", type = str, required = True)
    parser.add_argument("--output", type = str, default = "benchmark/results/v5/policy_replay_v5.json")
    parser.add_argument("--output-dir", type = str, default = "benchmark/results/v5")
    parser.add_argument("--mode", type = str, default = "both", choices = ["both", "baseline", "ace"])
    parser.add_argument("--baseline-policy-state", type = str, default = None)
    parser.add_argument("--ace-policy-state", type = str, default = None)
    parser.add_argument("--planner-epsilon", type = float, default = None)
    parser.add_argument("--planner-ucb-c", type = float, default = None)
    parser.add_argument("--planner-seed", type = int, default = 42)
    args = parser.parse_args()

    baseline_policy_path = args.baseline_policy_state or default_planner_state_path(
        role = "baseline",
        output_dir = args.output_dir,
    )
    ace_policy_path = args.ace_policy_state or default_planner_state_path(
        role = "ace",
        output_dir = args.output_dir,
    )

    baseline_rows = load_jsonl(args.baseline_graded) if args.mode in {"both", "baseline"} else []
    ace_rows = load_jsonl(args.ace_graded) if args.mode in {"both", "ace"} else []

    baseline_summary: Dict[str, Any] = {}
    ace_summary: Dict[str, Any] = {}

    if args.mode in {"both", "baseline"}:
        baseline_planner = PlannerPolicy(
            actions = list(PLANNER_ACTIONS),
            state_path = baseline_policy_path,
            epsilon = args.planner_epsilon,
            ucb_c = args.planner_ucb_c,
            seed = args.planner_seed,
        )
        baseline_summary = replay_rows(baseline_rows, baseline_planner, stream = "baseline_v5")
        baseline_summary["policy_state"] = baseline_planner.summary()

    if args.mode in {"both", "ace"}:
        ace_planner = PlannerPolicy(
            actions = list(PLANNER_ACTIONS),
            state_path = ace_policy_path,
            epsilon = args.planner_epsilon,
            ucb_c = args.planner_ucb_c,
            seed = args.planner_seed,
        )
        ace_summary = replay_rows(ace_rows, ace_planner, stream = "ace_v5")
        ace_summary["policy_state"] = ace_planner.summary()

    payload = {
        "generated_at": _utc_now_iso(),
        "mode": args.mode,
        "baseline_policy_path": baseline_policy_path,
        "ace_policy_path": ace_policy_path,
        "baseline": baseline_summary,
        "ace": ace_summary,
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok = True)
    with open(args.output, "w", encoding = "utf-8") as f:
        json.dump(payload, f, indent = 2, ensure_ascii = False)

    print(f"Policy replay summary: {args.output}")
    if baseline_summary:
        print(
            "Baseline replay:",
            f"rows={baseline_summary.get('rows', 0)}",
            f"updates={baseline_summary.get('updates', 0)}",
            f"mean_terminal_reward={baseline_summary.get('mean_terminal_reward', 0.0):.4f}",
        )
    if ace_summary:
        print(
            "ACE replay:",
            f"rows={ace_summary.get('rows', 0)}",
            f"updates={ace_summary.get('updates', 0)}",
            f"mean_terminal_reward={ace_summary.get('mean_terminal_reward', 0.0):.4f}",
        )


if __name__ == "__main__":
    main()
