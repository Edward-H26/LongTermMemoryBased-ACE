"""
Wait for v6 inference to complete, then run eval, error analysis, diagnostics, and comparison.
Run after infer_baseline_v6 and infer_ace_direct_v6 (or run_v6).
"""

import argparse
import os
import subprocess
import sys
import time
import json
from typing import Any, Dict, List, Tuple


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding = "utf-8") as f:
        return sum(1 for _ in f)


def _run_cmd(cmd: List[str]) -> int:
    proc = subprocess.Popen(cmd)
    return proc.wait()


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding = "utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(payload, f, indent = 2, ensure_ascii = False)


def _mark_phase_start(run_meta_path: str, phase_name: str) -> None:
    if not run_meta_path:
        return
    payload = _load_json(run_meta_path)
    phases = payload.get("phases", {})
    if not isinstance(phases, dict):
        phases = {}
    phase = phases.get(phase_name, {})
    if not isinstance(phase, dict):
        phase = {}
    phase["started_at"] = _utc_now_iso()
    phases[phase_name] = phase
    payload["phases"] = phases
    _write_json(run_meta_path, payload)


def _mark_phase_end(run_meta_path: str, phase_name: str) -> None:
    if not run_meta_path:
        return
    payload = _load_json(run_meta_path)
    phases = payload.get("phases", {})
    if not isinstance(phases, dict):
        phases = {}
    phase = phases.get(phase_name, {})
    if not isinstance(phase, dict):
        phase = {}
    phase["ended_at"] = _utc_now_iso()
    phases[phase_name] = phase
    payload["phases"] = phases
    payload["ended_at"] = _utc_now_iso()
    _write_json(run_meta_path, payload)


def _run_parallel_pair_with_retry(
    cmd_a: List[str],
    cmd_b: List[str],
    label_a: str,
    label_b: str,
    retry_max: int,
) -> None:
    p_a = subprocess.Popen(cmd_a)
    p_b = subprocess.Popen(cmd_b)
    code_a = p_a.wait()
    code_b = p_b.wait()

    retries_a = 0
    retries_b = 0
    while (code_a != 0 and retries_a < retry_max) or (code_b != 0 and retries_b < retry_max):
        if code_a != 0 and retries_a < retry_max:
            retries_a += 1
            print(f"Retrying {label_a} ({retries_a}/{retry_max})...")
            code_a = _run_cmd(cmd_a)
        if code_b != 0 and retries_b < retry_max:
            retries_b += 1
            print(f"Retrying {label_b} ({retries_b}/{retry_max})...")
            code_b = _run_cmd(cmd_b)

    if code_a != 0 or code_b != 0:
        raise RuntimeError(
            f"Parallel stage failed after retries: {label_a}={code_a}, {label_b}={code_b}"
        )


def _assert_task_id_parity(path_a: str, path_b: str) -> None:
    def _load_ids(path: str) -> List[str]:
        ids: List[str] = []
        if not os.path.exists(path):
            return ids
        with open(path, "r", encoding = "utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    import json
                    item = json.loads(raw)
                except Exception:
                    continue
                metadata = item.get("metadata", {})
                task_id = ""
                if isinstance(metadata, dict):
                    task_id = metadata.get("task_id", item.get("task_id", ""))
                if not task_id:
                    task_id = item.get("task_id", "")
                if isinstance(task_id, str) and task_id:
                    ids.append(task_id)
        return ids

    ids_a = set(_load_ids(path_a))
    ids_b = set(_load_ids(path_b))
    if ids_a != ids_b:
        missing_in_b = sorted(list(ids_a - ids_b))[:5]
        missing_in_a = sorted(list(ids_b - ids_a))[:5]
        raise RuntimeError(
            "Task-id parity check failed between baseline and ACE graded files: "
            f"missing_in_ace={missing_in_b}, missing_in_baseline={missing_in_a}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description = "V6 post-inference: eval, error analysis, diagnostics, compare")
    parser.add_argument("--output-dir", type = str, default = "benchmark/results/v6")
    parser.add_argument("--max-samples", type = int, default = 200)
    parser.add_argument("--judge-model", type = str, default = "gpt-5.1")
    parser.add_argument("--stage-retry-max", type = int, default = 1)
    parser.add_argument("--cost-mode", type = str, default = os.getenv("BENCHMARK_COST_MODE", "dual_source"), choices = ["legacy", "dual_source"])
    parser.add_argument("--billing-policy", type = str, default = os.getenv("BENCHMARK_BILLING_POLICY", "strict"), choices = ["strict", "off"])
    parser.add_argument("--run-meta", type = str, default = None)
    parser.add_argument("--openai-admin-key-env", type = str, default = "OPENAI_ADMIN_API_KEY")
    parser.add_argument("--openai-project-id-env", type = str, default = "OPENAI_COST_PROJECT_ID")
    parser.add_argument(
        "--skip-wait",
        action = "store_true",
        help = "Skip polling; assume inference is already complete (used when invoked by run_v6).",
    )
    args = parser.parse_args()

    if args.max_samples <= 0:
        print("Error: --max-samples must be > 0")
        sys.exit(1)

    baseline_path = os.path.join(args.output_dir, "baseline_v6.jsonl")
    ace_path = os.path.join(args.output_dir, "ace_v6.jsonl")
    baseline_graded_path = os.path.join(args.output_dir, "baseline_v6_graded.jsonl")
    ace_graded_path = os.path.join(args.output_dir, "ace_v6_graded.jsonl")
    baseline_errors_path = os.path.join(args.output_dir, "baseline_v6_graded_errors.jsonl")
    ace_errors_path = os.path.join(args.output_dir, "ace_v6_graded_errors.jsonl")
    baseline_eval_metrics_path = os.path.join(args.output_dir, "baseline_v6_graded_eval_metrics.json")
    ace_eval_metrics_path = os.path.join(args.output_dir, "ace_v6_graded_eval_metrics.json")
    baseline_error_metrics_path = os.path.join(args.output_dir, "baseline_v6_graded_errors_error_metrics.json")
    ace_error_metrics_path = os.path.join(args.output_dir, "ace_v6_graded_errors_error_metrics.json")
    policy_replay_path = os.path.join(args.output_dir, "policy_replay_v6.json")
    task_diagnostics_json_path = os.path.join(args.output_dir, "task_diagnostics_v6.jsonl")
    task_diagnostics_md_path = os.path.join(args.output_dir, "task_diagnostics_v6.md")
    run_meta_path = args.run_meta or os.path.join(args.output_dir, "run_v6_meta.json")

    if not args.skip_wait:
        print(f"Waiting for baseline_v6.jsonl and ace_v6.jsonl to reach {args.max_samples} lines...")
    while True:
        baseline_count = _count_lines(baseline_path)
        ace_count = _count_lines(ace_path)
        if not args.skip_wait:
            print(
                f"  Baseline: {baseline_count}/{args.max_samples}, "
                f"ACE: {ace_count}/{args.max_samples}",
                flush = True,
            )
        if baseline_count >= args.max_samples and ace_count >= args.max_samples:
            break
        if args.skip_wait:
            print(
                f"Error: inference incomplete. "
                f"Baseline: {baseline_count}/{args.max_samples}, ACE: {ace_count}/{args.max_samples}"
            )
            sys.exit(1)
        time.sleep(60)

    print("Inference complete. Running evaluation in parallel...")
    _mark_phase_start(run_meta_path, "evaluation")
    _run_parallel_pair_with_retry(
        [
            sys.executable,
            "-m",
            "benchmark.eval",
            "--input",
            baseline_path,
            "--output",
            baseline_graded_path,
            "--metrics-output",
            baseline_eval_metrics_path,
            "--judge-model",
            args.judge_model,
        ],
        [
            sys.executable,
            "-m",
            "benchmark.eval",
            "--input",
            ace_path,
            "--output",
            ace_graded_path,
            "--metrics-output",
            ace_eval_metrics_path,
            "--judge-model",
            args.judge_model,
        ],
        label_a = "baseline_eval",
        label_b = "ace_eval",
        retry_max = max(0, args.stage_retry_max),
    )
    _mark_phase_end(run_meta_path, "evaluation")

    if _count_lines(baseline_graded_path) < args.max_samples or _count_lines(ace_graded_path) < args.max_samples:
        raise RuntimeError(
            "Evaluation outputs are incomplete after retries: "
            f"baseline={_count_lines(baseline_graded_path)}/{args.max_samples}, "
            f"ace={_count_lines(ace_graded_path)}/{args.max_samples}"
        )

    print("Replaying terminal rewards into planner policy...")
    _mark_phase_start(run_meta_path, "policy_replay")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.v6.policy_replay",
            "--baseline-graded",
            baseline_graded_path,
            "--ace-graded",
            ace_graded_path,
            "--output",
            policy_replay_path,
            "--output-dir",
            args.output_dir,
            "--mode",
            "both",
        ],
        check = True,
    )
    _mark_phase_end(run_meta_path, "policy_replay")

    print("Running error analysis in parallel...")
    _mark_phase_start(run_meta_path, "error_analysis")
    _run_parallel_pair_with_retry(
        [
            sys.executable,
            "-m",
            "benchmark.error_analysis",
            "--input",
            baseline_graded_path,
            "--output",
            baseline_errors_path,
            "--metrics-output",
            baseline_error_metrics_path,
            "--judge-model",
            args.judge_model,
        ],
        [
            sys.executable,
            "-m",
            "benchmark.error_analysis",
            "--input",
            ace_graded_path,
            "--output",
            ace_errors_path,
            "--metrics-output",
            ace_error_metrics_path,
            "--judge-model",
            args.judge_model,
        ],
        label_a = "baseline_error_analysis",
        label_b = "ace_error_analysis",
        retry_max = max(0, args.stage_retry_max),
    )
    _mark_phase_end(run_meta_path, "error_analysis")

    print("Checking task-id parity before compare...")
    _assert_task_id_parity(baseline_graded_path, ace_graded_path)

    print("Generating comparison report...")
    _mark_phase_start(run_meta_path, "compare")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.compare",
            "--baseline",
            baseline_graded_path,
            "--ace",
            ace_graded_path,
            "--baseline-errors",
            baseline_errors_path,
            "--ace-errors",
            ace_errors_path,
            "--ace-aux-metrics",
            os.path.join(args.output_dir, "ace_v6_metrics.json"),
            "--baseline-eval-metrics",
            baseline_eval_metrics_path,
            "--ace-eval-metrics",
            ace_eval_metrics_path,
            "--baseline-error-metrics",
            baseline_error_metrics_path,
            "--ace-error-metrics",
            ace_error_metrics_path,
            "--cost-mode",
            args.cost_mode,
            "--billing-policy",
            args.billing_policy,
            "--run-meta",
            run_meta_path,
            "--openai-admin-key-env",
            args.openai_admin_key_env,
            "--openai-project-id-env",
            args.openai_project_id_env,
            "--output",
            os.path.join(args.output_dir, "comparison_report_v6.md"),
            "--title-label",
            "V6",
        ],
        check = True,
    )
    _mark_phase_end(run_meta_path, "compare")

    print("Generating per-task diagnostics...")
    _mark_phase_start(run_meta_path, "task_diagnostics")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.v6.task_diagnostics",
            "--baseline-graded",
            baseline_graded_path,
            "--ace-graded",
            ace_graded_path,
            "--baseline-errors",
            baseline_errors_path,
            "--ace-errors",
            ace_errors_path,
            "--output-jsonl",
            task_diagnostics_json_path,
            "--output-md",
            task_diagnostics_md_path,
        ],
        check = True,
    )
    _mark_phase_end(run_meta_path, "task_diagnostics")

    print("Sanitizing result artifacts...")
    _mark_phase_start(run_meta_path, "sanitize")
    sanitize_report = os.path.join(args.output_dir, "sanitize_report_v6.json")
    sanitize_cmd = [
        sys.executable,
        "-m",
        "benchmark.sanitize",
        "--input-root",
        args.output_dir,
        "--version",
        "all",
        "--report-path",
        sanitize_report,
        "--mode",
        "strict",
        "--in-place",
    ]
    subprocess.run(sanitize_cmd, check = True)
    _mark_phase_end(run_meta_path, "sanitize")
    print(f"Sanitization report: {sanitize_report}")
    print(f"Task diagnostics JSONL: {task_diagnostics_json_path}")
    print(f"Task diagnostics Markdown: {task_diagnostics_md_path}")

    print("V6 pipeline complete.")


if __name__ == "__main__":
    main()
