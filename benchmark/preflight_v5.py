"""
Preflight helpers for CL-bench V5 static checks, estimation, and mini smoke runs.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from benchmark.compare import GPT51_INPUT_PRICE, GPT51_OUTPUT_PRICE
from src.step_scoring import split_reasoning_steps


SUPPORTED_MEMORY_SCOPES = {"hybrid", "local", "global"}
SUPPORTED_SAMPLING_STRATEGIES = {"task_random", "context_dense", "context_dense_stratified"}
SUPPORTED_SMOKE_SAMPLES = {3, 4, 5}
VERSION_NAMES = {"v1", "v2", "v3", "v4", "v5"}

REQUIRED_ENV_KEYS = [
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
]

REQUIRED_MODULES = [
    "benchmark.infer_baseline_v5",
    "benchmark.infer_ace_direct_v5",
    "benchmark.complete_v5_pipeline",
    "benchmark.compare",
    "benchmark.eval",
    "benchmark.error_analysis",
]

REQUIRED_PACKAGES = [
    "openai",
    "neo4j",
    "datasets",
    "dotenv",
]

EVAL_SECONDS_PER_TASK = 21.0
ERROR_SECONDS_PER_FAILED_TASK = 7.0
EVAL_PROMPT_TOKENS_PER_TASK = 2200.0
EVAL_COMPLETION_TOKENS_PER_TASK = 350.0
ERROR_PROMPT_TOKENS_PER_FAILED_TASK = 1300.0
ERROR_COMPLETION_TOKENS_PER_FAILED_TASK = 220.0
COMPARE_SECONDS = 8.0

STEP_PROMPT_TOKENS_PER_CALL_FALLBACK = 220.0
STEP_COMPLETION_TOKENS_PER_CALL_FALLBACK = 40.0
STEP_SECONDS_PER_CALL_FALLBACK = 1.8

HEURISTIC_BASELINE = {
    "avg_prompt_tokens": 10008.315,
    "avg_completion_tokens": 2092.68,
    "avg_latency_ms": 24481.1036914296,
}
HEURISTIC_ACE = {
    "avg_prompt_tokens": 10140.82,
    "avg_completion_tokens": 2004.46,
    "avg_latency_ms": 24025.406068528537,
}
HEURISTIC_SOLVING = {
    "baseline_overall": 12.0,
    "ace_overall": 16.0,
}

UNSANITIZED_TOP_LEVEL_FIELDS = {
    "messages",
    "model_output",
    "rubrics",
    "grading_rationale",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding = "utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
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


def _find_unsanitized_jsonl_rows(root_dir: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    if not os.path.isdir(root_dir):
        return findings

    for current_root, _, files in os.walk(root_dir):
        for file_name in files:
            if not file_name.endswith(".jsonl") or file_name.endswith(".progress.jsonl"):
                continue
            path = os.path.join(current_root, file_name)
            detected_fields = set()
            inspected_rows = 0
            try:
                with open(path, "r", encoding = "utf-8") as handle:
                    for raw in handle:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(row, dict):
                            continue
                        inspected_rows += 1

                        for key in UNSANITIZED_TOP_LEVEL_FIELDS:
                            if key in row:
                                detected_fields.add(key)

                        metrics = row.get("metrics", {})
                        if isinstance(metrics, dict):
                            if "memory_error" in metrics:
                                detected_fields.add("metrics.memory_error")
                            step_scoring = metrics.get("step_scoring", {})
                            if isinstance(step_scoring, dict) and "steps" in step_scoring:
                                detected_fields.add("metrics.step_scoring.steps")

                        error_classification = row.get("error_classification", {})
                        if isinstance(error_classification, dict) and "per_rubric_classification" in error_classification:
                            detected_fields.add("error_classification.per_rubric_classification")

                        if inspected_rows >= 5:
                            break
            except Exception:
                continue

            if detected_fields:
                findings.append({
                    "path": path,
                    "detected_fields": sorted(detected_fields),
                    "inspected_rows": inspected_rows,
                })

    return findings


def _tracked_jsonl_violations() -> List[str]:
    if not os.path.isdir(".git"):
        return []
    try:
        result = subprocess.run(
            ["git", "ls-files", "benchmark/results"],
            check = False,
            capture_output = True,
            text = True,
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []

    violations: List[str] = []
    for raw in result.stdout.splitlines():
        path = raw.strip()
        if not path:
            continue
        if path.endswith(".jsonl") and not path.endswith(".progress.jsonl"):
            violations.append(path)
    return violations


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding = "utf-8") as f:
        return sum(1 for _ in f)


def _estimate_cost_usd(prompt_tokens: float, completion_tokens: float) -> float:
    return (
        (prompt_tokens / 1_000_000.0 * GPT51_INPUT_PRICE) +
        (completion_tokens / 1_000_000.0 * GPT51_OUTPUT_PRICE)
    )


def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok = True)
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w", encoding = "utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def _build_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    cache_dir = env.get("HF_DATASETS_CACHE")
    if not cache_dir:
        default_cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        if _is_writable_dir(default_cache_dir):
            cache_dir = default_cache_dir
        else:
            cache_dir = os.path.abspath(os.path.join("benchmark", "hf_cache"))
        env["HF_DATASETS_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok = True)
    return env


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path) if os.path.dirname(path) else "."
    os.makedirs(parent, exist_ok = True)


def _choose_source_path(estimate_source: str, output_dir: str) -> Tuple[str, str]:
    local_v5 = os.path.join(output_dir, "comparison_report_v5.json")
    default_v5 = os.path.join("benchmark", "results", "v5", "comparison_report_v5.json")
    local_v4 = os.path.join(output_dir, "comparison_report_v4.json")
    default_v4 = os.path.join("benchmark", "results", "v4", "comparison_report_v4.json")
    default_v3 = os.path.join("benchmark", "results", "v3", "comparison_report_v3.json")

    if estimate_source == "v5":
        if os.path.exists(local_v5):
            return "v5", local_v5
        if os.path.exists(default_v5):
            return "v5", default_v5
        if os.path.exists(local_v4):
            return "v4", local_v4
        if os.path.exists(default_v4):
            return "v4", default_v4
        return "heuristic", ""

    if estimate_source == "v4":
        if os.path.exists(local_v4):
            return "v4", local_v4
        if os.path.exists(default_v4):
            return "v4", default_v4
        return "heuristic", ""

    if estimate_source == "v3":
        if os.path.exists(default_v3):
            return "v3", default_v3
        return "heuristic", ""

    if estimate_source == "heuristic":
        return "heuristic", ""

    if os.path.exists(local_v4):
        return "v4", local_v4
    if os.path.exists(default_v4):
        return "v4", default_v4
    if os.path.exists(local_v5):
        return "v5", local_v5
    if os.path.exists(default_v5):
        return "v5", default_v5
    if os.path.exists(default_v3):
        return "v3", default_v3
    return "heuristic", ""


def _derive_step_inputs(source_tag: str, source_dir: str) -> Dict[str, Any]:
    fallback_info = {
        "mean_steps": 18.0,
        "non_empty_rate": 0.8,
        "source_paths": [],
        "fallback_used": True,
    }

    candidates: List[Tuple[str, str]] = []
    if source_tag == "v5":
        candidates.append((
            os.path.join(source_dir, "ace_v5.jsonl"),
            os.path.join(source_dir, "ace_v5_graded.jsonl"),
        ))
        candidates.append((
            os.path.join("benchmark", "results", "v5", "ace_v5.jsonl"),
            os.path.join("benchmark", "results", "v5", "ace_v5_graded.jsonl"),
        ))
    if source_tag == "v4":
        candidates.append((
            os.path.join(source_dir, "ace_v4.jsonl"),
            os.path.join(source_dir, "ace_v4_graded.jsonl"),
        ))
        candidates.append((
            os.path.join("benchmark", "results", "v4", "ace_v4.jsonl"),
            os.path.join("benchmark", "results", "v4", "ace_v4_graded.jsonl"),
        ))
    candidates.append((
        os.path.join("benchmark", "results", "v5", "ace_v5.jsonl"),
        os.path.join("benchmark", "results", "v5", "ace_v5_graded.jsonl"),
    ))
    candidates.append((
        os.path.join("benchmark", "results", "v4", "ace_v4.jsonl"),
        os.path.join("benchmark", "results", "v4", "ace_v4_graded.jsonl"),
    ))
    candidates.append((
        os.path.join("benchmark", "results", "v3", "ace_v3.jsonl"),
        os.path.join("benchmark", "results", "v3", "ace_v3_graded.jsonl"),
    ))

    for ace_path, graded_path in candidates:
        rows = _load_jsonl(ace_path)
        if not rows:
            continue
        graded_rows = _load_jsonl(graded_path)
        total_graded = len(graded_rows)
        no_output = 0
        if total_graded > 0:
            for item in graded_rows:
                rationale = str(item.get("grading_rationale", ""))
                if rationale.strip() == "No model output (score 0)":
                    no_output += 1
            non_empty_rate = max(0.0, min(1.0, (total_graded - no_output) / total_graded))
        else:
            non_empty_rate = 0.8

        step_counts: List[int] = []
        for item in rows:
            output = str(item.get("model_output", "")).strip()
            if not output:
                continue
            step_counts.append(len(split_reasoning_steps(output)))

        if step_counts:
            mean_steps = sum(step_counts) / len(step_counts)
            return {
                "mean_steps": mean_steps,
                "non_empty_rate": non_empty_rate,
                "source_paths": [ace_path, graded_path],
                "fallback_used": False,
                "sample_size": len(rows),
            }

    return fallback_info


def _derive_empirical_step_profile(source_tag: str, source_dir: str) -> Dict[str, Any]:
    fallback = {
        "prompt_tokens_per_call": STEP_PROMPT_TOKENS_PER_CALL_FALLBACK,
        "completion_tokens_per_call": STEP_COMPLETION_TOKENS_PER_CALL_FALLBACK,
        "seconds_per_call": STEP_SECONDS_PER_CALL_FALLBACK,
        "source_paths": [],
        "fallback_used": True,
    }

    candidates: List[str] = []
    if source_tag == "v5":
        candidates.append(os.path.join(source_dir, "ace_v5_metrics.json"))
        candidates.append(os.path.join("benchmark", "results", "v5", "ace_v5_metrics.json"))
    if source_tag == "v4":
        candidates.append(os.path.join(source_dir, "ace_v4_metrics.json"))
        candidates.append(os.path.join("benchmark", "results", "v4", "ace_v4_metrics.json"))
    candidates.extend(
        [
            os.path.join("benchmark", "results", "v5", "ace_v5_metrics.json"),
            os.path.join("benchmark", "results", "v4", "ace_v4_metrics.json"),
            os.path.join("benchmark", "results", "v3", "ace_v3_metrics.json"),
        ]
    )

    seen = set()
    for metrics_path in candidates:
        if metrics_path in seen:
            continue
        seen.add(metrics_path)
        payload = _load_json(metrics_path)
        summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {}
        total_calls = _safe_float(summary.get("total_calls", 0.0), 0.0)
        total_prompt = _safe_float(summary.get("total_prompt_tokens", 0.0), 0.0)
        total_completion = _safe_float(summary.get("total_completion_tokens", 0.0), 0.0)
        avg_latency_ms = _safe_float(summary.get("avg_latency_ms", 0.0), 0.0)
        if total_calls <= 0.0:
            continue
        prompt_per_call = total_prompt / total_calls if total_prompt > 0 else STEP_PROMPT_TOKENS_PER_CALL_FALLBACK
        completion_per_call = total_completion / total_calls if total_completion > 0 else STEP_COMPLETION_TOKENS_PER_CALL_FALLBACK
        seconds_per_call = (avg_latency_ms / 1000.0) if avg_latency_ms > 0 else STEP_SECONDS_PER_CALL_FALLBACK
        return {
            "prompt_tokens_per_call": prompt_per_call,
            "completion_tokens_per_call": completion_per_call,
            "seconds_per_call": seconds_per_call,
            "source_paths": [metrics_path],
            "fallback_used": False,
            "source_total_calls": total_calls,
        }

    return fallback


def run_static_checks(
    manifest_path: str,
    output_dir: str,
    progress_path: str,
    max_samples: int,
    sampling_strategy: str,
    memory_scope: str,
    smoke_samples: int,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    blocking_issues: List[str] = []
    warnings: List[str] = []

    if max_samples <= 0:
        blocking_issues.append("max_samples must be > 0.")
    checks.append({
        "name": "max_samples",
        "ok": max_samples > 0,
        "detail": f"max_samples={max_samples}",
    })

    smoke_ok = smoke_samples in SUPPORTED_SMOKE_SAMPLES
    if not smoke_ok:
        blocking_issues.append("smoke_samples must be one of 3, 4, 5.")
    checks.append({
        "name": "smoke_samples",
        "ok": smoke_ok,
        "detail": f"smoke_samples={smoke_samples}",
    })

    strategy_ok = sampling_strategy in SUPPORTED_SAMPLING_STRATEGIES
    if not strategy_ok:
        blocking_issues.append(
            f"sampling_strategy must be one of {sorted(SUPPORTED_SAMPLING_STRATEGIES)}."
        )
    checks.append({
        "name": "sampling_strategy",
        "ok": strategy_ok,
        "detail": sampling_strategy,
    })

    scope_ok = memory_scope in SUPPORTED_MEMORY_SCOPES
    if not scope_ok:
        blocking_issues.append(
            f"memory_scope must be one of {sorted(SUPPORTED_MEMORY_SCOPES)}."
        )
    checks.append({
        "name": "memory_scope",
        "ok": scope_ok,
        "detail": memory_scope,
    })

    missing_env = [key for key in REQUIRED_ENV_KEYS if not str(os.getenv(key, "")).strip()]
    if missing_env:
        blocking_issues.append(f"Missing required env keys: {', '.join(missing_env)}")
    checks.append({
        "name": "required_env",
        "ok": len(missing_env) == 0,
        "detail": "all present" if not missing_env else f"missing={missing_env}",
    })

    cost_mode = str(os.getenv("BENCHMARK_COST_MODE", "dual_source")).strip().lower()
    billing_policy = str(os.getenv("BENCHMARK_BILLING_POLICY", "strict")).strip().lower()
    strict_billing_needed = cost_mode == "dual_source" and billing_policy == "strict"
    strict_missing: List[str] = []
    if strict_billing_needed:
        for key in ["OPENAI_ADMIN_API_KEY", "OPENAI_COST_PROJECT_ID"]:
            if not str(os.getenv(key, "")).strip():
                strict_missing.append(key)
    if strict_missing:
        blocking_issues.append(
            "Strict dual-source billing reconciliation requires env keys: "
            + ", ".join(strict_missing)
        )
    checks.append({
        "name": "strict_billing_env",
        "ok": len(strict_missing) == 0,
        "detail": "strict billing prerequisites satisfied" if strict_billing_needed and not strict_missing else (
            "strict billing not active" if not strict_billing_needed else f"missing={strict_missing}"
        ),
    })

    missing_modules = [name for name in REQUIRED_MODULES if not _module_available(name)]
    if missing_modules:
        blocking_issues.append(f"Missing required benchmark modules: {', '.join(missing_modules)}")
    checks.append({
        "name": "benchmark_modules",
        "ok": len(missing_modules) == 0,
        "detail": "all importable" if not missing_modules else f"missing={missing_modules}",
    })

    missing_packages = [name for name in REQUIRED_PACKAGES if not _module_available(name)]
    if missing_packages:
        blocking_issues.append(f"Missing required packages: {', '.join(missing_packages)}")
    checks.append({
        "name": "runtime_packages",
        "ok": len(missing_packages) == 0,
        "detail": "all importable" if not missing_packages else f"missing={missing_packages}",
    })

    manifest_exists = os.path.exists(manifest_path)
    manifest_ok = True
    manifest_detail = "existing manifest validated"
    if manifest_exists:
        try:
            payload = _load_json(manifest_path)
            task_ids = payload.get("task_ids", [])
            if not isinstance(task_ids, list):
                manifest_ok = False
                manifest_detail = "manifest exists but task_ids is not a list"
            elif not task_ids:
                warnings.append("Manifest exists but task_ids is empty.")
                manifest_detail = "manifest exists with empty task_ids"
            else:
                manifest_detail = f"manifest exists with {len(task_ids)} task_ids"
        except Exception as exc:
            manifest_ok = False
            manifest_detail = f"manifest parse failed: {exc}"
    else:
        parent = os.path.dirname(manifest_path) if os.path.dirname(manifest_path) else "."
        if not os.path.isdir(parent):
            warnings.append(f"Manifest parent does not exist yet: {parent}")
        if os.path.exists(parent) and not os.access(parent, os.W_OK):
            manifest_ok = False
            manifest_detail = f"manifest parent not writable: {parent}"
        else:
            manifest_detail = "manifest will be created"
    if not manifest_ok:
        blocking_issues.append("Manifest path is not usable.")
    checks.append({
        "name": "manifest_path",
        "ok": manifest_ok,
        "detail": manifest_detail,
    })

    output_dir_ok = True
    output_detail = f"output_dir={output_dir}"
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        output_dir_ok = False
        output_detail = "output path exists and is not a directory"
    elif os.path.isdir(output_dir) and not os.access(output_dir, os.W_OK):
        output_dir_ok = False
        output_detail = "output directory not writable"
    elif not os.path.exists(output_dir):
        parent = os.path.dirname(output_dir) if os.path.dirname(output_dir) else "."
        if os.path.exists(parent) and not os.access(parent, os.W_OK):
            output_dir_ok = False
            output_detail = f"output parent not writable: {parent}"
        else:
            warnings.append(f"Output directory will be created: {output_dir}")
    if not output_dir_ok:
        blocking_issues.append("Output directory is not writable.")
    checks.append({
        "name": "output_dir",
        "ok": output_dir_ok,
        "detail": output_detail,
    })

    progress_ok = True
    progress_detail = f"progress_path={progress_path}"
    progress_parent = os.path.dirname(progress_path) if os.path.dirname(progress_path) else "."
    if os.path.exists(progress_path) and os.path.isdir(progress_path):
        progress_ok = False
        progress_detail = "progress path points to a directory"
    elif os.path.exists(progress_parent) and not os.access(progress_parent, os.W_OK):
        progress_ok = False
        progress_detail = f"progress parent not writable: {progress_parent}"
    if not progress_ok:
        blocking_issues.append("Progress path is not writable.")
    checks.append({
        "name": "progress_path",
        "ok": progress_ok,
        "detail": progress_detail,
    })

    checks.append({
        "name": "connectivity_probes",
        "ok": True,
        "detail": "optional probes skipped in static mode",
    })

    publish_root = output_dir
    if publish_root.endswith("/"):
        publish_root = publish_root[:-1]
    if os.path.basename(publish_root) in VERSION_NAMES:
        publish_root = os.path.dirname(publish_root) or publish_root

    unsanitized_findings = _find_unsanitized_jsonl_rows(publish_root)
    if unsanitized_findings:
        warnings.append(
            f"Unsanitized JSONL detected under publish root {publish_root}: "
            f"{len(unsanitized_findings)} file(s)."
        )
    checks.append({
        "name": "unsanitized_jsonl_scan",
        "ok": len(unsanitized_findings) == 0,
        "detail": "no unsanitized JSONL detected" if not unsanitized_findings else unsanitized_findings[:5],
    })

    tracked_violations = _tracked_jsonl_violations()
    if tracked_violations:
        warnings.append(
            "Tracked JSONL artifacts detected in benchmark/results; these are not safe to publish."
        )
    checks.append({
        "name": "tracked_jsonl_policy",
        "ok": len(tracked_violations) == 0,
        "detail": "no tracked JSONL artifacts" if not tracked_violations else tracked_violations[:10],
    })

    return {
        "checks": checks,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "status": "ok" if not blocking_issues else "failed",
    }


def build_estimate(
    max_samples: int,
    estimate_source: str,
    output_dir: str,
    context_workers: int,
    step_scoring_mode: str,
    step_score_workers: int,
) -> Dict[str, Any]:
    source_tag, source_path = _choose_source_path(estimate_source, output_dir)
    assumptions: List[str] = []

    if source_tag == "heuristic":
        baseline_avg_prompt = HEURISTIC_BASELINE["avg_prompt_tokens"]
        baseline_avg_completion = HEURISTIC_BASELINE["avg_completion_tokens"]
        baseline_avg_latency_s = HEURISTIC_BASELINE["avg_latency_ms"] / 1000.0
        ace_avg_prompt = HEURISTIC_ACE["avg_prompt_tokens"]
        ace_avg_completion = HEURISTIC_ACE["avg_completion_tokens"]
        ace_avg_latency_s = HEURISTIC_ACE["avg_latency_ms"] / 1000.0
        baseline_success = HEURISTIC_SOLVING["baseline_overall"] / 100.0
        ace_success = HEURISTIC_SOLVING["ace_overall"] / 100.0
        source_total = 200.0
        assumptions.append("Using heuristic fallback values because no comparison report source was found.")
    else:
        payload = _load_json(source_path)
        baseline_metrics = payload.get("baseline_metrics", {})
        ace_metrics = payload.get("ace_metrics", {})
        baseline_rates = payload.get("baseline_solving_rates", {})
        ace_rates = payload.get("ace_solving_rates", {})

        source_total = _safe_float(baseline_rates.get("total"), 0.0)
        if source_total <= 0:
            source_total = _safe_float(ace_rates.get("total"), 0.0)
        if source_total <= 0:
            source_total = 200.0
            assumptions.append("Source report missing task total; assumed 200.")

        baseline_avg_prompt = _safe_float(
            baseline_metrics.get("avg_prompt_tokens"),
            HEURISTIC_BASELINE["avg_prompt_tokens"],
        )
        baseline_avg_completion = _safe_float(
            baseline_metrics.get("avg_completion_tokens"),
            HEURISTIC_BASELINE["avg_completion_tokens"],
        )
        baseline_avg_latency_s = _safe_float(
            baseline_metrics.get("avg_latency_ms"),
            HEURISTIC_BASELINE["avg_latency_ms"],
        ) / 1000.0

        ace_avg_prompt = _safe_float(
            ace_metrics.get("avg_prompt_tokens"),
            HEURISTIC_ACE["avg_prompt_tokens"],
        )
        ace_avg_completion = _safe_float(
            ace_metrics.get("avg_completion_tokens"),
            HEURISTIC_ACE["avg_completion_tokens"],
        )
        ace_avg_latency_s = _safe_float(
            ace_metrics.get("avg_latency_ms"),
            HEURISTIC_ACE["avg_latency_ms"],
        ) / 1000.0

        baseline_success = _safe_float(baseline_rates.get("overall"), HEURISTIC_SOLVING["baseline_overall"]) / 100.0
        ace_success = _safe_float(ace_rates.get("overall"), HEURISTIC_SOLVING["ace_overall"]) / 100.0

    scale = max_samples / source_total if source_total > 0 else 1.0

    baseline_prompt_total = baseline_avg_prompt * max_samples
    baseline_completion_total = baseline_avg_completion * max_samples
    ace_prompt_total = ace_avg_prompt * max_samples
    ace_completion_total = ace_avg_completion * max_samples

    inference_cost = _estimate_cost_usd(
        baseline_prompt_total + ace_prompt_total,
        baseline_completion_total + ace_completion_total,
    )
    baseline_inference_seconds = baseline_avg_latency_s * max_samples
    ace_inference_seconds = ace_avg_latency_s * max_samples
    inference_wall_seconds = max(baseline_inference_seconds, ace_inference_seconds)

    source_dir = os.path.dirname(source_path) if source_path else output_dir
    step_inputs = _derive_step_inputs(source_tag = source_tag, source_dir = source_dir)
    step_profile = _derive_empirical_step_profile(source_tag = source_tag, source_dir = source_dir)
    non_empty_rate = _safe_float(step_inputs.get("non_empty_rate"), 0.8)
    mean_steps = _safe_float(step_inputs.get("mean_steps"), 18.0)

    step_mode = str(step_scoring_mode).strip().lower()
    if step_mode == "off":
        llm_steps_per_non_empty = 0.0
    elif step_mode == "full":
        llm_steps_per_non_empty = min(mean_steps, 40.0)
    else:
        llm_steps_per_non_empty = min(mean_steps, 24.0)

    step_calls_total = max_samples * non_empty_rate * llm_steps_per_non_empty
    step_prompt_tokens_per_call = _safe_float(
        step_profile.get("prompt_tokens_per_call"),
        STEP_PROMPT_TOKENS_PER_CALL_FALLBACK,
    )
    step_completion_tokens_per_call = _safe_float(
        step_profile.get("completion_tokens_per_call"),
        STEP_COMPLETION_TOKENS_PER_CALL_FALLBACK,
    )
    step_seconds_per_call = _safe_float(
        step_profile.get("seconds_per_call"),
        STEP_SECONDS_PER_CALL_FALLBACK,
    )
    step_prompt_total = step_calls_total * step_prompt_tokens_per_call
    step_completion_total = step_calls_total * step_completion_tokens_per_call
    step_cost = _estimate_cost_usd(step_prompt_total, step_completion_total)

    effective_step_workers = max(1, step_score_workers)
    per_task_step_seconds = non_empty_rate * (llm_steps_per_non_empty / effective_step_workers) * step_seconds_per_call
    effective_context_parallelism = max(1.0, min(float(max(context_workers, 1)), 6.0))
    step_wall_seconds = (max_samples * per_task_step_seconds) / effective_context_parallelism

    baseline_fail_count = max_samples * (1.0 - baseline_success)
    ace_fail_count = max_samples * (1.0 - ace_success)

    eval_prompt_total = 2.0 * max_samples * EVAL_PROMPT_TOKENS_PER_TASK
    eval_completion_total = 2.0 * max_samples * EVAL_COMPLETION_TOKENS_PER_TASK
    eval_cost = _estimate_cost_usd(eval_prompt_total, eval_completion_total)
    eval_wall_seconds = max_samples * EVAL_SECONDS_PER_TASK

    error_prompt_total = (
        baseline_fail_count * ERROR_PROMPT_TOKENS_PER_FAILED_TASK +
        ace_fail_count * ERROR_PROMPT_TOKENS_PER_FAILED_TASK
    )
    error_completion_total = (
        baseline_fail_count * ERROR_COMPLETION_TOKENS_PER_FAILED_TASK +
        ace_fail_count * ERROR_COMPLETION_TOKENS_PER_FAILED_TASK
    )
    error_cost = _estimate_cost_usd(error_prompt_total, error_completion_total)
    error_wall_seconds = max(
        baseline_fail_count * ERROR_SECONDS_PER_FAILED_TASK,
        ace_fail_count * ERROR_SECONDS_PER_FAILED_TASK,
    )

    total_cost = inference_cost + step_cost + eval_cost + error_cost
    total_wall_seconds = inference_wall_seconds + step_wall_seconds + eval_wall_seconds + error_wall_seconds + COMPARE_SECONDS

    low = {
        "cost_usd": total_cost * 0.80,
        "wall_seconds": total_wall_seconds * 0.80,
    }
    high = {
        "cost_usd": total_cost * 1.35,
        "wall_seconds": total_wall_seconds * 1.40,
    }

    assumptions.extend([
        f"Estimates scaled by ratio={scale:.4f} from source tasks to target tasks.",
        f"Eval stage assumes {EVAL_SECONDS_PER_TASK:.1f}s per task per side in parallel.",
        f"Error stage assumes {ERROR_SECONDS_PER_FAILED_TASK:.1f}s per failed task per side in parallel.",
        (
            "Step scoring token and latency assumptions use empirical auxiliary-call profile "
            f"from {step_profile.get('source_paths', [])}."
            if not step_profile.get("fallback_used")
            else (
                "Step scoring token and latency assumptions use fallback defaults "
                f"prompt={STEP_PROMPT_TOKENS_PER_CALL_FALLBACK:.0f}, completion={STEP_COMPLETION_TOKENS_PER_CALL_FALLBACK:.0f}, "
                f"seconds={STEP_SECONDS_PER_CALL_FALLBACK:.1f}."
            )
        ),
        f"Step scoring assumes {step_seconds_per_call:.2f}s latency per verifier call with {effective_step_workers} step workers.",
        f"Context parallelism factor for step overhead capped at {effective_context_parallelism:.1f}.",
    ])

    if step_inputs.get("fallback_used"):
        assumptions.append("Step scoring inputs used fallback defaults due missing local artifacts.")

    return {
        "source": {
            "selected": source_tag,
            "path": source_path,
        },
        "target_samples": max_samples,
        "phases": {
            "inference": {
                "cost_usd": inference_cost,
                "wall_seconds": inference_wall_seconds,
                "baseline_prompt_tokens": baseline_prompt_total,
                "baseline_completion_tokens": baseline_completion_total,
                "ace_prompt_tokens": ace_prompt_total,
                "ace_completion_tokens": ace_completion_total,
            },
            "step_scoring_overhead": {
                "cost_usd": step_cost,
                "wall_seconds": step_wall_seconds,
                "mode": step_mode,
                "total_llm_step_calls": step_calls_total,
                "non_empty_rate": non_empty_rate,
                "mean_steps": mean_steps,
                "llm_steps_per_non_empty": llm_steps_per_non_empty,
                "prompt_tokens_per_call": step_prompt_tokens_per_call,
                "completion_tokens_per_call": step_completion_tokens_per_call,
                "seconds_per_call": step_seconds_per_call,
            },
            "evaluation": {
                "cost_usd": eval_cost,
                "wall_seconds": eval_wall_seconds,
            },
            "error_analysis": {
                "cost_usd": error_cost,
                "wall_seconds": error_wall_seconds,
                "baseline_failed_est": baseline_fail_count,
                "ace_failed_est": ace_fail_count,
            },
            "compare": {
                "cost_usd": 0.0,
                "wall_seconds": COMPARE_SECONDS,
            },
        },
        "totals": {
            "base": {
                "cost_usd": total_cost,
                "wall_seconds": total_wall_seconds,
            },
            "low": low,
            "high": high,
        },
        "assumptions": assumptions,
        "step_inputs": step_inputs,
        "step_profile": step_profile,
    }


def run_smoke(
    smoke_samples: int,
    seed: int,
    sampling_strategy: str,
    memory_scope: str,
    smoke_output_dir: str,
    clear_db: bool,
    estimate_source: str,
    max_samples_for_scale: int,
    context_workers: int,
    step_scoring_mode: str,
    step_score_workers: int,
) -> Dict[str, Any]:
    if smoke_samples not in SUPPORTED_SMOKE_SAMPLES:
        raise ValueError("smoke_samples must be one of 3, 4, 5.")

    os.makedirs(smoke_output_dir, exist_ok = True)
    manifest_path = os.path.join(
        smoke_output_dir,
        f"subset_manifest_v5_seed{seed}_n{smoke_samples}.json",
    )
    baseline_path = os.path.join(smoke_output_dir, "baseline_v5.jsonl")
    ace_path = os.path.join(smoke_output_dir, "ace_v5.jsonl")
    report_json_path = os.path.join(smoke_output_dir, "comparison_report_v5.json")

    clear_paths = [
        baseline_path,
        ace_path,
        f"{ace_path}.progress.jsonl",
        f"{ace_path}.complete.json",
        os.path.join(smoke_output_dir, "ace_v5_metrics.json"),
        os.path.join(smoke_output_dir, "baseline_v5_graded.jsonl"),
        os.path.join(smoke_output_dir, "ace_v5_graded.jsonl"),
        os.path.join(smoke_output_dir, "baseline_v5_graded_errors.jsonl"),
        os.path.join(smoke_output_dir, "ace_v5_graded_errors.jsonl"),
        os.path.join(smoke_output_dir, "comparison_report_v5.md"),
        report_json_path,
    ]
    for path in clear_paths:
        if os.path.exists(path):
            os.remove(path)

    baseline_cmd = [
        sys.executable,
        "-m",
        "benchmark.infer_baseline_v5",
        "--manifest",
        os.path.abspath(manifest_path),
        "--max-samples",
        str(smoke_samples),
        "--seed",
        str(seed),
        "--sampling-strategy",
        sampling_strategy,
        "--output",
        os.path.abspath(baseline_path),
        "--no-clear-results",
    ]
    ace_cmd = [
        sys.executable,
        "-m",
        "benchmark.infer_ace_direct_v5",
        "--manifest",
        os.path.abspath(manifest_path),
        "--max-samples",
        str(smoke_samples),
        "--seed",
        str(seed),
        "--sampling-strategy",
        sampling_strategy,
        "--memory-scope",
        memory_scope,
        "--output",
        os.path.abspath(ace_path),
        "--no-clear-results",
        "--clear-db" if clear_db else "--no-clear-db",
    ]

    start_time = time.monotonic()
    subprocess_env = _build_subprocess_env()
    p_baseline = subprocess.Popen(baseline_cmd, env = subprocess_env)
    p_ace = subprocess.Popen(ace_cmd, env = subprocess_env)
    baseline_code = p_baseline.wait()
    ace_code = p_ace.wait()
    inference_end = time.monotonic()

    if baseline_code != 0 or ace_code != 0:
        return {
            "status": "failed",
            "error": f"inference_failed baseline_code={baseline_code}, ace_code={ace_code}",
            "commands": {
                "baseline": baseline_cmd,
                "ace": ace_cmd,
            },
        }

    baseline_lines = _count_lines(baseline_path)
    ace_lines = _count_lines(ace_path)
    if baseline_lines < smoke_samples or ace_lines < smoke_samples:
        return {
            "status": "failed",
            "error": "inference_output_incomplete",
            "line_counts": {
                "baseline": baseline_lines,
                "ace": ace_lines,
                "target": smoke_samples,
            },
        }

    post_cmd = [
        sys.executable,
        "-m",
        "benchmark.complete_v5_pipeline",
        "--output-dir",
        smoke_output_dir,
        "--max-samples",
        str(smoke_samples),
        "--skip-wait",
    ]
    post_result = subprocess.run(post_cmd, check = False, env = subprocess_env)
    end_time = time.monotonic()
    if post_result.returncode != 0:
        return {
            "status": "failed",
            "error": f"post_pipeline_failed code={post_result.returncode}",
            "post_command": post_cmd,
        }

    report_payload = _load_json(report_json_path)
    baseline_metrics = report_payload.get("baseline_metrics", {})
    ace_metrics = report_payload.get("ace_metrics", {})
    full_pipeline_metered = report_payload.get("full_pipeline_cost_metered", {})

    measured_inference_cost = _safe_float(baseline_metrics.get("cost_usd"), 0.0) + _safe_float(ace_metrics.get("cost_usd"), 0.0)
    measured_inference_tokens = _safe_float(baseline_metrics.get("total_tokens"), 0.0) + _safe_float(ace_metrics.get("total_tokens"), 0.0)
    measured_full_pipeline_cost = _safe_float(full_pipeline_metered.get("combined_total_cost_usd"), 0.0)
    if measured_full_pipeline_cost <= 0.0:
        measured_full_pipeline_cost = measured_inference_cost
    measured_inference_seconds = max(0.0, inference_end - start_time)
    measured_total_seconds = max(0.0, end_time - start_time)
    measured_post_seconds = max(0.0, measured_total_seconds - measured_inference_seconds)

    scaled_ratio = max_samples_for_scale / smoke_samples if smoke_samples > 0 else 0.0
    scaled_from_smoke = {
        "ratio": scaled_ratio,
        "inference_cost_usd": measured_inference_cost * scaled_ratio,
        "full_pipeline_cost_usd": measured_full_pipeline_cost * scaled_ratio,
        "total_wall_seconds": measured_total_seconds * scaled_ratio,
    }

    smoke_estimate = build_estimate(
        max_samples = smoke_samples,
        estimate_source = estimate_source,
        output_dir = smoke_output_dir,
        context_workers = context_workers,
        step_scoring_mode = step_scoring_mode,
        step_score_workers = step_score_workers,
    )

    return {
        "status": "ok",
        "paths": {
            "output_dir": smoke_output_dir,
            "manifest": manifest_path,
            "baseline": baseline_path,
            "ace": ace_path,
            "comparison_report_json": report_json_path,
        },
        "line_counts": {
            "baseline": baseline_lines,
            "ace": ace_lines,
            "target": smoke_samples,
        },
        "measured": {
            "inference_cost_usd": measured_inference_cost,
            "full_pipeline_cost_usd": measured_full_pipeline_cost,
            "inference_total_tokens": measured_inference_tokens,
            "inference_wall_seconds": measured_inference_seconds,
            "post_wall_seconds": measured_post_seconds,
            "total_wall_seconds": measured_total_seconds,
        },
        "scaled_full_run": scaled_from_smoke,
        "estimated_smoke": smoke_estimate,
    }


def write_preflight_report(path: str, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(payload, f, indent = 2, ensure_ascii = False)


def build_preflight_payload(
    preflight_mode: str,
    static_checks: Optional[Dict[str, Any]],
    estimate: Optional[Dict[str, Any]],
    smoke: Optional[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    blocking_issues: List[str] = []
    warnings: List[str] = []
    if static_checks:
        blocking_issues.extend(static_checks.get("blocking_issues", []))
        warnings.extend(static_checks.get("warnings", []))
    if smoke and smoke.get("status") != "ok":
        blocking_issues.append(str(smoke.get("error", "smoke_run_failed")))

    status = "ok" if not blocking_issues else "failed"

    return {
        "generated_at": _utc_now(),
        "preflight_mode": preflight_mode,
        "status": status,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "config": config,
        "static_checks": static_checks or {},
        "estimate": estimate or {},
        "smoke": smoke or {},
    }
