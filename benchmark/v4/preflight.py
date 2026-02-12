"""
Preflight helpers for CL-bench V4 static checks, estimation, and mini smoke runs.
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
SUPPORTED_SAMPLING_STRATEGIES = {"task_random", "context_dense"}
SUPPORTED_SMOKE_SAMPLES = {3, 4, 5}

REQUIRED_ENV_KEYS = [
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
]

REQUIRED_MODULES = [
    "benchmark.v4.infer_baseline",
    "benchmark.v4.infer_ace",
    "benchmark.v4.complete_pipeline",
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

STEP_PROMPT_TOKENS_PER_CALL = 220.0
STEP_COMPLETION_TOKENS_PER_CALL = 40.0
STEP_SECONDS_PER_CALL = 1.8

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
    local_v4 = os.path.join(output_dir, "comparison_report_v4.json")
    default_v4 = os.path.join("benchmark", "results", "v4", "comparison_report_v4.json")
    default_v3 = os.path.join("benchmark", "results", "v3", "comparison_report_v3.json")

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


def run_static_checks(
    manifest_path: str,
    output_dir: str,
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

    checks.append({
        "name": "connectivity_probes",
        "ok": True,
        "detail": "optional probes skipped in static mode",
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

    step_inputs = _derive_step_inputs(source_tag = source_tag, source_dir = os.path.dirname(source_path) if source_path else output_dir)
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
    step_prompt_total = step_calls_total * STEP_PROMPT_TOKENS_PER_CALL
    step_completion_total = step_calls_total * STEP_COMPLETION_TOKENS_PER_CALL
    step_cost = _estimate_cost_usd(step_prompt_total, step_completion_total)

    effective_step_workers = max(1, step_score_workers)
    per_task_step_seconds = non_empty_rate * (llm_steps_per_non_empty / effective_step_workers) * STEP_SECONDS_PER_CALL
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
        f"Step scoring assumes prompt={STEP_PROMPT_TOKENS_PER_CALL:.0f} and completion={STEP_COMPLETION_TOKENS_PER_CALL:.0f} tokens per scored step.",
        f"Step scoring assumes {STEP_SECONDS_PER_CALL:.1f}s latency per verifier call with {effective_step_workers} step workers.",
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
        f"subset_manifest_v4_seed{seed}_n{smoke_samples}.json",
    )
    baseline_path = os.path.join(smoke_output_dir, "baseline_v4.jsonl")
    ace_path = os.path.join(smoke_output_dir, "ace_v4.jsonl")
    report_json_path = os.path.join(smoke_output_dir, "comparison_report_v4.json")

    clear_paths = [
        baseline_path,
        ace_path,
        os.path.join(smoke_output_dir, "ace_v4_metrics.json"),
        os.path.join(smoke_output_dir, "baseline_v4_graded.jsonl"),
        os.path.join(smoke_output_dir, "ace_v4_graded.jsonl"),
        os.path.join(smoke_output_dir, "baseline_v4_graded_errors.jsonl"),
        os.path.join(smoke_output_dir, "ace_v4_graded_errors.jsonl"),
        os.path.join(smoke_output_dir, "comparison_report_v4.md"),
        report_json_path,
    ]
    for path in clear_paths:
        if os.path.exists(path):
            os.remove(path)

    baseline_cmd = [
        sys.executable,
        "-m",
        "benchmark.v4.infer_baseline",
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
        "benchmark.v4.infer_ace",
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
        "benchmark.v4.complete_pipeline",
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

    measured_inference_cost = _safe_float(baseline_metrics.get("cost_usd"), 0.0) + _safe_float(ace_metrics.get("cost_usd"), 0.0)
    measured_inference_tokens = _safe_float(baseline_metrics.get("total_tokens"), 0.0) + _safe_float(ace_metrics.get("total_tokens"), 0.0)
    measured_inference_seconds = max(0.0, inference_end - start_time)
    measured_total_seconds = max(0.0, end_time - start_time)
    measured_post_seconds = max(0.0, measured_total_seconds - measured_inference_seconds)

    scaled_ratio = max_samples_for_scale / smoke_samples if smoke_samples > 0 else 0.0
    scaled_from_smoke = {
        "ratio": scaled_ratio,
        "inference_cost_usd": measured_inference_cost * scaled_ratio,
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
