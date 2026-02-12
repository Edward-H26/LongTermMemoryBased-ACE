"""
Side-by-side comparison of baseline vs ACE benchmark results.

Produces:
- Table 1: Solving Rate by Category
- Table 2: Error Analysis Distribution
- Table 3: Inference Token/Latency/Estimated Cost
- Table 4: Per-category Token/Latency
- Table 5: Runtime diagnostics
- Table 6: Full pipeline actual metered cost by phase
- Table 7: OpenAI billed reconciliation
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from benchmark.costing import (
    compute_cost_usd,
    extract_ace_aux_metrics,
    fetch_openai_billed_reconciliation,
    load_json,
    load_phase_metric_file,
    merge_phase_summaries,
    resolve_run_window,
)

load_dotenv()

GPT51_INPUT_PRICE = 1.25
GPT51_OUTPUT_PRICE = 10.00
GEMINI_INPUT_PRICE = 0.50
GEMINI_OUTPUT_PRICE = 3.00

CATEGORY_DISPLAY_ORDER = [
    "Domain Knowledge Reasoning",
    "Rule System Application",
    "Procedural Task Execution",
    "Empirical Discovery & Simulation",
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding = "utf-8") as f:
        for line in f:
            raw = line.strip()
            if raw:
                data.append(json.loads(raw))
    return data


def get_task_id(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        task_id = metadata.get("task_id", item.get("task_id", ""))
        return task_id if isinstance(task_id, str) else ""
    task_id = item.get("task_id", "")
    return task_id if isinstance(task_id, str) else ""


def collect_task_ids(data: List[Dict[str, Any]]) -> List[str]:
    return [task_id for task_id in (get_task_id(item) for item in data) if task_id]


def compute_solving_rates(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(data)
    if total == 0:
        return {"overall": 0.0, "total": 0, "score_1": 0, "categories": {}}

    score_1 = sum(1 for item in data if item.get("score") == 1)

    category_stats: Dict[str, Dict[str, int]] = {}
    for item in data:
        metadata = item.get("metadata", {})
        category = metadata.get("context_category", "Unknown") if isinstance(metadata, dict) else "Unknown"
        stats = category_stats.setdefault(category, {"total": 0, "score_1": 0})
        stats["total"] += 1
        if item.get("score") == 1:
            stats["score_1"] += 1

    categories: Dict[str, float] = {}
    for category, stats in category_stats.items():
        categories[category] = stats["score_1"] / stats["total"] * 100 if stats["total"] > 0 else 0.0

    return {
        "overall": score_1 / total * 100 if total > 0 else 0.0,
        "total": total,
        "score_1": score_1,
        "categories": categories,
        "category_counts": {category: stats["total"] for category, stats in category_stats.items()},
    }


def compute_error_distribution(error_data: List[Dict[str, Any]], total_tasks: int) -> Dict[str, float]:
    error_counts = {"CONTEXT_IGNORED": 0, "CONTEXT_MISUSED": 0, "FORMAT_ERROR": 0, "REFUSAL": 0}
    for item in error_data:
        classification = item.get("error_classification", {})
        error_types = classification.get("task_error_types", []) if isinstance(classification, dict) else []
        if not isinstance(error_types, list):
            continue
        for error_type in error_types:
            if error_type in error_counts:
                error_counts[error_type] += 1

    distribution: Dict[str, float] = {}
    for error_type, count in error_counts.items():
        distribution[error_type] = count / total_tasks * 100 if total_tasks > 0 else 0.0
    return distribution


def percentile(sorted_values: List[float], percentile_value: int) -> float:
    if not sorted_values:
        return 0.0
    index = int(len(sorted_values) * percentile_value / 100)
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def format_percent_delta(baseline_value: Any, ace_value: Any) -> str:
    try:
        baseline = float(baseline_value)
        ace = float(ace_value)
    except (TypeError, ValueError):
        return "N/A"
    if baseline == 0.0:
        return "N/A"
    percent = ((ace - baseline) / baseline) * 100.0
    return f"{percent:+.1f}%"


def compute_metrics(data: List[Dict[str, Any]]) -> Dict[str, float]:
    latencies: List[float] = []
    prompt_tokens_list: List[float] = []
    completion_tokens_list: List[float] = []
    total_tokens_list: List[float] = []

    for item in data:
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        latency_ms = metrics.get("latency_ms", 0)
        if isinstance(latency_ms, (int, float)) and latency_ms > 0:
            latencies.append(float(latency_ms))

        prompt_tokens = metrics.get("prompt_tokens", 0)
        completion_tokens = metrics.get("completion_tokens", 0)
        total_tokens = metrics.get("total_tokens", 0)
        if isinstance(total_tokens, (int, float)) and total_tokens > 0:
            prompt_tokens_list.append(float(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else 0.0)
            completion_tokens_list.append(float(completion_tokens) if isinstance(completion_tokens, (int, float)) else 0.0)
            total_tokens_list.append(float(total_tokens))

    if not latencies:
        return {
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_tokens": 0.0,
            "total_tokens": 0.0,
            "total_prompt_tokens": 0.0,
            "total_completion_tokens": 0.0,
            "cost_usd": 0.0,
        }

    sorted_latency = sorted(latencies)
    total_prompt_tokens = sum(prompt_tokens_list)
    total_completion_tokens = sum(completion_tokens_list)
    total_tokens = sum(total_tokens_list)
    estimated_cost = (total_prompt_tokens / 1_000_000.0 * GPT51_INPUT_PRICE) + (total_completion_tokens / 1_000_000.0 * GPT51_OUTPUT_PRICE)

    return {
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": percentile(sorted_latency, 50),
        "p95_latency_ms": percentile(sorted_latency, 95),
        "avg_tokens": total_tokens / len(data) if data else 0.0,
        "avg_prompt_tokens": total_prompt_tokens / len(data) if data else 0.0,
        "avg_completion_tokens": total_completion_tokens / len(data) if data else 0.0,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "cost_usd": estimated_cost,
    }


def compute_metrics_by_category(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        metadata = item.get("metadata", {})
        category = metadata.get("context_category", "Unknown") if isinstance(metadata, dict) else "Unknown"
        grouped.setdefault(category, []).append(item)
    return {category: compute_metrics(items) for category, items in grouped.items()}


def compute_runtime_diagnostics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(data)
    if total == 0:
        return {}

    has_diagnostics = False
    has_v5_fields = False

    carryover_hits = 0
    learned_bullets_total = 0.0
    capped_count = 0
    step_scores: List[float] = []
    quality_gate_applied = 0
    progress_checkpoint_count = 0
    resume_recovered_count = 0
    memory_write_failure_count = 0

    for item in data:
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        if (
            "num_learned_bullets_retrieved" in metrics
            or "completion_capped" in metrics
            or "step_scoring" in metrics
        ):
            has_diagnostics = True
        if (
            "progress_written" in metrics
            or "resume_source" in metrics
            or "memory_write_failed" in metrics
            or "memory_write_retries" in metrics
        ):
            has_diagnostics = True
            has_v5_fields = True

        learned = float(metrics.get("num_learned_bullets_retrieved", 0) or 0)
        learned_bullets_total += learned
        if learned > 0:
            carryover_hits += 1

        if bool(metrics.get("completion_capped", False)):
            capped_count += 1

        step_scoring = metrics.get("step_scoring", {})
        if isinstance(step_scoring, dict):
            mean_step = step_scoring.get("mean_step_score")
            if isinstance(mean_step, (int, float)):
                step_scores.append(float(mean_step))

        quality_gate = metrics.get("quality_gate", {})
        if isinstance(quality_gate, dict) and bool(quality_gate.get("should_apply_update", False)):
            quality_gate_applied += 1

        if bool(metrics.get("progress_written", False)):
            progress_checkpoint_count += 1

        resume_source = str(metrics.get("resume_source", "")).strip().lower()
        if resume_source and resume_source != "live":
            resume_recovered_count += 1

        if bool(metrics.get("memory_write_failed", False)):
            memory_write_failure_count += 1

    if not has_diagnostics:
        return {}

    return {
        "carryover_coverage_pct": (carryover_hits / total * 100.0) if total else 0.0,
        "learned_retrieval_rate": (learned_bullets_total / total) if total else 0.0,
        "capped_output_rate_pct": (capped_count / total * 100.0) if total else 0.0,
        "mean_step_score": (sum(step_scores) / len(step_scores)) if step_scores else 0.0,
        "quality_gate_apply_rate_pct": (quality_gate_applied / total * 100.0) if total else 0.0,
        "resume_recovery_rate": (resume_recovered_count / total * 100.0) if total else 0.0,
        "memory_write_failure_rate": (memory_write_failure_count / total * 100.0) if total else 0.0,
        "progress_checkpoint_count": progress_checkpoint_count,
        "has_v5_fields": has_v5_fields,
        "num_rows": total,
    }


def normalize_phase_entry(
    phase_key: str,
    label: str,
    prompt_tokens: float,
    completion_tokens: float,
    provider: str,
    model: str,
    source_path: str = "",
) -> Dict[str, Any]:
    total_tokens = prompt_tokens + completion_tokens
    cost_usd = compute_cost_usd(
        prompt_tokens = prompt_tokens,
        completion_tokens = completion_tokens,
        provider = provider,
        model = model,
    )
    return {
        "phase_key": phase_key,
        "label": label,
        "provider": provider,
        "model": model,
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "total_tokens": float(total_tokens),
        "cost_usd": float(cost_usd),
        "source_path": source_path,
    }


def derive_default_path(base_path: Optional[str], suffix: str) -> Optional[str]:
    if not base_path:
        return None
    root, _ = os.path.splitext(base_path)
    candidate = f"{root}{suffix}"
    return candidate if os.path.exists(candidate) else None


def build_cost_breakdown(
    baseline_metrics: Dict[str, float],
    ace_metrics: Dict[str, float],
    ace_aux_metrics_path: Optional[str],
    baseline_eval_metrics_path: Optional[str],
    ace_eval_metrics_path: Optional[str],
    baseline_error_metrics_path: Optional[str],
    ace_error_metrics_path: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ace_aux = extract_ace_aux_metrics(ace_aux_metrics_path) if ace_aux_metrics_path else {
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "cost_usd": 0.0,
        "total_calls": 0,
        "by_provider": {},
        "source": "missing",
    }

    baseline_eval = load_phase_metric_file(baseline_eval_metrics_path, model_fallback = "gpt-5.1", provider_fallback = "openai")
    ace_eval = load_phase_metric_file(ace_eval_metrics_path, model_fallback = "gpt-5.1", provider_fallback = "openai")
    baseline_error = load_phase_metric_file(baseline_error_metrics_path, model_fallback = "gpt-5.1", provider_fallback = "openai")
    ace_error = load_phase_metric_file(ace_error_metrics_path, model_fallback = "gpt-5.1", provider_fallback = "openai")

    phase_rows = [
        normalize_phase_entry(
            "inference_baseline",
            "Inference (Baseline)",
            float(baseline_metrics.get("total_prompt_tokens", 0.0)),
            float(baseline_metrics.get("total_completion_tokens", 0.0)),
            "openai",
            "gpt-5.1",
            source_path = "",
        ),
        normalize_phase_entry(
            "inference_ace_primary",
            "Inference (ACE Primary)",
            float(ace_metrics.get("total_prompt_tokens", 0.0)),
            float(ace_metrics.get("total_completion_tokens", 0.0)),
            "openai",
            "gpt-5.1",
            source_path = "",
        ),
        normalize_phase_entry(
            "ace_auxiliary",
            "ACE Auxiliary (Reflector/Step)",
            float(ace_aux.get("prompt_tokens", 0.0)),
            float(ace_aux.get("completion_tokens", 0.0)),
            "openai",
            "gpt-5.1",
            source_path = ace_aux_metrics_path or "",
        ),
        normalize_phase_entry(
            "eval_baseline",
            "Evaluation (Baseline)",
            float(baseline_eval.get("prompt_tokens", 0.0)),
            float(baseline_eval.get("completion_tokens", 0.0)),
            str(baseline_eval.get("provider", "openai")),
            str(baseline_eval.get("model", "gpt-5.1")),
            source_path = str(baseline_eval.get("path", "")),
        ),
        normalize_phase_entry(
            "eval_ace",
            "Evaluation (ACE)",
            float(ace_eval.get("prompt_tokens", 0.0)),
            float(ace_eval.get("completion_tokens", 0.0)),
            str(ace_eval.get("provider", "openai")),
            str(ace_eval.get("model", "gpt-5.1")),
            source_path = str(ace_eval.get("path", "")),
        ),
        normalize_phase_entry(
            "error_baseline",
            "Error Analysis (Baseline)",
            float(baseline_error.get("prompt_tokens", 0.0)),
            float(baseline_error.get("completion_tokens", 0.0)),
            str(baseline_error.get("provider", "openai")),
            str(baseline_error.get("model", "gpt-5.1")),
            source_path = str(baseline_error.get("path", "")),
        ),
        normalize_phase_entry(
            "error_ace",
            "Error Analysis (ACE)",
            float(ace_error.get("prompt_tokens", 0.0)),
            float(ace_error.get("completion_tokens", 0.0)),
            str(ace_error.get("provider", "openai")),
            str(ace_error.get("model", "gpt-5.1")),
            source_path = str(ace_error.get("path", "")),
        ),
    ]

    baseline_total = merge_phase_summaries(
        [
            phase_rows[0],
            phase_rows[3],
            phase_rows[5],
        ]
    )
    ace_total = merge_phase_summaries(
        [
            phase_rows[1],
            phase_rows[2],
            phase_rows[4],
            phase_rows[6],
        ]
    )
    combined_total = merge_phase_summaries([baseline_total, ace_total])

    breakdown = {
        row["phase_key"]: row for row in phase_rows
    }
    breakdown["baseline_total"] = {"phase_key": "baseline_total", "label": "Baseline Total", **baseline_total}
    breakdown["ace_total"] = {"phase_key": "ace_total", "label": "ACE Total", **ace_total}
    breakdown["combined_total"] = {"phase_key": "combined_total", "label": "Combined Full Pipeline Total", **combined_total}

    metered = {
        "baseline_total_cost_usd": float(baseline_total.get("cost_usd", 0.0)),
        "ace_total_cost_usd": float(ace_total.get("cost_usd", 0.0)),
        "combined_total_cost_usd": float(combined_total.get("cost_usd", 0.0)),
        "baseline_total_tokens": float(baseline_total.get("total_tokens", 0.0)),
        "ace_total_tokens": float(ace_total.get("total_tokens", 0.0)),
        "combined_total_tokens": float(combined_total.get("total_tokens", 0.0)),
    }
    return breakdown, metered


def format_report(
    baseline_rates: Dict[str, Any],
    ace_rates: Dict[str, Any],
    baseline_errors: Dict[str, float],
    ace_errors: Dict[str, float],
    baseline_metrics: Dict[str, float],
    ace_metrics: Dict[str, float],
    baseline_cat_metrics: Dict[str, Dict[str, float]],
    ace_cat_metrics: Dict[str, Dict[str, float]],
    baseline_runtime_diag: Optional[Dict[str, Any]] = None,
    ace_runtime_diag: Optional[Dict[str, Any]] = None,
    title_label: str = "V2",
    cost_breakdown_by_phase: Optional[Dict[str, Any]] = None,
    full_pipeline_cost_metered: Optional[Dict[str, Any]] = None,
    billed_reconciliation: Optional[Dict[str, Any]] = None,
    cost_mode: str = "legacy",
    billing_policy: str = "off",
) -> str:
    lines: List[str] = []
    lines.append(f"# CL-bench Evaluation: Baseline vs ACE Comparison ({title_label})\n")

    lines.append("## Table 1: Solving Rate by Category\n")
    discovered_categories = set(list(baseline_rates["categories"].keys()) + list(ace_rates["categories"].keys()))
    ordered_known = [category for category in CATEGORY_DISPLAY_ORDER if category in discovered_categories]
    ordered_unknown = sorted([category for category in discovered_categories if category not in CATEGORY_DISPLAY_ORDER])
    all_categories = ordered_known + ordered_unknown

    category_abbrev = {
        "Domain Knowledge Reasoning": "DKR",
        "Rule System Application": "RSA",
        "Procedural Task Execution": "PTE",
        "Empirical Discovery & Simulation": "EDS",
    }

    header = "| Model Names | Overall (%) |"
    separator = "|---|---|"
    for category in all_categories:
        count = baseline_rates.get("category_counts", {}).get(category, 0) or ace_rates.get("category_counts", {}).get(category, 0)
        header += f" {category} n={count} (%) |"
        separator += "---|"
    lines.append(header)
    lines.append(separator)

    row_baseline = f"| GPT-5.1 (High) baseline | {baseline_rates['overall']:.1f} |"
    for category in all_categories:
        row_baseline += f" {baseline_rates['categories'].get(category, 0):.1f} |"
    lines.append(row_baseline)

    row_ace = f"| GPT-5.1 (High) + ACE | {ace_rates['overall']:.1f} |"
    for category in all_categories:
        row_ace += f" {ace_rates['categories'].get(category, 0):.1f} |"
    lines.append(row_ace)

    row_delta = f"| Delta | {format_percent_delta(baseline_rates['overall'], ace_rates['overall'])} |"
    for category in all_categories:
        row_delta += f" {format_percent_delta(baseline_rates['categories'].get(category, 0), ace_rates['categories'].get(category, 0))} |"
    lines.append(row_delta)

    if baseline_errors or ace_errors:
        lines.append("\n## Table 2: Error Analysis Distribution\n")
        lines.append("| Model Names | Context Ignored (%) | Context Misused (%) | Format Error (%) | Refusal (%) |")
        lines.append("|---|---|---|---|---|")
        if baseline_errors:
            lines.append(
                f"| Baseline | {baseline_errors.get('CONTEXT_IGNORED', 0):.1f} | {baseline_errors.get('CONTEXT_MISUSED', 0):.1f} | "
                f"{baseline_errors.get('FORMAT_ERROR', 0):.1f} | {baseline_errors.get('REFUSAL', 0):.1f} |"
            )
        if ace_errors:
            lines.append(
                f"| ACE | {ace_errors.get('CONTEXT_IGNORED', 0):.1f} | {ace_errors.get('CONTEXT_MISUSED', 0):.1f} | "
                f"{ace_errors.get('FORMAT_ERROR', 0):.1f} | {ace_errors.get('REFUSAL', 0):.1f} |"
            )

    lines.append("\n## Table 3: Token Usage, Latency, and Cost\n")
    lines.append("| Metric | Baseline | ACE | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Avg Tokens/Task | {baseline_metrics.get('avg_tokens', 0):,.0f} | {ace_metrics.get('avg_tokens', 0):,.0f} | "
        f"{format_percent_delta(baseline_metrics.get('avg_tokens', 0), ace_metrics.get('avg_tokens', 0))} |"
    )
    lines.append(
        f"| Avg Prompt Tokens | {baseline_metrics.get('avg_prompt_tokens', 0):,.0f} | {ace_metrics.get('avg_prompt_tokens', 0):,.0f} | "
        f"{format_percent_delta(baseline_metrics.get('avg_prompt_tokens', 0), ace_metrics.get('avg_prompt_tokens', 0))} |"
    )
    lines.append(
        f"| Avg Completion Tokens | {baseline_metrics.get('avg_completion_tokens', 0):,.0f} | {ace_metrics.get('avg_completion_tokens', 0):,.0f} | "
        f"{format_percent_delta(baseline_metrics.get('avg_completion_tokens', 0), ace_metrics.get('avg_completion_tokens', 0))} |"
    )
    lines.append(
        f"| Total Tokens | {baseline_metrics.get('total_tokens', 0):,} | {ace_metrics.get('total_tokens', 0):,} | "
        f"{format_percent_delta(baseline_metrics.get('total_tokens', 0), ace_metrics.get('total_tokens', 0))} |"
    )
    lines.append(
        f"| Avg Latency (ms) | {baseline_metrics.get('avg_latency_ms', 0):,.0f} | {ace_metrics.get('avg_latency_ms', 0):,.0f} | "
        f"{format_percent_delta(baseline_metrics.get('avg_latency_ms', 0), ace_metrics.get('avg_latency_ms', 0))} |"
    )
    lines.append(
        f"| p50 Latency (ms) | {baseline_metrics.get('p50_latency_ms', 0):,.0f} | {ace_metrics.get('p50_latency_ms', 0):,.0f} | "
        f"{format_percent_delta(baseline_metrics.get('p50_latency_ms', 0), ace_metrics.get('p50_latency_ms', 0))} |"
    )
    lines.append(
        f"| p95 Latency (ms) | {baseline_metrics.get('p95_latency_ms', 0):,.0f} | {ace_metrics.get('p95_latency_ms', 0):,.0f} | "
        f"{format_percent_delta(baseline_metrics.get('p95_latency_ms', 0), ace_metrics.get('p95_latency_ms', 0))} |"
    )
    lines.append(
        f"| Estimated Cost ($) | ${baseline_metrics.get('cost_usd', 0):.2f} | ${ace_metrics.get('cost_usd', 0):.2f} | "
        f"{format_percent_delta(baseline_metrics.get('cost_usd', 0), ace_metrics.get('cost_usd', 0))} |"
    )

    if baseline_cat_metrics or ace_cat_metrics:
        lines.append("\n## Table 4: Per-Category Token Usage and Latency\n")
        lines.append("| Category | Baseline Avg Tokens | ACE Avg Tokens | Baseline Avg Latency (ms) | ACE Avg Latency (ms) |")
        lines.append("|---|---|---|---|---|")
        for category in all_categories:
            short_name = category_abbrev.get(category, category[:3])
            baseline_cat = baseline_cat_metrics.get(category, {})
            ace_cat = ace_cat_metrics.get(category, {})
            lines.append(
                f"| {short_name} | {baseline_cat.get('avg_tokens', 0):,.0f} | {ace_cat.get('avg_tokens', 0):,.0f} | "
                f"{baseline_cat.get('avg_latency_ms', 0):,.0f} | {ace_cat.get('avg_latency_ms', 0):,.0f} |"
            )

    if baseline_runtime_diag or ace_runtime_diag:
        baseline_diag = baseline_runtime_diag or {}
        ace_diag = ace_runtime_diag or {}
        lines.append("\n## Table 5: Runtime Diagnostics\n")
        lines.append("| Metric | Baseline | ACE | Delta |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| Carryover Coverage (%) | {baseline_diag.get('carryover_coverage_pct', 0):.1f} | {ace_diag.get('carryover_coverage_pct', 0):.1f} | "
            f"{format_percent_delta(baseline_diag.get('carryover_coverage_pct', 0), ace_diag.get('carryover_coverage_pct', 0))} |"
        )
        lines.append(
            f"| Learned Retrieval Rate | {baseline_diag.get('learned_retrieval_rate', 0):.3f} | {ace_diag.get('learned_retrieval_rate', 0):.3f} | "
            f"{format_percent_delta(baseline_diag.get('learned_retrieval_rate', 0), ace_diag.get('learned_retrieval_rate', 0))} |"
        )
        lines.append(
            f"| Capped Output Rate (%) | {baseline_diag.get('capped_output_rate_pct', 0):.1f} | {ace_diag.get('capped_output_rate_pct', 0):.1f} | "
            f"{format_percent_delta(baseline_diag.get('capped_output_rate_pct', 0), ace_diag.get('capped_output_rate_pct', 0))} |"
        )
        lines.append(
            f"| Mean Step Score | {baseline_diag.get('mean_step_score', 0):.3f} | {ace_diag.get('mean_step_score', 0):.3f} | "
            f"{format_percent_delta(baseline_diag.get('mean_step_score', 0), ace_diag.get('mean_step_score', 0))} |"
        )
        lines.append(
            f"| Quality Gate Apply Rate (%) | {baseline_diag.get('quality_gate_apply_rate_pct', 0):.1f} | {ace_diag.get('quality_gate_apply_rate_pct', 0):.1f} | "
            f"{format_percent_delta(baseline_diag.get('quality_gate_apply_rate_pct', 0), ace_diag.get('quality_gate_apply_rate_pct', 0))} |"
        )
        if bool(baseline_diag.get("has_v5_fields", False)) or bool(ace_diag.get("has_v5_fields", False)):
            lines.append(
                f"| Resume Recovery Rate (%) | {baseline_diag.get('resume_recovery_rate', 0):.1f} | {ace_diag.get('resume_recovery_rate', 0):.1f} | "
                f"{format_percent_delta(baseline_diag.get('resume_recovery_rate', 0), ace_diag.get('resume_recovery_rate', 0))} |"
            )
            lines.append(
                f"| Memory Write Failure Rate (%) | {baseline_diag.get('memory_write_failure_rate', 0):.1f} | {ace_diag.get('memory_write_failure_rate', 0):.1f} | "
                f"{format_percent_delta(baseline_diag.get('memory_write_failure_rate', 0), ace_diag.get('memory_write_failure_rate', 0))} |"
            )
            lines.append(
                f"| Progress Checkpoint Count | {baseline_diag.get('progress_checkpoint_count', 0):,.0f} | {ace_diag.get('progress_checkpoint_count', 0):,.0f} | "
                f"{format_percent_delta(baseline_diag.get('progress_checkpoint_count', 0), ace_diag.get('progress_checkpoint_count', 0))} |"
            )

    if cost_breakdown_by_phase and full_pipeline_cost_metered:
        lines.append("\n## Table 6: Full Pipeline Actual Metered Cost\n")
        lines.append("| Phase | Prompt Tokens | Completion Tokens | Total Tokens | Actual Metered Cost ($) |")
        lines.append("|---|---|---|---|---|")
        phase_order = [
            "inference_baseline",
            "inference_ace_primary",
            "ace_auxiliary",
            "eval_baseline",
            "eval_ace",
            "error_baseline",
            "error_ace",
        ]
        for key in phase_order:
            row = cost_breakdown_by_phase.get(key, {})
            lines.append(
                f"| {row.get('label', key)} | {row.get('prompt_tokens', 0):,.0f} | {row.get('completion_tokens', 0):,.0f} | "
                f"{row.get('total_tokens', 0):,.0f} | ${row.get('cost_usd', 0):.2f} |"
            )
        lines.append(
            f"| Baseline Total | {cost_breakdown_by_phase.get('baseline_total', {}).get('prompt_tokens', 0):,.0f} | "
            f"{cost_breakdown_by_phase.get('baseline_total', {}).get('completion_tokens', 0):,.0f} | "
            f"{cost_breakdown_by_phase.get('baseline_total', {}).get('total_tokens', 0):,.0f} | "
            f"${full_pipeline_cost_metered.get('baseline_total_cost_usd', 0):.2f} |"
        )
        lines.append(
            f"| ACE Total | {cost_breakdown_by_phase.get('ace_total', {}).get('prompt_tokens', 0):,.0f} | "
            f"{cost_breakdown_by_phase.get('ace_total', {}).get('completion_tokens', 0):,.0f} | "
            f"{cost_breakdown_by_phase.get('ace_total', {}).get('total_tokens', 0):,.0f} | "
            f"${full_pipeline_cost_metered.get('ace_total_cost_usd', 0):.2f} |"
        )
        lines.append(
            f"| Combined Total | {cost_breakdown_by_phase.get('combined_total', {}).get('prompt_tokens', 0):,.0f} | "
            f"{cost_breakdown_by_phase.get('combined_total', {}).get('completion_tokens', 0):,.0f} | "
            f"{cost_breakdown_by_phase.get('combined_total', {}).get('total_tokens', 0):,.0f} | "
            f"${full_pipeline_cost_metered.get('combined_total_cost_usd', 0):.2f} |"
        )

    if billed_reconciliation is not None:
        lines.append("\n## Table 7: OpenAI Billed Reconciliation\n")
        lines.append("| Item | Value |")
        lines.append("|---|---|")
        lines.append(f"| Cost Mode | {cost_mode} |")
        lines.append(f"| Billing Policy | {billing_policy} |")
        lines.append(f"| Reconciliation Status | {billed_reconciliation.get('status', 'unknown')} |")
        lines.append(f"| Project Scope | {billed_reconciliation.get('project_id', '') or 'N/A'} |")
        lines.append(f"| Run Window Start (UTC) | {billed_reconciliation.get('window_start', '') or 'N/A'} |")
        lines.append(f"| Run Window End (UTC) | {billed_reconciliation.get('window_end', '') or 'N/A'} |")
        lines.append(f"| Metered Cost ($) | ${billed_reconciliation.get('metered_cost_usd', 0):.2f} |")
        billed_cost = billed_reconciliation.get("billed_cost_usd")
        lines.append(f"| Billed Cost ($) | ${billed_cost:.2f} |" if isinstance(billed_cost, (int, float)) else "| Billed Cost ($) | N/A |")
        delta_usd = billed_reconciliation.get("delta_usd")
        lines.append(f"| Reconciliation Delta ($) | ${delta_usd:+.2f} |" if isinstance(delta_usd, (int, float)) else "| Reconciliation Delta ($) | N/A |")
        delta_pct = billed_reconciliation.get("delta_pct")
        lines.append(f"| Reconciliation Delta (%) | {delta_pct:+.2f}% |" if isinstance(delta_pct, (int, float)) else "| Reconciliation Delta (%) | N/A |")
        lines.append(f"| Notes | {billed_reconciliation.get('note', '') or 'N/A'} |")

    return "\n".join(lines)


def resolve_phase_metric_paths(
    args: argparse.Namespace,
) -> Dict[str, Optional[str]]:
    return {
        "ace_aux_metrics": args.ace_aux_metrics,
        "baseline_eval_metrics": args.baseline_eval_metrics or derive_default_path(args.baseline, "_eval_metrics.json"),
        "ace_eval_metrics": args.ace_eval_metrics or derive_default_path(args.ace, "_eval_metrics.json"),
        "baseline_error_metrics": args.baseline_error_metrics or derive_default_path(args.baseline_errors, "_error_metrics.json"),
        "ace_error_metrics": args.ace_error_metrics or derive_default_path(args.ace_errors, "_error_metrics.json"),
    }


def run_billing_reconciliation(
    args: argparse.Namespace,
    full_pipeline_cost_metered: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    if args.cost_mode != "dual_source":
        return {
            "status": "disabled_legacy_mode",
            "metered_cost_usd": float(full_pipeline_cost_metered.get("combined_total_cost_usd", 0.0)),
            "note": "Billing reconciliation skipped because cost mode is legacy.",
        }, "disabled_legacy_mode"
    if args.billing_policy == "off":
        return {
            "status": "disabled_policy_off",
            "metered_cost_usd": float(full_pipeline_cost_metered.get("combined_total_cost_usd", 0.0)),
            "note": "Billing reconciliation disabled by policy.",
        }, "disabled_policy_off"

    run_meta_path = args.run_meta
    if not run_meta_path or not os.path.exists(run_meta_path):
        raise ValueError("Strict billing policy requires --run-meta with a readable run metadata file.")
    run_meta = load_json(run_meta_path)
    start_iso, end_iso = resolve_run_window(run_meta)
    if not start_iso or not end_iso:
        raise ValueError("Strict billing policy requires run metadata with usable start/end timestamps.")

    admin_key = str(os.getenv(args.openai_admin_key_env, "")).strip()
    project_id = str(os.getenv(args.openai_project_id_env, "")).strip()
    if not admin_key:
        raise ValueError(f"Strict billing policy requires env {args.openai_admin_key_env} to be set.")
    if not project_id:
        raise ValueError(f"Strict billing policy requires env {args.openai_project_id_env} to be set.")

    result = fetch_openai_billed_reconciliation(
        admin_api_key = admin_key,
        project_id = project_id,
        start_iso = start_iso,
        end_iso = end_iso,
    )
    if not result.get("success", False):
        raise ValueError(f"Strict billing reconciliation failed: {result.get('error', 'unknown_error')}")

    metered = float(full_pipeline_cost_metered.get("combined_total_cost_usd", 0.0))
    billed = float(result.get("cost_usd", 0.0))
    delta_usd = billed - metered
    delta_pct = (delta_usd / metered * 100.0) if metered != 0 else None
    payload = {
        "status": "reconciled",
        "project_id": project_id,
        "window_start": start_iso,
        "window_end": end_iso,
        "metered_cost_usd": metered,
        "billed_cost_usd": billed,
        "delta_usd": delta_usd,
        "delta_pct": delta_pct,
        "usage": result.get("usage", {}),
        "note": "OpenAI notes that usage and cost views may not reconcile perfectly in all windows.",
        "endpoint_meta": result.get("endpoints", {}),
    }
    return payload, "reconciled"


def main() -> None:
    parser = argparse.ArgumentParser(description = "CL-bench Baseline vs ACE Comparison")
    parser.add_argument("--baseline", type = str, required = True)
    parser.add_argument("--ace", type = str, required = True)
    parser.add_argument("--baseline-errors", type = str, default = None)
    parser.add_argument("--ace-errors", type = str, default = None)
    parser.add_argument("--output", type = str, default = "benchmark/results/v2/comparison_report_v2.md")
    parser.add_argument("--title-label", type = str, default = "V2")

    parser.add_argument(
        "--cost-mode",
        type = str,
        default = "legacy",
        choices = ["legacy", "dual_source"],
    )
    parser.add_argument(
        "--billing-policy",
        type = str,
        default = "off",
        choices = ["strict", "off"],
    )
    parser.add_argument("--run-meta", type = str, default = None)
    parser.add_argument("--ace-aux-metrics", type = str, default = None)
    parser.add_argument("--baseline-eval-metrics", type = str, default = None)
    parser.add_argument("--ace-eval-metrics", type = str, default = None)
    parser.add_argument("--baseline-error-metrics", type = str, default = None)
    parser.add_argument("--ace-error-metrics", type = str, default = None)
    parser.add_argument("--openai-admin-key-env", type = str, default = "OPENAI_ADMIN_API_KEY")
    parser.add_argument("--openai-project-id-env", type = str, default = "OPENAI_COST_PROJECT_ID")
    args = parser.parse_args()

    baseline_data = load_jsonl(args.baseline)
    ace_data = load_jsonl(args.ace)

    baseline_ids = set(collect_task_ids(baseline_data))
    ace_ids = set(collect_task_ids(ace_data))
    if baseline_ids != ace_ids:
        missing_in_ace = sorted(list(baseline_ids - ace_ids))
        missing_in_baseline = sorted(list(ace_ids - baseline_ids))
        raise ValueError(
            "Baseline and ACE graded files do not contain the same task_id set. "
            f"missing_in_ace={missing_in_ace[:5]} missing_in_baseline={missing_in_baseline[:5]}"
        )

    baseline_rates = compute_solving_rates(baseline_data)
    ace_rates = compute_solving_rates(ace_data)

    baseline_error_dist: Dict[str, float] = {}
    ace_error_dist: Dict[str, float] = {}
    if args.baseline_errors:
        baseline_error_data = load_jsonl(args.baseline_errors)
        baseline_error_dist = compute_error_distribution(baseline_error_data, len(baseline_data))
    if args.ace_errors:
        ace_error_data = load_jsonl(args.ace_errors)
        ace_error_dist = compute_error_distribution(ace_error_data, len(ace_data))

    baseline_metrics = compute_metrics(baseline_data)
    ace_metrics = compute_metrics(ace_data)
    baseline_cat_metrics = compute_metrics_by_category(baseline_data)
    ace_cat_metrics = compute_metrics_by_category(ace_data)
    baseline_runtime_diag = compute_runtime_diagnostics(baseline_data)
    ace_runtime_diag = compute_runtime_diagnostics(ace_data)

    metric_paths = resolve_phase_metric_paths(args)
    if not metric_paths["ace_aux_metrics"]:
        ace_dir = os.path.dirname(args.ace) if os.path.dirname(args.ace) else "."
        version_tag = "v5" if "_v5_" in os.path.basename(args.ace) or args.title_label.strip().upper() == "V5" else "v4"
        default_aux_path = os.path.join(ace_dir, f"ace_{version_tag}_metrics.json")
        metric_paths["ace_aux_metrics"] = default_aux_path if os.path.exists(default_aux_path) else None

    cost_breakdown_by_phase, full_pipeline_cost_metered = build_cost_breakdown(
        baseline_metrics = baseline_metrics,
        ace_metrics = ace_metrics,
        ace_aux_metrics_path = metric_paths["ace_aux_metrics"],
        baseline_eval_metrics_path = metric_paths["baseline_eval_metrics"],
        ace_eval_metrics_path = metric_paths["ace_eval_metrics"],
        baseline_error_metrics_path = metric_paths["baseline_error_metrics"],
        ace_error_metrics_path = metric_paths["ace_error_metrics"],
    )

    billed_reconciliation, cost_reconciliation_status = run_billing_reconciliation(
        args = args,
        full_pipeline_cost_metered = full_pipeline_cost_metered,
    )

    report = format_report(
        baseline_rates = baseline_rates,
        ace_rates = ace_rates,
        baseline_errors = baseline_error_dist,
        ace_errors = ace_error_dist,
        baseline_metrics = baseline_metrics,
        ace_metrics = ace_metrics,
        baseline_cat_metrics = baseline_cat_metrics,
        ace_cat_metrics = ace_cat_metrics,
        baseline_runtime_diag = baseline_runtime_diag,
        ace_runtime_diag = ace_runtime_diag,
        title_label = args.title_label,
        cost_breakdown_by_phase = cost_breakdown_by_phase,
        full_pipeline_cost_metered = full_pipeline_cost_metered,
        billed_reconciliation = billed_reconciliation,
        cost_mode = args.cost_mode,
        billing_policy = args.billing_policy,
    )

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok = True)
    with open(args.output, "w", encoding = "utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {args.output}")

    json_output = args.output.replace(".md", ".json")
    summary = {
        "baseline_solving_rates": baseline_rates,
        "ace_solving_rates": ace_rates,
        "baseline_error_distribution": baseline_error_dist,
        "ace_error_distribution": ace_error_dist,
        "baseline_metrics": baseline_metrics,
        "ace_metrics": ace_metrics,
        "baseline_category_metrics": {key: value for key, value in baseline_cat_metrics.items()},
        "ace_category_metrics": {key: value for key, value in ace_cat_metrics.items()},
        "baseline_v4_diagnostics": baseline_runtime_diag,
        "ace_v4_diagnostics": ace_runtime_diag,
        "baseline_v5_diagnostics": baseline_runtime_diag,
        "ace_v5_diagnostics": ace_runtime_diag,
        "full_pipeline_cost_metered": full_pipeline_cost_metered,
        "cost_breakdown_by_phase": cost_breakdown_by_phase,
        "openai_billed_reconciliation": billed_reconciliation,
        "cost_reconciliation_status": cost_reconciliation_status,
        "cost_mode": args.cost_mode,
        "billing_policy": args.billing_policy,
        "metric_paths": metric_paths,
    }
    with open(json_output, "w", encoding = "utf-8") as f:
        json.dump(summary, f, indent = 2, ensure_ascii = False)
    print(f"JSON summary saved to: {json_output}")


if __name__ == "__main__":
    main()
