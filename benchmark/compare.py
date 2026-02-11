"""
Side-by-side comparison of baseline vs ACE benchmark results.

Produces Table 1 (Solving Rate), Table 2 (Error Analysis), Table 3 (Token/Latency/Cost).

Usage:
    python -m benchmark.compare \
        --baseline benchmark/results/baseline_v2_graded.jsonl \
        --ace benchmark/results/ace_v2_graded.jsonl \
        --baseline-errors benchmark/results/baseline_v2_graded_errors.jsonl \
        --ace-errors benchmark/results/ace_v2_graded_errors.jsonl \
        --output benchmark/results/comparison_report_v2.md
"""

import json
import os
import argparse

from dotenv import load_dotenv
load_dotenv()

GPT51_INPUT_PRICE = 1.25
GPT51_OUTPUT_PRICE = 10.00
GEMINI_INPUT_PRICE = 0.50
GEMINI_OUTPUT_PRICE = 3.00


def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_task_id(item):
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        task_id = metadata.get("task_id", item.get("task_id", ""))
        return task_id if isinstance(task_id, str) else ""
    task_id = item.get("task_id", "")
    return task_id if isinstance(task_id, str) else ""


def collect_task_ids(data):
    return [task_id for task_id in (get_task_id(item) for item in data) if task_id]


def compute_solving_rates(data):
    total = len(data)
    if total == 0:
        return {"overall": 0.0, "total": 0, "score_1": 0, "categories": {}}

    score_1 = sum(1 for item in data if item.get("score") == 1)

    category_stats = {}
    for item in data:
        metadata = item.get("metadata", {})
        category = metadata.get("context_category", "Unknown") if isinstance(metadata, dict) else "Unknown"
        stats = category_stats.setdefault(category, {"total": 0, "score_1": 0})
        stats["total"] += 1
        if item.get("score") == 1:
            stats["score_1"] += 1

    categories = {}
    for cat, stats in category_stats.items():
        categories[cat] = stats["score_1"] / stats["total"] * 100 if stats["total"] > 0 else 0

    return {
        "overall": score_1 / total * 100 if total > 0 else 0,
        "total": total,
        "score_1": score_1,
        "categories": categories,
        "category_counts": {cat: stats["total"] for cat, stats in category_stats.items()},
    }


def compute_error_distribution(error_data, total_tasks):
    error_counts = {"CONTEXT_IGNORED": 0, "CONTEXT_MISUSED": 0, "FORMAT_ERROR": 0, "REFUSAL": 0}
    for item in error_data:
        error_types = item.get("error_classification", {}).get("task_error_types", [])
        for et in error_types:
            if et in error_counts:
                error_counts[et] += 1

    distribution = {}
    for et, count in error_counts.items():
        distribution[et] = count / total_tasks * 100 if total_tasks > 0 else 0
    return distribution


def percentile(sorted_values, p):
    if not sorted_values:
        return 0
    idx = int(len(sorted_values) * p / 100)
    return sorted_values[min(idx, len(sorted_values) - 1)]


def compute_metrics(data):
    latencies = []
    prompt_tokens_list = []
    completion_tokens_list = []
    total_tokens_list = []

    for item in data:
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        lat = metrics.get("latency_ms", 0)
        if lat:
            latencies.append(lat)
        pt = metrics.get("prompt_tokens", 0)
        ct = metrics.get("completion_tokens", 0)
        tt = metrics.get("total_tokens", 0)
        if tt:
            prompt_tokens_list.append(pt)
            completion_tokens_list.append(ct)
            total_tokens_list.append(tt)

    if not latencies:
        return {"avg_latency_ms": 0, "p50_latency_ms": 0, "p95_latency_ms": 0,
                "avg_tokens": 0, "total_tokens": 0, "total_prompt_tokens": 0,
                "total_completion_tokens": 0, "cost_usd": 0}

    sorted_lat = sorted(latencies)
    total_prompt = sum(prompt_tokens_list)
    total_completion = sum(completion_tokens_list)
    total_tok = sum(total_tokens_list)

    cost = (total_prompt / 1_000_000 * GPT51_INPUT_PRICE) + (total_completion / 1_000_000 * GPT51_OUTPUT_PRICE)

    return {
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": percentile(sorted_lat, 50),
        "p95_latency_ms": percentile(sorted_lat, 95),
        "avg_tokens": total_tok / len(data) if data else 0,
        "avg_prompt_tokens": total_prompt / len(data) if data else 0,
        "avg_completion_tokens": total_completion / len(data) if data else 0,
        "total_tokens": total_tok,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "cost_usd": cost,
    }


def compute_metrics_by_category(data):
    cat_data = {}
    for item in data:
        metadata = item.get("metadata", {})
        category = metadata.get("context_category", "Unknown") if isinstance(metadata, dict) else "Unknown"
        cat_data.setdefault(category, []).append(item)

    result = {}
    for cat, items in cat_data.items():
        result[cat] = compute_metrics(items)
    return result


def format_report(
    baseline_rates,
    ace_rates,
    baseline_errors,
    ace_errors,
    baseline_metrics,
    ace_metrics,
    baseline_cat_metrics,
    ace_cat_metrics,
    title_label = "V2",
):
    lines = []
    lines.append(f"# CL-bench Evaluation: Baseline vs ACE Comparison ({title_label})\n")

    lines.append("## Table 1: Solving Rate by Category\n")
    all_cats = sorted(set(list(baseline_rates["categories"].keys()) + list(ace_rates["categories"].keys())))

    cat_abbrev = {
        "Domain Knowledge Reasoning": "DKR",
        "Rule System Application": "RSA",
        "Procedural Task Execution": "PTE",
        "Empirical Discovery & Simulation": "EDS",
    }

    header = "| Model | Overall (%) |"
    sep = "|---|---|"
    for cat in all_cats:
        short = cat_abbrev.get(cat, cat[:3])
        n = baseline_rates.get("category_counts", {}).get(cat, 0) or ace_rates.get("category_counts", {}).get(cat, 0)
        header += f" {short} n={n} (%) |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)

    row_b = f"| GPT-5.1 (High) baseline | {baseline_rates['overall']:.1f} |"
    for cat in all_cats:
        row_b += f" {baseline_rates['categories'].get(cat, 0):.1f} |"
    lines.append(row_b)

    row_a = f"| GPT-5.1 (High) + ACE | {ace_rates['overall']:.1f} |"
    for cat in all_cats:
        row_a += f" {ace_rates['categories'].get(cat, 0):.1f} |"
    lines.append(row_a)

    diff_row = f"| Delta | {ace_rates['overall'] - baseline_rates['overall']:+.1f} |"
    for cat in all_cats:
        b = baseline_rates["categories"].get(cat, 0)
        a = ace_rates["categories"].get(cat, 0)
        diff_row += f" {a - b:+.1f} |"
    lines.append(diff_row)

    if baseline_errors or ace_errors:
        lines.append("\n## Table 2: Error Analysis Distribution\n")
        lines.append("| Model | Context Ignored (%) | Context Misused (%) | Format Error (%) | Refusal (%) |")
        lines.append("|---|---|---|---|---|")

        if baseline_errors:
            lines.append(
                f"| Baseline | {baseline_errors.get('CONTEXT_IGNORED', 0):.1f} | "
                f"{baseline_errors.get('CONTEXT_MISUSED', 0):.1f} | "
                f"{baseline_errors.get('FORMAT_ERROR', 0):.1f} | "
                f"{baseline_errors.get('REFUSAL', 0):.1f} |"
            )
        if ace_errors:
            lines.append(
                f"| ACE | {ace_errors.get('CONTEXT_IGNORED', 0):.1f} | "
                f"{ace_errors.get('CONTEXT_MISUSED', 0):.1f} | "
                f"{ace_errors.get('FORMAT_ERROR', 0):.1f} | "
                f"{ace_errors.get('REFUSAL', 0):.1f} |"
            )

    lines.append("\n## Table 3: Token Usage, Latency, and Cost\n")
    lines.append("| Metric | Baseline | ACE | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Avg Tokens/Task | {baseline_metrics.get('avg_tokens', 0):,.0f} | {ace_metrics.get('avg_tokens', 0):,.0f} | {ace_metrics.get('avg_tokens', 0) - baseline_metrics.get('avg_tokens', 0):+,.0f} |")
    lines.append(f"| Avg Prompt Tokens | {baseline_metrics.get('avg_prompt_tokens', 0):,.0f} | {ace_metrics.get('avg_prompt_tokens', 0):,.0f} | {ace_metrics.get('avg_prompt_tokens', 0) - baseline_metrics.get('avg_prompt_tokens', 0):+,.0f} |")
    lines.append(f"| Avg Completion Tokens | {baseline_metrics.get('avg_completion_tokens', 0):,.0f} | {ace_metrics.get('avg_completion_tokens', 0):,.0f} | {ace_metrics.get('avg_completion_tokens', 0) - baseline_metrics.get('avg_completion_tokens', 0):+,.0f} |")
    lines.append(f"| Total Tokens | {baseline_metrics.get('total_tokens', 0):,} | {ace_metrics.get('total_tokens', 0):,} | {ace_metrics.get('total_tokens', 0) - baseline_metrics.get('total_tokens', 0):+,} |")
    lines.append(f"| Avg Latency (ms) | {baseline_metrics.get('avg_latency_ms', 0):,.0f} | {ace_metrics.get('avg_latency_ms', 0):,.0f} | {ace_metrics.get('avg_latency_ms', 0) - baseline_metrics.get('avg_latency_ms', 0):+,.0f} |")
    lines.append(f"| p50 Latency (ms) | {baseline_metrics.get('p50_latency_ms', 0):,.0f} | {ace_metrics.get('p50_latency_ms', 0):,.0f} | {ace_metrics.get('p50_latency_ms', 0) - baseline_metrics.get('p50_latency_ms', 0):+,.0f} |")
    lines.append(f"| p95 Latency (ms) | {baseline_metrics.get('p95_latency_ms', 0):,.0f} | {ace_metrics.get('p95_latency_ms', 0):,.0f} | {ace_metrics.get('p95_latency_ms', 0) - baseline_metrics.get('p95_latency_ms', 0):+,.0f} |")
    lines.append(f"| Estimated Cost ($) | ${baseline_metrics.get('cost_usd', 0):.2f} | ${ace_metrics.get('cost_usd', 0):.2f} | ${ace_metrics.get('cost_usd', 0) - baseline_metrics.get('cost_usd', 0):+.2f} |")

    if baseline_cat_metrics or ace_cat_metrics:
        lines.append("\n## Table 4: Per-Category Token Usage and Latency\n")
        lines.append("| Category | Baseline Avg Tokens | ACE Avg Tokens | Baseline Avg Latency (ms) | ACE Avg Latency (ms) |")
        lines.append("|---|---|---|---|---|")
        for cat in all_cats:
            short = cat_abbrev.get(cat, cat[:3])
            bm = baseline_cat_metrics.get(cat, {})
            am = ace_cat_metrics.get(cat, {})
            lines.append(
                f"| {short} | {bm.get('avg_tokens', 0):,.0f} | {am.get('avg_tokens', 0):,.0f} | "
                f"{bm.get('avg_latency_ms', 0):,.0f} | {am.get('avg_latency_ms', 0):,.0f} |"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CL-bench Baseline vs ACE Comparison")
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--ace", type=str, required=True)
    parser.add_argument("--baseline-errors", type=str, default=None)
    parser.add_argument("--ace-errors", type=str, default=None)
    parser.add_argument("--output", type=str, default="benchmark/results/comparison_report_v2.md")
    parser.add_argument("--title-label", type=str, default="V2")
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

    baseline_error_dist = {}
    ace_error_dist = {}
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

    report = format_report(
        baseline_rates,
        ace_rates,
        baseline_error_dist,
        ace_error_dist,
        baseline_metrics,
        ace_metrics,
        baseline_cat_metrics,
        ace_cat_metrics,
        title_label = args.title_label,
    )

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
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
        "baseline_category_metrics": {k: v for k, v in baseline_cat_metrics.items()},
        "ace_category_metrics": {k: v for k, v in ace_cat_metrics.items()},
    }
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"JSON summary saved to: {json_output}")


if __name__ == "__main__":
    main()
