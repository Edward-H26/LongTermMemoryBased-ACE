"""
Backfill full-pipeline cost reporting for existing v4/v5 artifacts.

This utility is non-destructive by default and writes:
- benchmark/results/<version>/cost_backfill_<version>.md
- benchmark/results/<version>/cost_backfill_<version>.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(payload, f, indent = 2, ensure_ascii = False)


def build_paths(version: str, version_dir: str) -> Dict[str, str]:
    return {
        "baseline": os.path.join(version_dir, f"baseline_{version}_graded.jsonl"),
        "ace": os.path.join(version_dir, f"ace_{version}_graded.jsonl"),
        "baseline_errors": os.path.join(version_dir, f"baseline_{version}_graded_errors.jsonl"),
        "ace_errors": os.path.join(version_dir, f"ace_{version}_graded_errors.jsonl"),
        "ace_aux_metrics": os.path.join(version_dir, f"ace_{version}_metrics.json"),
        "baseline_eval_metrics": os.path.join(version_dir, f"baseline_{version}_graded_eval_metrics.json"),
        "ace_eval_metrics": os.path.join(version_dir, f"ace_{version}_graded_eval_metrics.json"),
        "baseline_error_metrics": os.path.join(version_dir, f"baseline_{version}_graded_errors_error_metrics.json"),
        "ace_error_metrics": os.path.join(version_dir, f"ace_{version}_graded_errors_error_metrics.json"),
        "run_meta": os.path.join(version_dir, "run_v5_meta.json"),
        "backfill_md": os.path.join(version_dir, f"cost_backfill_{version}.md"),
        "backfill_json": os.path.join(version_dir, f"cost_backfill_{version}.json"),
        "report_md": os.path.join(version_dir, f"comparison_report_{version}.md"),
        "report_json": os.path.join(version_dir, f"comparison_report_{version}.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description = "Backfill cost reporting for v4/v5 artifacts")
    parser.add_argument("--version", type = str, required = True, choices = ["v4", "v5"])
    parser.add_argument("--output-root", type = str, default = "benchmark/results")
    parser.add_argument("--cost-mode", type = str, default = os.getenv("BENCHMARK_COST_MODE", "dual_source"), choices = ["legacy", "dual_source"])
    parser.add_argument("--billing-policy", type = str, default = os.getenv("BENCHMARK_BILLING_POLICY", "strict"), choices = ["strict", "off"])
    parser.add_argument("--run-meta", type = str, default = None)
    parser.add_argument("--window-start", type = str, default = None)
    parser.add_argument("--window-end", type = str, default = None)
    parser.add_argument("--openai-admin-key-env", type = str, default = "OPENAI_ADMIN_API_KEY")
    parser.add_argument("--openai-project-id-env", type = str, default = "OPENAI_COST_PROJECT_ID")
    parser.add_argument(
        "--overwrite-original-report",
        action = argparse.BooleanOptionalAction,
        default = False,
        help = "If true, copy backfill outputs over comparison_report_<version> files.",
    )
    args = parser.parse_args()

    version_dir = os.path.join(args.output_root, args.version)
    paths = build_paths(args.version, version_dir)
    os.makedirs(version_dir, exist_ok = True)

    for required in [paths["baseline"], paths["ace"]]:
        if not os.path.exists(required):
            raise FileNotFoundError(f"Missing required input file: {required}")

    run_meta_path = args.run_meta or (paths["run_meta"] if os.path.exists(paths["run_meta"]) else "")
    if not run_meta_path and args.window_start and args.window_end:
        run_meta_path = os.path.join(version_dir, f"cost_backfill_window_{args.version}.json")
        write_json(
            run_meta_path,
            {
                "run_id": f"backfill-{args.version}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                "started_at": args.window_start,
                "ended_at": args.window_end,
                "phases": {
                    "inference": {"started_at": args.window_start},
                    "compare": {"ended_at": args.window_end},
                },
                "generated_at": utc_now_iso(),
                "source": "backfill_window_override",
            },
        )

    compare_cmd = [
        sys.executable,
        "-m",
        "benchmark.compare",
        "--baseline",
        paths["baseline"],
        "--ace",
        paths["ace"],
        "--output",
        paths["backfill_md"],
        "--title-label",
        args.version.upper(),
        "--cost-mode",
        args.cost_mode,
        "--billing-policy",
        args.billing_policy,
        "--openai-admin-key-env",
        args.openai_admin_key_env,
        "--openai-project-id-env",
        args.openai_project_id_env,
    ]

    if os.path.exists(paths["baseline_errors"]):
        compare_cmd.extend(["--baseline-errors", paths["baseline_errors"]])
    if os.path.exists(paths["ace_errors"]):
        compare_cmd.extend(["--ace-errors", paths["ace_errors"]])
    if os.path.exists(paths["ace_aux_metrics"]):
        compare_cmd.extend(["--ace-aux-metrics", paths["ace_aux_metrics"]])
    if os.path.exists(paths["baseline_eval_metrics"]):
        compare_cmd.extend(["--baseline-eval-metrics", paths["baseline_eval_metrics"]])
    if os.path.exists(paths["ace_eval_metrics"]):
        compare_cmd.extend(["--ace-eval-metrics", paths["ace_eval_metrics"]])
    if os.path.exists(paths["baseline_error_metrics"]):
        compare_cmd.extend(["--baseline-error-metrics", paths["baseline_error_metrics"]])
    if os.path.exists(paths["ace_error_metrics"]):
        compare_cmd.extend(["--ace-error-metrics", paths["ace_error_metrics"]])
    if run_meta_path:
        compare_cmd.extend(["--run-meta", run_meta_path])

    subprocess.run(compare_cmd, check = True)

    if args.overwrite_original_report:
        shutil.copyfile(paths["backfill_md"], paths["report_md"])
        shutil.copyfile(paths["backfill_json"], paths["report_json"])

    print(f"Backfill markdown: {paths['backfill_md']}")
    print(f"Backfill json: {paths['backfill_json']}")
    if args.overwrite_original_report:
        print(f"Updated report markdown: {paths['report_md']}")
        print(f"Updated report json: {paths['report_json']}")


if __name__ == "__main__":
    main()
