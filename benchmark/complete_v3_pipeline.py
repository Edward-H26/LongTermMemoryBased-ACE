"""
Wait for v3 inference to complete, then run eval, error analysis, and comparison report.
Run after infer_baseline_v3 and infer_ace_direct_v3 (or run_v3).
"""

import argparse
import os
import subprocess
import sys
import time

RESULTS_DIR = "benchmark/results"
TARGET = 200


def main():
    parser = argparse.ArgumentParser(description="V3 post-inference: eval, error analysis, compare")
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Skip polling; assume inference is already complete (used when invoked by run_v3).",
    )
    args = parser.parse_args()

    baseline_path = os.path.join(RESULTS_DIR, "baseline_v3.jsonl")
    ace_path = os.path.join(RESULTS_DIR, "ace_v3.jsonl")

    if not args.skip_wait:
        print("Waiting for baseline_v3.jsonl and ace_v3.jsonl to reach 200 lines...")
    while True:
        b = 0
        a = 0
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                b = sum(1 for _ in f)
        if os.path.exists(ace_path):
            with open(ace_path) as f:
                a = sum(1 for _ in f)
        if not args.skip_wait:
            print(f"  Baseline: {b}/{TARGET}, ACE: {a}/{TARGET}", flush=True)
        if b >= TARGET and a >= TARGET:
            break
        if args.skip_wait:
            print(f"Error: inference incomplete. Baseline: {b}/{TARGET}, ACE: {a}/{TARGET}")
            sys.exit(1)
        time.sleep(60)

    print("Inference complete. Running evaluation...")

    subprocess.run([
        sys.executable, "-m", "benchmark.eval",
        "--input", baseline_path,
        "--output", os.path.join(RESULTS_DIR, "baseline_v3_graded.jsonl"),
        "--judge-model", "gpt-5.1",
    ], check=True)

    subprocess.run([
        sys.executable, "-m", "benchmark.eval",
        "--input", ace_path,
        "--output", os.path.join(RESULTS_DIR, "ace_v3_graded.jsonl"),
        "--judge-model", "gpt-5.1",
    ], check=True)

    print("Running error analysis...")
    subprocess.run([
        sys.executable, "-m", "benchmark.error_analysis",
        "--input", os.path.join(RESULTS_DIR, "baseline_v3_graded.jsonl"),
        "--output", os.path.join(RESULTS_DIR, "baseline_v3_graded_errors.jsonl"),
        "--judge-model", "gpt-5.1",
    ], check=True)

    subprocess.run([
        sys.executable, "-m", "benchmark.error_analysis",
        "--input", os.path.join(RESULTS_DIR, "ace_v3_graded.jsonl"),
        "--output", os.path.join(RESULTS_DIR, "ace_v3_graded_errors.jsonl"),
        "--judge-model", "gpt-5.1",
    ], check=True)

    print("Generating comparison report...")
    subprocess.run([
        sys.executable, "-m", "benchmark.compare",
        "--baseline", os.path.join(RESULTS_DIR, "baseline_v3_graded.jsonl"),
        "--ace", os.path.join(RESULTS_DIR, "ace_v3_graded.jsonl"),
        "--baseline-errors", os.path.join(RESULTS_DIR, "baseline_v3_graded_errors.jsonl"),
        "--ace-errors", os.path.join(RESULTS_DIR, "ace_v3_graded_errors.jsonl"),
        "--output", os.path.join(RESULTS_DIR, "comparison_report_v3.md"),
        "--title-label", "V3",
    ], check=True)

    print("V3 pipeline complete.")


if __name__ == "__main__":
    main()
