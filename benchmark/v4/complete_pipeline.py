"""
Wait for v4 inference to complete, then run eval, error analysis, and comparison report.
Run after infer_baseline_v4 and infer_ace_direct_v4 (or run_v4).
"""

import argparse
import os
import subprocess
import sys
import time


def _count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding = "utf-8") as f:
        return sum(1 for _ in f)


def _run_parallel(cmd_a, cmd_b) -> None:
    p_a = subprocess.Popen(cmd_a)
    p_b = subprocess.Popen(cmd_b)
    code_a = p_a.wait()
    code_b = p_b.wait()
    if code_a != 0 or code_b != 0:
        raise RuntimeError(f"Parallel stage failed: code_a={code_a}, code_b={code_b}")


def main() -> None:
    parser = argparse.ArgumentParser(description = "V4 post-inference: eval, error analysis, compare")
    parser.add_argument("--output-dir", type = str, default = "benchmark/results/v4")
    parser.add_argument("--max-samples", type = int, default = 200)
    parser.add_argument("--judge-model", type = str, default = "gpt-5.1")
    parser.add_argument(
        "--skip-wait",
        action = "store_true",
        help = "Skip polling; assume inference is already complete (used when invoked by run_v4).",
    )
    args = parser.parse_args()

    if args.max_samples <= 0:
        print("Error: --max-samples must be > 0")
        sys.exit(1)

    baseline_path = os.path.join(args.output_dir, "baseline_v4.jsonl")
    ace_path = os.path.join(args.output_dir, "ace_v4.jsonl")

    if not args.skip_wait:
        print(f"Waiting for baseline_v4.jsonl and ace_v4.jsonl to reach {args.max_samples} lines...")
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

    print("Inference complete. Running evaluation...")
    _run_parallel(
        [
            sys.executable,
            "-m",
            "benchmark.eval",
            "--input",
            baseline_path,
            "--output",
            os.path.join(args.output_dir, "baseline_v4_graded.jsonl"),
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
            os.path.join(args.output_dir, "ace_v4_graded.jsonl"),
            "--judge-model",
            args.judge_model,
        ],
    )

    print("Running error analysis...")
    _run_parallel(
        [
            sys.executable,
            "-m",
            "benchmark.error_analysis",
            "--input",
            os.path.join(args.output_dir, "baseline_v4_graded.jsonl"),
            "--output",
            os.path.join(args.output_dir, "baseline_v4_graded_errors.jsonl"),
            "--judge-model",
            args.judge_model,
        ],
        [
            sys.executable,
            "-m",
            "benchmark.error_analysis",
            "--input",
            os.path.join(args.output_dir, "ace_v4_graded.jsonl"),
            "--output",
            os.path.join(args.output_dir, "ace_v4_graded_errors.jsonl"),
            "--judge-model",
            args.judge_model,
        ],
    )

    print("Generating comparison report...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.compare",
            "--baseline",
            os.path.join(args.output_dir, "baseline_v4_graded.jsonl"),
            "--ace",
            os.path.join(args.output_dir, "ace_v4_graded.jsonl"),
            "--baseline-errors",
            os.path.join(args.output_dir, "baseline_v4_graded_errors.jsonl"),
            "--ace-errors",
            os.path.join(args.output_dir, "ace_v4_graded_errors.jsonl"),
            "--output",
            os.path.join(args.output_dir, "comparison_report_v4.md"),
            "--title-label",
            "V4",
        ],
        check = True,
    )

    print("V4 pipeline complete.")


if __name__ == "__main__":
    main()
