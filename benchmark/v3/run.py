"""
CL-bench V3: Run baseline and ACE inference in parallel.

Clears results and Neo4j database by default. Use --no-clear-results and
--no-clear-db to resume existing runs.

Usage:
    python -m benchmark.v3.run \
        --manifest benchmark/results/v3/subset_manifest_v3_seed42_n200.json \
        --max-samples 200 \
        --seed 42
"""

import argparse
import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()


def clear_neo4j_all():
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI") or os.getenv("NEXT_PUBLIC_NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEXT_PUBLIC_NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD") or os.getenv("NEXT_PUBLIC_NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE") or None

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session(database=db) as session:
        result = session.run("MATCH (n) DETACH DELETE n")
        result.consume()
    driver.close()


def main():
    parser = argparse.ArgumentParser(description="CL-bench V3: Parallel baseline + ACE inference")
    parser.add_argument("--manifest", type=str, default="benchmark/results/v3/subset_manifest_v3_seed42_n200.json")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="benchmark/results/v3")
    parser.add_argument(
        "--clear-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete v3 result files before starting (default: True). Use --no-clear-results to resume.",
    )
    parser.add_argument(
        "--clear-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wipe all Neo4j nodes and relationships before starting (default: True). Use --no-clear-db to resume.",
    )
    parser.add_argument(
        "--with-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run eval, error analysis, and comparison report after inference (default: True).",
    )
    args = parser.parse_args()

    baseline_path = os.path.join(args.output_dir, "baseline_v3.jsonl")
    ace_path = os.path.join(args.output_dir, "ace_v3.jsonl")
    ace_metrics_path = os.path.join(args.output_dir, "ace_v3_metrics.json")

    if args.clear_results:
        paths_to_clear = [
            baseline_path, ace_path, ace_metrics_path,
            os.path.join(args.output_dir, "baseline_v3_graded.jsonl"),
            os.path.join(args.output_dir, "ace_v3_graded.jsonl"),
            os.path.join(args.output_dir, "baseline_v3_graded_errors.jsonl"),
            os.path.join(args.output_dir, "ace_v3_graded_errors.jsonl"),
            os.path.join(args.output_dir, "comparison_report_v3.md"),
            os.path.join(args.output_dir, "comparison_report_v3.json"),
        ]
        for path in paths_to_clear:
            if os.path.exists(path):
                os.remove(path)
                print(f"Cleared: {path}")

    if args.clear_db:
        if os.getenv("NEO4J_URI") or os.getenv("NEXT_PUBLIC_NEO4J_URI"):
            try:
                clear_neo4j_all()
                print("Cleared all Neo4j nodes and relationships")
            except Exception as e:
                print(f"Warning: Neo4j clear failed ({e}). Proceeding anyway.")
        else:
            print("Skipping DB clear (Neo4j not configured)")

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_abs = os.path.abspath(args.manifest)
    baseline_abs = os.path.abspath(baseline_path)
    ace_abs = os.path.abspath(ace_path)

    baseline_cmd = [
        sys.executable, "-m", "benchmark.v3.infer_baseline",
        "--manifest", manifest_abs,
        "--max-samples", str(args.max_samples),
        "--seed", str(args.seed),
        "--output", baseline_abs,
        "--no-clear-results",
    ]
    ace_cmd = [
        sys.executable, "-m", "benchmark.v3.infer_ace",
        "--manifest", manifest_abs,
        "--max-samples", str(args.max_samples),
        "--seed", str(args.seed),
        "--output", ace_abs,
        "--no-clear-results",
        "--no-clear-db",
    ]

    print("Starting baseline v3 and ACE v3 in parallel...")
    p_baseline = subprocess.Popen(baseline_cmd)
    p_ace = subprocess.Popen(ace_cmd)

    print("Waiting for both to complete...")
    code_b = p_baseline.wait()
    code_a = p_ace.wait()

    if code_b != 0:
        print(f"Baseline exited with code {code_b}")
    if code_a != 0:
        print(f"ACE exited with code {code_a}")

    if code_b != 0 or code_a != 0:
        sys.exit(1)

    def count_lines(path):
        return sum(1 for _ in open(path)) if os.path.exists(path) else 0

    b_count = count_lines(baseline_abs)
    a_count = count_lines(ace_abs)
    if b_count < args.max_samples or a_count < args.max_samples:
        print(f"Error: incomplete inference. Baseline: {b_count}/{args.max_samples}, ACE: {a_count}/{args.max_samples}")
        sys.exit(1)

    if args.with_report:
        subprocess.run(
            [sys.executable, "-m", "benchmark.v3.complete_pipeline", "--skip-wait"],
            check=True,
        )

    print("V3 pipeline complete.")


if __name__ == "__main__":
    main()
