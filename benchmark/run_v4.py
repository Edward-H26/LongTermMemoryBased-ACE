"""
CL-bench V4: Run baseline and ACE inference in parallel.

Default behavior:
- Clear v4 result artifacts
- Optionally clear Neo4j database
- Run baseline_v4 and ace_v4 inference in parallel
- Optionally run full post-inference report pipeline
"""

import argparse
import os
import subprocess
import sys

from dotenv import load_dotenv
from benchmark.preflight_v4 import (
    build_estimate,
    build_preflight_payload,
    run_smoke,
    run_static_checks,
    write_preflight_report,
)

load_dotenv()


def clear_neo4j_all() -> None:
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI") or os.getenv("NEXT_PUBLIC_NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEXT_PUBLIC_NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD") or os.getenv("NEXT_PUBLIC_NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE") or None

    driver = GraphDatabase.driver(uri, auth = (user, pwd))
    with driver.session(database = db) as session:
        result = session.run("MATCH (n) DETACH DELETE n")
        result.consume()
    driver.close()


def count_lines(path: str) -> int:
    return sum(1 for _ in open(path, "r", encoding = "utf-8")) if os.path.exists(path) else 0


def is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok = True)
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w", encoding = "utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def build_subprocess_env() -> dict:
    env = os.environ.copy()
    cache_dir = env.get("HF_DATASETS_CACHE")
    if not cache_dir:
        default_cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        if is_writable_dir(default_cache_dir):
            cache_dir = default_cache_dir
        else:
            cache_dir = os.path.abspath(os.path.join("benchmark", "hf_cache"))
        env["HF_DATASETS_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok = True)
    return env


def run_main_pipeline(args: argparse.Namespace) -> None:
    baseline_path = os.path.join(args.output_dir, "baseline_v4.jsonl")
    ace_path = os.path.join(args.output_dir, "ace_v4.jsonl")
    ace_metrics_path = os.path.join(args.output_dir, "ace_v4_metrics.json")

    if args.clear_results:
        paths_to_clear = [
            baseline_path,
            ace_path,
            ace_metrics_path,
            os.path.join(args.output_dir, "baseline_v4_graded.jsonl"),
            os.path.join(args.output_dir, "ace_v4_graded.jsonl"),
            os.path.join(args.output_dir, "baseline_v4_graded_errors.jsonl"),
            os.path.join(args.output_dir, "ace_v4_graded_errors.jsonl"),
            os.path.join(args.output_dir, "comparison_report_v4.md"),
            os.path.join(args.output_dir, "comparison_report_v4.json"),
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
            except Exception as exc:
                print(f"Warning: Neo4j clear failed ({exc}). Proceeding anyway.")
        else:
            print("Skipping DB clear (Neo4j not configured)")

    os.makedirs(args.output_dir, exist_ok = True)
    manifest_abs = os.path.abspath(args.manifest)
    baseline_abs = os.path.abspath(baseline_path)
    ace_abs = os.path.abspath(ace_path)

    baseline_cmd = [
        sys.executable,
        "-m",
        "benchmark.infer_baseline_v4",
        "--manifest",
        manifest_abs,
        "--max-samples",
        str(args.max_samples),
        "--seed",
        str(args.seed),
        "--sampling-strategy",
        args.sampling_strategy,
        "--output",
        baseline_abs,
        "--no-clear-results",
    ]
    ace_cmd = [
        sys.executable,
        "-m",
        "benchmark.infer_ace_direct_v4",
        "--manifest",
        manifest_abs,
        "--max-samples",
        str(args.max_samples),
        "--seed",
        str(args.seed),
        "--sampling-strategy",
        args.sampling_strategy,
        "--memory-scope",
        args.memory_scope,
        "--output",
        ace_abs,
        "--no-clear-results",
        "--no-clear-db",
    ]

    print("Starting baseline v4 and ACE v4 in parallel...")
    subprocess_env = build_subprocess_env()
    p_baseline = subprocess.Popen(baseline_cmd, env = subprocess_env)
    p_ace = subprocess.Popen(ace_cmd, env = subprocess_env)

    print("Waiting for both to complete...")
    code_b = p_baseline.wait()
    code_a = p_ace.wait()

    if code_b != 0:
        print(f"Baseline exited with code {code_b}")
    if code_a != 0:
        print(f"ACE exited with code {code_a}")
    if code_b != 0 or code_a != 0:
        sys.exit(1)

    baseline_count = count_lines(baseline_abs)
    ace_count = count_lines(ace_abs)
    if baseline_count < args.max_samples or ace_count < args.max_samples:
        print(
            f"Error: incomplete inference. "
            f"Baseline: {baseline_count}/{args.max_samples}, ACE: {ace_count}/{args.max_samples}"
        )
        sys.exit(1)

    if args.with_report:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmark.complete_v4_pipeline",
                "--output-dir",
                args.output_dir,
                "--skip-wait",
                "--max-samples",
                str(args.max_samples),
            ],
            check = True,
            env = subprocess_env,
        )

    print("V4 pipeline complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description = "CL-bench V4: Parallel baseline + ACE inference")
    parser.add_argument("--manifest", type = str, default = "benchmark/results/v4/subset_manifest_v4_seed42_n200.json")
    parser.add_argument("--max-samples", type = int, default = 200)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--sampling-strategy", type = str, default = "context_dense", choices = ["task_random", "context_dense"])
    parser.add_argument("--memory-scope", type = str, default = os.getenv("ACE_MEMORY_SCOPE_MODE", "hybrid"), choices = ["hybrid", "local", "global"])
    parser.add_argument("--output-dir", type = str, default = "benchmark/results/v4")
    parser.add_argument(
        "--clear-results",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "Delete v4 result files before starting (default: True). Use --no-clear-results to resume.",
    )
    parser.add_argument(
        "--clear-db",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "Wipe all Neo4j nodes and relationships before starting (default: True). Use --no-clear-db to resume.",
    )
    parser.add_argument(
        "--with-report",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "Run eval, error analysis, and comparison report after inference (default: True).",
    )
    parser.add_argument(
        "--preflight-mode",
        type = str,
        default = "off",
        choices = ["off", "static", "smoke", "both"],
        help = "Preflight mode: static checks, mini smoke run, or both.",
    )
    parser.add_argument(
        "--smoke-samples",
        type = int,
        default = 5,
        choices = [3, 4, 5],
        help = "Number of tasks for mini live smoke in preflight mode.",
    )
    parser.add_argument(
        "--smoke-output-dir",
        type = str,
        default = "benchmark/results/v4/smoke_v4",
        help = "Output directory for smoke artifacts.",
    )
    parser.add_argument(
        "--estimate-source",
        type = str,
        default = "auto",
        choices = ["auto", "v4", "v3", "heuristic"],
        help = "Source to use for static cost/time estimation.",
    )
    parser.add_argument(
        "--preflight-report",
        type = str,
        default = "benchmark/results/v4/preflight_v4.json",
        help = "JSON report path for preflight output.",
    )
    args = parser.parse_args()

    if args.max_samples <= 0:
        print("Error: --max-samples must be > 0")
        sys.exit(1)

    if args.preflight_mode == "off":
        run_main_pipeline(args)
        return

    context_workers = int(os.getenv("ACE_CONTEXT_WORKERS", "6"))
    step_scoring_mode = str(os.getenv("ACE_STEP_SCORING_MODE", "near_full")).strip().lower()
    step_score_workers = int(os.getenv("ACE_STEP_SCORE_WORKERS", "8"))

    static_checks = run_static_checks(
        manifest_path = args.manifest,
        output_dir = args.output_dir,
        max_samples = args.max_samples,
        sampling_strategy = args.sampling_strategy,
        memory_scope = args.memory_scope,
        smoke_samples = args.smoke_samples,
    )
    estimate = build_estimate(
        max_samples = args.max_samples,
        estimate_source = args.estimate_source,
        output_dir = args.output_dir,
        context_workers = context_workers,
        step_scoring_mode = step_scoring_mode,
        step_score_workers = step_score_workers,
    )
    smoke = None

    has_blocking = bool(static_checks.get("blocking_issues", []))
    should_run_smoke = args.preflight_mode in {"smoke", "both"}
    if should_run_smoke and not has_blocking:
        smoke = run_smoke(
            smoke_samples = args.smoke_samples,
            seed = args.seed,
            sampling_strategy = args.sampling_strategy,
            memory_scope = args.memory_scope,
            smoke_output_dir = args.smoke_output_dir,
            clear_db = bool(args.clear_db),
            estimate_source = args.estimate_source,
            max_samples_for_scale = args.max_samples,
            context_workers = context_workers,
            step_scoring_mode = step_scoring_mode,
            step_score_workers = step_score_workers,
        )

    report_payload = build_preflight_payload(
        preflight_mode = args.preflight_mode,
        static_checks = static_checks,
        estimate = estimate,
        smoke = smoke,
        config = {
            "manifest": args.manifest,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "sampling_strategy": args.sampling_strategy,
            "memory_scope": args.memory_scope,
            "smoke_samples": args.smoke_samples,
            "smoke_output_dir": args.smoke_output_dir,
            "estimate_source": args.estimate_source,
            "context_workers": context_workers,
            "step_scoring_mode": step_scoring_mode,
            "step_score_workers": step_score_workers,
        },
    )
    write_preflight_report(args.preflight_report, report_payload)
    print(f"Preflight report written: {args.preflight_report}")
    if report_payload.get("status") != "ok":
        print("Preflight failed.")
        sys.exit(1)

    print("Preflight complete.")


if __name__ == "__main__":
    main()
