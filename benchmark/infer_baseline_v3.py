"""
CL-bench Baseline Inference V3: Raw GPT-5.1 (High), no ACE.

Adds deterministic seeded sampling with optional manifest reuse so
baseline and ACE runs can share the exact same subset and order.

Usage:
    python -m benchmark.infer_baseline_v3 \
        --max-samples 200 \
        --seed 42 \
        --manifest benchmark/results/v3/subset_manifest_v3_seed42_n200.json \
        --output benchmark/results/v3/baseline_v3.jsonl
"""

import argparse
import json
import os
import time

from dotenv import load_dotenv

# #region agent log
DEBUG_LOG = "/Users/edwardhu/Desktop/INFO490/LongTermMemoryBasedSelfEvolvingAlgorithm/.cursor/debug.log"
def _dbg(msg, data):
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"message": msg, "data": data, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception:
        pass
# #endregion

load_dotenv()

from openai import OpenAI
from tqdm import tqdm

from benchmark.sampling import (
    CLBENCH_DATASET_NAME,
    CLBENCH_SPLIT,
    get_task_id,
    load_clbench_dataset,
    resolve_subset_with_manifest,
)


def load_jsonl(path):
    data = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def append_jsonl(item, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def call_api(client, messages, model, max_retries = 3):
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            response = client.chat.completions.create(
                model = model,
                messages = messages,
                max_completion_tokens = 4096,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            content = response.choices[0].message.content
            usage = response.usage
            metrics = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "latency_ms": latency_ms,
            }
            return content, metrics, None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None, {}, str(e)
    return None, {}, "Unknown error"


def main():
    parser = argparse.ArgumentParser(description = "CL-bench Baseline V3 Inference")
    parser.add_argument("--model", type = str, default = "gpt-5.1")
    parser.add_argument("--output", type = str, default = "benchmark/results/v3/baseline_v3.jsonl")
    parser.add_argument("--max-samples", type = int, default = None)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--manifest", type = str, default = None)
    parser.add_argument("--api-key", type = str, default = None)
    parser.add_argument("--base-url", type = str, default = None)
    parser.add_argument(
        "--clear-results",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "Delete output file before starting (default: True). Use --no-clear-results to resume.",
    )
    args = parser.parse_args()

    if args.clear_results and os.path.exists(args.output):
        os.remove(args.output)
        print(f"Cleared existing output: {args.output}")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    print("Loading CL-bench dataset...")
    all_data = load_clbench_dataset(dataset_name = CLBENCH_DATASET_NAME, split = CLBENCH_SPLIT)
    data, manifest, reused_manifest = resolve_subset_with_manifest(
        all_data,
        max_samples = args.max_samples,
        seed = args.seed,
        manifest_path = args.manifest,
        dataset_name = CLBENCH_DATASET_NAME,
        split = CLBENCH_SPLIT,
    )
    print(f"Loaded {len(all_data)} tasks, selected {len(data)} tasks")
    if args.manifest:
        status = "reused existing" if reused_manifest else "created new"
        print(f"Manifest ({status}): {args.manifest}")

    completed_ids = set()
    if os.path.exists(args.output):
        existing = load_jsonl(args.output)
        completed_ids = {get_task_id(item) for item in existing if get_task_id(item)}
        print(f"Found {len(completed_ids)} completed, resuming remaining")

    pending = [item for item in data if get_task_id(item) not in completed_ids]
    if not pending:
        print("All samples already processed")
        return

    print(f"Running inference on {len(pending)} tasks with {args.model}...")
    success_count = 0
    fail_count = 0
    run_start = time.perf_counter()
    # #region agent log
    _dbg("run_start", {"stream": "baseline", "total_pending": len(pending), "completed_from_resume": len(completed_ids), "hypothesisId": "H2"})
    # #endregion

    for item in tqdm(pending, desc = "Baseline V3"):
        messages = item.get("messages", [])
        if not messages:
            fail_count += 1
            # #region agent log
            _dbg("task_skip_empty", {"stream": "baseline", "task_id": get_task_id(item), "hypothesisId": "H5"})
            # #endregion
            continue

        api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        content, metrics, error = call_api(client, api_messages, args.model)

        if error:
            fail_count += 1
            # #region agent log
            wall_ms = (time.perf_counter() - run_start) * 1000
            _dbg("task_fail", {"stream": "baseline", "tasks_processed": success_count + fail_count, "fail_count": fail_count, "wall_elapsed_ms": wall_ms, "hypothesisId": "H5"})
            # #endregion
            continue

        result = {
            "task_id": get_task_id(item),
            "messages": messages,
            "model_output": content,
            "rubrics": item.get("rubrics", []),
            "metadata": item.get("metadata", {}),
            "metrics": metrics,
        }
        append_jsonl(result, args.output)
        success_count += 1
        # #region agent log
        wall_ms = (time.perf_counter() - run_start) * 1000
        _dbg("task_complete", {"stream": "baseline", "task_index": success_count + fail_count, "task_id": get_task_id(item), "latency_ms": metrics.get("latency_ms"), "wall_elapsed_ms": wall_ms, "total_written": success_count, "hypothesisId": "H1,H3"})
        # #endregion

    print(f"\nDone: {success_count} success, {fail_count} failed")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
