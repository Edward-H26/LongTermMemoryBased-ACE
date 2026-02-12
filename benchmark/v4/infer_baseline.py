"""
CL-bench Baseline Inference V4: Raw GPT-5.1 (High), no ACE.

V4 additions:
- Deterministic subset with selectable sampling strategy
- Capped-output retry for empty completion-capped responses
- Deterministic manifest reuse support

Usage:
    python -m benchmark.v4.infer_baseline \
        --max-samples 200 \
        --seed 42 \
        --manifest benchmark/results/v4/subset_manifest_v4_seed42_n200.json \
        --sampling-strategy context_dense \
        --max-completion-tokens 8192 \
        --empty-output-retry-max-tokens 16384 \
        --output benchmark/results/v4/baseline_v4.jsonl
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from tqdm import tqdm

from benchmark.sampling import (
    CLBENCH_DATASET_NAME,
    CLBENCH_SPLIT,
    DEFAULT_SAMPLING_STRATEGY,
    get_task_id,
    load_clbench_dataset,
    resolve_subset_with_manifest,
)


def _safe_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if os.path.exists(path):
        with open(path, "r", encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def append_jsonl(item: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok = True)
    with open(path, "a", encoding = "utf-8") as f:
        f.write(json.dumps(item, ensure_ascii = False) + "\n")


def call_api(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    max_completion_tokens: int,
    max_retries: int = 3,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            response = client.chat.completions.create(
                model = model,
                messages = messages,
                max_completion_tokens = max_completion_tokens,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            choice = response.choices[0]
            content = choice.message.content or ""
            usage = response.usage
            completion_tokens = usage.completion_tokens if usage else 0
            metrics = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.total_tokens if usage else 0,
                "latency_ms": latency_ms,
                "finish_reason": choice.finish_reason or "",
                "completion_capped": bool(completion_tokens >= max_completion_tokens),
            }
            return content, metrics, None
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return "", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": 0,
                    "finish_reason": "error",
                    "completion_capped": False,
                }, str(exc)
    return "", {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_ms": 0,
        "finish_reason": "error",
        "completion_capped": False,
    }, "Unknown error"


def infer_with_retry(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    max_completion_tokens: int,
    retry_max_tokens: int,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    content, metrics, error = call_api(
        client = client,
        messages = messages,
        model = model,
        max_completion_tokens = max_completion_tokens,
        max_retries = 3,
    )

    aggregate = dict(metrics)
    retry_count = 0

    should_retry = (
        not error
        and not (content or "").strip()
        and metrics.get("completion_tokens", 0) == max_completion_tokens
        and retry_max_tokens > max_completion_tokens
    )

    if should_retry:
        retry_count = 1
        retry_messages = list(messages)
        retry_messages.append({
            "role": "user",
            "content": "Your previous response was empty. Provide the final answer now, fully and directly.",
        })
        retry_content, retry_metrics, retry_error = call_api(
            client = client,
            messages = retry_messages,
            model = model,
            max_completion_tokens = retry_max_tokens,
            max_retries = 1,
        )

        aggregate["prompt_tokens"] = aggregate.get("prompt_tokens", 0) + retry_metrics.get("prompt_tokens", 0)
        aggregate["completion_tokens"] = aggregate.get("completion_tokens", 0) + retry_metrics.get("completion_tokens", 0)
        aggregate["total_tokens"] = aggregate.get("total_tokens", 0) + retry_metrics.get("total_tokens", 0)
        aggregate["latency_ms"] = aggregate.get("latency_ms", 0) + retry_metrics.get("latency_ms", 0)
        aggregate["finish_reason"] = retry_metrics.get("finish_reason", aggregate.get("finish_reason", ""))
        aggregate["completion_capped"] = bool(
            aggregate.get("completion_capped", False) or retry_metrics.get("completion_capped", False)
        )

        if not retry_error:
            content = retry_content
        else:
            error = retry_error

    aggregate["empty_output_retry_count"] = retry_count
    return content, aggregate, error


def main() -> None:
    parser = argparse.ArgumentParser(description = "CL-bench Baseline V4 Inference")
    parser.add_argument("--model", type = str, default = "gpt-5.1")
    parser.add_argument("--output", type = str, default = "benchmark/results/v4/baseline_v4.jsonl")
    parser.add_argument("--max-samples", type = int, default = None)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--manifest", type = str, default = None)
    parser.add_argument(
        "--sampling-strategy",
        type = str,
        default = "context_dense",
        choices = ["task_random", "context_dense"],
    )
    parser.add_argument("--api-key", type = str, default = None)
    parser.add_argument("--base-url", type = str, default = None)
    parser.add_argument(
        "--max-completion-tokens",
        type = int,
        default = _safe_env_int("ACE_MAX_COMPLETION_TOKENS", 8192),
    )
    parser.add_argument(
        "--empty-output-retry-max-tokens",
        type = int,
        default = _safe_env_int("ACE_EMPTY_OUTPUT_RETRY_MAX_TOKENS", 16384),
    )
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
    data, _, reused_manifest = resolve_subset_with_manifest(
        all_data,
        max_samples = args.max_samples,
        seed = args.seed,
        manifest_path = args.manifest,
        dataset_name = CLBENCH_DATASET_NAME,
        split = CLBENCH_SPLIT,
        sampling_strategy = args.sampling_strategy,
    )
    print(f"Loaded {len(all_data)} tasks, selected {len(data)} tasks")
    if args.manifest:
        status = "reused existing" if reused_manifest else "created new"
        print(f"Manifest ({status}): {args.manifest}")
    print(f"Sampling strategy: {args.sampling_strategy or DEFAULT_SAMPLING_STRATEGY}")

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
    written_count = 0
    fail_count = 0

    for item in tqdm(pending, desc = "Baseline V4"):
        messages = item.get("messages", [])
        task_id = get_task_id(item)
        api_messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]

        if not api_messages:
            fail_count += 1
            result = {
                "task_id": task_id,
                "messages": messages,
                "model_output": "",
                "rubrics": item.get("rubrics", []),
                "metadata": item.get("metadata", {}),
                "metrics": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": 0,
                    "finish_reason": "empty_messages",
                    "completion_capped": False,
                    "empty_output_retry_count": 0,
                    "api_error": "missing_messages",
                },
            }
            append_jsonl(result, args.output)
            written_count += 1
            continue

        content, metrics, error = infer_with_retry(
            client = client,
            messages = api_messages,
            model = args.model,
            max_completion_tokens = args.max_completion_tokens,
            retry_max_tokens = args.empty_output_retry_max_tokens,
        )

        if error:
            fail_count += 1
            metrics["api_error"] = error

        result = {
            "task_id": task_id,
            "messages": messages,
            "model_output": content,
            "rubrics": item.get("rubrics", []),
            "metadata": item.get("metadata", {}),
            "metrics": metrics,
        }
        append_jsonl(result, args.output)
        written_count += 1

    print(f"\nDone: {written_count} written, {fail_count} with API/format failures")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
