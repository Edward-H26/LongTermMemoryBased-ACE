"""
CL-bench Baseline Inference V2: Raw GPT-5.1 (High), no ACE.

FIX from v1: Preserves ALL messages including assistant (reference solutions
for prior tasks in multi-turn sequences), matching the official CL-bench
infer.py protocol.

Usage:
    python -m benchmark.infer_baseline_v2 --max-samples 200 --output benchmark/results/v2/baseline_v2.jsonl
"""

import json
import os
import time
import argparse

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from tqdm import tqdm


def load_clbench(max_samples=None):
    from datasets import load_dataset
    ds = load_dataset("tencent/CL-bench", split="train")
    data = [dict(row) for row in ds]
    if max_samples:
        data = data[:max_samples]
    return data


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


def get_task_id(item):
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get("task_id", "")
    return ""


def call_api(client, messages, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=4096,
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
    parser = argparse.ArgumentParser(description="CL-bench Baseline V2 Inference")
    parser.add_argument("--model", type=str, default="gpt-5.1")
    parser.add_argument("--output", type=str, default="benchmark/results/v2/baseline_v2.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    print("Loading CL-bench dataset...")
    data = load_clbench(max_samples=args.max_samples)
    print(f"Loaded {len(data)} tasks")

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

    for item in tqdm(pending, desc="Baseline V2"):
        messages = item.get("messages", [])
        if not messages:
            fail_count += 1
            continue

        api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        content, metrics, error = call_api(client, api_messages, args.model)

        if error:
            fail_count += 1
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

    print(f"\nDone: {success_count} success, {fail_count} failed")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
