"""
CL-bench ACE Inference: GPT-5.1 (High) + ACE with all 5 enhancements.

Groups tasks by context_id and processes sequentially within each context
to enable intra-context learning (Enhancement 1).

Usage:
    python -m benchmark.infer_ace --max-samples 500 --output benchmark/results/ace.jsonl
"""

import json
import os
import sys
import time
import argparse
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("ACE_INJECTION_MODE", "post_context")
os.environ.setdefault("ACE_WEIGHT_RELEVANCE", "0.55")
os.environ.setdefault("ACE_WEIGHT_STRENGTH", "0.25")
os.environ.setdefault("ACE_WEIGHT_TYPE", "0.20")
os.environ.setdefault("ACE_SEED_META_STRATEGIES", "true")
os.environ.setdefault("LLM_BACKEND", "openai")

from tqdm import tqdm

from src.agent import build_ace_graph
from src.llm import get_metrics_collector


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


def get_context_id(item):
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get("context_id", "")
    return ""


def main():
    parser = argparse.ArgumentParser(description="CL-bench ACE Inference")
    parser.add_argument("--output", type=str, default="benchmark/results/ace.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model", type=str, default=None, help="Solver model (default: OPENAI_MODEL env)")
    args = parser.parse_args()

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

    context_groups = defaultdict(list)
    for item in pending:
        context_groups[get_context_id(item)].append(item)

    print(f"Processing {len(pending)} tasks across {len(context_groups)} contexts...")

    app = build_ace_graph()
    collector = get_metrics_collector()
    collector.reset()
    success_count = 0
    fail_count = 0

    pbar = tqdm(total=len(pending), desc="ACE Inference")

    for context_id, tasks in context_groups.items():
        for item in tasks:
            task_id = get_task_id(item)
            messages = item.get("messages", [])
            if not messages:
                fail_count += 1
                pbar.update(1)
                continue

            clean_messages = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") != "assistant"]

            try:
                cfg = {"configurable": {"thread_id": f"ace-bench-{task_id}"}}
                records_before = len(collector.records)
                start = time.perf_counter()

                state = {
                    "messages": clean_messages,
                    "mode": "",
                    "scratch": {
                        "ace_online_learning": True,
                        "learner_id": "benchmark_user",
                        "context_scope_id": context_id,
                    },
                    "result": {},
                }

                output = app.invoke(state, config=cfg)
                latency_ms = (time.perf_counter() - start) * 1000

                new_records = collector.records[records_before:]
                task_prompt_tokens = sum(r.prompt_tokens for r in new_records)
                task_completion_tokens = sum(r.completion_tokens for r in new_records)
                task_total_tokens = sum(r.total_tokens for r in new_records)
                task_num_calls = len(new_records)

                model_output = output.get("result", {}).get("answer", "")
                ace_delta = output.get("scratch", {}).get("ace_delta")

                result = {
                    "task_id": task_id,
                    "messages": messages,
                    "model_output": model_output,
                    "rubrics": item.get("rubrics", []),
                    "metadata": item.get("metadata", {}),
                    "metrics": {
                        "prompt_tokens": task_prompt_tokens,
                        "completion_tokens": task_completion_tokens,
                        "total_tokens": task_total_tokens,
                        "num_llm_calls": task_num_calls,
                        "latency_ms": latency_ms,
                        "mode": output.get("mode", ""),
                        "ace_delta": ace_delta,
                    },
                }
                append_jsonl(result, args.output)
                success_count += 1

            except Exception as e:
                fail_count += 1

            pbar.update(1)

    pbar.close()

    metrics_path = args.output.replace(".jsonl", "_metrics.json")
    collector.export_json(metrics_path)

    print(f"\nDone: {success_count} success, {fail_count} failed")
    print(f"Output: {args.output}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
