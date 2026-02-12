"""
CL-bench ACE Direct Inference V2: GPT-5.1 + ACE memory enrichment.

Direct API call with ACE Generator/Reflector/Curator roles properly implemented:
- Generator: Retrieves ACE bullets, creates guidance preamble, injects before last user message
- Reflector: Extracts lessons from execution trace using Gemini 3-flash-preview
- Curator: Applies delta updates to context-scoped ACE memory (heuristic mode)

FIX from v1:
- Preserves ALL messages including assistant (multi-turn reference solutions)
- No LangGraph pipeline, no CoT/ReAct wrapping, no system prompt replacement
- No <final> tag extraction, no output truncation
- Context-scoped memory isolation per context_id
- Database clearing for fresh experiments

Usage:
    python -m benchmark.v2.infer_ace --max-samples 200 --clear-memory --output benchmark/results/v2/ace_v2.jsonl
"""

import json
import os
import sys
import time
import argparse
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from tqdm import tqdm

from src.ace_memory import ACEMemory, Bullet
from src.ace_components import ACEPipeline, ExecutionTrace, Reflector, Curator
from src.storage import Neo4jMemoryStore
from src.llm import LLM, get_metrics_collector


META_STRATEGY_SEEDS = [
    Bullet(
        id="", content="Before answering, re-read all constraints, rules, and procedures explicitly stated in the provided context. Base your answer entirely on the context, not prior knowledge.",
        tags=["meta_strategy", "procedural"], memory_type="procedural", helpful_count=5,
    ),
    Bullet(
        id="", content="Follow the exact output format and structure specified in the system prompt and context. Do not add extraneous text or deviate from the requested format.",
        tags=["meta_strategy", "procedural"], memory_type="procedural", helpful_count=5,
    ),
    Bullet(
        id="", content="Do not rely on pre-trained knowledge when the context provides explicit rules, definitions, or information that may differ from common knowledge. Adhere strictly to what the context states.",
        tags=["meta_strategy", "procedural"], memory_type="procedural", helpful_count=5,
    ),
]


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


def clear_ace_memory(learner_id):
    from neo4j import GraphDatabase
    uri = os.getenv("NEO4J_URI") or os.getenv("NEXT_PUBLIC_NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEXT_PUBLIC_NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD") or os.getenv("NEXT_PUBLIC_NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE") or None

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session(database=db) as session:
        result = session.run(
            "MATCH (u:User {id: $userId})-[:HAS_ACE_MEMORY]->(m:AceMemoryState) "
            "DETACH DELETE m RETURN count(m) AS deleted",
            {"userId": learner_id},
        )
        record = result.single()
        deleted = record["deleted"] if record else 0
    driver.close()
    return deleted


_MEMORY_CACHE = {}


def get_context_memory(learner_id, context_id):
    cache_key = f"{learner_id}:{context_id}"
    if cache_key in _MEMORY_CACHE:
        return _MEMORY_CACHE[cache_key]

    storage = Neo4jMemoryStore(f"{learner_id}_ctx_{context_id[:8]}")
    memory = ACEMemory(max_bullets=100, dedup_threshold=0.85, prune_threshold=0.3, storage=storage)

    if not memory.bullets:
        for seed in META_STRATEGY_SEEDS:
            seed_copy = Bullet(
                id="", content=seed.content, tags=list(seed.tags),
                memory_type=seed.memory_type, helpful_count=seed.helpful_count,
                context_scope_id=context_id,
            )
            memory._merge_or_add_bullet(seed_copy)
        memory._save_memory()

    _MEMORY_CACHE[cache_key] = memory
    return memory


def format_guidance(bullets):
    if not bullets:
        return ""
    parts = ["=== Guidance from Prior Experience ===", "Based on lessons learned from similar tasks:"]
    for idx, bullet in enumerate(bullets, 1):
        parts.append(f"{idx}. {bullet.format_for_prompt()}")
    parts.append("===")
    return "\n".join(parts)


def inject_guidance(messages, guidance):
    if not guidance:
        return messages

    enriched = [dict(m) for m in messages]
    last_user_idx = None
    for i in range(len(enriched) - 1, -1, -1):
        if enriched[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is not None:
        enriched[last_user_idx] = dict(enriched[last_user_idx])
        enriched[last_user_idx]["content"] = f"{guidance}\n\n{enriched[last_user_idx]['content']}"

    return enriched


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


def run_reflector(reflector_llm, question, model_answer):
    trace = ExecutionTrace(
        question=question,
        model_answer=model_answer or "",
        success=bool(model_answer),
        trace_messages=[],
        metadata={},
    )
    reflector = Reflector(reflector_llm)
    try:
        lessons = reflector.reflect(trace, max_refinement_rounds=2)
        return lessons
    except Exception:
        return []


def apply_lessons_to_memory(memory, lessons, context_id):
    curator = Curator(None, memory)
    delta = curator._lessons_to_delta(lessons, learner_id="benchmark_user")
    for bullet in delta.new_bullets:
        bullet.context_scope_id = context_id
    memory.apply_delta(delta)
    return {
        "num_new_bullets": len(delta.new_bullets),
        "num_updates": len(delta.update_bullets),
        "num_removals": len(delta.remove_bullets),
    }


def main():
    parser = argparse.ArgumentParser(description="CL-bench ACE Direct V2 Inference")
    parser.add_argument("--output", type=str, default="benchmark/results/v2/ace_v2.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--clear-memory", action="store_true", help="Clear all ACE memory before starting")
    args = parser.parse_args()

    model = args.model or os.getenv("OPENAI_MODEL", "gpt-5.1")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_API_BASE")
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    reflector_llm = LLM(backend="gemini", temperature=0.3)

    if args.clear_memory:
        deleted = clear_ace_memory("benchmark_user")
        print(f"Cleared {deleted} ACE memory nodes")

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

    print(f"Processing {len(pending)} tasks across {len(context_groups)} contexts with {model}...")

    collector = get_metrics_collector()
    collector.reset()
    success_count = 0
    fail_count = 0

    pbar = tqdm(total=len(pending), desc="ACE V2")

    for context_id, tasks in context_groups.items():
        memory = get_context_memory("benchmark_user", context_id)

        for item in tasks:
            task_id = get_task_id(item)
            messages = item.get("messages", [])
            if not messages:
                fail_count += 1
                pbar.update(1)
                continue

            api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

            user_messages = [m["content"] for m in messages if m.get("role") == "user"]
            last_user_text = user_messages[-1] if user_messages else ""

            retrieved_bullets = memory.retrieve_relevant_bullets(
                last_user_text, top_k=5, context_scope_id=context_id,
            )
            guidance = format_guidance(retrieved_bullets)
            enriched_messages = inject_guidance(api_messages, guidance)

            records_before = len(collector.records)
            content, metrics, error = call_api(client, enriched_messages, model)

            if error:
                fail_count += 1
                pbar.update(1)
                continue

            lessons = run_reflector(reflector_llm, last_user_text, content)
            ace_delta = {}
            if lessons:
                ace_delta = apply_lessons_to_memory(memory, lessons, context_id)

            new_records = collector.records[records_before:]
            reflector_tokens = sum(r.total_tokens for r in new_records)

            result = {
                "task_id": task_id,
                "messages": messages,
                "model_output": content,
                "rubrics": item.get("rubrics", []),
                "metadata": item.get("metadata", {}),
                "metrics": {
                    "prompt_tokens": metrics.get("prompt_tokens", 0),
                    "completion_tokens": metrics.get("completion_tokens", 0),
                    "total_tokens": metrics.get("total_tokens", 0),
                    "reflector_tokens": reflector_tokens,
                    "num_bullets_retrieved": len(retrieved_bullets),
                    "num_lessons_extracted": len(lessons),
                    "latency_ms": metrics.get("latency_ms", 0),
                    "ace_delta": ace_delta,
                },
            }
            append_jsonl(result, args.output)
            success_count += 1
            pbar.update(1)

    pbar.close()

    metrics_path = args.output.replace(".jsonl", "_metrics.json")
    collector.export_json(metrics_path)

    print(f"\nDone: {success_count} success, {fail_count} failed")
    print(f"Output: {args.output}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
