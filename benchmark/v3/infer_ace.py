"""
CL-bench ACE Direct Inference V3: GPT-5.1 + ACE memory enrichment.

Adds deterministic seeded sampling and quality-gated online updates.

Usage:
    python -m benchmark.v3.infer_ace \
        --max-samples 200 \
        --seed 42 \
        --manifest benchmark/results/v3/subset_manifest_v3_seed42_n200.json \
        --qg-gate-score-min 0.65 \
        --qg-lesson-score-min 0.60 \
        --qg-overlap-min 0.10 \
        --qg-max-accepted-lessons 2 \
        --output benchmark/results/v3/ace_v3.jsonl
"""

import argparse
import json
import os
import time
from collections import defaultdict

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
from src.ace_components import (
    Curator,
    ExecutionTrace,
    QualityGateConfig,
    Reflector,
    apply_quality_gate,
)
from src.ace_memory import ACEMemory, Bullet
from src.llm import LLM, get_metrics_collector
from src.storage import Neo4jMemoryStore


META_STRATEGY_SEEDS = [
    Bullet(
        id = "",
        content = "Before answering, re-read all constraints, rules, and procedures explicitly stated in the provided context. Base your answer entirely on the context, not prior knowledge.",
        tags = ["meta_strategy", "procedural"],
        memory_type = "procedural",
        helpful_count = 5,
    ),
    Bullet(
        id = "",
        content = "Follow the exact output format and structure specified in the system prompt and context. Do not add extraneous text or deviate from the requested format.",
        tags = ["meta_strategy", "procedural"],
        memory_type = "procedural",
        helpful_count = 5,
    ),
    Bullet(
        id = "",
        content = "Do not rely on pre-trained knowledge when the context provides explicit rules, definitions, or information that may differ from common knowledge. Adhere strictly to what the context states.",
        tags = ["meta_strategy", "procedural"],
        memory_type = "procedural",
        helpful_count = 5,
    ),
]


def load_jsonl(path):
    data = []
    if os.path.exists(path):
        with open(path, "r", encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def append_jsonl(item, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok = True)
    with open(path, "a", encoding = "utf-8") as f:
        f.write(json.dumps(item, ensure_ascii = False) + "\n")


def get_context_id(item):
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get("context_id", "")
    return ""


def clear_neo4j_all():
    """
    Delete all nodes and relationships in the Neo4j database.
    WARNING: Destructive. Use only when the instance is dedicated to the benchmark.
    """
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
    return True


_MEMORY_CACHE = {}


def get_context_memory(learner_id, context_id):
    cache_key = f"{learner_id}:{context_id}"
    if cache_key in _MEMORY_CACHE:
        return _MEMORY_CACHE[cache_key]

    storage = Neo4jMemoryStore(f"{learner_id}_ctx_{context_id[:8]}")
    memory = ACEMemory(max_bullets = 100, dedup_threshold = 0.85, prune_threshold = 0.3, storage = storage)

    if not memory.bullets:
        for seed in META_STRATEGY_SEEDS:
            seed_copy = Bullet(
                id = "",
                content = seed.content,
                tags = list(seed.tags),
                memory_type = seed.memory_type,
                helpful_count = seed.helpful_count,
                context_scope_id = context_id,
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
        original_content = enriched[last_user_idx]["content"]
        enriched[last_user_idx]["content"] = f"{guidance}\n\n{original_content}"

    return enriched


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


def run_reflector(reflector_llm, question, model_answer):
    trace = ExecutionTrace(
        question = question,
        model_answer = model_answer or "",
        success = bool(model_answer and str(model_answer).strip()),
        trace_messages = [],
        metadata = {},
    )
    reflector = Reflector(reflector_llm)
    try:
        lessons = reflector.reflect(trace, max_refinement_rounds = 2)
        return lessons
    except Exception:
        return []


def apply_lessons_to_memory(memory, lessons, context_id):
    curator = Curator(None, memory)
    delta = curator._lessons_to_delta(lessons, learner_id = "benchmark_user")
    for bullet in delta.new_bullets:
        bullet.context_scope_id = context_id
    memory.apply_delta(delta)
    return {
        "num_new_bullets": len(delta.new_bullets),
        "num_updates": len(delta.update_bullets),
        "num_removals": len(delta.remove_bullets),
    }


def main():
    default_qg = QualityGateConfig.from_env()

    parser = argparse.ArgumentParser(description = "CL-bench ACE Direct V3 Inference")
    parser.add_argument("--output", type = str, default = "benchmark/results/v3/ace_v3.jsonl")
    parser.add_argument("--max-samples", type = int, default = None)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--manifest", type = str, default = None)
    parser.add_argument("--model", type = str, default = None)
    parser.add_argument(
        "--clear-results",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "Delete output and metrics files before starting (default: True). Use --no-clear-results to resume.",
    )
    parser.add_argument(
        "--clear-db",
        action = argparse.BooleanOptionalAction,
        default = True,
        help = "Delete all Neo4j nodes and relationships before starting (default: True). WARNING: Wipes entire DB. Use --no-clear-db to resume.",
    )
    parser.add_argument("--qg-gate-score-min", type = float, default = default_qg.gate_score_min)
    parser.add_argument("--qg-lesson-score-min", type = float, default = default_qg.lesson_score_min)
    parser.add_argument("--qg-overlap-min", type = float, default = default_qg.overlap_min)
    parser.add_argument("--qg-max-accepted-lessons", type = int, default = default_qg.max_accepted_lessons)
    args = parser.parse_args()

    qg_config = QualityGateConfig(
        gate_score_min = args.qg_gate_score_min,
        lesson_score_min = args.qg_lesson_score_min,
        overlap_min = args.qg_overlap_min,
        max_accepted_lessons = args.qg_max_accepted_lessons,
    )

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

    reflector_llm = LLM(backend = "gemini", temperature = 0.3)

    if args.clear_results:
        for path in [args.output, args.output.replace(".jsonl", "_metrics.json")]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Cleared: {path}")

    if args.clear_db:
        try:
            clear_neo4j_all()
            _MEMORY_CACHE.clear()
            print("Cleared all Neo4j nodes and relationships")
        except Exception as e:
            print(f"Warning: Neo4j clear failed ({e}). Proceeding anyway.")

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

    context_groups = defaultdict(list)
    for item in pending:
        context_groups[get_context_id(item)].append(item)

    print(f"Processing {len(pending)} tasks across {len(context_groups)} contexts with {model}...")

    collector = get_metrics_collector()
    collector.reset()
    success_count = 0
    fail_count = 0
    run_start = time.perf_counter()
    # #region agent log
    _dbg("run_start", {"stream": "ace", "total_pending": len(pending), "completed_from_resume": len(completed_ids), "num_contexts": len(context_groups), "hypothesisId": "H2,H4"})
    # #endregion

    pbar = tqdm(total = len(pending), desc = "ACE V3")

    for context_id, tasks in context_groups.items():
        memory = get_context_memory("benchmark_user", context_id)

        for item in tasks:
            task_id = get_task_id(item)
            messages = item.get("messages", [])
            if not messages:
                fail_count += 1
                # #region agent log
                _dbg("task_skip_empty", {"stream": "ace", "task_id": task_id, "hypothesisId": "H5"})
                # #endregion
                pbar.update(1)
                continue

            api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

            user_messages = [m["content"] for m in messages if m.get("role") == "user"]
            last_user_text = user_messages[-1] if user_messages else ""

            retrieved_bullets = memory.retrieve_relevant_bullets(
                last_user_text,
                top_k = 5,
                context_scope_id = context_id,
            )
            guidance = format_guidance(retrieved_bullets)
            enriched_messages = inject_guidance(api_messages, guidance)

            records_before = len(collector.records)
            content, metrics, error = call_api(client, enriched_messages, model)

            if error:
                fail_count += 1
                # #region agent log
                wall_ms = (time.perf_counter() - run_start) * 1000
                _dbg("task_fail", {"stream": "ace", "tasks_processed": success_count + fail_count, "fail_count": fail_count, "wall_elapsed_ms": wall_ms, "hypothesisId": "H5"})
                # #endregion
                pbar.update(1)
                continue

            lessons = run_reflector(reflector_llm, last_user_text, content)
            qg_eval = apply_quality_gate(
                question = last_user_text,
                model_answer = content,
                lessons = lessons,
                config = qg_config,
            )
            accepted_lessons = qg_eval.get("accepted_lessons", [])
            quality_gate = qg_eval.get("diagnostics", {})

            ace_delta = {
                "num_new_bullets": 0,
                "num_updates": 0,
                "num_removals": 0,
            }
            if quality_gate.get("should_apply_update") and accepted_lessons:
                ace_delta = apply_lessons_to_memory(memory, accepted_lessons, context_id)

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
                    "num_lessons_accepted": len(accepted_lessons),
                    "latency_ms": metrics.get("latency_ms", 0),
                    "ace_delta": ace_delta,
                    "quality_gate": quality_gate,
                },
            }
            append_jsonl(result, args.output)
            success_count += 1
            # #region agent log
            wall_ms = (time.perf_counter() - run_start) * 1000
            _dbg("task_complete", {"stream": "ace", "task_index": success_count + fail_count, "task_id": task_id, "latency_ms": metrics.get("latency_ms"), "wall_elapsed_ms": wall_ms, "total_written": success_count, "context_id": context_id[:8] if context_id else "", "hypothesisId": "H1,H3,H4"})
            # #endregion
            pbar.update(1)

    pbar.close()

    metrics_path = args.output.replace(".jsonl", "_metrics.json")
    collector.export_json(metrics_path)

    print(f"\nDone: {success_count} success, {fail_count} failed")
    print(f"Output: {args.output}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
