"""
CL-bench ACE Direct Inference V6: GPT-5.1 solver with Gemini-first ACE internals.

V6 additions:
- Isolated v6 artifacts and policy state
- Strict model routing checks (Gemini internals, GPT-5.1 solver)
- Lower quality-gate confidence threshold defaults for restored learning
- Strict final-answer recursion prompt style
- Tuned planner action budget to reduce recursion regressions
- Per-task routing telemetry
"""

import argparse
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

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
from src.ace_components import Curator, ExecutionTrace, QualityGateConfig, Reflector, apply_quality_gate
from src.ace_memory import ACEMemory, Bullet
from src.llm import LLM, get_metrics_collector
from src.planner_policy import PlannerPolicy, compute_shaped_reward
from src.reasoning_loop import ReasoningLoopConfig, run_recursive_openai_reasoning
from src.step_scoring import StepScoringConfig
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

UUID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_LOCAL_MEMORY_CACHE: Dict[str, ACEMemory] = {}
_GLOBAL_MEMORY_CACHE: Dict[str, ACEMemory] = {}
_LOCAL_CACHE_LOCK = threading.Lock()
_GLOBAL_CACHE_LOCK = threading.Lock()
_GLOBAL_MEMORY_LOCK = threading.Lock()
_REFLECTOR_LOCK = threading.Lock()

PLANNER_ACTIONS = {
    "direct": {"rounds": 1, "candidates": 1},
    "explore": {"rounds": 1, "candidates": 2},
    "refine": {"rounds": 1, "candidates": 2},
    "deep_refine": {"rounds": 2, "candidates": 2},
}


def _safe_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _safe_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_backend(value: str) -> str:
    return str(value or "").strip().lower()


def _normalize_model(value: str) -> str:
    return str(value or "").strip().lower()


def _build_routing_snapshot(
    solver_model: str,
    reflector_backend: str,
    reflector_model: str,
    step_scorer_backend: str,
    step_scorer_model: str,
) -> Dict[str, str]:
    return {
        "solver_model": solver_model,
        "reflector_backend": reflector_backend,
        "reflector_model": reflector_model,
        "step_scorer_backend": step_scorer_backend,
        "step_scorer_model": step_scorer_model,
    }


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


def load_progress_entries(path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return entries
    with open(path, "r", encoding = "utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def append_progress_entry(path: str, entry: Dict[str, Any], lock: threading.Lock) -> bool:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok = True)
    try:
        with lock:
            with open(path, "a", encoding = "utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii = False) + "\n")
        return True
    except Exception:
        return False


def finalize_output_from_sources(
    output_path: str,
    ordered_task_ids: List[str],
    output_rows: List[Dict[str, Any]],
    progress_entries: List[Dict[str, Any]],
    finalize_order: bool,
) -> Dict[str, Any]:
    row_by_task: Dict[str, Dict[str, Any]] = {}
    source_by_task: Dict[str, str] = {}

    for row in output_rows:
        task_id = get_task_id(row)
        if not task_id:
            continue
        row_by_task[task_id] = row
        source_by_task[task_id] = "output"

    for entry in progress_entries:
        task_id = str(entry.get("task_id", "")).strip()
        row = entry.get("row")
        if not task_id or not isinstance(row, dict):
            continue
        row_by_task[task_id] = row
        source_by_task[task_id] = "progress"

    ordered_rows: List[Dict[str, Any]] = []
    for task_id in ordered_task_ids:
        row = row_by_task.get(task_id)
        if not row:
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
            row["metrics"] = metrics
        src = source_by_task.get(task_id, "live")
        current_resume_source = str(metrics.get("resume_source", "")).strip().lower()
        if src == "progress" and current_resume_source in {"", "live"}:
            metrics["resume_source"] = "progress_journal"
        elif src == "output" and current_resume_source == "":
            metrics["resume_source"] = "output_file"
        ordered_rows.append(row)

    if finalize_order:
        tmp_path = f"{output_path}.tmp"
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok = True)
        with open(tmp_path, "w", encoding = "utf-8") as f:
            for row in ordered_rows:
                f.write(json.dumps(row, ensure_ascii = False) + "\n")
        os.replace(tmp_path, output_path)
    else:
        existing_ids = {get_task_id(row) for row in output_rows if get_task_id(row)}
        for row in ordered_rows:
            task_id = get_task_id(row)
            if task_id in existing_ids:
                continue
            append_jsonl(row, output_path)
            existing_ids.add(task_id)

    return {
        "rows": len(ordered_rows),
        "task_ids": [get_task_id(row) for row in ordered_rows if get_task_id(row)],
    }


def write_completion_marker(
    output_path: str,
    total_selected: int,
    completed_count: int,
    progress_path: str,
) -> str:
    marker_path = f"{output_path}.complete.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_path": output_path,
        "progress_path": progress_path,
        "selected_count": total_selected,
        "completed_count": completed_count,
        "is_complete": bool(completed_count >= total_selected),
    }
    with open(marker_path, "w", encoding = "utf-8") as f:
        json.dump(payload, f, indent = 2, ensure_ascii = False)
    return marker_path


def get_context_id(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        context_id = metadata.get("context_id", "")
        if isinstance(context_id, str):
            return context_id
    return ""


def clear_neo4j_all() -> bool:
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


def _seed_memory_if_empty(memory: ACEMemory, context_scope_id: Optional[str]) -> None:
    if memory.bullets:
        return
    for seed in META_STRATEGY_SEEDS:
        seed_copy = Bullet(
            id = "",
            content = seed.content,
            tags = list(seed.tags),
            memory_type = seed.memory_type,
            helpful_count = seed.helpful_count,
            context_scope_id = context_scope_id,
        )
        memory._merge_or_add_bullet(seed_copy)
    memory._save_memory()


def get_local_memory(context_id: str) -> ACEMemory:
    with _LOCAL_CACHE_LOCK:
        if context_id in _LOCAL_MEMORY_CACHE:
            return _LOCAL_MEMORY_CACHE[context_id]

    storage_id = f"benchmark_user_ctx_{context_id}"
    storage = Neo4jMemoryStore(storage_id)
    memory = ACEMemory(max_bullets = 100, dedup_threshold = 0.85, prune_threshold = 0.3, storage = storage)
    _seed_memory_if_empty(memory, context_scope_id = context_id)

    with _LOCAL_CACHE_LOCK:
        _LOCAL_MEMORY_CACHE[context_id] = memory
    return memory


def get_global_memory() -> ACEMemory:
    cache_key = "benchmark_user_global"
    with _GLOBAL_CACHE_LOCK:
        if cache_key in _GLOBAL_MEMORY_CACHE:
            return _GLOBAL_MEMORY_CACHE[cache_key]

    storage = Neo4jMemoryStore(cache_key)
    memory = ACEMemory(max_bullets = 100, dedup_threshold = 0.85, prune_threshold = 0.3, storage = storage)
    if _safe_env_bool("ACE_SEED_GLOBAL_META_STRATEGIES", False):
        _seed_memory_if_empty(memory, context_scope_id = None)

    with _GLOBAL_CACHE_LOCK:
        _GLOBAL_MEMORY_CACHE[cache_key] = memory
    return memory


def format_guidance(bullets: List[Bullet]) -> str:
    if not bullets:
        return ""
    parts = ["=== Guidance from Prior Experience ===", "Based on lessons learned from similar tasks:"]
    for idx, bullet in enumerate(bullets, 1):
        parts.append(f"{idx}. {bullet.format_for_prompt()}")
    parts.append("===")
    return "\n".join(parts)


def inject_guidance(messages: List[Dict[str, str]], guidance: str) -> List[Dict[str, str]]:
    if not guidance:
        return messages

    enriched = [dict(m) for m in messages]
    last_user_idx = None
    for idx in range(len(enriched) - 1, -1, -1):
        if enriched[idx].get("role") == "user":
            last_user_idx = idx
            break

    if last_user_idx is not None:
        enriched[last_user_idx] = dict(enriched[last_user_idx])
        original = enriched[last_user_idx].get("content", "")
        enriched[last_user_idx]["content"] = f"{guidance}\n\n{original}"

    return enriched


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


def run_reflector(
    reflector_llm: LLM,
    collector,
    question: str,
    model_answer: str,
) -> Tuple[List[Dict[str, Any]], int]:
    trace = ExecutionTrace(
        question = question,
        model_answer = model_answer or "",
        success = bool(model_answer and str(model_answer).strip()),
        trace_messages = [],
        metadata = {},
    )
    reflector = Reflector(reflector_llm)

    with _REFLECTOR_LOCK:
        records_before = len(collector.records)
        try:
            lessons = reflector.reflect(trace, max_refinement_rounds = 2)
        except Exception:
            lessons = []
        new_records = collector.records[records_before:]
    reflector_tokens = sum(record.total_tokens for record in new_records)
    return lessons, reflector_tokens


def apply_lessons_to_memory(
    memory: ACEMemory,
    lessons: List[Dict[str, Any]],
    context_id: Optional[str],
    learner_id: str,
) -> Dict[str, Any]:
    curator = Curator(None, memory)
    delta = curator._lessons_to_delta(lessons, learner_id = learner_id)
    for bullet in delta.new_bullets:
        if context_id:
            bullet.context_scope_id = context_id
    persist_status = memory.apply_delta(delta)
    if not isinstance(persist_status, dict):
        persist_status = memory.get_last_persist_status()
    return {
        "num_new_bullets": len(delta.new_bullets),
        "num_updates": len(delta.update_bullets),
        "num_removals": len(delta.remove_bullets),
        "persist_ok": bool(persist_status.get("ok", True)),
        "persist_retries": int(persist_status.get("retries", 0)),
        "persist_error": str(persist_status.get("error", "")),
    }


def _is_transferable_lesson(lesson: Dict[str, Any]) -> bool:
    content = str(lesson.get("content", "")).strip()
    if not content:
        return False
    lesson_type = str(lesson.get("type", "")).lower()
    if lesson_type not in {"success", "failure", "domain", "tool"}:
        return False
    tokens = TOKEN_PATTERN.findall(content.lower())
    if len(tokens) < 8:
        return False
    if UUID_PATTERN.search(content):
        return False
    if re.search(r"\b\d{6,}\b", content):
        return False
    return True


def filter_transferable_lessons(lessons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [lesson for lesson in lessons if _is_transferable_lesson(lesson)]


def merge_retrieved_bullets(local_bullets: List[Bullet], global_bullets: List[Bullet]) -> List[Bullet]:
    merged: List[Bullet] = []
    seen = set()

    def add_all(items: List[Bullet]) -> None:
        for bullet in items:
            key = bullet.content_hash or bullet.id
            if key in seen:
                continue
            seen.add(key)
            merged.append(bullet)

    add_all(local_bullets)
    add_all(global_bullets)
    return merged


def count_seed_and_learned(bullets: List[Bullet]) -> Tuple[int, int]:
    seed_count = 0
    learned_count = 0
    for bullet in bullets:
        tags = {tag.lower() for tag in bullet.tags}
        if "meta_strategy" in tags:
            seed_count += 1
        else:
            learned_count += 1
    return seed_count, learned_count


def create_failure_result(
    item: Dict[str, Any],
    task_id: str,
    api_error: str,
    memory_scope: str,
    routing_snapshot: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "messages": item.get("messages", []),
        "model_output": "",
        "rubrics": item.get("rubrics", []),
        "metadata": item.get("metadata", {}),
        "metrics": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reflector_tokens": 0,
            "num_bullets_retrieved": 0,
            "num_local_bullets_retrieved": 0,
            "num_global_bullets_retrieved": 0,
            "num_seed_bullets_retrieved": 0,
            "num_learned_bullets_retrieved": 0,
            "num_lessons_extracted": 0,
            "num_lessons_accepted": 0,
            "latency_ms": 0,
            "finish_reason": "error",
            "completion_capped": False,
            "empty_output_retry_count": 0,
            "memory_scope_mode": memory_scope,
            "ace_delta": {
                "local": {"num_new_bullets": 0, "num_updates": 0, "num_removals": 0},
                "global": {"num_new_bullets": 0, "num_updates": 0, "num_removals": 0},
                "num_new_bullets": 0,
                "num_updates": 0,
                "num_removals": 0,
            },
            "quality_gate": {},
            "step_scoring": {
                "num_steps": 0,
                "num_llm_scored": 0,
                "mean_step_score": 0.0,
                "min_step_score": 0.0,
                "max_step_score": 0.0,
                "mean_step_confidence": 0.0,
                "min_step_confidence": 0.0,
                "max_step_confidence": 0.0,
                "overall_confidence": 0.0,
                "steps": [],
            },
            "step_score_mean": 0.0,
            "recursion": {
                "rounds_planned": 0,
                "rounds_used": 0,
                "candidates_per_round": 0,
                "candidate_calls": 0,
                "initial_score": 0.0,
                "final_score": 0.0,
                "improvement": 0.0,
                "improved": False,
                "exit_reason": "failure",
            },
            "planner": {
                "action_id": "",
                "explore": False,
                "scores": {},
                "reward_proxy": 0.0,
                "reward_confidence": 0.0,
                "policy_update": {"updated": False},
            },
            "model_routing": dict(routing_snapshot),
            "progress_written": False,
            "resume_source": "live",
            "memory_write_retries": 0,
            "memory_write_failed": False,
            "memory_error": "",
            "api_error": api_error,
        },
    }


def main() -> None:
    default_qg = QualityGateConfig.from_env()
    default_step_cfg = StepScoringConfig.from_env()
    default_step_backend = _normalize_backend(os.getenv("ACE_STEP_SCORER_BACKEND", "gemini"))
    default_step_model = str(os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")).strip()
    default_qg_confidence_min = _safe_env_float("ACE_V6_QG_CONFIDENCE_MIN", 0.60)
    strict_routing_default = _safe_env_bool("ACE_V6_STRICT_ROUTING", True)

    parser = argparse.ArgumentParser(description = "CL-bench ACE Direct V6 Inference")
    parser.add_argument("--output", type = str, default = "benchmark/results/v6/ace_v6.jsonl")
    parser.add_argument("--max-samples", type = int, default = None)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--manifest", type = str, default = None)
    parser.add_argument("--model", type = str, default = None)
    parser.add_argument(
        "--sampling-strategy",
        type = str,
        default = "context_dense",
        choices = ["task_random", "context_dense", "context_dense_stratified"],
    )
    parser.add_argument(
        "--memory-scope",
        type = str,
        default = os.getenv("ACE_MEMORY_SCOPE_MODE", "hybrid"),
        choices = ["hybrid", "local", "global"],
    )
    parser.add_argument("--local-top-k", type = int, default = _safe_env_int("ACE_LOCAL_TOP_K", 8))
    parser.add_argument("--global-top-k", type = int, default = _safe_env_int("ACE_GLOBAL_TOP_K", 5))
    parser.add_argument("--context-workers", type = int, default = _safe_env_int("ACE_CONTEXT_WORKERS", 6))
    parser.add_argument("--global-gate-score-min", type = float, default = _safe_env_float("ACE_GLOBAL_GATE_SCORE_MIN", 0.80))
    parser.add_argument("--step-scoring-mode", type = str, default = default_step_cfg.mode, choices = ["off", "near_full", "full"])
    parser.add_argument("--step-scorer-backend", type = str, default = default_step_backend, choices = ["openai", "gemini"])
    parser.add_argument("--step-scorer-model", type = str, default = default_step_model)
    parser.add_argument("--step-score-workers", type = int, default = default_step_cfg.workers)
    parser.add_argument("--step-score-min", type = float, default = default_step_cfg.min_score)
    parser.add_argument("--max-completion-tokens", type = int, default = _safe_env_int("ACE_MAX_COMPLETION_TOKENS", 16384))
    parser.add_argument("--empty-output-retry-max-tokens", type = int, default = _safe_env_int("ACE_EMPTY_OUTPUT_RETRY_MAX_TOKENS", 32768))
    parser.add_argument("--enforce-model-routing", action = argparse.BooleanOptionalAction, default = strict_routing_default)
    parser.add_argument("--allow-non-gpt-solver", action = argparse.BooleanOptionalAction, default = False)
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
        help = "Delete all Neo4j nodes and relationships before starting (default: True). Use --no-clear-db to resume.",
    )
    parser.add_argument("--qg-gate-score-min", type = float, default = default_qg.gate_score_min)
    parser.add_argument("--qg-lesson-score-min", type = float, default = default_qg.lesson_score_min)
    parser.add_argument("--qg-overlap-min", type = float, default = default_qg.overlap_min)
    parser.add_argument("--qg-confidence-min", type = float, default = default_qg_confidence_min)
    parser.add_argument("--qg-max-accepted-lessons", type = int, default = default_qg.max_accepted_lessons)
    parser.add_argument(
        "--progress-path",
        type = str,
        default = None,
        help = "Durable per-task progress journal path (default: <output>.progress.jsonl).",
    )
    parser.add_argument(
        "--resume-from-progress",
        action = argparse.BooleanOptionalAction,
        default = _safe_env_bool("ACE_V6_RESUME_FROM_PROGRESS", _safe_env_bool("ACE_V5_RESUME_FROM_PROGRESS", True)),
        help = "Resume from progress journal in addition to output file (default: True).",
    )
    parser.add_argument(
        "--finalize-order",
        action = argparse.BooleanOptionalAction,
        default = _safe_env_bool("ACE_V6_FINALIZE_ORDER", _safe_env_bool("ACE_V5_FINALIZE_ORDER", True)),
        help = "Rewrite final output in manifest order using output + progress sources (default: True).",
    )
    args = parser.parse_args()

    solver_model = args.model or os.getenv("OPENAI_MODEL", "gpt-5.1")
    reflector_backend = "gemini"
    reflector_model = str(os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")).strip()
    expected_gemini_model = reflector_model
    step_scorer_backend = _normalize_backend(args.step_scorer_backend)
    step_scorer_model = str(args.step_scorer_model).strip()

    if _normalize_model(solver_model) != "gpt-5.1" and not args.allow_non_gpt_solver:
        print(
            "Error: solver model must be gpt-5.1 for ACE v6. "
            "Use --allow-non-gpt-solver to override for experiments."
        )
        return

    if args.enforce_model_routing:
        route_errors: List[str] = []
        if step_scorer_backend != "gemini":
            route_errors.append(f"step scorer backend must be gemini, got={step_scorer_backend}")
        if _normalize_backend(reflector_backend) != "gemini":
            route_errors.append(f"reflector backend must be gemini, got={reflector_backend}")
        if step_scorer_model != expected_gemini_model:
            route_errors.append(
                f"step scorer model must match GEMINI_MODEL ({expected_gemini_model}), got={step_scorer_model}"
            )
        if reflector_model != expected_gemini_model:
            route_errors.append(
                f"reflector model must match GEMINI_MODEL ({expected_gemini_model}), got={reflector_model}"
            )
        if _normalize_model(solver_model) != "gpt-5.1" and not args.allow_non_gpt_solver:
            route_errors.append(f"solver model must be gpt-5.1, got={solver_model}")
        if route_errors:
            print("Error: model routing validation failed:")
            for issue in route_errors:
                print(f"  - {issue}")
            return

    routing_snapshot = _build_routing_snapshot(
        solver_model = solver_model,
        reflector_backend = reflector_backend,
        reflector_model = reflector_model,
        step_scorer_backend = step_scorer_backend,
        step_scorer_model = step_scorer_model,
    )

    qg_config = QualityGateConfig(
        gate_score_min = args.qg_gate_score_min,
        lesson_score_min = args.qg_lesson_score_min,
        overlap_min = args.qg_overlap_min,
        confidence_min = args.qg_confidence_min,
        max_accepted_lessons = args.qg_max_accepted_lessons,
    )
    step_config = StepScoringConfig(
        mode = args.step_scoring_mode,
        scorer_model = step_scorer_model,
        scorer_backend = step_scorer_backend,
        workers = args.step_score_workers,
        min_score = args.step_score_min,
    )
    recursion_config = ReasoningLoopConfig.from_env()

    progress_path = (
        args.progress_path
        or os.getenv("ACE_V6_PROGRESS_PATH")
        or os.getenv("ACE_V5_PROGRESS_PATH")
        or f"{args.output}.progress.jsonl"
    )
    completion_marker_path = f"{args.output}.complete.json"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return

    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_API_BASE")
    if base_url:
        client_kwargs["base_url"] = base_url

    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
    planner_state_path = os.getenv("ACE_PLANNER_STATE_PATH_ACE")
    if not planner_state_path:
        planner_state_path = os.path.join(output_dir, "planner_policy_ace_v6.json")
    planner_policy = PlannerPolicy(
        actions = list(PLANNER_ACTIONS.keys()),
        state_path = planner_state_path,
        epsilon = _safe_env_float("ACE_PLANNER_EPSILON", 0.08),
        ucb_c = _safe_env_float("ACE_PLANNER_UCB_C", 1.10),
        seed = args.seed,
    )

    if args.clear_results:
        for path in [args.output, args.output.replace(".jsonl", "_metrics.json"), progress_path, completion_marker_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Cleared: {path}")

    if args.clear_db:
        try:
            clear_neo4j_all()
            with _LOCAL_CACHE_LOCK:
                _LOCAL_MEMORY_CACHE.clear()
            with _GLOBAL_CACHE_LOCK:
                _GLOBAL_MEMORY_CACHE.clear()
            print("Cleared all Neo4j nodes and relationships")
        except Exception as exc:
            print(f"Warning: Neo4j clear failed ({exc}). Proceeding anyway.")

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

    output_rows_existing: List[Dict[str, Any]] = load_jsonl(args.output) if os.path.exists(args.output) else []
    output_completed_ids = {get_task_id(item) for item in output_rows_existing if get_task_id(item)}
    progress_entries_existing = load_progress_entries(progress_path) if args.resume_from_progress else []
    progress_completed_ids = set()
    for entry in progress_entries_existing:
        task_id = str(entry.get("task_id", "")).strip()
        if task_id:
            progress_completed_ids.add(task_id)

    completed_ids = set(output_completed_ids).union(progress_completed_ids)
    if completed_ids:
        print(
            f"Found completed tasks: output={len(output_completed_ids)}, "
            f"progress={len(progress_completed_ids)}, merged={len(completed_ids)}"
        )

    pending = [item for item in data if get_task_id(item) not in completed_ids]
    if not pending:
        print("All samples already processed")
        ordered_task_ids = [get_task_id(item) for item in data if get_task_id(item)]
        final_stats = finalize_output_from_sources(
            output_path = args.output,
            ordered_task_ids = ordered_task_ids,
            output_rows = output_rows_existing,
            progress_entries = progress_entries_existing,
            finalize_order = args.finalize_order,
        )
        marker_path = write_completion_marker(
            output_path = args.output,
            total_selected = len(ordered_task_ids),
            completed_count = final_stats["rows"],
            progress_path = progress_path,
        )
        print(f"Completion marker: {marker_path}")
        return

    context_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pending_order: List[str] = []
    for item in pending:
        task_id = get_task_id(item)
        pending_order.append(task_id)
        context_groups[get_context_id(item)].append(item)

    task_index = {task_id: idx for idx, task_id in enumerate(pending_order)}
    print(
        f"Processing {len(pending)} tasks across {len(context_groups)} contexts with {solver_model} "
        f"(memory_scope={args.memory_scope}, workers={args.context_workers})..."
    )
    print(
        f"Durable progress journal: {progress_path} "
        f"(resume_from_progress={args.resume_from_progress}, finalize_order={args.finalize_order})"
    )
    print(
        "Model routing:"
        f" solver={solver_model},"
        f" reflector={reflector_backend}:{reflector_model},"
        f" step_scorer={step_scorer_backend}:{step_scorer_model}"
    )

    collector = get_metrics_collector()
    collector.reset()

    reflector_llm = LLM(model = reflector_model, backend = reflector_backend, temperature = 0.3)
    global_memory = get_global_memory() if args.memory_scope in {"hybrid", "global"} else None

    progress_lock = threading.Lock()

    def process_context_tasks(context_id: str, tasks: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, Dict[str, Any]]], int]:
        local_memory = get_local_memory(context_id) if args.memory_scope in {"hybrid", "local"} else None
        client = OpenAI(**client_kwargs)

        context_results: List[Tuple[int, Dict[str, Any]]] = []
        context_failures = 0

        for item in tasks:
            task_id = get_task_id(item)
            index = task_index.get(task_id, 10**9)
            messages = item.get("messages", [])

            if not messages:
                context_failures += 1
                failure_row = create_failure_result(
                    item,
                    task_id,
                    "missing_messages",
                    args.memory_scope,
                    routing_snapshot,
                )
                failure_row["metrics"]["progress_written"] = True
                progress_written = append_progress_entry(
                    progress_path,
                    {"task_id": task_id, "index": index, "row": failure_row},
                    progress_lock,
                )
                failure_row["metrics"]["progress_written"] = progress_written
                context_results.append((index, failure_row))
                continue

            api_messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
            user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
            last_user_text = user_messages[-1] if user_messages else ""
            planner_choice = planner_policy.choose_action(
                features = {
                    "question_length": len(last_user_text),
                    "num_messages": len(messages),
                    "memory_scope_local": int(args.memory_scope in {"local", "hybrid"}),
                    "memory_scope_global": int(args.memory_scope in {"global", "hybrid"}),
                }
            )
            action_id = str(planner_choice.get("action_id", "direct"))
            action_cfg = PLANNER_ACTIONS.get(action_id, PLANNER_ACTIONS["direct"])
            memory_errors: List[str] = []

            local_retrieved: List[Bullet] = []
            global_retrieved: List[Bullet] = []

            if local_memory is not None:
                try:
                    local_retrieved = local_memory.retrieve_relevant_bullets(
                        last_user_text,
                        top_k = args.local_top_k,
                        context_scope_id = context_id,
                    )
                except Exception as exc:
                    memory_errors.append(f"local_retrieve:{exc}")

            if global_memory is not None:
                try:
                    with _GLOBAL_MEMORY_LOCK:
                        global_retrieved = global_memory.retrieve_relevant_bullets(
                            last_user_text,
                            top_k = args.global_top_k,
                        )
                except Exception as exc:
                    memory_errors.append(f"global_retrieve:{exc}")

            if args.memory_scope == "local":
                retrieved_bullets = local_retrieved
            elif args.memory_scope == "global":
                retrieved_bullets = global_retrieved
            else:
                retrieved_bullets = merge_retrieved_bullets(local_retrieved, global_retrieved)

            seed_count, learned_count = count_seed_and_learned(retrieved_bullets)
            guidance = format_guidance(retrieved_bullets)
            enriched_messages = inject_guidance(api_messages, guidance)

            recursive_output = run_recursive_openai_reasoning(
                question = last_user_text,
                base_messages = enriched_messages,
                generate_once = lambda generated_messages: infer_with_retry(
                    client = client,
                    messages = generated_messages,
                    model = solver_model,
                    max_completion_tokens = args.max_completion_tokens,
                    retry_max_tokens = args.empty_output_retry_max_tokens,
                ),
                step_config = step_config,
                loop_config = recursion_config,
                action_overrides = {
                    "rounds": int(action_cfg.get("rounds", recursion_config.max_rounds)),
                    "candidates": int(action_cfg.get("candidates", recursion_config.candidates)),
                },
                prompt_style = "strict_final",
            )
            content = str(recursive_output.get("content", ""))
            metrics = recursive_output.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            error = str(recursive_output.get("error", ""))
            step_summary = recursive_output.get("step_summary", {})
            if not isinstance(step_summary, dict):
                step_summary = {}
            recursion_diag = recursive_output.get("recursion", {})
            if not isinstance(recursion_diag, dict):
                recursion_diag = {}

            if error and not content.strip():
                context_failures += 1
            if error:
                metrics["api_error"] = error

            if (content or "").strip():
                lessons, reflector_tokens = run_reflector(
                    reflector_llm = reflector_llm,
                    collector = collector,
                    question = last_user_text,
                    model_answer = content,
                )
            else:
                lessons = []
                reflector_tokens = 0

            qg_eval = apply_quality_gate(
                question = last_user_text,
                model_answer = content,
                lessons = lessons,
                step_summary = step_summary,
                config = qg_config,
            )
            accepted_lessons = qg_eval.get("accepted_lessons", [])
            quality_gate = qg_eval.get("diagnostics", {})

            ace_delta_local: Dict[str, Any] = {
                "num_new_bullets": 0,
                "num_updates": 0,
                "num_removals": 0,
                "persist_ok": True,
                "persist_retries": 0,
                "persist_error": "",
            }
            ace_delta_global: Dict[str, Any] = {
                "num_new_bullets": 0,
                "num_updates": 0,
                "num_removals": 0,
                "persist_ok": True,
                "persist_retries": 0,
                "persist_error": "",
            }

            should_local_update = bool(quality_gate.get("should_apply_update", False) and accepted_lessons)
            if should_local_update and local_memory is not None:
                try:
                    ace_delta_local = apply_lessons_to_memory(
                        memory = local_memory,
                        lessons = accepted_lessons,
                        context_id = context_id,
                        learner_id = "benchmark_user_local",
                    )
                except Exception as exc:
                    ace_delta_local["persist_ok"] = False
                    ace_delta_local["persist_error"] = str(exc)
                    memory_errors.append(f"local_update:{exc}")

            should_global_update = bool(
                quality_gate.get("should_apply_update", False)
                and accepted_lessons
                and float(quality_gate.get("gate_score", 0.0)) >= args.global_gate_score_min
            )
            if should_global_update and global_memory is not None:
                transferable = filter_transferable_lessons(accepted_lessons)
                if transferable:
                    try:
                        with _GLOBAL_MEMORY_LOCK:
                            ace_delta_global = apply_lessons_to_memory(
                                memory = global_memory,
                                lessons = transferable,
                                context_id = None,
                                learner_id = "benchmark_user_global",
                            )
                    except Exception as exc:
                        ace_delta_global["persist_ok"] = False
                        ace_delta_global["persist_error"] = str(exc)
                        memory_errors.append(f"global_update:{exc}")

            reward_payload = compute_shaped_reward(
                step_score = float(step_summary.get("mean_step_score", 0.0) or 0.0),
                output_valid = bool((content or "").strip()),
                quality_gate_applied = bool(quality_gate.get("should_apply_update", False)),
                recursion_improved = bool(recursion_diag.get("improved", False)),
                step_confidence = float(step_summary.get("overall_confidence", 0.7) or 0.7),
            )
            planner_update = planner_policy.update(
                action_id = action_id,
                reward = reward_payload.get("proxy_reward", 0.0),
                confidence = reward_payload.get("confidence", 0.7),
                metadata = {
                    "task_id": task_id,
                    "stream": "ace_v6",
                    "memory_scope": args.memory_scope,
                },
            )

            ace_delta = {
                "local": {
                    "num_new_bullets": ace_delta_local.get("num_new_bullets", 0),
                    "num_updates": ace_delta_local.get("num_updates", 0),
                    "num_removals": ace_delta_local.get("num_removals", 0),
                },
                "global": {
                    "num_new_bullets": ace_delta_global.get("num_new_bullets", 0),
                    "num_updates": ace_delta_global.get("num_updates", 0),
                    "num_removals": ace_delta_global.get("num_removals", 0),
                },
                "num_new_bullets": ace_delta_local.get("num_new_bullets", 0) + ace_delta_global.get("num_new_bullets", 0),
                "num_updates": ace_delta_local.get("num_updates", 0) + ace_delta_global.get("num_updates", 0),
                "num_removals": ace_delta_local.get("num_removals", 0) + ace_delta_global.get("num_removals", 0),
            }

            memory_write_retries = int(ace_delta_local.get("persist_retries", 0)) + int(ace_delta_global.get("persist_retries", 0))
            memory_write_failed = bool(not ace_delta_local.get("persist_ok", True) or not ace_delta_global.get("persist_ok", True))
            if ace_delta_local.get("persist_error"):
                memory_errors.append("local_persist:" + str(ace_delta_local.get("persist_error")))
            if ace_delta_global.get("persist_error"):
                memory_errors.append("global_persist:" + str(ace_delta_global.get("persist_error")))

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
                    "num_local_bullets_retrieved": len(local_retrieved),
                    "num_global_bullets_retrieved": len(global_retrieved),
                    "num_seed_bullets_retrieved": seed_count,
                    "num_learned_bullets_retrieved": learned_count,
                    "num_lessons_extracted": len(lessons),
                    "num_lessons_accepted": len(accepted_lessons),
                    "latency_ms": metrics.get("latency_ms", 0),
                    "finish_reason": metrics.get("finish_reason", ""),
                    "completion_capped": bool(metrics.get("completion_capped", False)),
                    "empty_output_retry_count": metrics.get("empty_output_retry_count", 0),
                    "memory_scope_mode": args.memory_scope,
                    "ace_delta": ace_delta,
                    "quality_gate": quality_gate,
                    "step_scoring": step_summary,
                    "step_score_mean": float(step_summary.get("mean_step_score", 0.0) or 0.0),
                    "recursion": recursion_diag,
                    "planner": {
                        "action_id": action_id,
                        "explore": bool(planner_choice.get("explore", False)),
                        "scores": planner_choice.get("scores", {}),
                        "reward_proxy": reward_payload.get("proxy_reward", 0.0),
                        "reward_confidence": reward_payload.get("confidence", 0.0),
                        "policy_update": planner_update,
                    },
                    "model_routing": dict(routing_snapshot),
                    "resume_source": "live",
                    "memory_write_retries": memory_write_retries,
                    "memory_write_failed": memory_write_failed,
                    "memory_error": "; ".join(memory_errors),
                    "progress_written": False,
                },
            }
            if error:
                result["metrics"]["api_error"] = error

            result["metrics"]["progress_written"] = True
            progress_written = append_progress_entry(
                progress_path,
                {"task_id": task_id, "index": index, "row": result},
                progress_lock,
            )
            result["metrics"]["progress_written"] = progress_written
            context_results.append((index, result))

        return context_results, context_failures

    all_results: List[Tuple[int, Dict[str, Any]]] = []
    total_failures = 0

    progress = tqdm(total = len(pending), desc = "ACE V6")
    with ThreadPoolExecutor(max_workers = max(1, args.context_workers)) as executor:
        future_map = {
            executor.submit(process_context_tasks, context_id, tasks): context_id
            for context_id, tasks in context_groups.items()
        }
        for future in as_completed(future_map):
            context_id = future_map[future]
            context_tasks = context_groups.get(context_id, [])
            try:
                context_results, context_failures = future.result()
                all_results.extend(context_results)
                total_failures += context_failures
                progress.update(len(context_results))
            except Exception as exc:
                total_failures += len(context_tasks)
                for item in context_tasks:
                    task_id = get_task_id(item)
                    index = task_index.get(task_id, 10**9)
                    failed_row = create_failure_result(
                        item,
                        task_id,
                        f"context_worker_exception:{exc}",
                        args.memory_scope,
                        routing_snapshot,
                    )
                    failed_row["metrics"]["progress_written"] = True
                    progress_written = append_progress_entry(
                        progress_path,
                        {"task_id": task_id, "index": index, "row": failed_row},
                        progress_lock,
                    )
                    failed_row["metrics"]["progress_written"] = progress_written
                    all_results.append((index, failed_row))
                progress.update(len(context_tasks))
    progress.close()

    all_results.sort(key = lambda row: row[0])
    for _, row in all_results:
        if row.get("metrics", {}).get("progress_written"):
            continue
        task_id = get_task_id(row)
        index = task_index.get(task_id, 10**9)
        row.setdefault("metrics", {})["progress_written"] = True
        progress_written = append_progress_entry(
            progress_path,
            {"task_id": task_id, "index": index, "row": row},
            progress_lock,
        )
        row.setdefault("metrics", {})["progress_written"] = progress_written

    metrics_path = args.output.replace(".jsonl", "_metrics.json")
    collector.export_json(metrics_path)

    progress_entries_final = load_progress_entries(progress_path)
    ordered_task_ids = [get_task_id(item) for item in data if get_task_id(item)]
    final_stats = finalize_output_from_sources(
        output_path = args.output,
        ordered_task_ids = ordered_task_ids,
        output_rows = output_rows_existing,
        progress_entries = progress_entries_final,
        finalize_order = args.finalize_order,
    )
    marker_path = write_completion_marker(
        output_path = args.output,
        total_selected = len(ordered_task_ids),
        completed_count = final_stats["rows"],
        progress_path = progress_path,
    )

    finalized_rows = int(final_stats.get("rows", 0))
    print(f"\nDone: {finalized_rows} finalized rows, {total_failures} with API/format failures")
    print(f"Output: {args.output}")
    print(f"Progress: {progress_path}")
    print(f"Completion marker: {marker_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
