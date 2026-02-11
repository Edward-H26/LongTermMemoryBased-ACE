"""
ACE-Enhanced LangGraph Agent

Integrates Agentic Context Engineering into the LangGraph workflow:
- Uses ACE memory to enrich prompts before generation
- Applies ACE pipeline after execution to learn from traces
- Enhancement 1: Context-scoped memory isolation
- Enhancement 2: Post-context bullet injection
- Enhancement 4: Meta-strategy seed bullets
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Any, Dict, List, Optional
import time
import os
import re

from src.solvers import GraphState, solve_cot, solve_tot, solve_react, _extract_final
from src.ace_memory import ACEMemory, Bullet
from src.ace_components import ACEPipeline, ExecutionTrace
from src.storage import Neo4jMemoryStore
from src.llm import LLM
from src.tools import _calculator_schema, _google_search_schema, _neo4j_retrieveqa_schema


_ACE_CACHE: Dict[str, Dict[str, Any]] = {}

META_STRATEGY_SEEDS = [
    Bullet(
        id="",
        content="Before answering, re-read all constraints, rules, and procedures explicitly stated in the provided context.",
        tags=["meta_strategy", "procedural"],
        memory_type="procedural",
        helpful_count=5,
    ),
    Bullet(
        id="",
        content="Follow the exact output format specified in the system prompt and context. Do not add extraneous text or deviate from the requested structure.",
        tags=["meta_strategy", "procedural"],
        memory_type="procedural",
        helpful_count=5,
    ),
    Bullet(
        id="",
        content="Do not rely on pre-trained knowledge when the context provides explicit rules, definitions, or information that may differ from common knowledge.",
        tags=["meta_strategy", "procedural"],
        memory_type="procedural",
        helpful_count=5,
    ),
]


def _extract_retrieval_facets(message: str, scratch: Dict[str, Any]) -> Dict[str, Any]:
    facets: Dict[str, Any] = {}
    lowered = (message or "").lower()

    if any(token in lowered for token in ("visual", "diagram", "picture", "draw", "show me")):
        facets["needs_visual"] = True

    persona = scratch.get("persona_request") or scratch.get("voice")
    if persona:
        facets["persona_request"] = str(persona).lower()

    if "next step" in lowered or ("step" in lowered and "what" in lowered):
        facets["next_step_flag"] = True

    return facets


def _infer_topic(question: str) -> Optional[str]:
    return None


def get_ace_system(
    learner_id: Optional[str] = None,
    context_scope_id: Optional[str] = None,
    seed_meta_strategies: Optional[bool] = None,
):
    """Get or create the ACE system for a specific learner/context scope."""
    if seed_meta_strategies is None:
        seed_meta_strategies = os.getenv("ACE_SEED_META_STRATEGIES", "true").lower() in {"1", "true", "yes"}

    cache_key = f"{learner_id or 'global'}:{context_scope_id or 'global'}"
    entry = _ACE_CACHE.get(cache_key)
    if entry is None:
        entry = {}
        _ACE_CACHE[cache_key] = entry

    target_model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    try:
        target_temperature = float(os.getenv("ACE_LLM_TEMPERATURE", "0.2"))
    except ValueError:
        target_temperature = 0.2

    memory = entry.get("memory")
    if memory is None:
        storage_id = learner_id or "default_user"
        try:
            storage = Neo4jMemoryStore(storage_id)
        except Exception as exc:
            raise RuntimeError(
                f"[ACE Memory] CRITICAL: Neo4j storage initialization failed for learner={storage_id}. "
                f"Error: {exc}. Please check Neo4j credentials and connection."
            )

        memory = ACEMemory(
            max_bullets=100,
            dedup_threshold=0.85,
            prune_threshold=0.3,
            storage=storage,
        )

        if seed_meta_strategies and not memory.bullets:
            for seed in META_STRATEGY_SEEDS:
                seed_copy = Bullet(
                    id="",
                    content=seed.content,
                    tags=list(seed.tags),
                    memory_type=seed.memory_type,
                    helpful_count=seed.helpful_count,
                    context_scope_id=context_scope_id,
                )
                memory._merge_or_add_bullet(seed_copy)
            memory._save_memory()

        entry["memory"] = memory

    pipeline = entry.get("pipeline")
    if (
        pipeline is None
        or getattr(pipeline, "_llm_model", None) != target_model
        or getattr(pipeline, "_llm_temperature", None) != target_temperature
    ):
        llm = LLM(model=target_model, temperature=target_temperature, backend="gemini")
        pipeline = ACEPipeline(llm, memory)
        pipeline._llm_model = target_model
        pipeline._llm_temperature = target_temperature
        entry["pipeline"] = pipeline

    return memory, pipeline


def router_node(state: GraphState) -> GraphState:
    if state.get("mode"):
        return state

    scratch = state.setdefault("scratch", {})
    user_messages = [m.get("content", "") for m in state.get("messages", []) if m.get("role") == "user"]
    user_text = user_messages[-1] if user_messages else ""
    full_user_text = " ".join(user_messages)
    normalized_text = full_user_text.lower()
    learner_id = scratch.get("learner_id")
    context_scope_id = scratch.get("context_scope_id")
    topic = scratch.get("topic") or _infer_topic(user_text)
    if topic:
        scratch["topic"] = topic
        state["scratch"] = scratch

    memory, _ = get_ace_system(learner_id, context_scope_id=context_scope_id)
    if not scratch.get("_ace_memory_loaded"):
        if not memory.consume_fresh_init_flag():
            memory.reload_from_storage()
        scratch["_ace_memory_loaded"] = True
        state["scratch"] = scratch

    retrieval_facets = _extract_retrieval_facets(user_text, scratch)
    scratch["ace_retrieval_facets"] = retrieval_facets
    state["scratch"] = scratch

    relevant_bullets = memory.retrieve_relevant_bullets(
        user_text,
        top_k=5,
        learner_id=learner_id,
        topic=topic,
        facets=retrieval_facets,
        context_scope_id=context_scope_id,
    )
    state.setdefault("scratch", {})["ace_bullets"] = [b.to_dict() for b in relevant_bullets]

    if any(w in normalized_text for w in ["chapter", "unit", "textbook", "quiz", "user", "session", "group", "message", "database", "graph"]):
        state["mode"] = "react"
    elif any(w in normalized_text for w in ["calculate", "sum", "difference", "product", "ratio", "percent", "%", "number", "verify", "double check", "make sure", "confirm", "check if"]):
        state["mode"] = "react"
    elif any(w in normalized_text for w in ["search", "web", "internet", "current", "latest", "trending", "news", "online", "look up", "find out", "recent", "today"]):
        state["mode"] = "react"
    elif any(w in normalized_text for w in ["plan", "options", "steps", "strategy", "search space"]):
        state["mode"] = "tot"
    else:
        state["mode"] = "cot"

    return state


def planner_node(state: GraphState) -> GraphState:
    mode = state["mode"]
    scratch = dict(state.get("scratch", {}))

    if mode == "tot":
        scratch.setdefault("breadth", 3)
        scratch.setdefault("depth", 2)
        scratch.setdefault("temperature", 0.2)
    elif mode == "react":
        scratch.setdefault("max_turns", 8)
        tool_schemas = [_calculator_schema(), _google_search_schema(), _neo4j_retrieveqa_schema()]
        scratch["tool_names"] = [t["function"]["name"] for t in tool_schemas]
        scratch.setdefault("temperature", 0.2)
    elif mode == "cot":
        scratch.setdefault("k", 1)
        scratch.setdefault("temperature", 0.2 if scratch["k"] == 1 else 0.7)

    scratch["use_ace_context"] = True
    state["scratch"] = scratch
    return state


def solver_node_with_ace(state: GraphState) -> GraphState:
    """
    Solver with ACE memory enrichment.
    Enhancement 2: Post-context injection (controlled by ACE_INJECTION_MODE).
    """
    mode = state["mode"]
    scratch = state.get("scratch", {})
    user_messages = [m.get("content", "") for m in state.get("messages", []) if m.get("role") == "user"]
    question = user_messages[-1] if user_messages else ""
    full_question = " ".join(user_messages)
    learner_id = scratch.get("learner_id")
    context_scope_id = scratch.get("context_scope_id")
    topic = scratch.get("topic") or _infer_topic(question)
    if topic and scratch.get("topic") != topic:
        scratch["topic"] = topic
        state["scratch"] = scratch

    memory, pipeline = get_ace_system(learner_id, context_scope_id=context_scope_id)
    facets = scratch.get("ace_retrieval_facets") or _extract_retrieval_facets(question, scratch)
    scratch["ace_retrieval_facets"] = facets
    scratch["ace_full_question"] = full_question
    scratch["ace_latest_question"] = question
    state["scratch"] = scratch

    if scratch.get("use_ace_context", True):
        retrieved_bullets = memory.retrieve_relevant_bullets(
            question,
            top_k=10,
            learner_id=learner_id,
            topic=topic,
            facets=facets,
            context_scope_id=context_scope_id,
        )
        if retrieved_bullets:
            context_parts = ["=== Relevant Strategies and Lessons ==="]
            for idx, bullet in enumerate(retrieved_bullets, 1):
                suffix = []
                if bullet.memory_type:
                    suffix.append(f"type={bullet.memory_type}")
                if bullet.topic:
                    suffix.append(f"topic={bullet.topic}")
                annotation = f" ({', '.join(suffix)})" if suffix else ""
                context_parts.append(f"{idx}. {bullet.format_for_prompt()}{annotation}")
            context_parts.append("=" * 50)
            context = "\n".join(context_parts)

            messages = state.get("messages", [])
            injection_mode = os.getenv("ACE_INJECTION_MODE", "post_context")

            if injection_mode == "post_context":
                last_user_idx = None
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        last_user_idx = i
                        break
                if last_user_idx is not None:
                    messages[last_user_idx]["content"] = f"{context}\n\n{messages[last_user_idx]['content']}"
            else:
                context_injected = False
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        messages[i]["content"] = f"{context}\n\n{msg['content']}"
                        context_injected = True
                        break
                if not context_injected and messages:
                    if messages[0].get("role") == "user":
                        messages[0]["content"] = f"{context}\n\n{messages[0]['content']}"

            state["messages"] = messages

    if mode == "cot":
        result = solve_cot(state)
    elif mode == "tot":
        result = solve_tot(state)
    elif mode == "react":
        result = solve_react(state)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    state["result"] = result
    return state


def critic_node(state: GraphState) -> GraphState:
    res = dict(state.get("result", {}))
    ans = str(res.get("answer", "")).strip()
    m = re.search(r"Answer\s*[:\-]\s*(.*)$", ans, flags=re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
    inner = _extract_final(ans)
    if inner:
        ans = inner
    res["answer"] = ans
    state["result"] = res
    return state


def ace_learning_node(state: GraphState) -> GraphState:
    """
    ACE Learning Node: Applies the ACE pipeline to learn from execution.
    """
    try:
        messages = state.get("messages", [])
        result = state.get("result", {})
        mode = state.get("mode", "unknown")
        scratch = state.get("scratch", {})
        learner_id = scratch.get("learner_id")
        context_scope_id = scratch.get("context_scope_id")

        user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
        question = scratch.get("ace_full_question") or " ".join(user_messages).strip()
        model_answer = result.get("answer", "")
        success = bool(model_answer and model_answer != "(no final)")
        ground_truth = scratch.get("ground_truth")

        trace = ExecutionTrace(
            question=question,
            model_answer=model_answer,
            ground_truth=ground_truth,
            success=success,
            trace_messages=result.get("trace", messages),
            metadata={"mode": mode, "scratch": scratch},
        )

        _, pipeline = get_ace_system(learner_id, context_scope_id=context_scope_id)
        apply_update = scratch.get("ace_online_learning", True)

        rubric_feedback = scratch.get("rubric_feedback")
        delta = pipeline.process_execution(trace, apply_update=apply_update, rubric_feedback=rubric_feedback)

        if delta:
            quality_gate = {}
            if isinstance(delta.metadata, dict):
                quality_gate = delta.metadata.get("quality_gate", {})
            state.setdefault("scratch", {})["ace_delta"] = {
                "num_new_bullets": len(delta.new_bullets),
                "num_updates": len(delta.update_bullets),
                "num_removals": len(delta.remove_bullets),
                "quality_gate": quality_gate,
            }

    except Exception as e:
        import traceback
        traceback.print_exc()

    return state


def build_ace_graph() -> any:
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("planner", planner_node)
    graph.add_node("solver", solver_node_with_ace)
    graph.add_node("critic", critic_node)
    graph.add_node("ace_learning", ace_learning_node)

    graph.add_edge(START, "router")
    graph.add_edge("router", "planner")
    graph.add_edge("planner", "solver")
    graph.add_edge("solver", "critic")
    graph.add_edge("critic", "ace_learning")
    graph.add_edge("ace_learning", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    return app
