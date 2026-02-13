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
import os
import re

from src.solvers import GraphState, solve_cot, solve_tot, solve_react, _extract_final
from src.ace_memory import ACEMemory, Bullet
from src.ace_components import ACEPipeline, ExecutionTrace
from src.storage import Neo4jMemoryStore
from src.llm import LLM
from src.step_scoring import StepScoringConfig
from src.reasoning_loop import ReasoningLoopConfig, run_recursive_text_refinement
from src.planner_policy import PlannerPolicy, compute_shaped_reward, default_planner_state_path
from src.tools import _calculator_schema, _google_search_schema, _neo4j_retrieveqa_schema


_ACE_CACHE: Dict[str, Dict[str, Any]] = {}
_PLANNER_POLICY: Optional[PlannerPolicy] = None

PLANNER_ACTIONS: Dict[str, Dict[str, Any]] = {
    "cot_fast": {
        "mode": "cot",
        "scratch": {"k": 1, "temperature": 0.2},
    },
    "cot_deliberate": {
        "mode": "cot",
        "scratch": {"k": 3, "temperature": 0.7},
    },
    "tot_balanced": {
        "mode": "tot",
        "scratch": {"breadth": 3, "depth": 2, "temperature": 0.2},
    },
    "tot_deep": {
        "mode": "tot",
        "scratch": {"breadth": 4, "depth": 3, "temperature": 0.2},
    },
    "react_tool": {
        "mode": "react",
        "scratch": {"max_turns": 8, "temperature": 0.2},
    },
}

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


def _safe_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _extract_step_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    step_summary = result.get("step_scoring")
    if isinstance(step_summary, dict):
        return step_summary
    sample_summaries = result.get("sample_step_scoring")
    selected_index = int(result.get("selected_sample_index", 0) or 0)
    if isinstance(sample_summaries, list) and sample_summaries:
        if 0 <= selected_index < len(sample_summaries):
            selected = sample_summaries[selected_index]
            if isinstance(selected, dict):
                return selected
        first = sample_summaries[0]
        if isinstance(first, dict):
            return first
    return {}


def get_planner_policy() -> PlannerPolicy:
    global _PLANNER_POLICY
    if _PLANNER_POLICY is None:
        state_path = os.getenv("ACE_PLANNER_STATE_PATH")
        if not state_path:
            state_path = default_planner_state_path(role = "runtime")
        _PLANNER_POLICY = PlannerPolicy(
            actions = list(PLANNER_ACTIONS.keys()),
            state_path = state_path,
            epsilon = _safe_env_float("ACE_PLANNER_EPSILON", 0.08),
            ucb_c = _safe_env_float("ACE_PLANNER_UCB_C", 1.10),
            seed = _safe_env_int("ACE_PLANNER_SEED", 42),
        )
    return _PLANNER_POLICY


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
    scratch = dict(state.get("scratch", {}))
    messages = state.get("messages", [])
    user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
    user_text = user_messages[-1] if user_messages else ""
    features = {
        "question_length": len(user_text),
        "num_messages": len(messages),
        "has_tools_hint": int(any(token in user_text.lower() for token in ["search", "web", "database", "calculate"])),
    }

    planner_policy = get_planner_policy()
    choice = planner_policy.choose_action(features = features)
    action_id = choice.get("action_id", "cot_fast")
    action_config = PLANNER_ACTIONS.get(action_id, PLANNER_ACTIONS["cot_fast"])
    mode = action_config.get("mode", "cot")
    state["mode"] = mode

    planned_scratch = action_config.get("scratch", {})
    if isinstance(planned_scratch, dict):
        for key, value in planned_scratch.items():
            scratch[key] = value

    if mode == "react":
        tool_schemas = [_calculator_schema(), _google_search_schema(), _neo4j_retrieveqa_schema()]
        scratch["tool_names"] = [tool["function"]["name"] for tool in tool_schemas]

    scratch["planner_action_id"] = action_id
    scratch["planner_explore"] = bool(choice.get("explore", False))
    scratch["planner_scores"] = choice.get("scores", {})
    scratch["planner_features"] = features
    scratch["planner_policy_type"] = "bandit"
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

    recursion_cfg = ReasoningLoopConfig.from_env()
    step_cfg = StepScoringConfig.from_state(scratch)
    if recursion_cfg.max_rounds > 1 and mode in {"cot", "tot"}:
        initial_answer = str(result.get("answer", "")).strip()
        if initial_answer:
            llm = LLM(temperature = 0.2)

            def _runtime_generate_candidate(prompt: str) -> str:
                response = llm.chat(
                    [
                        {"role": "system", "content": "You are a precise reasoner. Return only the final answer."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature = 0.2,
                    max_tokens = 1200,
                )
                text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                return str(text or "").strip()

            refinement = run_recursive_text_refinement(
                question = full_question or question,
                initial_answer = initial_answer,
                generate_candidate = _runtime_generate_candidate,
                step_config = step_cfg,
                loop_config = recursion_cfg,
            )
            improved_answer = str(refinement.get("answer", "")).strip()
            if improved_answer:
                result["answer"] = improved_answer
                result["step_scoring"] = refinement.get("step_summary", {})
                result["recursion"] = refinement.get("recursion", {})

    step_summary = _extract_step_summary(result)
    if step_summary:
        scratch["latest_step_scoring"] = step_summary
        state["scratch"] = scratch
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
            metadata={
                "mode": mode,
                "scratch": scratch,
                "step_scoring": _extract_step_summary(result),
            },
        )

        _, pipeline = get_ace_system(learner_id, context_scope_id=context_scope_id)
        apply_update = scratch.get("ace_online_learning", True)

        rubric_feedback = scratch.get("rubric_feedback")
        delta = pipeline.process_execution(trace, apply_update=apply_update, rubric_feedback=rubric_feedback)

        quality_gate = {}
        if delta:
            if isinstance(delta.metadata, dict):
                quality_gate = delta.metadata.get("quality_gate", {})
            state.setdefault("scratch", {})["ace_delta"] = {
                "num_new_bullets": len(delta.new_bullets),
                "num_updates": len(delta.update_bullets),
                "num_removals": len(delta.remove_bullets),
                "quality_gate": quality_gate,
            }

        planner_action_id = str(scratch.get("planner_action_id", "")).strip()
        if planner_action_id:
            step_summary = _extract_step_summary(result)
            step_score = float(step_summary.get("mean_step_score", 0.0) or 0.0)
            step_confidence = float(step_summary.get("overall_confidence", 0.7) or 0.7)
            recursion_diag = result.get("recursion", {})
            recursion_improved = bool(
                isinstance(recursion_diag, dict) and recursion_diag.get("improved", False)
            )
            gate_applied = bool(isinstance(quality_gate, dict) and quality_gate.get("should_apply_update", False))
            reward_payload = compute_shaped_reward(
                step_score = step_score,
                output_valid = success,
                quality_gate_applied = gate_applied,
                recursion_improved = recursion_improved,
                step_confidence = step_confidence,
            )
            planner_policy = get_planner_policy()
            update_result = planner_policy.update(
                action_id = planner_action_id,
                reward = reward_payload.get("final_reward", reward_payload.get("proxy_reward", 0.0)),
                confidence = reward_payload.get("confidence", 0.7),
                metadata = {
                    "mode": mode,
                    "success": success,
                    "quality_gate_applied": gate_applied,
                },
            )
            state.setdefault("scratch", {})["planner_reward"] = {
                "action_id": planner_action_id,
                "proxy_reward": reward_payload.get("proxy_reward", 0.0),
                "final_reward": reward_payload.get("final_reward", 0.0),
                "confidence": reward_payload.get("confidence", 0.0),
                "updated": bool(update_result.get("updated", False)),
                "action_mean_reward": float(update_result.get("action_mean_reward", 0.0)),
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
