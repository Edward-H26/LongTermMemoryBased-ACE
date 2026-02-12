"""
Solver functions for the LangGraph agent.

Provides: GraphState, ConversationMemory, solve_cot, solve_tot, solve_react,
_extract_final, _finalize_answer
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict
import re
import json
import time
import os
from datetime import datetime
from pathlib import Path

from src.llm import LLM
from src.tools import (
    _calculator_run, _calculator_schema,
    _google_search_run, _google_search_schema,
    _neo4j_retrieveqa_run, _neo4j_retrieveqa_schema,
    _deep_research_run, _deep_research_schema,
)
from src.prompts.reasoning_prompts import COT_PROMPT, TOT_EXPAND_TEMPLATE, TOT_VALUE_TEMPLATE, REACT_SYSTEM
from src.step_scoring import StepScoringConfig, score_reasoning_candidates, score_reasoning_text


class GraphState(TypedDict):
    messages: List[Dict[str, Any]]
    mode: Literal["cot", "tot", "react"]
    scratch: Dict[str, Any]
    result: Dict[str, Any]


class ConversationMemory:
    """Simple persistent memory to track conversation history."""

    def __init__(self, memory_file: str = "conversation_memory.json"):
        self.memory_file = Path(memory_file)
        self.conversations: List[Dict[str, Any]] = []
        self._load_memory()

    def _load_memory(self):
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    self.conversations = json.load(f)
            except Exception:
                self.conversations = []
        else:
            self.conversations = []

    def _save_memory(self):
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.conversations, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def add_conversation(self, question: str, answer: str, mode: str, metadata: Optional[Dict[str, Any]] = None):
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "mode": mode,
            "metadata": metadata or {},
        }
        self.conversations.append(conversation)
        self._save_memory()

    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.conversations[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        if not self.conversations:
            return {"total_conversations": 0, "mode_distribution": {}}
        mode_counter = Counter(conv.get("mode", "unknown") for conv in self.conversations)
        return {
            "total_conversations": len(self.conversations),
            "mode_distribution": dict(mode_counter),
        }


def _extract_final(text: str) -> Optional[str]:
    m = re.search(r"<final>(.*?)</final>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def _finalize_answer(text: str) -> str:
    final = _extract_final(text)
    if final:
        return final.strip()
    cleaned = re.sub(r"<scratchpad>.*?</scratchpad>", "", text, flags=re.DOTALL)
    stripped = cleaned.strip()
    if not stripped:
        return ""
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    return lines[-1] if lines else stripped


def solve_cot(state: GraphState) -> Dict[str, Any]:
    params = state["scratch"]
    k = int(params.get("k", 1))
    temp = float(params.get("temperature", 0.2 if k == 1 else 0.7))
    step_config = StepScoringConfig.from_state(params)
    llm = LLM(temperature=temp)
    base_msgs = [{"role": "system", "content": COT_PROMPT}] + state["messages"]
    user = next((m["content"] for m in state["messages"] if m["role"] == "user"), "")
    if k == 1:
        resp = llm.chat(base_msgs)
        text = resp["choices"][0]["message"]["content"]
        cleaned = re.sub(r"<scratchpad>.*?</scratchpad>", "", text, flags=re.DOTALL)
        step_summary = score_reasoning_text(
            question = user,
            reasoning_text = text,
            config = step_config,
        )
        return {
            "answer": _finalize_answer(cleaned),
            "raw": text,
            "step_scoring": step_summary,
        }
    answers: List[str] = []
    raws: List[str] = []
    for _ in range(k):
        resp = llm.chat(base_msgs, temperature=temp)
        text = resp["choices"][0]["message"]["content"]
        cleaned = re.sub(r"<scratchpad>.*?</scratchpad>", "", text, flags=re.DOTALL)
        raws.append(text)
        answers.append(_finalize_answer(cleaned))

    sample_scores = score_reasoning_candidates(
        question = user,
        candidate_texts = raws,
        config = step_config,
    )
    best_idx = 0
    best_score = -1.0
    for idx, summary in enumerate(sample_scores):
        score = float(summary.get("mean_step_score", 0.0))
        if score > best_score:
            best_score = score
            best_idx = idx

    return {
        "answer": answers[best_idx],
        "raw_samples": raws,
        "sample_step_scoring": sample_scores,
        "selected_sample_index": best_idx,
    }


def solve_tot(state: GraphState) -> Dict[str, Any]:
    params = state["scratch"]
    breadth = int(params.get("breadth", 3))
    depth = int(params.get("depth", 2))
    temp = float(params.get("temperature", 0.2))
    step_config = StepScoringConfig.from_state(params)
    llm = LLM(temperature=temp)
    user = next((m["content"] for m in state["messages"] if m["role"] == "user"), "")
    beam: List[Dict[str, Any]] = [{
        "pad": "",
        "value_score": 0.0,
        "normalized_value_score": 0.0,
        "step_mean": 0.0,
        "combined_rank": 0.0,
    }]
    depth_diagnostics: List[Dict[str, Any]] = []
    for d in range(depth):
        candidates: List[Dict[str, Any]] = []
        for beam_item in beam:
            scratchpad = str(beam_item.get("pad", ""))
            sys = "You are exploring solution trees. Expand concise next-steps. Do not jump to the final answer yet."
            expand_prompt = TOT_EXPAND_TEMPLATE.format(k=breadth)
            msgs = [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
                {"role": "assistant", "content": f"<partial>{scratchpad}</partial>\n{expand_prompt}"},
            ]
            exp = llm.chat(msgs, temperature=max(0.7, temp))
            text = exp["choices"][0]["message"]["content"] or ""
            next_thoughts: List[str] = []
            try:
                j = json.loads(text)
                if isinstance(j, dict) and isinstance(j.get("thoughts"), list):
                    next_thoughts = [str(t) for t in j["thoughts"]][:breadth]
            except Exception:
                pass
            if not next_thoughts:
                next_thoughts = [
                    ln.strip(" -\t")
                    for ln in text.splitlines()
                    if ln.strip().startswith(("-", "1.", "2.", "3."))
                ][:breadth]
                if not next_thoughts:
                    next_thoughts = [text.strip().split("\n")[0]]
            for thought in next_thoughts:
                new_pad = (scratchpad + "\n" if scratchpad else "") + f"Thought: {thought}"
                val_msgs = [
                    {"role": "system", "content": "You are a strict evaluator."},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": f"<partial>{new_pad}</partial>\n{TOT_VALUE_TEMPLATE}"},
                ]
                val = llm.chat(val_msgs, temperature=0.0)
                score_text = val["choices"][0]["message"]["content"] or "5"
                try:
                    score = float(re.findall(r"-?\d+(?:\.\d+)?", score_text)[0])
                except Exception:
                    score = 5.0
                candidates.append({
                    "pad": new_pad,
                    "value_score": score,
                })

        if not candidates:
            continue

        raw_values = [float(c.get("value_score", 0.0)) for c in candidates]
        min_value = min(raw_values)
        max_value = max(raw_values)
        if max_value > min_value:
            normalized_values = [
                (value - min_value) / (max_value - min_value)
                for value in raw_values
            ]
        else:
            normalized_values = [0.5 for _ in raw_values]

        step_summaries = score_reasoning_candidates(
            question = user,
            candidate_texts = [str(c.get("pad", "")) for c in candidates],
            config = step_config,
        )

        enriched: List[Dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            step_summary = step_summaries[idx] if idx < len(step_summaries) else {}
            step_mean = float(step_summary.get("mean_step_score", 0.0))
            normalized_value_score = normalized_values[idx]
            combined_rank = 0.6 * normalized_value_score + 0.4 * step_mean
            enriched.append({
                "pad": candidate.get("pad", ""),
                "value_score": float(candidate.get("value_score", 0.0)),
                "normalized_value_score": normalized_value_score,
                "step_mean": step_mean,
                "combined_rank": combined_rank,
                "step_scoring": step_summary,
            })

        eligible = [row for row in enriched if row.get("step_mean", 0.0) >= step_config.min_score]
        if not eligible:
            eligible = enriched

        eligible.sort(key = lambda row: row.get("combined_rank", 0.0), reverse = True)
        beam = eligible[:breadth]
        depth_diagnostics.append({
            "depth": d + 1,
            "num_candidates": len(candidates),
            "num_eligible_after_prune": len(eligible),
            "top_combined_rank": beam[0].get("combined_rank", 0.0) if beam else 0.0,
        })

    best_pad = str(beam[0].get("pad", "")) if beam else ""
    final_msgs = [
        {"role": "system", "content": COT_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": f"<scratchpad>{best_pad}</scratchpad>\nNow conclude."},
    ]
    fin = llm.chat(final_msgs, temperature=0.0)
    text = fin["choices"][0]["message"]["content"]
    text = re.sub(r"<scratchpad>.*?</scratchpad>", "", text, flags=re.DOTALL)
    final_step_summary = score_reasoning_text(
        question = user,
        reasoning_text = text,
        config = step_config,
    )
    return {
        "answer": _finalize_answer(text),
        "scratchpad": best_pad,
        "tot_scoring": depth_diagnostics,
        "step_scoring": final_step_summary,
    }


def solve_react(state: GraphState) -> Dict[str, Any]:
    params = state["scratch"]
    max_turns = int(params.get("max_turns", 8))
    temp = float(params.get("temperature", 0.2))
    step_config = StepScoringConfig.from_state(params)
    tool_schemas = [_calculator_schema(), _google_search_schema(), _neo4j_retrieveqa_schema()]
    llm = LLM(temperature=temp)
    messages = [{"role": "system", "content": REACT_SYSTEM}] + state["messages"]
    user = next((m["content"] for m in state["messages"] if m["role"] == "user"), "")

    content = ""
    step_scores: List[Dict[str, Any]] = []
    for turn in range(max_turns):
        max_tokens = 1200 if turn > 0 else 800
        resp = llm.chat(messages, tools=tool_schemas, tool_choice="auto", max_tokens=max_tokens)
        choice = resp["choices"][0]["message"]
        content = choice.get("content") or ""
        tool_calls = choice.get("tool_calls") or []

        if content.strip():
            assistant_summary = score_reasoning_text(
                question = user,
                reasoning_text = content,
                config = step_config,
            )
            step_scores.append({
                "turn": turn + 1,
                "kind": "assistant",
                "summary": assistant_summary,
            })

        if content:
            final = _extract_final(content)
            if final:
                messages.append({"role": "assistant", "content": content})
                return {"answer": final, "trace": messages, "step_scores": step_scores}

        assistant_msg = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if tool_calls:
            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"].get("arguments")
                try:
                    parsed = json.loads(args) if isinstance(args, str) else (args or {})
                except Exception:
                    parsed = {}

                if name == "calculator":
                    out = _calculator_run(parsed)
                elif name == "google_search":
                    out = _google_search_run(parsed)
                elif name == "deep_research":
                    out = _deep_research_run(parsed)
                elif name == "neo4j_retrieveqa":
                    out = _neo4j_retrieveqa_run(parsed)
                else:
                    out = f"Unknown tool: {name}"

                tool_summary = score_reasoning_text(
                    question = user,
                    reasoning_text = f"Tool {name} output: {str(out)[:1200]}",
                    config = step_config,
                )
                step_scores.append({
                    "turn": turn + 1,
                    "kind": "tool",
                    "tool_name": name,
                    "summary": tool_summary,
                })

                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": out if isinstance(out, str) else json.dumps(out),
                    "tool_call_id": tc.get("id", ""),
                })

            if turn >= max_turns - 2:
                messages.append({
                    "role": "user",
                    "content": "Based on the tool results above, please provide your final answer wrapped in <final></final> tags.",
                })

    return {
        "answer": _finalize_answer(content) if content else "(no final)",
        "trace": messages,
        "step_scores": step_scores,
    }
