"""
Solver functions for the LangGraph agent.

Provides: GraphState, ConversationMemory, solve_cot, solve_tot, solve_react,
_extract_final, _finalize_answer
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict
import re
import json
import time
import os
from datetime import datetime
from pathlib import Path
from collections import Counter

from src.llm import LLM
from src.tools import (
    _calculator_run, _calculator_schema,
    _google_search_run, _google_search_schema,
    _neo4j_retrieveqa_run, _neo4j_retrieveqa_schema,
    _deep_research_run, _deep_research_schema,
)
from src.prompts.reasoning_prompts import COT_PROMPT, TOT_EXPAND_TEMPLATE, TOT_VALUE_TEMPLATE, REACT_SYSTEM


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
    llm = LLM(temperature=temp)
    base_msgs = [{"role": "system", "content": COT_PROMPT}] + state["messages"]
    if k == 1:
        resp = llm.chat(base_msgs)
        text = resp["choices"][0]["message"]["content"]
        cleaned = re.sub(r"<scratchpad>.*?</scratchpad>", "", text, flags=re.DOTALL)
        return {"answer": _finalize_answer(cleaned), "raw": text}
    answers: List[str] = []
    raws: List[str] = []
    for _ in range(k):
        resp = llm.chat(base_msgs, temperature=temp)
        text = resp["choices"][0]["message"]["content"]
        cleaned = re.sub(r"<scratchpad>.*?</scratchpad>", "", text, flags=re.DOTALL)
        raws.append(text)
        answers.append(_finalize_answer(cleaned))

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    tally = Counter(norm(a) for a in answers)
    best_norm, _ = tally.most_common(1)[0]
    chosen = next(a for a in answers if norm(a) == best_norm)
    return {"answer": chosen, "raw_samples": raws}


def solve_tot(state: GraphState) -> Dict[str, Any]:
    params = state["scratch"]
    breadth = int(params.get("breadth", 3))
    depth = int(params.get("depth", 2))
    temp = float(params.get("temperature", 0.2))
    llm = LLM(temperature=temp)
    user = next((m["content"] for m in state["messages"] if m["role"] == "user"), "")
    beam: List[Tuple[str, float]] = [("", 0.0)]
    for d in range(depth):
        candidates: List[Tuple[str, float]] = []
        for scratchpad, _ in beam:
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
                candidates.append((new_pad, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:breadth] if candidates else beam
    best_pad = beam[0][0]
    final_msgs = [
        {"role": "system", "content": COT_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": f"<scratchpad>{best_pad}</scratchpad>\nNow conclude."},
    ]
    fin = llm.chat(final_msgs, temperature=0.0)
    text = fin["choices"][0]["message"]["content"]
    text = re.sub(r"<scratchpad>.*?</scratchpad>", "", text, flags=re.DOTALL)
    return {"answer": _finalize_answer(text), "scratchpad": best_pad}


def solve_react(state: GraphState) -> Dict[str, Any]:
    params = state["scratch"]
    max_turns = int(params.get("max_turns", 8))
    temp = float(params.get("temperature", 0.2))
    tool_schemas = [_calculator_schema(), _google_search_schema(), _neo4j_retrieveqa_schema()]
    llm = LLM(temperature=temp)
    messages = [{"role": "system", "content": REACT_SYSTEM}] + state["messages"]

    content = ""
    for turn in range(max_turns):
        max_tokens = 1200 if turn > 0 else 800
        resp = llm.chat(messages, tools=tool_schemas, tool_choice="auto", max_tokens=max_tokens)
        choice = resp["choices"][0]["message"]
        content = choice.get("content") or ""
        tool_calls = choice.get("tool_calls") or []

        if content:
            final = _extract_final(content)
            if final:
                messages.append({"role": "assistant", "content": content})
                return {"answer": final, "trace": messages}

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

    return {"answer": _finalize_answer(content) if content else "(no final)", "trace": messages}
