"""
Runtime entrypoint for the ACE-enabled LangGraph agent.

Reads a JSON payload from stdin, invokes the agent, writes JSON response to stdout.
"""

from __future__ import annotations

import json
import sys
import time
import io
import re
from contextlib import redirect_stdout
from typing import Any, Optional
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
load_dotenv()

from src.agent import build_ace_graph
from src.solvers import _extract_final


def _clean_answer(answer: Optional[str]) -> Optional[str]:
    if not answer:
        return answer
    final = _extract_final(answer)
    if final:
        return final.strip()
    cleaned = re.sub(r"<scratchpad>.*?</scratchpad>", "", answer, flags=re.DOTALL)
    stripped_lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if stripped_lines:
        return stripped_lines[-1]
    return cleaned.strip() or answer


def _sanitize_answer(answer: Optional[str]) -> Optional[str]:
    if not answer:
        return answer
    lines = answer.splitlines()
    filtered = [ln for ln in lines if not re.fullmatch(r"\s*\d+\s*", ln)]
    if filtered:
        return "\n".join(filtered).strip()
    return answer.strip()


def _log(message: str) -> None:
    sys.stderr.write(f"[ACE Runner] {message}\n")
    sys.stderr.flush()


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("No input received on stdin")
    payload = json.loads(raw)
    _log("Received payload")

    messages = payload.get("messages") or []
    mode = payload.get("mode") or ""
    scratch = payload.get("scratch") or {}
    scratch.setdefault("ace_online_learning", True)

    thread_id = payload.get("thread_id") or f"ace-thread-{int(time.time() * 1000)}"

    app = build_ace_graph()
    config = {"configurable": {"thread_id": thread_id}}

    state = {
        "messages": messages,
        "mode": mode,
        "scratch": scratch,
        "result": {},
    }

    _log("Invoking LangGraph application")
    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        output = app.invoke(state, config=config)

    response = {
        "answer": _sanitize_answer(_clean_answer(output.get("result", {}).get("answer"))),
        "mode": output.get("mode"),
        "result": output.get("result", {}),
        "scratch": output.get("scratch", {}),
    }

    json.dump(response, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        error_payload = {"error": str(exc)}
        json.dump(error_payload, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise SystemExit(1)
