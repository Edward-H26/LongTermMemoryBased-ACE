"""
Multi-backend LLM wrapper with token and latency tracking.

Supports:
- GeminiBackend: Google Gemini REST API (for ACE Reflector)
- OpenAIBackend: OpenAI SDK (for GPT-5.1 solver, HuggingFace endpoints)

Backend selected via LLM_BACKEND env var. Every call records metrics
to the global MetricsCollector.
"""

from __future__ import annotations

import json
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RequestMetrics:
    task_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    backend: str = ""
    model: str = ""
    num_llm_calls: int = 1


class MetricsCollector:
    """Thread-safe aggregator for LLM call metrics."""

    def __init__(self):
        self._records: List[RequestMetrics] = []
        self._lock = threading.Lock()

    def record(self, m: RequestMetrics):
        with self._lock:
            self._records.append(m)

    @property
    def records(self) -> List[RequestMetrics]:
        with self._lock:
            return list(self._records)

    def summary(self) -> Dict[str, Any]:
        recs = self.records
        if not recs:
            return {"total_calls": 0}
        total_prompt = sum(r.prompt_tokens for r in recs)
        total_completion = sum(r.completion_tokens for r in recs)
        latencies = [r.latency_ms for r in recs]
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        return {
            "total_calls": len(recs),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "median_latency_ms": sorted_lat[len(sorted_lat) // 2],
            "p95_latency_ms": sorted_lat[min(p95_idx, len(sorted_lat) - 1)],
        }

    def export_json(self, path: str):
        import json as _json
        data = {
            "summary": self.summary(),
            "records": [
                {
                    "task_id": r.task_id,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "latency_ms": r.latency_ms,
                    "backend": r.backend,
                    "model": r.model,
                }
                for r in self.records
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(data, f, indent=2, ensure_ascii=False)

    def reset(self):
        with self._lock:
            self._records.clear()


_GLOBAL_COLLECTOR = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    return _GLOBAL_COLLECTOR


class GeminiBackend:
    """Google Gemini via REST API."""

    def __init__(self, model: str = "gemini-3-flash-preview", temperature: float = 0.2):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not configured.")
        self.model = model
        self.temperature = temperature
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    @staticmethod
    def _text_parts(content: str) -> List[Dict[str, Any]]:
        return [{"text": str(content)}]

    @staticmethod
    def _tool_response_part(name: str, raw_content: Any) -> Dict[str, Any]:
        if isinstance(raw_content, str):
            try:
                parsed = json.loads(raw_content)
            except json.JSONDecodeError:
                parsed = {"output": raw_content}
        elif isinstance(raw_content, dict):
            parsed = raw_content
        else:
            parsed = {"output": raw_content}
        return {"functionResponse": {"name": name or "tool", "response": parsed}}

    @staticmethod
    def _convert_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        declarations: List[Dict[str, Any]] = []
        for tool in tools:
            fn = tool.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            declarations.append({
                "name": name,
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        if not declarations:
            return None
        return [{"function_declarations": declarations}]

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        system_instruction = None
        contents: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                if content and system_instruction is None:
                    system_instruction = {"parts": [{"text": str(content)}]}
                    continue
                role = "user"

            if role == "assistant":
                contents.append({"role": "model", "parts": self._text_parts(content)})
            elif role == "tool":
                name = msg.get("name") or msg.get("tool_name") or msg.get("tool_call_id") or "tool"
                part = self._tool_response_part(name, content)
                contents.append({"role": "user", "parts": [part]})
            else:
                contents.append({"role": "user", "parts": self._text_parts(content)})

        return system_instruction, contents

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1000,
        retry: int = 3,
        response_mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for _ in range(retry):
            try:
                system_instruction, contents = self._convert_messages(messages)
                body: Dict[str, Any] = {
                    "contents": contents,
                    "generationConfig": {
                        "temperature": temperature if temperature is not None else self.temperature,
                        "maxOutputTokens": max_tokens,
                    },
                }
                if response_mime_type:
                    body.setdefault("generationConfig", {})["responseMimeType"] = response_mime_type
                tools_spec = self._convert_tools(tools)
                if tools_spec:
                    body["tools"] = tools_spec
                    if tool_choice == "auto":
                        body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
                if system_instruction:
                    body["systemInstruction"] = system_instruction

                resp = requests.post(
                    self.endpoint, params={"key": self.api_key}, json=body, timeout=60,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Gemini API error: {resp.status_code} {resp.text}")

                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"Gemini API error: {data['error']}")

                candidates = data.get("candidates") or []
                if not candidates:
                    raise RuntimeError("Gemini API returned no candidates.")

                candidate = candidates[0]
                finish_reason = candidate.get("finishReason")
                if finish_reason and finish_reason not in {"STOP", "MAX_TOKENS"}:
                    raise RuntimeError(f"Generation halted by Gemini (reason={finish_reason}).")

                content_obj = candidate.get("content") or {}
                parts = content_obj.get("parts") or []
                text_chunks: List[str] = []
                tool_calls: List[Dict[str, Any]] = []

                for part in parts:
                    if "text" in part:
                        text_chunks.append(part["text"])
                    if "functionCall" in part:
                        fc = part["functionCall"] or {}
                        name = fc.get("name") or "tool"
                        args = fc.get("args") or {}
                        tool_calls.append({
                            "id": name,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        })

                message: Dict[str, Any] = {"content": "\n".join(text_chunks).strip()}
                if tool_calls:
                    message["tool_calls"] = tool_calls

                usage = data.get("usageMetadata") or {}
                prompt_tokens = usage.get("promptTokenCount", 0)
                completion_tokens = usage.get("candidatesTokenCount", 0)

                return {
                    "choices": [{"message": message}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            except Exception as exc:
                last_err = exc
                time.sleep(0.5)

        raise RuntimeError(f"Gemini call failed after retries: {last_err}")


class OpenAIBackend:
    """OpenAI SDK client for GPT-5.1, HuggingFace endpoints, or any compatible API."""

    def __init__(self, model: str = "gpt-5.1", temperature: float = 0.2):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured.")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1000,
        retry: int = 3,
        response_mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        clean_messages = []
        for msg in messages:
            clean_msg = {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            if msg.get("tool_calls"):
                clean_msg["tool_calls"] = msg["tool_calls"]
            if msg.get("role") == "tool":
                clean_msg["tool_call_id"] = msg.get("tool_call_id", "")
            clean_messages.append(clean_msg)

        last_err: Optional[Exception] = None
        for _ in range(retry):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": clean_messages,
                    "temperature": temperature if temperature is not None else self.temperature,
                    "max_completion_tokens": max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                    if tool_choice:
                        kwargs["tool_choice"] = tool_choice

                response = self._client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                message_dict: Dict[str, Any] = {"content": choice.message.content or ""}
                if choice.message.tool_calls:
                    message_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in choice.message.tool_calls
                    ]

                usage = response.usage
                return {
                    "choices": [{"message": message_dict}],
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                    },
                }
            except Exception as exc:
                last_err = exc
                time.sleep(0.5)

        raise RuntimeError(f"OpenAI call failed after retries: {last_err}")


class LLM:
    """
    Unified LLM wrapper. Delegates to configured backend and records
    token/latency metrics to the global MetricsCollector.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        backend: Optional[str] = None,
    ):
        backend = backend or os.getenv("LLM_BACKEND", "gemini")
        if backend == "openai":
            model = model or os.getenv("OPENAI_MODEL", "gpt-5.1")
            self._backend = OpenAIBackend(model=model, temperature=temperature)
        else:
            model = model or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
            self._backend = GeminiBackend(model=model, temperature=temperature)
        self.model = model
        self.temperature = temperature
        self._backend_name = backend
        self._collector = get_metrics_collector()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1000,
        retry: int = 3,
        response_mime_type: Optional[str] = None,
        task_id: str = "",
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        result = self._backend.chat(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            retry=retry,
            response_mime_type=response_mime_type,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        usage = result.get("usage", {})
        self._collector.record(RequestMetrics(
            task_id=task_id,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=elapsed_ms,
            backend=self._backend_name,
            model=self.model,
        ))
        return result
