"""OpenAI chat-completion wrapper with capped-output retry.

Provides ``call_api`` (single request with retry) and
``infer_with_retry`` (automatic retry on capped empty output).
Used by v4+ inference scripts; earlier versions define simpler
local variants.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


def call_api(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    max_completion_tokens: int,
    max_retries: int = 3,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """Send a chat-completion request and return (content, metrics, error).

    Parameters
    ----------
    client : OpenAI
        Authenticated OpenAI client instance.
    messages : list[dict]
        Conversation messages in OpenAI format.
    model : str
        Model identifier (e.g. ``"gpt-5.1"``).
    max_completion_tokens : int
        Hard cap on completion tokens.
    max_retries : int
        Number of retry attempts on transient failure.

    Returns
    -------
    tuple[str, dict, str | None]
        ``(content, metrics_dict, error_string_or_None)``
    """
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
            metrics: Dict[str, Any] = {
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
    """Call ``call_api`` and retry once if the output was capped and empty.

    Used by v4+ inference scripts to handle models that sometimes
    return zero content when they hit the token ceiling.
    """
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
