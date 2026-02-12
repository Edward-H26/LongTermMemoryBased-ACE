"""LLM response parsing helpers.

Shared by ``eval.py`` and ``error_analysis.py`` for cleaning
model outputs and extracting token-usage metadata.
"""

from typing import Any, Dict


def parse_response_text(result_text: str) -> str:
    """Strip Markdown code fences from an LLM response.

    Handles ``\\ ```json ... \\ ``` `` and bare ``\\ ``` ... \\ ``` `` wrappers.
    """
    cleaned = result_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def extract_usage(response: Any) -> Dict[str, int]:
    """Extract token-usage counters from an OpenAI response object."""
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(
        getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
        or (prompt_tokens + completion_tokens)
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
