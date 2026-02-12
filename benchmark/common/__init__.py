"""Shared utilities for the CL-bench benchmark pipeline.

This module provides canonical implementations of functions that are used
across multiple benchmark versions, reducing duplication and ensuring
consistent behavior.

Submodules
----------
io            – JSONL / JSON file I/O helpers.
identifiers   – Task and context ID extraction.
api           – OpenAI chat-completion wrapper with retry logic.
env           – Environment-variable parsing and timestamp helpers.
neo4j_utils   – Neo4j database management.
ace_shared    – ACE-specific constants (META_STRATEGY_SEEDS, guidance
                formatting, memory seeding, transferability filters).
pipeline      – Orchestrator utilities (line counting, subprocess env,
                run-metadata phase tracking).
llm_utils     – LLM response parsing helpers.
"""

from benchmark.common.io import load_jsonl, append_jsonl, load_json, write_json
from benchmark.common.identifiers import get_task_id, get_context_id, get_context_category
from benchmark.common.env import utc_now_iso, safe_env_int, safe_env_float, safe_env_bool
from benchmark.common.neo4j_utils import clear_neo4j_all
from benchmark.common.pipeline import count_lines, is_writable_dir, build_subprocess_env
from benchmark.common.llm_utils import parse_response_text, extract_usage

__all__ = [
    "load_jsonl",
    "append_jsonl",
    "load_json",
    "write_json",
    "get_task_id",
    "get_context_id",
    "get_context_category",
    "utc_now_iso",
    "safe_env_int",
    "safe_env_float",
    "safe_env_bool",
    "clear_neo4j_all",
    "count_lines",
    "is_writable_dir",
    "build_subprocess_env",
    "parse_response_text",
    "extract_usage",
]
