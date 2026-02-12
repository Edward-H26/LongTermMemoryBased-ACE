"""Task and context identifier extraction.

Consistent helpers for extracting task_id, context_id, and
context_category from CL-bench data rows.
"""

from typing import Any, Dict


def get_task_id(item: Dict[str, Any]) -> str:
    """Return the canonical task identifier for a CL-bench row.

    Checks ``task_id`` at the top level first, then falls back to
    ``metadata.task_id``.
    """
    task_id = item.get("task_id")
    if isinstance(task_id, str) and task_id:
        return task_id
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        task_id = metadata.get("task_id", "")
        if isinstance(task_id, str):
            return task_id
    return ""


def get_context_id(item: Dict[str, Any]) -> str:
    """Return the context identifier from a CL-bench row's metadata."""
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        context_id = metadata.get("context_id", "")
        if isinstance(context_id, str):
            return context_id
    return ""


def get_context_category(item: Dict[str, Any]) -> str:
    """Return the context category from a CL-bench row's metadata."""
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        category = metadata.get("context_category", "")
        if isinstance(category, str):
            return category
    return ""
