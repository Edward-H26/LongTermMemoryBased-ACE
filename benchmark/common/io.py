"""JSONL and JSON file I/O helpers.

These functions are used across all benchmark versions for reading,
appending, and writing structured data files.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of parsed rows.

    Returns an empty list when the file does not exist.
    """
    data: List[Dict[str, Any]] = []
    if os.path.exists(path):
        with open(path, "r", encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def append_jsonl(item: Dict[str, Any], path: str) -> None:
    """Append a single JSON object as a new line to a JSONL file.

    Creates parent directories if they do not exist.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok = True)
    with open(path, "a", encoding = "utf-8") as f:
        f.write(json.dumps(item, ensure_ascii = False) + "\n")


def load_json(path: str) -> Dict[str, Any]:
    """Read a JSON file and return the parsed dict.

    Returns an empty dict when the file does not exist or does not
    contain a JSON object.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding = "utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write a dict to a JSON file with pretty-printing.

    Creates parent directories if they do not exist.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(payload, f, indent = 2, ensure_ascii = False)
