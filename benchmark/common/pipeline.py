"""Orchestrator utilities for benchmark pipelines.

Provides line counting, directory writability checks, subprocess
environment construction, and run-metadata phase tracking — shared
across ``run_v*.py`` and ``complete_v*_pipeline.py`` scripts.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

from benchmark.common.env import utc_now_iso
from benchmark.common.io import load_json, write_json


def count_lines(path: str) -> int:
    """Return the number of lines in *path*, or 0 if it does not exist."""
    if not os.path.exists(path):
        return 0
    return sum(1 for _ in open(path, "r", encoding = "utf-8"))


def is_writable_dir(path: str) -> bool:
    """Return True if *path* can be created and written to."""
    try:
        os.makedirs(path, exist_ok = True)
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w", encoding = "utf-8") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def build_subprocess_env() -> Dict[str, str]:
    """Return a copy of ``os.environ`` with ``HF_DATASETS_CACHE`` set.

    Falls back to ``~/.cache/huggingface/datasets`` or a local
    ``benchmark/hf_cache`` directory when the variable is unset.
    """
    env = os.environ.copy()
    cache_dir = env.get("HF_DATASETS_CACHE")
    if not cache_dir:
        default_cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        if is_writable_dir(default_cache_dir):
            cache_dir = default_cache_dir
        else:
            cache_dir = os.path.abspath(os.path.join("benchmark", "hf_cache"))
        env["HF_DATASETS_CACHE"] = cache_dir
    os.makedirs(cache_dir, exist_ok = True)
    return env


# ---------------------------------------------------------------------------
# Run-metadata phase tracking (v5+)
# ---------------------------------------------------------------------------

def init_run_meta(path: str, version: str, args: Any) -> None:
    """Initialise or update the run-metadata JSON at *path*.

    Creates a new metadata document with run-id, timestamps, and
    parameters derived from *args*.
    """
    payload = load_json(path)
    if not payload:
        payload = {
            "run_id": f"{version}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            "started_at": utc_now_iso(),
            "ended_at": "",
            "manifest": os.path.abspath(args.manifest),
            "seed": int(args.seed),
            "max_samples": int(args.max_samples),
            "sampling_strategy": args.sampling_strategy,
            "memory_scope": getattr(args, "memory_scope", "hybrid"),
            "output_dir": os.path.abspath(args.output_dir),
            "phases": {},
        }
    payload["started_at"] = payload.get("started_at", utc_now_iso()) or utc_now_iso()
    write_json(path, payload)


def mark_phase_start(path: str, phase_name: str) -> None:
    """Record the start timestamp for *phase_name* in run metadata."""
    payload = load_json(path)
    phases = payload.get("phases", {})
    if not isinstance(phases, dict):
        phases = {}
    phase = phases.get(phase_name, {})
    if not isinstance(phase, dict):
        phase = {}
    phase["started_at"] = utc_now_iso()
    phases[phase_name] = phase
    payload["phases"] = phases
    write_json(path, payload)


def mark_phase_end(path: str, phase_name: str) -> None:
    """Record the end timestamp for *phase_name* in run metadata."""
    payload = load_json(path)
    phases = payload.get("phases", {})
    if not isinstance(phases, dict):
        phases = {}
    phase = phases.get(phase_name, {})
    if not isinstance(phase, dict):
        phase = {}
    phase["ended_at"] = utc_now_iso()
    phases[phase_name] = phase
    payload["phases"] = phases
    payload["ended_at"] = utc_now_iso()
    write_json(path, payload)
