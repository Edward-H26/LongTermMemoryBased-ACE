"""
Deterministic CL-bench subset sampling with manifest reuse.
"""

import json
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


CLBENCH_DATASET_NAME = "tencent/CL-bench"
CLBENCH_SPLIT = "train"


def get_task_id(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        task_id = metadata.get("task_id")
        if isinstance(task_id, str):
            return task_id
    task_id = item.get("task_id", "")
    return task_id if isinstance(task_id, str) else ""


def load_clbench_dataset(
    dataset_name: str = CLBENCH_DATASET_NAME,
    split: str = CLBENCH_SPLIT,
) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    return [dict(row) for row in ds]


def sample_deterministic_subset(
    data: List[Dict[str, Any]],
    max_samples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(data):
        return list(data)
    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(data)), max_samples))
    return [data[idx] for idx in sampled_indices]


def create_manifest_payload(
    task_ids: List[str],
    seed: int,
    max_samples: Optional[int],
    dataset_name: str,
    split: str,
) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "split": split,
        "seed": seed,
        "max_samples": max_samples,
        "selected_count": len(task_ids),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_ids": task_ids,
    }


def write_manifest(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest format at {path}")
    task_ids = data.get("task_ids")
    if not isinstance(task_ids, list):
        raise ValueError(f"Manifest missing task_ids list: {path}")
    return data


def subset_from_manifest(
    data: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> List[Dict[str, Any]]:
    task_ids = manifest.get("task_ids", [])
    if not isinstance(task_ids, list):
        raise ValueError("Manifest task_ids must be a list")
    index: Dict[str, Dict[str, Any]] = {}
    for item in data:
        task_id = get_task_id(item)
        if task_id and task_id not in index:
            index[task_id] = item
    missing = [task_id for task_id in task_ids if task_id not in index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Manifest task_ids not found in dataset: {preview}")
    return [index[task_id] for task_id in task_ids]


def resolve_subset_with_manifest(
    data: List[Dict[str, Any]],
    max_samples: Optional[int],
    seed: int,
    manifest_path: Optional[str],
    dataset_name: str = CLBENCH_DATASET_NAME,
    split: str = CLBENCH_SPLIT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
    if manifest_path and os.path.exists(manifest_path):
        manifest = load_manifest(manifest_path)
        subset = subset_from_manifest(data, manifest)
        return subset, manifest, True

    subset = sample_deterministic_subset(data, max_samples=max_samples, seed=seed)
    task_ids = [get_task_id(item) for item in subset if get_task_id(item)]
    manifest = create_manifest_payload(
        task_ids=task_ids,
        seed=seed,
        max_samples=max_samples,
        dataset_name=dataset_name,
        split=split,
    )
    if manifest_path:
        write_manifest(manifest_path, manifest)
    return subset, manifest, False
