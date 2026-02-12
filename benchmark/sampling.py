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
DEFAULT_SAMPLING_STRATEGY = "task_random"


def validate_max_samples(max_samples: Optional[int]) -> None:
    if max_samples is None:
        return
    if max_samples <= 0:
        raise ValueError("max_samples must be > 0. Use None to run the full dataset.")


def get_task_id(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        task_id = metadata.get("task_id")
        if isinstance(task_id, str):
            return task_id
    task_id = item.get("task_id", "")
    return task_id if isinstance(task_id, str) else ""


def get_context_id(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        context_id = metadata.get("context_id")
        if isinstance(context_id, str):
            return context_id
    return ""


def get_context_category(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        context_category = metadata.get("context_category")
        if isinstance(context_category, str) and context_category.strip():
            return context_category.strip()
    return "Unknown"


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
    validate_max_samples(max_samples)
    if max_samples is None or max_samples >= len(data):
        return list(data)
    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(len(data)), max_samples))
    return [data[idx] for idx in sampled_indices]


def sample_context_dense_subset(
    data: List[Dict[str, Any]],
    max_samples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    validate_max_samples(max_samples)
    if max_samples is None or max_samples >= len(data):
        return list(data)

    rng = random.Random(seed)

    context_to_indices: Dict[str, List[int]] = {}
    context_order: List[str] = []
    for idx, item in enumerate(data):
        context_id = get_context_id(item)
        if context_id not in context_to_indices:
            context_to_indices[context_id] = []
            context_order.append(context_id)
        context_to_indices[context_id].append(idx)

    dense_contexts = [ctx for ctx in context_order if len(context_to_indices.get(ctx, [])) >= 2]
    rng.shuffle(dense_contexts)

    selected: List[int] = []
    selected_set = set()

    for context_id in dense_contexts:
        for idx in context_to_indices[context_id]:
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= max_samples:
                break
        if len(selected) >= max_samples:
            break

    if len(selected) < max_samples:
        remaining = [idx for idx in range(len(data)) if idx not in selected_set]
        rng.shuffle(remaining)
        needed = max_samples - len(selected)
        selected.extend(remaining[:needed])

    selected = sorted(selected[:max_samples])
    return [data[idx] for idx in selected]


def sample_context_dense_stratified_subset(
    data: List[Dict[str, Any]],
    max_samples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    validate_max_samples(max_samples)
    if max_samples is None or max_samples >= len(data):
        return list(data)

    rng = random.Random(seed)
    category_to_indices: Dict[str, List[int]] = {}
    for idx, item in enumerate(data):
        category = get_context_category(item)
        category_to_indices.setdefault(category, []).append(idx)

    categories = sorted(category_to_indices.keys())
    total = len(data)
    allocations: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    allocated = 0

    for category in categories:
        category_size = len(category_to_indices[category])
        exact = (max_samples * category_size) / total
        base = int(exact)
        allocations[category] = base
        allocated += base
        remainders.append((exact - base, category))

    remaining = max_samples - allocated
    for _, category in sorted(remainders, key = lambda row: (-row[0], row[1])):
        if remaining <= 0:
            break
        allocations[category] += 1
        remaining -= 1

    selected_indices: List[int] = []
    selected_set = set()

    for category in categories:
        candidates = list(category_to_indices[category])
        rng.shuffle(candidates)
        take = min(allocations.get(category, 0), len(candidates))
        picked = candidates[:take]
        selected_indices.extend(picked)
        selected_set.update(picked)

    if len(selected_indices) < max_samples:
        remaining_indices = [idx for idx in range(len(data)) if idx not in selected_set]
        rng.shuffle(remaining_indices)
        selected_indices.extend(remaining_indices[:(max_samples - len(selected_indices))])

    selected_indices = sorted(selected_indices[:max_samples])
    return [data[idx] for idx in selected_indices]


def sample_subset_by_strategy(
    data: List[Dict[str, Any]],
    max_samples: Optional[int],
    seed: int,
    sampling_strategy: str = DEFAULT_SAMPLING_STRATEGY,
) -> List[Dict[str, Any]]:
    strategy = (sampling_strategy or DEFAULT_SAMPLING_STRATEGY).strip().lower()
    if strategy == "context_dense_stratified":
        return sample_context_dense_stratified_subset(data = data, max_samples = max_samples, seed = seed)
    if strategy == "context_dense":
        return sample_context_dense_subset(data = data, max_samples = max_samples, seed = seed)
    return sample_deterministic_subset(data = data, max_samples = max_samples, seed = seed)


def create_manifest_payload(
    task_ids: List[str],
    seed: int,
    max_samples: Optional[int],
    dataset_name: str,
    split: str,
    sampling_strategy: str = DEFAULT_SAMPLING_STRATEGY,
) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "split": split,
        "seed": seed,
        "max_samples": max_samples,
        "sampling_strategy": sampling_strategy,
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
    sampling_strategy: str = DEFAULT_SAMPLING_STRATEGY,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
    validate_max_samples(max_samples)

    if manifest_path and os.path.exists(manifest_path):
        manifest = load_manifest(manifest_path)
        subset = subset_from_manifest(data, manifest)
        return subset, manifest, True

    subset = sample_subset_by_strategy(
        data = data,
        max_samples = max_samples,
        seed = seed,
        sampling_strategy = sampling_strategy,
    )
    task_ids = [get_task_id(item) for item in subset if get_task_id(item)]
    manifest = create_manifest_payload(
        task_ids=task_ids,
        seed=seed,
        max_samples=max_samples,
        dataset_name=dataset_name,
        split=split,
        sampling_strategy=sampling_strategy,
    )
    if manifest_path:
        write_manifest(manifest_path, manifest)
    return subset, manifest, False
