"""
Costing utilities for benchmark reporting and reconciliation.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

OPENAI_DEFAULT_INPUT_PRICE = 1.25
OPENAI_DEFAULT_OUTPUT_PRICE = 10.0
GEMINI_DEFAULT_INPUT_PRICE = 0.50
GEMINI_DEFAULT_OUTPUT_PRICE = 3.00

PRICE_TABLE: Dict[str, Dict[str, Dict[str, float]]] = {
    "openai": {
        "gpt-5.1": {"input": OPENAI_DEFAULT_INPUT_PRICE, "output": OPENAI_DEFAULT_OUTPUT_PRICE},
        "default": {"input": OPENAI_DEFAULT_INPUT_PRICE, "output": OPENAI_DEFAULT_OUTPUT_PRICE},
    },
    "gemini": {
        "gemini-3-flash-preview": {"input": GEMINI_DEFAULT_INPUT_PRICE, "output": GEMINI_DEFAULT_OUTPUT_PRICE},
        "default": {"input": GEMINI_DEFAULT_INPUT_PRICE, "output": GEMINI_DEFAULT_OUTPUT_PRICE},
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding = "utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def dump_json(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(path) if os.path.dirname(path) else "."
    os.makedirs(parent, exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(payload, f, indent = 2, ensure_ascii = False)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_provider(provider: str, model: str) -> str:
    provider_norm = str(provider or "").strip().lower()
    model_norm = str(model or "").strip().lower()
    if provider_norm:
        return provider_norm
    if model_norm.startswith("gemini"):
        return "gemini"
    return "openai"


def resolve_price(provider: str, model: str) -> Dict[str, float]:
    provider_norm = _normalize_provider(provider, model)
    model_norm = str(model or "default").strip().lower()

    provider_prices = PRICE_TABLE.get(provider_norm)
    if not provider_prices:
        provider_prices = PRICE_TABLE["openai"]

    row = provider_prices.get(model_norm) or provider_prices.get("default") or PRICE_TABLE["openai"]["default"]
    return {
        "input": float(row.get("input", OPENAI_DEFAULT_INPUT_PRICE)),
        "output": float(row.get("output", OPENAI_DEFAULT_OUTPUT_PRICE)),
        "provider": provider_norm,
        "model": model_norm,
    }


def compute_cost_usd(
    prompt_tokens: float,
    completion_tokens: float,
    provider: str,
    model: str,
) -> float:
    price = resolve_price(provider = provider, model = model)
    return (
        (_safe_float(prompt_tokens) / 1_000_000.0 * price["input"]) +
        (_safe_float(completion_tokens) / 1_000_000.0 * price["output"])
    )


def aggregate_token_cost_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_prompt = 0.0
    total_completion = 0.0
    total_tokens = 0.0
    total_cost = 0.0
    by_provider: Dict[str, Dict[str, float]] = {}

    for entry in entries:
        provider = str(entry.get("provider", "")).strip().lower()
        model = str(entry.get("model", "")).strip()
        prompt_tokens = _safe_float(entry.get("prompt_tokens", 0.0), 0.0)
        completion_tokens = _safe_float(entry.get("completion_tokens", 0.0), 0.0)
        total_tokens_entry = _safe_float(entry.get("total_tokens", prompt_tokens + completion_tokens), prompt_tokens + completion_tokens)

        entry_cost = compute_cost_usd(
            prompt_tokens = prompt_tokens,
            completion_tokens = completion_tokens,
            provider = provider,
            model = model,
        )

        total_prompt += prompt_tokens
        total_completion += completion_tokens
        total_tokens += total_tokens_entry
        total_cost += entry_cost

        key = provider or "openai"
        provider_bucket = by_provider.setdefault(
            key,
            {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "cost_usd": 0.0,
            },
        )
        provider_bucket["prompt_tokens"] += prompt_tokens
        provider_bucket["completion_tokens"] += completion_tokens
        provider_bucket["total_tokens"] += total_tokens_entry
        provider_bucket["cost_usd"] += entry_cost

    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "cost_usd": total_cost,
        "by_provider": by_provider,
    }


def merge_phase_summaries(phases: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_prompt = 0.0
    total_completion = 0.0
    total_tokens = 0.0
    total_cost = 0.0
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        total_prompt += _safe_float(phase.get("prompt_tokens", 0.0), 0.0)
        total_completion += _safe_float(phase.get("completion_tokens", 0.0), 0.0)
        total_tokens += _safe_float(phase.get("total_tokens", 0.0), 0.0)
        total_cost += _safe_float(phase.get("cost_usd", 0.0), 0.0)
    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "cost_usd": total_cost,
    }


def extract_ace_aux_metrics(path: str) -> Dict[str, Any]:
    payload = load_json(path)
    summary = payload.get("summary", {}) if isinstance(payload.get("summary", {}), dict) else {}
    records = payload.get("records", []) if isinstance(payload.get("records", []), list) else []

    if records:
        entries: List[Dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            entries.append({
                "provider": record.get("backend", ""),
                "model": record.get("model", ""),
                "prompt_tokens": record.get("prompt_tokens", 0),
                "completion_tokens": record.get("completion_tokens", 0),
                "total_tokens": record.get("total_tokens", 0),
            })
        agg = aggregate_token_cost_entries(entries)
        agg["source"] = "records"
        agg["total_calls"] = len(entries)
        return agg

    if summary:
        prompt_tokens = _safe_float(summary.get("total_prompt_tokens", 0.0), 0.0)
        completion_tokens = _safe_float(summary.get("total_completion_tokens", 0.0), 0.0)
        total_tokens = _safe_float(summary.get("total_tokens", prompt_tokens + completion_tokens), prompt_tokens + completion_tokens)
        cost = compute_cost_usd(
            prompt_tokens = prompt_tokens,
            completion_tokens = completion_tokens,
            provider = "openai",
            model = "gpt-5.1",
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost,
            "by_provider": {
                "openai": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost_usd": cost,
                },
            },
            "source": "summary",
            "total_calls": int(summary.get("total_calls", 0) or 0),
        }

    return {
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "cost_usd": 0.0,
        "by_provider": {},
        "source": "missing",
        "total_calls": 0,
    }


def normalize_phase_metric_payload(
    payload: Dict[str, Any],
    model_fallback: str = "gpt-5.1",
    provider_fallback: str = "openai",
) -> Dict[str, Any]:
    prompt_tokens = _safe_float(payload.get("prompt_tokens", 0.0), 0.0)
    completion_tokens = _safe_float(payload.get("completion_tokens", 0.0), 0.0)
    total_tokens = _safe_float(payload.get("total_tokens", prompt_tokens + completion_tokens), prompt_tokens + completion_tokens)
    provider = str(payload.get("provider", provider_fallback)).strip().lower() or provider_fallback
    model = str(payload.get("model", model_fallback)).strip() or model_fallback
    cost_usd = compute_cost_usd(
        prompt_tokens = prompt_tokens,
        completion_tokens = completion_tokens,
        provider = provider,
        model = model,
    )
    normalized = {
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }
    passthrough_keys = [
        "total_calls",
        "api_fail_count",
        "json_fail_count",
        "graded_ok_count",
        "classified_count",
        "fallback_count",
        "no_output_count",
        "started_at",
        "ended_at",
        "wall_seconds",
    ]
    for key in passthrough_keys:
        if key in payload:
            normalized[key] = payload.get(key)
    return normalized


def load_phase_metric_file(
    path: Optional[str],
    model_fallback: str = "gpt-5.1",
    provider_fallback: str = "openai",
) -> Dict[str, Any]:
    if not path:
        return {
            "exists": False,
            "path": "",
            **normalize_phase_metric_payload({}, model_fallback = model_fallback, provider_fallback = provider_fallback),
        }
    payload = load_json(path)
    normalized = normalize_phase_metric_payload(payload, model_fallback = model_fallback, provider_fallback = provider_fallback)
    normalized["exists"] = bool(payload)
    normalized["path"] = path
    return normalized


def parse_iso_to_unix_seconds(timestamp: str) -> Optional[int]:
    if not timestamp:
        return None
    ts = str(timestamp).strip()
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo = timezone.utc)
    return int(dt.timestamp())


def _sum_openai_cost_from_payload(payload: Dict[str, Any]) -> float:
    total = 0.0
    data_rows = payload.get("data", []) if isinstance(payload.get("data", []), list) else []
    for bucket in data_rows:
        if not isinstance(bucket, dict):
            continue
        results = bucket.get("results", []) if isinstance(bucket.get("results", []), list) else []
        for row in results:
            if not isinstance(row, dict):
                continue
            amount = row.get("amount", {})
            if isinstance(amount, dict):
                total += _safe_float(amount.get("value", 0.0), 0.0)
    return total


def _sum_openai_usage_from_payload(payload: Dict[str, Any]) -> Dict[str, float]:
    totals = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "total_tokens": 0.0,
        "num_model_requests": 0.0,
    }
    data_rows = payload.get("data", []) if isinstance(payload.get("data", []), list) else []
    for bucket in data_rows:
        if not isinstance(bucket, dict):
            continue
        results = bucket.get("results", []) if isinstance(bucket.get("results", []), list) else []
        for row in results:
            if not isinstance(row, dict):
                continue
            totals["input_tokens"] += _safe_float(row.get("input_tokens", 0.0), 0.0)
            totals["output_tokens"] += _safe_float(row.get("output_tokens", 0.0), 0.0)
            totals["num_model_requests"] += _safe_float(row.get("num_model_requests", 0.0), 0.0)
    totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return totals


def fetch_openai_billed_reconciliation(
    admin_api_key: str,
    project_id: str,
    start_iso: str,
    end_iso: str,
    base_url: Optional[str] = None,
    timeout_sec: int = 30,
) -> Dict[str, Any]:
    start_ts = parse_iso_to_unix_seconds(start_iso)
    end_ts = parse_iso_to_unix_seconds(end_iso)
    if start_ts is None or end_ts is None:
        return {
            "success": False,
            "error": "invalid_time_window",
            "cost_usd": 0.0,
            "usage": {},
            "window": {"start": start_iso, "end": end_iso},
        }
    if end_ts <= start_ts:
        return {
            "success": False,
            "error": "non_positive_time_window",
            "cost_usd": 0.0,
            "usage": {},
            "window": {"start": start_iso, "end": end_iso},
        }

    api_base = (base_url or os.getenv("OPENAI_ADMIN_API_BASE") or "https://api.openai.com/v1").rstrip("/")
    headers = {
        "Authorization": f"Bearer {admin_api_key}",
        "Content-Type": "application/json",
    }

    cost_url = f"{api_base}/organization/costs"
    usage_url = f"{api_base}/organization/usage/completions"

    base_params = {
        "start_time": start_ts,
        "end_time": end_ts,
        "bucket_width": "1d",
        "project_ids[]": project_id,
    }

    try:
        cost_resp = requests.get(cost_url, params = base_params, headers = headers, timeout = timeout_sec)
    except Exception as exc:
        return {
            "success": False,
            "error": f"cost_request_failed:{exc}",
            "cost_usd": 0.0,
            "usage": {},
            "window": {"start": start_iso, "end": end_iso},
            "endpoints": {"cost_url": cost_url, "usage_url": usage_url},
        }

    if cost_resp.status_code != 200:
        return {
            "success": False,
            "error": f"cost_status_{cost_resp.status_code}:{cost_resp.text[:500]}",
            "cost_usd": 0.0,
            "usage": {},
            "window": {"start": start_iso, "end": end_iso},
            "endpoints": {"cost_url": cost_url, "usage_url": usage_url},
        }

    try:
        cost_payload = cost_resp.json()
    except Exception as exc:
        return {
            "success": False,
            "error": f"cost_json_parse_failed:{exc}",
            "cost_usd": 0.0,
            "usage": {},
            "window": {"start": start_iso, "end": end_iso},
            "endpoints": {"cost_url": cost_url, "usage_url": usage_url},
        }

    try:
        usage_resp = requests.get(usage_url, params = base_params, headers = headers, timeout = timeout_sec)
    except Exception as exc:
        return {
            "success": False,
            "error": f"usage_request_failed:{exc}",
            "cost_usd": 0.0,
            "usage": {},
            "window": {"start": start_iso, "end": end_iso},
            "endpoints": {"cost_url": cost_url, "usage_url": usage_url},
            "raw": {"cost": cost_payload},
        }

    if usage_resp.status_code != 200:
        return {
            "success": False,
            "error": f"usage_status_{usage_resp.status_code}:{usage_resp.text[:500]}",
            "cost_usd": 0.0,
            "usage": {},
            "window": {"start": start_iso, "end": end_iso},
            "endpoints": {"cost_url": cost_url, "usage_url": usage_url},
            "raw": {"cost": cost_payload},
        }

    try:
        usage_payload = usage_resp.json()
    except Exception as exc:
        return {
            "success": False,
            "error": f"usage_json_parse_failed:{exc}",
            "cost_usd": 0.0,
            "usage": {},
            "window": {"start": start_iso, "end": end_iso},
            "endpoints": {"cost_url": cost_url, "usage_url": usage_url},
            "raw": {"cost": cost_payload},
        }

    cost_usd = _sum_openai_cost_from_payload(cost_payload)
    usage_totals = _sum_openai_usage_from_payload(usage_payload)

    if cost_usd <= 0.0 and usage_totals.get("total_tokens", 0.0) <= 0.0:
        return {
            "success": False,
            "error": "empty_billing_payload",
            "cost_usd": 0.0,
            "usage": usage_totals,
            "window": {"start": start_iso, "end": end_iso},
            "endpoints": {"cost_url": cost_url, "usage_url": usage_url},
            "raw": {"cost": cost_payload, "usage": usage_payload},
        }

    return {
        "success": True,
        "error": "",
        "cost_usd": cost_usd,
        "usage": usage_totals,
        "window": {"start": start_iso, "end": end_iso},
        "endpoints": {"cost_url": cost_url, "usage_url": usage_url},
    }


def resolve_run_window(run_meta: Dict[str, Any]) -> Tuple[str, str]:
    phases = run_meta.get("phases", {}) if isinstance(run_meta.get("phases", {}), dict) else {}
    default_start = str(run_meta.get("started_at", "")).strip()
    default_end = str(run_meta.get("ended_at", "")).strip()

    start_candidates = [
        str(phases.get("inference", {}).get("started_at", "")).strip() if isinstance(phases.get("inference", {}), dict) else "",
        default_start,
    ]
    end_candidates = [
        str(phases.get("compare", {}).get("ended_at", "")).strip() if isinstance(phases.get("compare", {}), dict) else "",
        str(phases.get("error_analysis", {}).get("ended_at", "")).strip() if isinstance(phases.get("error_analysis", {}), dict) else "",
        str(phases.get("evaluation", {}).get("ended_at", "")).strip() if isinstance(phases.get("evaluation", {}), dict) else "",
        default_end,
    ]

    start_iso = next((value for value in start_candidates if value), "")
    end_iso = next((value for value in end_candidates if value), "")
    return start_iso, end_iso
