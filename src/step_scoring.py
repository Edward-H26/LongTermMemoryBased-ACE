"""
Process-level step scoring utilities for CoT, ToT, and ReAct flows.

Implements a near-full process scoring approach inspired by process supervision:
- Split reasoning into steps
- Score each step with deterministic checks
- Optionally add LLM verification per step
- Blend both signals for robust intermediate quality scoring
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.llm import LLM

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class StepScoringConfig:
    mode: str = "full"
    scorer_model: str = "gpt-5.1"
    workers: int = 12
    min_score: float = 0.40
    llm_weight: float = 0.85
    deterministic_weight: float = 0.15
    max_steps_with_full_llm: int = 24

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            return default
        return parsed if parsed > 0 else default

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @classmethod
    def from_env(cls) -> "StepScoringConfig":
        return cls(
            mode = str(os.getenv("ACE_STEP_SCORING_MODE", "full")).strip().lower(),
            scorer_model = str(os.getenv("ACE_STEP_SCORER_MODEL", "gpt-5.1")).strip(),
            workers = cls._safe_int(os.getenv("ACE_STEP_SCORE_WORKERS", "12"), 12),
            min_score = cls._safe_float(os.getenv("ACE_STEP_SCORE_MIN", "0.40"), 0.40),
        )

    @classmethod
    def from_state(cls, scratch: Optional[Dict[str, Any]] = None) -> "StepScoringConfig":
        cfg = cls.from_env()
        if not isinstance(scratch, dict):
            return cfg
        mode = scratch.get("step_scoring_mode")
        scorer_model = scratch.get("step_scorer_model")
        workers = scratch.get("step_score_workers")
        min_score = scratch.get("step_score_min")
        if mode:
            cfg.mode = str(mode).strip().lower()
        if scorer_model:
            cfg.scorer_model = str(scorer_model).strip()
        if workers is not None:
            cfg.workers = cls._safe_int(workers, cfg.workers)
        if min_score is not None:
            cfg.min_score = cls._safe_float(min_score, cfg.min_score)
        return cfg


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


def split_reasoning_steps(text: str, max_steps: int = 40) -> List[str]:
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets: List[str] = []
    for line in lines:
        cleaned = re.sub(r"^\s*(?:\d+[\).\s]+|[-*•]\s+)", "", line).strip()
        if cleaned:
            bullets.append(cleaned)

    if len(bullets) <= 1:
        sentence_candidates = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
        bullets = sentence_candidates if sentence_candidates else bullets

    if not bullets:
        bullets = [text.strip()]

    deduped: List[str] = []
    seen = set()
    for step in bullets:
        key = re.sub(r"\s+", " ", step.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(step)
        if len(deduped) >= max_steps:
            break
    return deduped


def _deterministic_step_score(question: str, step: str, index: int, total_steps: int) -> float:
    step_tokens = _tokenize(step)
    question_tokens = set(_tokenize(question))
    step_token_set = set(step_tokens)

    token_count = len(step_tokens)
    token_score = min(token_count / 24.0, 1.0)

    overlap_score = 0.0
    union = len(question_tokens.union(step_token_set))
    if union > 0:
        overlap_score = len(question_tokens.intersection(step_token_set)) / union

    lower = step.lower()
    reasoning_markers = [
        "because",
        "therefore",
        "so",
        "thus",
        "verify",
        "check",
        "derive",
        "calculate",
        "assume",
        "constraint",
        "format",
    ]
    marker_score = 1.0 if any(marker in lower for marker in reasoning_markers) else 0.0

    if total_steps <= 1:
        position_score = 0.8
    else:
        normalized_pos = index / max(total_steps - 1, 1)
        position_score = 1.0 - abs(normalized_pos - 0.6)
        position_score = _clamp(position_score)

    score = (
        0.35 * token_score +
        0.30 * overlap_score +
        0.20 * marker_score +
        0.15 * position_score
    )
    return _clamp(score)


def _parse_llm_score(content: str) -> Tuple[Optional[float], str]:
    if not content:
        return None, "empty_response"
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    try:
        data = json.loads(text)
    except Exception:
        match = JSON_OBJECT_PATTERN.search(text)
        if not match:
            return None, "json_parse_failed"
        try:
            data = json.loads(match.group(0))
        except Exception:
            return None, "json_parse_failed"

    score = data.get("score")
    try:
        parsed = float(score)
    except Exception:
        return None, "missing_score"
    return _clamp(parsed), str(data.get("reason", "")).strip()


def _score_step_with_llm(
    question: str,
    step: str,
    index: int,
    total_steps: int,
    model: str,
) -> Tuple[Optional[float], str]:
    llm = LLM(model = model, backend = "openai", temperature = 0.0)
    prompt = (
        "Score one reasoning step for correctness and usefulness toward solving the task.\n"
        "Return strict JSON only: {\"score\": float_between_0_and_1, \"reason\": \"short\"}.\n"
        "Scoring rubric:\n"
        "1.0 means highly relevant, logically valid, and moves toward completion.\n"
        "0.5 means partially useful or uncertain.\n"
        "0.0 means irrelevant, incorrect, or harmful.\n"
        f"Question:\n{question}\n\n"
        f"Step {index + 1}/{total_steps}:\n{step}\n"
    )
    try:
        response = llm.chat(
            [
                {"role": "system", "content": "You are a strict process verifier. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens = 220,
            temperature = 0.0,
        )
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return _parse_llm_score(content)
    except Exception as exc:
        return None, f"llm_error:{exc}"


def _llm_selection_mask(steps: List[str], config: StepScoringConfig) -> List[bool]:
    if config.mode == "full":
        return [True for _ in steps]
    if config.mode == "off":
        return [False for _ in steps]

    total = len(steps)
    if total <= config.max_steps_with_full_llm:
        return [True for _ in steps]

    ranked = sorted(
        list(enumerate(steps)),
        key = lambda row: len(_tokenize(row[1])),
        reverse = True,
    )
    allowed = {idx for idx, _ in ranked[:config.max_steps_with_full_llm]}
    return [idx in allowed for idx in range(total)]


def score_steps(
    question: str,
    steps: List[str],
    config: Optional[StepScoringConfig] = None,
) -> Dict[str, Any]:
    cfg = config or StepScoringConfig.from_env()
    safe_steps = [step for step in steps if isinstance(step, str) and step.strip()]
    if cfg.mode == "off":
        return {
            "config": {
                "mode": cfg.mode,
                "scorer_model": cfg.scorer_model,
                "workers": cfg.workers,
                "min_score": cfg.min_score,
            },
            "num_steps": len(safe_steps),
            "num_llm_scored": 0,
            "mean_step_score": 0.0,
            "min_step_score": 0.0,
            "max_step_score": 0.0,
            "mean_step_confidence": 0.0,
            "min_step_confidence": 0.0,
            "max_step_confidence": 0.0,
            "overall_confidence": 0.0,
            "steps": [],
        }
    if not safe_steps:
        return {
            "config": {
                "mode": cfg.mode,
                "scorer_model": cfg.scorer_model,
                "workers": cfg.workers,
                "min_score": cfg.min_score,
            },
            "num_steps": 0,
            "num_llm_scored": 0,
            "mean_step_score": 0.0,
            "min_step_score": 0.0,
            "max_step_score": 0.0,
            "mean_step_confidence": 0.0,
            "min_step_confidence": 0.0,
            "max_step_confidence": 0.0,
            "overall_confidence": 0.0,
            "steps": [],
        }

    llm_mask = _llm_selection_mask(safe_steps, cfg)
    llm_results: Dict[int, Tuple[Optional[float], str]] = {}

    if any(llm_mask):
        with ThreadPoolExecutor(max_workers = max(1, cfg.workers)) as executor:
            future_map = {}
            for idx, step in enumerate(safe_steps):
                if not llm_mask[idx]:
                    continue
                future = executor.submit(
                    _score_step_with_llm,
                    question,
                    step,
                    idx,
                    len(safe_steps),
                    cfg.scorer_model,
                )
                future_map[future] = idx
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    llm_results[idx] = future.result()
                except Exception as exc:
                    llm_results[idx] = (None, f"executor_error:{exc}")

    rows: List[Dict[str, Any]] = []
    combined_scores: List[float] = []
    confidence_scores: List[float] = []
    for idx, step in enumerate(safe_steps):
        deterministic_score = _deterministic_step_score(
            question = question,
            step = step,
            index = idx,
            total_steps = len(safe_steps),
        )
        llm_score, llm_reason = llm_results.get(idx, (None, "not_scored_by_llm"))

        if llm_score is None:
            final_score = deterministic_score
            confidence = _clamp(0.55 + 0.45 * deterministic_score)
        else:
            final_score = _clamp(
                cfg.llm_weight * llm_score + cfg.deterministic_weight * deterministic_score
            )
            agreement = _clamp(1.0 - abs(llm_score - deterministic_score))
            confidence = _clamp(0.5 * agreement + 0.5 * final_score)

        combined_scores.append(final_score)
        confidence_scores.append(confidence)
        rows.append({
            "index": idx,
            "step": step,
            "deterministic_score": deterministic_score,
            "llm_score": llm_score,
            "score": final_score,
            "confidence": confidence,
            "selected_for_llm": bool(llm_mask[idx]),
            "llm_reason": llm_reason,
        })

    mean_score = sum(combined_scores) / len(combined_scores)
    mean_confidence = sum(confidence_scores) / len(confidence_scores)
    return {
        "config": {
            "mode": cfg.mode,
            "scorer_model": cfg.scorer_model,
            "workers": cfg.workers,
            "min_score": cfg.min_score,
        },
        "num_steps": len(rows),
        "num_llm_scored": sum(1 for row in rows if row.get("selected_for_llm")),
        "mean_step_score": mean_score,
        "min_step_score": min(combined_scores),
        "max_step_score": max(combined_scores),
        "mean_step_confidence": mean_confidence,
        "min_step_confidence": min(confidence_scores),
        "max_step_confidence": max(confidence_scores),
        "overall_confidence": _clamp(0.6 * mean_confidence + 0.4 * mean_score),
        "steps": rows,
    }


def score_reasoning_text(
    question: str,
    reasoning_text: str,
    config: Optional[StepScoringConfig] = None,
) -> Dict[str, Any]:
    steps = split_reasoning_steps(reasoning_text)
    return score_steps(question = question, steps = steps, config = config)


def score_reasoning_candidates(
    question: str,
    candidate_texts: List[str],
    config: Optional[StepScoringConfig] = None,
) -> List[Dict[str, Any]]:
    cfg = config or StepScoringConfig.from_env()
    if not candidate_texts:
        return []

    results: List[Dict[str, Any]] = [{} for _ in candidate_texts]
    with ThreadPoolExecutor(max_workers = max(1, cfg.workers)) as executor:
        future_map = {}
        for idx, text in enumerate(candidate_texts):
            future = executor.submit(
                score_reasoning_text,
                question,
                text,
                cfg,
            )
            future_map[future] = idx
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = {
                    "num_steps": 0,
                    "num_llm_scored": 0,
                    "mean_step_score": 0.0,
                    "min_step_score": 0.0,
                    "max_step_score": 0.0,
                    "mean_step_confidence": 0.0,
                    "min_step_confidence": 0.0,
                    "max_step_confidence": 0.0,
                    "overall_confidence": 0.0,
                    "steps": [],
                    "error": str(exc),
                }
    return results
