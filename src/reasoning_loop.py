"""
Recursive test-time reasoning utilities.

Provides reusable recursive refinement loops for:
- benchmark direct inference (OpenAI callables)
- runtime agent answer refinement (text-level generator callables)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.step_scoring import StepScoringConfig, score_reasoning_text

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _safe_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class ReasoningLoopConfig:
    max_rounds: int = 2
    candidates: int = 3
    improve_min: float = 0.03

    @classmethod
    def from_env(cls) -> "ReasoningLoopConfig":
        return cls(
            max_rounds = _safe_int(os.getenv("ACE_RECURSION_MAX_ROUNDS", "2"), 2),
            candidates = _safe_int(os.getenv("ACE_RECURSION_CANDIDATES", "3"), 3),
            improve_min = _safe_float(os.getenv("ACE_RECURSION_IMPROVE_MIN", "0.03"), 0.03),
        )


def _empty_step_summary(step_config: Optional[StepScoringConfig] = None) -> Dict[str, Any]:
    cfg = step_config or StepScoringConfig.from_env()
    return {
        "config": {
            "mode": cfg.mode,
            "scorer_model": cfg.scorer_model,
            "scorer_backend": cfg.scorer_backend,
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


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


def _context_overlap_score(question: str, content: str) -> float:
    question_tokens = set(_tokenize(question))
    content_tokens = set(_tokenize(content))
    if not question_tokens or not content_tokens:
        return 0.0
    union = len(question_tokens.union(content_tokens))
    if union <= 0:
        return 0.0
    intersection = len(question_tokens.intersection(content_tokens))
    return _clamp(intersection / union, 0.0, 1.0)


def _candidate_quality_score(
    question: str,
    content: str,
    step_config: StepScoringConfig,
    score_style: str = "legacy",
) -> Tuple[float, Dict[str, Any]]:
    if not str(content or "").strip():
        return 0.0, _empty_step_summary(step_config)
    summary = score_reasoning_text(
        question = question,
        reasoning_text = content,
        config = step_config,
    )
    step_mean = float(summary.get("mean_step_score", 0.0) or 0.0)
    output_valid = 1.0 if str(content).strip() else 0.0
    style = str(score_style or "legacy").strip().lower()
    if style == "strict_final":
        context_overlap = _context_overlap_score(question = question, content = content)
        score = _clamp(0.55 * step_mean + 0.25 * output_valid + 0.20 * context_overlap, 0.0, 1.0)
    else:
        score = _clamp(0.75 * step_mean + 0.25 * output_valid, 0.0, 1.0)
    return score, summary


def _aggregate_metrics(metrics_rows: List[Dict[str, Any]], selected_index: int) -> Dict[str, Any]:
    prompt_tokens = sum(int(row.get("prompt_tokens", 0) or 0) for row in metrics_rows)
    completion_tokens = sum(int(row.get("completion_tokens", 0) or 0) for row in metrics_rows)
    total_tokens = sum(int(row.get("total_tokens", 0) or 0) for row in metrics_rows)
    latency_ms = sum(float(row.get("latency_ms", 0.0) or 0.0) for row in metrics_rows)
    completion_capped = any(bool(row.get("completion_capped", False)) for row in metrics_rows)
    empty_output_retry_count = sum(int(row.get("empty_output_retry_count", 0) or 0) for row in metrics_rows)
    selected = metrics_rows[selected_index] if 0 <= selected_index < len(metrics_rows) else {}
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": latency_ms,
        "finish_reason": str(selected.get("finish_reason", "")),
        "completion_capped": completion_capped,
        "empty_output_retry_count": empty_output_retry_count,
    }


def run_recursive_openai_reasoning(
    question: str,
    base_messages: List[Dict[str, str]],
    generate_once: Callable[[List[Dict[str, str]]], Tuple[str, Dict[str, Any], Optional[str]]],
    step_config: Optional[StepScoringConfig] = None,
    loop_config: Optional[ReasoningLoopConfig] = None,
    action_overrides: Optional[Dict[str, Any]] = None,
    prompt_style: str = "legacy",
) -> Dict[str, Any]:
    """
    Run recursive candidate generation with scorer-guided selection.

    ``generate_once`` must return ``(content, metrics, error)``.
    """
    cfg = loop_config or ReasoningLoopConfig.from_env()
    step_cfg = step_config or StepScoringConfig.from_env()
    overrides = action_overrides or {}
    style = str(prompt_style or "legacy").strip().lower()
    rounds_planned = max(1, _safe_int(overrides.get("rounds", cfg.max_rounds), cfg.max_rounds))
    candidates_per_round = max(1, _safe_int(overrides.get("candidates", cfg.candidates), cfg.candidates))
    improve_min = _safe_float(overrides.get("improve_min", cfg.improve_min), cfg.improve_min)

    all_candidate_rows: List[Dict[str, Any]] = []
    diagnostics_rounds: List[Dict[str, Any]] = []
    errors: List[str] = []
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    initial_score = 0.0
    exit_reason = "max_rounds_reached"

    for round_idx in range(rounds_planned):
        round_rows: List[Dict[str, Any]] = []
        for candidate_idx in range(candidates_per_round):
            prompt_messages = [dict(message) for message in base_messages]
            if style == "strict_final":
                prompt_messages.append({
                    "role": "user",
                    "content": (
                        "Return only the final answer. Follow all required constraints and output format exactly. "
                        "Do not include rationale unless explicitly requested."
                    ),
                })
            if best is not None and round_idx > 0:
                prior = str(best.get("content", "")).strip()
                if candidate_idx == 0:
                    if style == "strict_final":
                        critique = (
                            "Refine your previous answer and output only the final answer. "
                            "Preserve the exact required format from the task. "
                            "Do not include analysis or rationale unless explicitly requested."
                        )
                    else:
                        critique = (
                            "Refine your previous answer. Fix weak reasoning steps, strengthen factual grounding, "
                            "and produce a complete final answer."
                        )
                else:
                    if style == "strict_final":
                        critique = (
                            "Generate an alternative improved final answer from scratch. "
                            "Preserve all constraints and exact output format. "
                            "Output only the final answer text."
                        )
                    else:
                        critique = (
                            "Generate an alternative improved solution path from scratch, while preserving strict "
                            "format and constraints."
                        )
                prompt_messages.append({
                    "role": "user",
                    "content": f"{critique}\n\nPrevious answer:\n{prior}",
                })
            elif candidate_idx > 0:
                if style == "strict_final":
                    prompt_messages.append({
                        "role": "user",
                        "content": (
                            "Provide an alternative final answer only. Keep all constraints and output format "
                            "requirements exactly."
                        ),
                    })
                else:
                    prompt_messages.append({
                        "role": "user",
                        "content": (
                            "Provide an alternative reasoning path and final answer. Keep all constraints and "
                            "format requirements."
                        ),
                    })

            content, metrics, error = generate_once(prompt_messages)
            if error:
                errors.append(str(error))

            score, step_summary = _candidate_quality_score(
                question = question,
                content = content,
                step_config = step_cfg,
                score_style = style,
            )
            row = {
                "round": round_idx + 1,
                "candidate_index": candidate_idx,
                "content": content,
                "score": score,
                "step_summary": step_summary,
                "metrics": metrics,
                "error": error or "",
            }
            round_rows.append(row)
            all_candidate_rows.append(row)

        if not round_rows:
            exit_reason = "no_candidates"
            break

        round_rows.sort(key = lambda row: row.get("score", 0.0), reverse = True)
        round_best = round_rows[0]
        round_best_score = float(round_best.get("score", 0.0) or 0.0)
        diagnostics_rounds.append({
            "round": round_idx + 1,
            "num_candidates": len(round_rows),
            "best_score": round_best_score,
        })

        if round_idx == 0:
            initial_score = round_best_score
            best = round_best
            best_score = round_best_score
            continue

        improvement = round_best_score - best_score
        if improvement > 0:
            best = round_best
            best_score = round_best_score

        if improvement < improve_min:
            exit_reason = "improvement_below_threshold"
            break

    if best is None and all_candidate_rows:
        all_candidate_rows.sort(key = lambda row: row.get("score", 0.0), reverse = True)
        best = all_candidate_rows[0]
        best_score = float(best.get("score", 0.0) or 0.0)
        initial_score = best_score

    if best is None:
        return {
            "content": "",
            "metrics": _aggregate_metrics([], selected_index = -1),
            "step_summary": _empty_step_summary(step_cfg),
            "error": "; ".join(errors),
            "recursion": {
                "rounds_planned": rounds_planned,
                "rounds_used": 0,
                "candidates_per_round": candidates_per_round,
                "candidate_calls": 0,
                "initial_score": 0.0,
                "final_score": 0.0,
                "improvement": 0.0,
                "improved": False,
                "exit_reason": "no_valid_candidate",
                "round_diagnostics": diagnostics_rounds,
            },
        }

    metrics_rows = [row.get("metrics", {}) for row in all_candidate_rows]
    selected_index = all_candidate_rows.index(best)
    aggregated_metrics = _aggregate_metrics(metrics_rows, selected_index = selected_index)
    recursion = {
        "rounds_planned": rounds_planned,
        "rounds_used": len(diagnostics_rounds),
        "candidates_per_round": candidates_per_round,
        "candidate_calls": len(all_candidate_rows),
        "initial_score": initial_score,
        "final_score": best_score,
        "improvement": best_score - initial_score,
        "improved": bool(best_score > initial_score),
        "exit_reason": exit_reason,
        "round_diagnostics": diagnostics_rounds,
    }
    return {
        "content": str(best.get("content", "")),
        "metrics": aggregated_metrics,
        "step_summary": best.get("step_summary", _empty_step_summary(step_cfg)),
        "error": "; ".join(error for error in errors if error),
        "recursion": recursion,
    }


def run_recursive_text_refinement(
    question: str,
    initial_answer: str,
    generate_candidate: Callable[[str], str],
    step_config: Optional[StepScoringConfig] = None,
    loop_config: Optional[ReasoningLoopConfig] = None,
) -> Dict[str, Any]:
    """
    Runtime text-only recursion helper.
    """
    cfg = loop_config or ReasoningLoopConfig.from_env()
    step_cfg = step_config or StepScoringConfig.from_env()

    initial_score, initial_step_summary = _candidate_quality_score(
        question = question,
        content = initial_answer,
        step_config = step_cfg,
    )
    best_answer = initial_answer
    best_score = initial_score
    best_step_summary = initial_step_summary
    exit_reason = "max_rounds_reached"
    rounds_used = 0

    for round_idx in range(max(cfg.max_rounds - 1, 0)):
        rounds_used += 1
        candidate_answers: List[str] = []
        for candidate_idx in range(max(1, cfg.candidates)):
            if candidate_idx == 0:
                critique = (
                    "Improve the answer by correcting weak reasoning steps and preserving all explicit constraints."
                )
            else:
                critique = "Provide a distinct alternative improved answer with the same required format."
            prompt = f"{critique}\n\nQuestion:\n{question}\n\nCurrent answer:\n{best_answer}"
            candidate_answers.append(generate_candidate(prompt))

        candidate_rows: List[Tuple[float, str, Dict[str, Any]]] = []
        for answer in candidate_answers:
            score, step_summary = _candidate_quality_score(
                question = question,
                content = answer,
                step_config = step_cfg,
            )
            candidate_rows.append((score, answer, step_summary))
        candidate_rows.sort(key = lambda row: row[0], reverse = True)

        if not candidate_rows:
            exit_reason = "no_candidates"
            break
        round_best_score, round_best_answer, round_best_step_summary = candidate_rows[0]
        improvement = round_best_score - best_score
        if improvement > 0:
            best_score = round_best_score
            best_answer = round_best_answer
            best_step_summary = round_best_step_summary
        if improvement < cfg.improve_min:
            exit_reason = "improvement_below_threshold"
            break

    return {
        "answer": best_answer,
        "step_summary": best_step_summary,
        "recursion": {
            "rounds_planned": max(1, cfg.max_rounds),
            "rounds_used": rounds_used + 1,
            "candidates_per_round": max(1, cfg.candidates),
            "initial_score": initial_score,
            "final_score": best_score,
            "improvement": best_score - initial_score,
            "improved": bool(best_score > initial_score),
            "exit_reason": exit_reason,
        },
    }
