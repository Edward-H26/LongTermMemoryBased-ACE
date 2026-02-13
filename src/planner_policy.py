"""
Planner policy utilities for ACE runtime and benchmark flows.

Implements a lightweight contextual bandit controller that:
- selects planner actions with epsilon exploration and UCB exploitation
- applies clipped online reward updates
- persists policy state for resume and post-run replay
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def default_planner_state_path(role: str = "runtime", output_dir: Optional[str] = None) -> str:
    role_name = (role or "runtime").strip().lower()
    safe_role = role_name.replace(" ", "_")
    base_dir = output_dir or os.path.join("benchmark", "results", "v5")
    return os.path.join(base_dir, f"planner_policy_{safe_role}_v5.json")


@dataclass
class ActionStats:
    pulls: int = 0
    updates: int = 0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0
    last_reward: float = 0.0
    last_confidence: float = 0.0
    last_updated_at: str = ""

    def mean_reward(self) -> float:
        if self.updates <= 0:
            return 0.5
        return self.reward_sum / float(self.updates)

    def variance(self) -> float:
        if self.updates <= 1:
            return 0.0
        mean = self.mean_reward()
        return max((self.reward_sq_sum / float(self.updates)) - (mean * mean), 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pulls": int(self.pulls),
            "updates": int(self.updates),
            "reward_sum": float(self.reward_sum),
            "reward_sq_sum": float(self.reward_sq_sum),
            "last_reward": float(self.last_reward),
            "last_confidence": float(self.last_confidence),
            "last_updated_at": self.last_updated_at,
            "mean_reward": self.mean_reward(),
            "variance": self.variance(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionStats":
        return cls(
            pulls = _safe_int(data.get("pulls"), 0),
            updates = _safe_int(data.get("updates"), 0),
            reward_sum = _safe_float(data.get("reward_sum"), 0.0),
            reward_sq_sum = _safe_float(data.get("reward_sq_sum"), 0.0),
            last_reward = _safe_float(data.get("last_reward"), 0.0),
            last_confidence = _safe_float(data.get("last_confidence"), 0.0),
            last_updated_at = str(data.get("last_updated_at", "")),
        )


class PlannerPolicy:
    """
    Contextual bandit planner policy with thread-safe online updates.
    """

    def __init__(
        self,
        actions: List[str],
        state_path: Optional[str] = None,
        epsilon: Optional[float] = None,
        ucb_c: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        if not actions:
            raise ValueError("PlannerPolicy requires at least one action.")
        self.actions = sorted(set(str(action).strip() for action in actions if str(action).strip()))
        if not self.actions:
            raise ValueError("PlannerPolicy actions are empty after normalization.")

        self.state_path = state_path or os.getenv("ACE_PLANNER_STATE_PATH", "")
        self.epsilon = _clamp(
            _safe_float(
                epsilon if epsilon is not None else os.getenv("ACE_PLANNER_EPSILON", "0.08"),
                0.08,
            ),
            0.0,
            1.0,
        )
        self.ucb_c = max(
            _safe_float(
                ucb_c if ucb_c is not None else os.getenv("ACE_PLANNER_UCB_C", "1.10"),
                1.10,
            ),
            0.0,
        )
        self.policy_type = str(os.getenv("ACE_PLANNER_POLICY", "bandit")).strip().lower()
        self._rng = random.Random(
            _safe_int(seed if seed is not None else os.getenv("ACE_PLANNER_SEED", "42"), 42)
        )
        self._lock = threading.Lock()
        self.total_pulls = 0
        self.total_updates = 0
        self.action_stats: Dict[str, ActionStats] = {action: ActionStats() for action in self.actions}
        self.last_choice: Dict[str, Any] = {}
        self._load_state()

    def _state_payload(self) -> Dict[str, Any]:
        return {
            "version": "v5_bandit_1",
            "policy_type": self.policy_type,
            "epsilon": float(self.epsilon),
            "ucb_c": float(self.ucb_c),
            "total_pulls": int(self.total_pulls),
            "total_updates": int(self.total_updates),
            "actions": {action: self.action_stats[action].to_dict() for action in self.actions},
            "updated_at": _utc_now_iso(),
        }

    def _load_state(self) -> None:
        if not self.state_path:
            return
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding = "utf-8") as f:
                payload = json.load(f)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        actions_payload = payload.get("actions", {})
        if not isinstance(actions_payload, dict):
            return
        with self._lock:
            self.total_pulls = _safe_int(payload.get("total_pulls"), 0)
            self.total_updates = _safe_int(payload.get("total_updates"), 0)
            for action in self.actions:
                raw = actions_payload.get(action)
                if isinstance(raw, dict):
                    self.action_stats[action] = ActionStats.from_dict(raw)

    def _save_state(self) -> None:
        if not self.state_path:
            return
        payload = self._state_payload()
        os.makedirs(os.path.dirname(self.state_path) if os.path.dirname(self.state_path) else ".", exist_ok = True)
        with open(self.state_path, "w", encoding = "utf-8") as f:
            json.dump(payload, f, indent = 2, ensure_ascii = False)

    def _ucb_score(self, action: str) -> float:
        stats = self.action_stats[action]
        mean = stats.mean_reward()
        bonus = self.ucb_c * math.sqrt(
            math.log(float(self.total_pulls + len(self.actions) + 1)) / float(stats.pulls + 1)
        )
        return mean + bonus

    def choose_action(
        self,
        features: Optional[Dict[str, Any]] = None,
        allowed_actions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.policy_type != "bandit":
            fallback = self.actions[0]
            return {
                "action_id": fallback,
                "explore": False,
                "scores": {fallback: 0.0},
                "features": features or {},
            }

        with self._lock:
            candidates = self.actions
            if allowed_actions:
                allowed = [a for a in allowed_actions if a in self.action_stats]
                if allowed:
                    candidates = sorted(set(allowed))
            explore = self._rng.random() < self.epsilon
            scores = {action: self._ucb_score(action) for action in candidates}
            if explore:
                action_id = self._rng.choice(candidates)
            else:
                action_id = sorted(
                    candidates,
                    key = lambda action: (scores.get(action, 0.0), action),
                    reverse = True,
                )[0]
            self.total_pulls += 1
            self.action_stats[action_id].pulls += 1
            choice = {
                "action_id": action_id,
                "explore": bool(explore),
                "scores": scores,
                "features": features or {},
                "chosen_at": _utc_now_iso(),
            }
            self.last_choice = dict(choice)
            self._save_state()
            return choice

    def update(
        self,
        action_id: str,
        reward: float,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if action_id not in self.action_stats:
                return {
                    "updated": False,
                    "error": f"unknown_action:{action_id}",
                }
            clipped_reward = _clamp(_safe_float(reward, 0.0), 0.0, 1.0)
            clipped_confidence = _clamp(_safe_float(confidence, 1.0), 0.0, 1.0)
            weighted_reward = clipped_reward * clipped_confidence
            stats = self.action_stats[action_id]
            stats.updates += 1
            stats.reward_sum += weighted_reward
            stats.reward_sq_sum += weighted_reward * weighted_reward
            stats.last_reward = clipped_reward
            stats.last_confidence = clipped_confidence
            stats.last_updated_at = _utc_now_iso()
            self.total_updates += 1
            self._save_state()
            return {
                "updated": True,
                "action_id": action_id,
                "reward": clipped_reward,
                "confidence": clipped_confidence,
                "weighted_reward": weighted_reward,
                "total_updates": int(self.total_updates),
                "action_updates": int(stats.updates),
                "action_mean_reward": float(stats.mean_reward()),
                "metadata": metadata or {},
            }

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            action_rows: Dict[str, Any] = {}
            for action in self.actions:
                stats = self.action_stats[action]
                action_rows[action] = stats.to_dict()
                action_rows[action]["ucb_score"] = self._ucb_score(action)
            return {
                "policy_type": self.policy_type,
                "epsilon": float(self.epsilon),
                "ucb_c": float(self.ucb_c),
                "total_pulls": int(self.total_pulls),
                "total_updates": int(self.total_updates),
                "actions": action_rows,
                "state_path": self.state_path,
            }


def compute_shaped_reward(
    step_score: float,
    output_valid: bool,
    quality_gate_applied: bool,
    recursion_improved: bool = False,
    terminal_score: Optional[float] = None,
    step_confidence: float = 0.7,
) -> Dict[str, float]:
    clipped_step = _clamp(_safe_float(step_score, 0.0), 0.0, 1.0)
    clipped_confidence = _clamp(_safe_float(step_confidence, 0.7), 0.0, 1.0)
    output_term = 1.0 if output_valid else 0.0
    gate_term = 1.0 if quality_gate_applied else 0.0
    recursion_term = 1.0 if recursion_improved else 0.0

    proxy_reward = (
        0.55 * clipped_step +
        0.20 * output_term +
        0.15 * gate_term +
        0.10 * recursion_term
    )
    proxy_reward = _clamp(proxy_reward, 0.0, 1.0)

    if terminal_score is None:
        final_reward = proxy_reward
    else:
        final_reward = _clamp(0.60 * proxy_reward + 0.40 * _clamp(float(terminal_score), 0.0, 1.0), 0.0, 1.0)

    return {
        "proxy_reward": proxy_reward,
        "final_reward": final_reward,
        "confidence": clipped_confidence,
    }
