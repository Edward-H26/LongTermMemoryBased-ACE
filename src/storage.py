"""
Neo4j-backed persistence for ACE memory.

Stores each learner's playbook as a single JSON blob attached to an
AceMemoryState node to keep the schema simple while enabling per-user
isolation.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError, SessionExpired

_DRIVER = None
_DRIVER_LOCK = threading.Lock()


def _get_driver():
    global _DRIVER
    if _DRIVER is not None:
        return _DRIVER

    uri = os.getenv("NEO4J_URI") or os.getenv("NEXT_PUBLIC_NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEXT_PUBLIC_NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD") or os.getenv("NEXT_PUBLIC_NEO4J_PASSWORD")

    if not uri or not user or not password:
        raise RuntimeError(
            "Neo4j credentials are not configured. "
            "Set NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD (or NEXT_PUBLIC_*)."
        )

    with _DRIVER_LOCK:
        if _DRIVER is None:
            _DRIVER = GraphDatabase.driver(uri, auth=(user, password))
    return _DRIVER


def _get_database() -> Optional[str]:
    return os.getenv("NEO4J_DATABASE") or None


def _safe_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed >= 0 else default


def _safe_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if parsed >= 0.0 else default


def _safe_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _reset_driver() -> None:
    global _DRIVER
    with _DRIVER_LOCK:
        if _DRIVER is not None:
            try:
                _DRIVER.close()
            except Exception:
                pass
            _DRIVER = None


class Neo4jMemoryStore:
    """Persist ACE memory state for a specific learner in Neo4j."""

    def __init__(self, learner_id: str):
        if not learner_id:
            raise ValueError("learner_id is required for Neo4jMemoryStore")
        self.learner_id = learner_id
        self._database = _get_database()
        self._retry_max = _safe_env_int("ACE_NEO4J_RETRY_MAX", 2)
        self._retry_backoff_sec = _safe_env_float("ACE_NEO4J_RETRY_BACKOFF_SEC", 1.0)
        self._reconnect_on_session_expired = _safe_env_bool(
            "ACE_NEO4J_RECONNECT_ON_SESSION_EXPIRED",
            True,
        )

    def _run_with_retry(
        self,
        operation_name: str,
        fn,
    ) -> Dict[str, Any]:
        max_attempts = self._retry_max + 1
        last_error = ""
        for attempt in range(max_attempts):
            try:
                value = fn()
                return {
                    "ok": True,
                    "value": value,
                    "attempts": attempt + 1,
                    "retries": attempt,
                    "error": "",
                }
            except (Neo4jError, DriverError) as exc:
                last_error = str(exc)
                is_last = attempt >= max_attempts - 1
                should_reconnect = (
                    self._reconnect_on_session_expired
                    and isinstance(exc, SessionExpired)
                )
                if should_reconnect:
                    _reset_driver()
                if is_last:
                    print(
                        f"[ACE Memory] Warning: Neo4j {operation_name} failed for learner={self.learner_id} "
                        f"after {attempt + 1} attempt(s): {exc}",
                        flush = True,
                    )
                    return {
                        "ok": False,
                        "value": None,
                        "attempts": attempt + 1,
                        "retries": attempt,
                        "error": last_error,
                    }
                sleep_sec = self._retry_backoff_sec * (2 ** attempt)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
            except Exception as exc:
                last_error = str(exc)
                print(
                    f"[ACE Memory] Warning: Neo4j {operation_name} failed for learner={self.learner_id}: {exc}",
                    flush = True,
                )
                return {
                    "ok": False,
                    "value": None,
                    "attempts": attempt + 1,
                    "retries": attempt,
                    "error": last_error,
                }
        return {
            "ok": False,
            "value": None,
            "attempts": max_attempts,
            "retries": max_attempts - 1,
            "error": last_error,
        }

    def load(self) -> Optional[Dict[str, Any]]:
        def _op() -> Optional[Dict[str, Any]]:
            driver = _get_driver()
            with driver.session(database=self._database) as session:
                record = session.run(
                    """
                    MATCH (u:User {id: $userId})
                    MERGE (u)-[:HAS_ACE_MEMORY]->(m:AceMemoryState)
                    ON CREATE SET
                        m.id = coalesce($memoryId, randomUUID()),
                        m.memory_json = $emptyPayload,
                        m.access_clock = 0,
                        m.created_at = datetime(),
                        m.updated_at = datetime()
                    RETURN m.memory_json AS memory_json,
                           m.access_clock AS access_clock
                    """,
                    {
                        "userId": self.learner_id,
                        "memoryId": None,
                        "emptyPayload": json.dumps(
                            {"bullets": [], "access_clock": 0}, ensure_ascii=False
                        ),
                    },
                ).single()
                if not record:
                    return None
                raw = record.get("memory_json")
                if not raw:
                    return None
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    return None
                access_clock = record.get("access_clock")
                if access_clock is not None:
                    try:
                        access_clock = int(access_clock)
                    except (TypeError, ValueError):
                        pass
                if access_clock is not None and "access_clock" not in data:
                    data["access_clock"] = access_clock
                return data
        result = self._run_with_retry("load", _op)
        if not result.get("ok"):
            return None
        return result.get("value")

    def save(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = json.dumps(data, ensure_ascii=False)
        access_clock = int(data.get("access_clock", 0))
        def _op() -> None:
            driver = _get_driver()
            with driver.session(database=self._database) as session:
                session.run(
                    """
                    MERGE (u:User {id: $userId})
                    ON CREATE SET u.created_at = datetime()
                    MERGE (u)-[:HAS_ACE_MEMORY]->(m:AceMemoryState)
                    ON CREATE SET
                        m.id = randomUUID(),
                        m.created_at = datetime()
                    SET m.memory_json = $memory_json,
                        m.access_clock = $access_clock,
                        m.updated_at = datetime()
                    """,
                    {
                        "userId": self.learner_id,
                        "memory_json": payload,
                        "access_clock": access_clock,
                    },
                )
            return None
        result = self._run_with_retry("save", _op)
        return {
            "ok": bool(result.get("ok", False)),
            "attempts": int(result.get("attempts", 1)),
            "retries": int(result.get("retries", 0)),
            "error": str(result.get("error", "")),
        }
