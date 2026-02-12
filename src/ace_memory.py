"""
ACE (Agentic Context Engineering) Memory System

Implements the core memory system from the ACE paper:
- Structured bullets with metadata (ID, counters, content)
- Incremental delta updates (not monolithic rewrites)
- Grow-and-refine mechanism to prevent context collapse
- Semantic deduplication
- Context-scoped memory (Enhancement 1)
- Configurable retrieval weights (Enhancement 3)
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import math
import os

DEFAULT_MEMORY_STRENGTH = float(os.getenv("ACE_MEMORY_BASE_STRENGTH", "100.0"))

RELEVANCE_WEIGHT = float(os.getenv("ACE_WEIGHT_RELEVANCE", "0.25"))
STRENGTH_WEIGHT = float(os.getenv("ACE_WEIGHT_STRENGTH", "0.55"))
TYPE_WEIGHT = float(os.getenv("ACE_WEIGHT_TYPE", "0.20"))


@dataclass
class Bullet:
    """
    A single memory bullet with metadata and content.
    Represents a reusable strategy, lesson, or domain concept.
    """
    id: str
    content: str
    helpful_count: int = 0
    harmful_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    semantic_strength: float = 0.0
    episodic_strength: float = 0.0
    procedural_strength: float = 0.0
    semantic_last_access: Optional[str] = None
    episodic_last_access: Optional[str] = None
    procedural_last_access: Optional[str] = None
    semantic_access_index: Optional[int] = None
    episodic_access_index: Optional[int] = None
    procedural_access_index: Optional[int] = None
    learner_id: Optional[str] = None
    topic: Optional[str] = None
    concept: Optional[str] = None
    memory_type: Optional[str] = None
    ttl_days: Optional[int] = None
    content_hash: Optional[str] = None
    context_scope_id: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id(self.content)
        all_zero = (
            self.semantic_strength == 0.0
            and self.episodic_strength == 0.0
            and self.procedural_strength == 0.0
        )
        if not self.memory_type:
            self.memory_type = "semantic"
        else:
            self.memory_type = self.memory_type.lower()

        if all_zero:
            baseline = self._baseline_strength()
            if self.memory_type == "episodic":
                self.episodic_strength = baseline
                self.semantic_strength = 0.0
                self.procedural_strength = 0.0
            elif self.memory_type == "procedural":
                self.procedural_strength = baseline
                self.semantic_strength = 0.0
                self.episodic_strength = 0.0
            else:
                self.semantic_strength = baseline
                self.episodic_strength = 0.0
                self.procedural_strength = 0.0

        now_iso = datetime.now().isoformat()
        if self.semantic_strength > 0 and not self.semantic_last_access:
            self.semantic_last_access = self.last_used or now_iso
        if self.episodic_strength > 0 and not self.episodic_last_access:
            self.episodic_last_access = self.last_used or now_iso
        if self.procedural_strength > 0 and not self.procedural_last_access:
            self.procedural_last_access = self.last_used or now_iso
        self.content_hash = self.content_hash or self._compute_hash(self.content)

    @staticmethod
    def _generate_id(content: str) -> str:
        normalized = content.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    @staticmethod
    def _compute_hash(text: str) -> str:
        normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "tags": self.tags,
            "semantic_strength": self.semantic_strength,
            "episodic_strength": self.episodic_strength,
            "procedural_strength": self.procedural_strength,
            "semantic_last_access": self.semantic_last_access,
            "episodic_last_access": self.episodic_last_access,
            "procedural_last_access": self.procedural_last_access,
            "semantic_access_index": self.semantic_access_index,
            "episodic_access_index": self.episodic_access_index,
            "procedural_access_index": self.procedural_access_index,
            "learner_id": self.learner_id,
            "topic": self.topic,
            "concept": self.concept,
            "memory_type": self.memory_type,
            "ttl_days": self.ttl_days,
            "content_hash": self.content_hash,
            "context_scope_id": self.context_scope_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bullet":
        return cls(
            id=data.get("id", ""),
            content=data["content"],
            helpful_count=data.get("helpful_count", 0),
            harmful_count=data.get("harmful_count", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_used=data.get("last_used"),
            tags=data.get("tags", []),
            semantic_strength=float(data.get("semantic_strength", 0.0)),
            episodic_strength=float(data.get("episodic_strength", 0.0)),
            procedural_strength=float(data.get("procedural_strength", 0.0)),
            semantic_last_access=data.get("semantic_last_access"),
            episodic_last_access=data.get("episodic_last_access"),
            procedural_last_access=data.get("procedural_last_access"),
            semantic_access_index=data.get("semantic_access_index"),
            episodic_access_index=data.get("episodic_access_index"),
            procedural_access_index=data.get("procedural_access_index"),
            learner_id=data.get("learner_id"),
            topic=data.get("topic"),
            concept=data.get("concept"),
            memory_type=data.get("memory_type"),
            ttl_days=data.get("ttl_days"),
            content_hash=data.get("content_hash"),
            context_scope_id=data.get("context_scope_id"),
        )

    def score(self) -> float:
        total = self.helpful_count + self.harmful_count
        if total == 0:
            return 0.5
        return self.helpful_count / total

    def format_for_prompt(self) -> str:
        score_str = f"[+{self.helpful_count}/-{self.harmful_count}]"
        return f"{score_str} {self.content}"

    def _baseline_strength(self) -> float:
        delta = max(self.helpful_count - self.harmful_count, 0)
        return max(DEFAULT_MEMORY_STRENGTH, DEFAULT_MEMORY_STRENGTH + float(delta))


@dataclass
class DeltaUpdate:
    """
    Represents a delta update to the context.
    Contains new bullets or modifications to existing ones.
    """
    new_bullets: List[Bullet] = field(default_factory=list)
    update_bullets: Dict[str, Dict[str, int]] = field(default_factory=dict)
    remove_bullets: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ACEMemory:
    """
    ACE Memory System with evolving playbook of structured bullets.

    Key features:
    - Incremental delta updates (not monolithic rewrites)
    - Grow-and-refine to prevent context collapse
    - Semantic deduplication
    - Context-scoped memory isolation (Enhancement 1)
    - Configurable retrieval scoring weights (Enhancement 3)
    """

    def __init__(
        self,
        max_bullets: int = 100,
        dedup_threshold: float = 0.85,
        prune_threshold: float = 0.3,
        decay_rates: Optional[Dict[str, float]] = None,
        storage: Any = None,
    ):
        if storage is None:
            raise ValueError(
                "[ACE Memory] Error: storage adapter is required. "
                "Pass Neo4jMemoryStore instance. JSON fallback has been removed."
            )
        self._storage = storage
        self.max_bullets = max_bullets
        self.dedup_threshold = dedup_threshold
        self.prune_threshold = prune_threshold
        default_decay = {
            "semantic": 0.01,
            "episodic": 0.05,
            "procedural": 0.002,
        }
        if decay_rates:
            default_decay.update(decay_rates)
        self.decay_rates = {k: max(0.0, min(1.0, v)) for k, v in default_decay.items()}
        self.access_clock = 0

        self.bullets: Dict[str, Bullet] = {}
        self.categories: Dict[str, List[str]] = defaultdict(list)
        self.hash_index: Dict[str, Set[str]] = defaultdict(set)
        self._fresh_from_init = False
        self._loaded_once = False
        self._last_persist_status: Dict[str, Any] = {
            "ok": True,
            "attempts": 0,
            "retries": 0,
            "error": "",
            "timestamp": None,
        }

        self._load_memory()
        self._fresh_from_init = True
        self._loaded_once = True

    @staticmethod
    def _normalized_hash(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def consume_fresh_init_flag(self) -> bool:
        fresh = bool(getattr(self, "_fresh_from_init", False))
        self._fresh_from_init = False
        return fresh

    def is_loaded(self) -> bool:
        return bool(getattr(self, "_loaded_once", False))

    def _register_bullet(self, bullet: Bullet):
        bullet.content_hash = bullet.content_hash or self._normalized_hash(bullet.content)
        self.hash_index[bullet.content_hash].add(bullet.id)

    def _unregister_bullet(self, bullet_id: str):
        for hash_val, ids in list(self.hash_index.items()):
            if bullet_id in ids:
                ids.remove(bullet_id)
                if not ids:
                    self.hash_index.pop(hash_val, None)

    def _is_duplicate(self, bullet: Bullet) -> Optional[str]:
        candidate_hash = bullet.content_hash or self._normalized_hash(bullet.content)
        ids = self.hash_index.get(candidate_hash)
        if not ids:
            return None
        for existing_id in ids:
            existing = self.bullets.get(existing_id)
            if not existing:
                continue
            if (
                existing.memory_type == (bullet.memory_type or existing.memory_type)
                and existing.learner_id == bullet.learner_id
                and existing.topic == bullet.topic
            ):
                return existing_id
        return None

    def _component_score(
        self,
        strength: float,
        last_index: Optional[int],
        decay_key: str,
    ) -> float:
        if strength <= 0:
            return 0.0
        decay_rate = self.decay_rates.get(decay_key, 0.0)
        base = max(0.0, min(1.0, 1.0 - decay_rate))
        last_index = last_index if last_index is not None else self.access_clock
        t = max(self.access_clock - last_index, 0)
        return strength * math.pow(base, t)

    def _compute_score(self, bullet: Bullet, now: Optional[datetime] = None) -> float:
        semantic = self._component_score(
            bullet.semantic_strength, bullet.semantic_access_index, "semantic",
        )
        episodic = self._component_score(
            bullet.episodic_strength, bullet.episodic_access_index, "episodic",
        )
        procedural = self._component_score(
            bullet.procedural_strength, bullet.procedural_access_index, "procedural",
        )
        return semantic + episodic + procedural

    def _next_access_index(self) -> int:
        self.access_clock += 1
        return self.access_clock

    def _touch_bullets(
        self,
        bullets: List[Bullet],
        timestamp: Optional[datetime] = None,
        access_index: Optional[int] = None,
    ):
        if not bullets:
            return
        if access_index is None:
            access_index = self._next_access_index()
        else:
            if access_index > self.access_clock:
                self.access_clock = access_index
        ts = timestamp or datetime.now()
        iso_ts = ts.isoformat()
        for bullet in bullets:
            bullet.last_used = iso_ts
            if bullet.semantic_strength > 0:
                bullet.semantic_last_access = iso_ts
                bullet.semantic_access_index = access_index
            if bullet.episodic_strength > 0:
                bullet.episodic_last_access = iso_ts
                bullet.episodic_access_index = access_index
            if bullet.procedural_strength > 0:
                bullet.procedural_last_access = iso_ts
                bullet.procedural_access_index = access_index

    def _touch_bullet(
        self,
        bullet: Bullet,
        timestamp: Optional[datetime] = None,
        access_index: Optional[int] = None,
    ):
        self._touch_bullets([bullet], timestamp=timestamp, access_index=access_index)

    @staticmethod
    def _sync_strengths(bullet: Bullet):
        valid = {"semantic", "episodic", "procedural"}
        mt = (bullet.memory_type or "semantic").lower()
        if mt not in valid:
            mt = "semantic"
        bullet.memory_type = mt
        baseline = bullet._baseline_strength()
        if mt == "semantic":
            bullet.semantic_strength = max(float(bullet.semantic_strength or 0.0), baseline)
            bullet.episodic_strength = 0.0
            bullet.procedural_strength = 0.0
        elif mt == "episodic":
            bullet.episodic_strength = max(float(bullet.episodic_strength or 0.0), baseline)
            bullet.semantic_strength = 0.0
            bullet.procedural_strength = 0.0
        else:
            bullet.procedural_strength = max(float(bullet.procedural_strength or 0.0), baseline)
            bullet.semantic_strength = 0.0
            bullet.episodic_strength = 0.0

    @staticmethod
    def _ensure_memory_tags(bullet: Bullet):
        valid = {"semantic", "episodic", "procedural"}
        tags = [t for t in bullet.tags if t.lower() not in valid]
        if bullet.memory_type:
            tags.insert(0, bullet.memory_type)
        seen = set()
        bullet.tags = [t for t in tags if not (t.lower() in seen or seen.add(t.lower()))]

    def _sync_categories(self, bullet: Bullet):
        for tag in bullet.tags:
            if bullet.id not in self.categories[tag]:
                self.categories[tag].append(bullet.id)

    def _normalise_bullet(
        self,
        bullet: Bullet,
        access_index: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ):
        tag_set = {tag.lower() for tag in bullet.tags}
        valid = {"semantic", "episodic", "procedural"}
        if not bullet.memory_type or bullet.memory_type.lower() not in valid:
            if "episodic" in tag_set:
                bullet.memory_type = "episodic"
            elif "procedural" in tag_set:
                bullet.memory_type = "procedural"
            else:
                bullet.memory_type = "semantic"
        bullet.memory_type = bullet.memory_type.lower()
        self._sync_strengths(bullet)
        self._ensure_memory_tags(bullet)
        self._touch_bullet(bullet, timestamp=timestamp, access_index=access_index)
        bullet.content_hash = bullet.content_hash or self._normalized_hash(bullet.content)

    def _finalize_bullet(self, bullet: Bullet):
        self._sync_strengths(bullet)
        self._ensure_memory_tags(bullet)
        bullet.content_hash = self._normalized_hash(bullet.content)

    @staticmethod
    def _parse_created_at(bullet: Bullet) -> datetime:
        try:
            return datetime.fromisoformat(bullet.created_at)
        except Exception:
            return datetime.fromtimestamp(0)

    def _select_canonical_bullet(self, a: Bullet, b: Bullet) -> Tuple[Bullet, Bullet]:
        score_a = a.helpful_count - a.harmful_count
        score_b = b.helpful_count - b.harmful_count
        if score_a != score_b:
            return (a, b) if score_a > score_b else (b, a)
        created_a = self._parse_created_at(a)
        created_b = self._parse_created_at(b)
        if created_a >= created_b:
            return a, b
        return b, a

    def _merge_bullet_into(self, keep: Bullet, drop: Bullet):
        keep.helpful_count += drop.helpful_count
        keep.harmful_count += drop.harmful_count
        if not keep.learner_id:
            keep.learner_id = drop.learner_id
        if not keep.topic:
            keep.topic = drop.topic
        if not keep.concept:
            keep.concept = drop.concept
        if not keep.memory_type and drop.memory_type:
            keep.memory_type = drop.memory_type
        keep.tags = list(dict.fromkeys(keep.tags + drop.tags))
        self._finalize_bullet(keep)
        self._sync_categories(keep)
        self._register_bullet(keep)

    def _merge_or_add_bullet(
        self,
        bullet: Bullet,
        access_index: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[Bullet, bool]:
        self._finalize_bullet(bullet)
        duplicate_id = self._is_duplicate(bullet)
        existing = None
        if duplicate_id:
            existing = self.bullets.get(duplicate_id)
        else:
            candidates = list(self.hash_index.get(bullet.content_hash, []))
            if candidates:
                existing = self.bullets.get(candidates[0])
        if not existing:
            existing, score = self.find_similar_bullet(
                bullet.content,
                learner_id=bullet.learner_id,
                topic=bullet.topic,
                threshold=float(os.getenv("ACE_MEMORY_SIMILARITY_MERGE", "0.9")),
                return_score=True,
            )
        else:
            score = 1.0

        if existing:
            self._merge_bullet_into(existing, bullet)
            self._touch_bullet(existing, timestamp=timestamp, access_index=access_index)
            return existing, False

        self._normalise_bullet(bullet, access_index=access_index, timestamp=timestamp)
        self.bullets[bullet.id] = bullet
        self._sync_categories(bullet)
        self._register_bullet(bullet)
        return bullet, True

    def find_similar_bullet(
        self,
        content: str,
        learner_id: Optional[str] = None,
        topic: Optional[str] = None,
        threshold: float = 0.9,
        return_score: bool = False,
    ):
        if not content:
            return (None, 0.0) if return_score else None
        best = None
        best_score = threshold
        for bullet in self.bullets.values():
            if learner_id and bullet.learner_id and bullet.learner_id != learner_id:
                continue
            if topic and bullet.topic and bullet.topic != topic:
                continue
            score = self._text_similarity(content, bullet.content)
            if score >= best_score:
                best = bullet
                best_score = score
        if return_score:
            return best, (best_score if best is not None else 0.0)
        return best

    def _populate_from_data(self, data: Dict[str, Any]):
        self.bullets.clear()
        self.categories = defaultdict(list)
        self.hash_index = defaultdict(set)

        for bullet_data in data.get("bullets", []):
            bullet = Bullet.from_dict(bullet_data)
            if bullet.semantic_strength > 0 and bullet.semantic_access_index is None:
                bullet.semantic_access_index = 0
            if bullet.episodic_strength > 0 and bullet.episodic_access_index is None:
                bullet.episodic_access_index = 0
            if bullet.procedural_strength > 0 and bullet.procedural_access_index is None:
                bullet.procedural_access_index = 0
            self._ensure_memory_tags(bullet)
            self.bullets[bullet.id] = bullet
            for tag in bullet.tags:
                self.categories[tag].append(bullet.id)
            self._register_bullet(bullet)

        self.access_clock = int(data.get("access_clock", len(self.bullets)))
        for bullet in self.bullets.values():
            if bullet.semantic_strength > 0 and (bullet.semantic_access_index is None or bullet.semantic_access_index == 0):
                bullet.semantic_access_index = self.access_clock
            if bullet.episodic_strength > 0 and (bullet.episodic_access_index is None or bullet.episodic_access_index == 0):
                bullet.episodic_access_index = self.access_clock
            if bullet.procedural_strength > 0 and (bullet.procedural_access_index is None or bullet.procedural_access_index == 0):
                bullet.procedural_access_index = self.access_clock

    def _load_memory(self):
        if self._storage:
            try:
                stored = self._storage.load()
            except Exception as exc:
                print(f"[ACE Memory] Warning: Storage load failed: {exc}", flush=True)
            else:
                if stored:
                    self._populate_from_data(stored)
                    learner = getattr(self._storage, "learner_id", "unknown")
                    print(
                        f"[ACE Memory] Loaded {len(self.bullets)} bullets from Neo4j for learner={learner}",
                        flush=True,
                    )
                else:
                    learner = getattr(self._storage, "learner_id", "unknown")
                    print(
                        f"[ACE Memory] No existing Neo4j memory for learner={learner}; starting fresh",
                        flush=True,
                    )
                self._loaded_once = True
                return

    def reload_from_storage(self):
        try:
            stored = self._storage.load()
        except Exception as exc:
            print(f"[ACE Memory] ERROR: Storage reload failed: {exc}", flush=True)
            raise

        if not stored:
            learner = getattr(self._storage, "learner_id", "unknown")
            print(
                f"[ACE Memory] No existing Neo4j memory for learner={learner}; starting fresh",
                flush=True,
            )
            return

        self._populate_from_data(stored)
        learner = getattr(self._storage, "learner_id", "unknown")
        print(
            f"[ACE Memory] Reloaded {len(self.bullets)} bullets from Neo4j for learner={learner}",
            flush=True,
        )
        self._loaded_once = True
        self._fresh_from_init = False

    def get_last_persist_status(self) -> Dict[str, Any]:
        return dict(self._last_persist_status)

    def _save_memory(self, raise_on_failure: bool = False) -> Dict[str, Any]:
        data = {
            "bullets": [bullet.to_dict() for bullet in self.bullets.values()],
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "access_clock": self.access_clock,
        }
        try:
            save_result = self._storage.save(data)
        except Exception as exc:
            save_result = {
                "ok": False,
                "attempts": 1,
                "retries": 0,
                "error": str(exc),
            }

        status: Dict[str, Any]
        if isinstance(save_result, dict):
            status = {
                "ok": bool(save_result.get("ok", False)),
                "attempts": int(save_result.get("attempts", 1)),
                "retries": int(save_result.get("retries", 0)),
                "error": str(save_result.get("error", "")),
                "timestamp": datetime.now().isoformat(),
            }
        else:
            status = {
                "ok": bool(save_result is not False),
                "attempts": 1,
                "retries": 0,
                "error": "" if save_result is not False else "save_returned_false",
                "timestamp": datetime.now().isoformat(),
            }

        self._last_persist_status = status
        if not status["ok"]:
            print(
                "[ACE Memory] Warning: Failed to persist memory state to Neo4j "
                f"(retries={status['retries']}, error={status['error']})",
                flush = True,
            )
            if raise_on_failure:
                raise RuntimeError(status["error"] or "neo4j_save_failed")
        return status

    def apply_delta(self, delta: DeltaUpdate):
        has_changes = bool(
            delta.new_bullets or delta.update_bullets or delta.remove_bullets
        )
        event_index = self._next_access_index() if has_changes else None
        event_ts = datetime.now() if has_changes else None

        for bullet in delta.new_bullets:
            self._merge_or_add_bullet(
                bullet, access_index=event_index, timestamp=event_ts,
            )

        for bullet_id, updates in delta.update_bullets.items():
            if bullet_id in self.bullets:
                bullet = self.bullets[bullet_id]
                bullet.helpful_count += updates.get("helpful", 0)
                bullet.harmful_count += updates.get("harmful", 0)
                self._finalize_bullet(bullet)
                self._touch_bullet(bullet, timestamp=event_ts, access_index=event_index)
                self._register_bullet(bullet)

        for bullet_id in delta.remove_bullets:
            if bullet_id in self.bullets:
                bullet = self.bullets.pop(bullet_id)
                for tag in bullet.tags:
                    if bullet_id in self.categories[tag]:
                        self.categories[tag].remove(bullet_id)
                self._unregister_bullet(bullet_id)

        self._refine()
        return self._save_memory()

    def _refine(self):
        self._deduplicate_bullets()
        if len(self.bullets) > self.max_bullets:
            self._prune_bullets()

    def _deduplicate_bullets(self):
        bullets_list = list(self.bullets.values())
        to_remove = set()

        for i in range(len(bullets_list)):
            if bullets_list[i].id in to_remove:
                continue
            for j in range(i + 1, len(bullets_list)):
                if bullets_list[j].id in to_remove:
                    continue
                similarity = self._text_similarity(
                    bullets_list[i].content, bullets_list[j].content
                )
                if similarity > self.dedup_threshold:
                    keep, drop = self._select_canonical_bullet(bullets_list[i], bullets_list[j])
                    self._merge_bullet_into(keep, drop)
                    to_remove.add(drop.id)
                    if keep is bullets_list[j]:
                        bullets_list[i], bullets_list[j] = bullets_list[j], bullets_list[i]

        for bullet_id in to_remove:
            if bullet_id in self.bullets:
                bullet = self.bullets.pop(bullet_id)
                for tag in bullet.tags:
                    if bullet_id in self.categories[tag]:
                        self.categories[tag].remove(bullet_id)
                self._unregister_bullet(bullet_id)

    def _text_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)

    def _prune_bullets(self):
        now = datetime.now()
        bullets_list = sorted(
            self.bullets.values(),
            key=lambda b: (self._compute_score(b, now), b.helpful_count),
            reverse=True,
        )
        to_keep = set(b.id for b in bullets_list[:self.max_bullets])
        to_remove = set(self.bullets.keys()) - to_keep
        for bullet_id in to_remove:
            bullet = self.bullets.pop(bullet_id)
            for tag in bullet.tags:
                if bullet_id in self.categories[tag]:
                    self.categories[tag].remove(bullet_id)
            self._unregister_bullet(bullet_id)

    def retrieve_relevant_bullets(
        self,
        query: str,
        top_k: int = 10,
        tags: Optional[List[str]] = None,
        min_score: float = 0.0,
        learner_id: Optional[str] = None,
        topic: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        facets: Optional[Dict[str, Any]] = None,
        context_scope_id: Optional[str] = None,
    ) -> List[Bullet]:
        facets = facets or {}

        if tags:
            candidate_ids = set()
            for tag in tags:
                candidate_ids.update(self.categories.get(tag, []))
            candidates = [self.bullets[bid] for bid in candidate_ids if bid in self.bullets]
        else:
            candidates = list(self.bullets.values())

        if memory_types:
            allowed = {mt.lower() for mt in memory_types}
            candidates = [b for b in candidates if (b.memory_type or "semantic") in allowed]

        if learner_id:
            candidates = [b for b in candidates if not b.learner_id or b.learner_id == learner_id]

        if topic:
            candidates = [b for b in candidates if not b.topic or b.topic == topic]

        if context_scope_id:
            candidates = [b for b in candidates if not b.context_scope_id or b.context_scope_id == context_scope_id]

        score_cache = {b.id: self._compute_score(b) for b in candidates}
        candidates = [b for b in candidates if score_cache.get(b.id, 0.0) >= min_score]

        if not candidates:
            return []

        query_terms: List[str] = []
        if query:
            query_terms.append(query)
        persona = facets.get("persona_request")
        if persona:
            query_terms.append(str(persona))
        if facets.get("next_step_flag"):
            query_terms.append("next step")

        query_text = " ".join(term for term in query_terms if term).strip()

        memory_weight = {"procedural": 1.0, "episodic": 0.7, "semantic": 0.4}

        scored_bullets = []
        for bullet in candidates:
            mt = (bullet.memory_type or "semantic").lower()
            bullet_tags = {t.lower() for t in bullet.tags}
            base_score = score_cache.get(bullet.id, 0.0)
            normalized_strength = base_score / max(DEFAULT_MEMORY_STRENGTH, 1.0)

            relevance = self._text_similarity(query_text, bullet.content) if query_text else 0.0
            type_priority = memory_weight.get(mt, 0.3)

            bonus = 0.0
            if facets.get("needs_visual") and ("visual" in bullet_tags or "diagram" in bullet_tags):
                bonus += 0.2
            if persona and persona in bullet_tags:
                bonus += 0.1

            combined_score = (
                RELEVANCE_WEIGHT * relevance +
                STRENGTH_WEIGHT * normalized_strength +
                TYPE_WEIGHT * type_priority +
                bonus
            )
            scored_bullets.append((combined_score, bullet))

        scored_bullets.sort(key=lambda x: x[0], reverse=True)
        top_bullets = [bullet for _, bullet in scored_bullets[:top_k]]
        self._touch_bullets(top_bullets)
        return top_bullets

    def format_context(
        self,
        query: str,
        top_k: int = 10,
        tags: Optional[List[str]] = None,
        learner_id: Optional[str] = None,
        topic: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
    ) -> str:
        bullets = self.retrieve_relevant_bullets(
            query, top_k=top_k, tags=tags, learner_id=learner_id, topic=topic, memory_types=memory_types,
        )
        if not bullets:
            return ""
        context_parts = ["=== Relevant Strategies and Lessons ==="]
        for i, bullet in enumerate(bullets, 1):
            context_parts.append(f"{i}. {bullet.format_for_prompt()}")
        context_parts.append("=" * 50)
        return "\n".join(context_parts)

    def get_statistics(self) -> Dict[str, Any]:
        if not self.bullets:
            return {"total_bullets": 0, "avg_score": 0.0, "categories": {}}
        now = datetime.now()
        scores = [self._compute_score(b, now) for b in self.bullets.values()]
        return {
            "total_bullets": len(self.bullets),
            "avg_score": sum(scores) / len(scores),
            "avg_helpful": sum(b.helpful_count for b in self.bullets.values()) / len(self.bullets),
            "avg_harmful": sum(b.harmful_count for b in self.bullets.values()) / len(self.bullets),
            "categories": {tag: len(ids) for tag, ids in self.categories.items()},
        }

    def clear(self):
        self.bullets.clear()
        self.categories.clear()
        self._save_memory()
