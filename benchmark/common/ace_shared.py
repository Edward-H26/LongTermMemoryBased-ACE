"""ACE-specific shared constants and helpers.

Contains ``META_STRATEGY_SEEDS``, guidance formatting/injection,
memory seeding, lesson-transferability filtering, bullet merging,
and failure-result construction — all of which are duplicated across
the v2–v5 ACE inference scripts.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.ace_memory import ACEMemory, Bullet


# ---------------------------------------------------------------------------
# Meta-strategy seed bullets (injected into empty memories)
# ---------------------------------------------------------------------------

META_STRATEGY_SEEDS: List[Bullet] = [
    Bullet(
        id = "",
        content = (
            "Before answering, re-read all constraints, rules, and "
            "procedures explicitly stated in the provided context. "
            "Base your answer entirely on the context, not prior knowledge."
        ),
        tags = ["meta_strategy", "procedural"],
        memory_type = "procedural",
        helpful_count = 5,
    ),
    Bullet(
        id = "",
        content = (
            "Follow the exact output format and structure specified in "
            "the system prompt and context. Do not add extraneous text "
            "or deviate from the requested format."
        ),
        tags = ["meta_strategy", "procedural"],
        memory_type = "procedural",
        helpful_count = 5,
    ),
    Bullet(
        id = "",
        content = (
            "Do not rely on pre-trained knowledge when the context "
            "provides explicit rules, definitions, or information that "
            "may differ from common knowledge. Adhere strictly to what "
            "the context states."
        ),
        tags = ["meta_strategy", "procedural"],
        memory_type = "procedural",
        helpful_count = 5,
    ),
]


# ---------------------------------------------------------------------------
# Guidance formatting / injection
# ---------------------------------------------------------------------------

def format_guidance(bullets: List[Bullet]) -> str:
    """Format retrieved bullets into a labelled guidance block."""
    if not bullets:
        return ""
    parts = [
        "=== Guidance from Prior Experience ===",
        "Based on lessons learned from similar tasks:",
    ]
    for idx, bullet in enumerate(bullets, 1):
        parts.append(f"{idx}. {bullet.format_for_prompt()}")
    parts.append("===")
    return "\n".join(parts)


def inject_guidance(
    messages: List[Dict[str, str]],
    guidance: str,
) -> List[Dict[str, str]]:
    """Prepend *guidance* to the last user message in *messages*.

    Returns a new list (does not mutate the original).
    """
    if not guidance:
        return messages

    enriched = [dict(m) for m in messages]
    last_user_idx = None
    for idx in range(len(enriched) - 1, -1, -1):
        if enriched[idx].get("role") == "user":
            last_user_idx = idx
            break

    if last_user_idx is not None:
        enriched[last_user_idx] = dict(enriched[last_user_idx])
        original = enriched[last_user_idx].get("content", "")
        enriched[last_user_idx]["content"] = f"{guidance}\n\n{original}"

    return enriched


# ---------------------------------------------------------------------------
# Memory seeding
# ---------------------------------------------------------------------------

def seed_memory_if_empty(
    memory: ACEMemory,
    context_scope_id: Optional[str],
) -> None:
    """Inject ``META_STRATEGY_SEEDS`` into *memory* when it has no bullets."""
    if memory.bullets:
        return
    for seed in META_STRATEGY_SEEDS:
        seed_copy = Bullet(
            id = "",
            content = seed.content,
            tags = list(seed.tags),
            memory_type = seed.memory_type,
            helpful_count = seed.helpful_count,
            context_scope_id = context_scope_id,
        )
        memory._merge_or_add_bullet(seed_copy)
    memory._save_memory()


# ---------------------------------------------------------------------------
# Lesson transferability filtering (v4+)
# ---------------------------------------------------------------------------

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)


def _is_transferable_lesson(lesson: Dict[str, Any]) -> bool:
    """Return True if *lesson* is generic enough to transfer across tasks."""
    content = str(lesson.get("content", "")).strip()
    if not content:
        return False
    lesson_type = str(lesson.get("type", "")).lower()
    if lesson_type not in {"success", "failure", "domain", "tool"}:
        return False
    tokens = TOKEN_PATTERN.findall(content.lower())
    if len(tokens) < 8:
        return False
    if UUID_PATTERN.search(content):
        return False
    if re.search(r"\b\d{6,}\b", content):
        return False
    return True


def filter_transferable_lessons(
    lessons: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep only lessons that are generic enough to transfer."""
    return [lesson for lesson in lessons if _is_transferable_lesson(lesson)]


# ---------------------------------------------------------------------------
# Bullet merging / counting (v4+)
# ---------------------------------------------------------------------------

def merge_retrieved_bullets(
    local_bullets: List[Bullet],
    global_bullets: List[Bullet],
) -> List[Bullet]:
    """Merge local and global bullet lists, deduplicating by content hash."""
    merged: List[Bullet] = []
    seen: set = set()

    def add_all(items: List[Bullet]) -> None:
        for bullet in items:
            key = bullet.content_hash or bullet.id
            if key in seen:
                continue
            seen.add(key)
            merged.append(bullet)

    add_all(local_bullets)
    add_all(global_bullets)
    return merged


def count_seed_and_learned(
    bullets: List[Bullet],
) -> Tuple[int, int]:
    """Count seed (meta_strategy) vs learned bullets."""
    seed_count = 0
    learned_count = 0
    for bullet in bullets:
        tags = {tag.lower() for tag in bullet.tags}
        if "meta_strategy" in tags:
            seed_count += 1
        else:
            learned_count += 1
    return seed_count, learned_count
