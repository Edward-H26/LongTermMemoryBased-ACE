"""
ACE Components: Generator, Reflector, and Curator

Implements the three-role architecture from the ACE paper:
1. Generator: Produces reasoning trajectories with feedback on bullets
2. Reflector: Extracts lessons from trajectories (Enhancement 5: rubric-informed)
3. Curator: Synthesizes lessons into delta updates
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import json
import os
import re

from src.ace_memory import Bullet, DeltaUpdate, ACEMemory
from src.prompts.ace_prompts import REFLECTOR_PROMPT, CURATOR_PROMPT


@dataclass
class ExecutionTrace:
    """Represents the execution trace of a single query"""
    question: str
    model_answer: str
    ground_truth: Optional[str] = None
    success: bool = False
    trace_messages: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.trace_messages is None:
            self.trace_messages = []
        if self.metadata is None:
            self.metadata = {}

    def format_trace(self, max_length: int = 5000) -> str:
        parts = []
        for msg in self.trace_messages[-10:]:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))[:500]
            if role == "tool":
                tool_name = msg.get("name", "unknown_tool")
                parts.append(f"[TOOL: {tool_name}] {content}")
            else:
                parts.append(f"[{role.upper()}] {content}")
        trace_str = "\n".join(parts)
        if len(trace_str) > max_length:
            trace_str = trace_str[:max_length] + "... (truncated)"
        return trace_str


@dataclass
class QualityGateConfig:
    gate_score_min: float = 0.65
    lesson_score_min: float = 0.60
    overlap_min: float = 0.10
    max_accepted_lessons: int = 2

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            return default
        return parsed if parsed > 0 else default

    @classmethod
    def from_env(cls) -> "QualityGateConfig":
        return cls(
            gate_score_min = cls._safe_float(os.getenv("ACE_QG_GATE_SCORE_MIN", "0.65"), 0.65),
            lesson_score_min = cls._safe_float(os.getenv("ACE_QG_LESSON_SCORE_MIN", "0.60"), 0.60),
            overlap_min = cls._safe_float(os.getenv("ACE_QG_OVERLAP_MIN", "0.10"), 0.10),
            max_accepted_lessons = cls._safe_int(os.getenv("ACE_QG_MAX_ACCEPTED_LESSONS", "2"), 2),
        )


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize_text(text: str) -> Set[str]:
    if not text:
        return set()
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _lesson_overlap_score(question: str, lesson_content: str) -> float:
    question_tokens = _tokenize_text(question)
    lesson_tokens = _tokenize_text(lesson_content)
    if not question_tokens or not lesson_tokens:
        return 0.0
    intersection = len(question_tokens.intersection(lesson_tokens))
    union = len(question_tokens.union(lesson_tokens))
    if union == 0:
        return 0.0
    return intersection / union


def _lesson_quality_score(lesson: Dict[str, Any]) -> float:
    content = str(lesson.get("content", "")).strip()
    if not content:
        return 0.0

    token_count = len(_tokenize_text(content))
    token_score = min(token_count / 20.0, 1.0) * 0.6

    tags = lesson.get("tags") or []
    has_tags = isinstance(tags, list) and len(tags) > 0
    tags_score = 0.2 if has_tags else 0.0

    lesson_type = str(lesson.get("type", "")).lower()
    has_known_type = lesson_type in {"success", "failure", "domain", "tool"}
    type_score = 0.2 if has_known_type else 0.0

    return min(token_score + tags_score + type_score, 1.0)


def apply_quality_gate(
    question: str,
    model_answer: str,
    lessons: List[Dict[str, Any]],
    config: Optional[QualityGateConfig] = None,
) -> Dict[str, Any]:
    cfg = config or QualityGateConfig.from_env()
    output_valid = bool(model_answer and str(model_answer).strip())

    accepted_candidates: List[Tuple[float, float, Dict[str, Any]]] = []
    rejected_entries: List[Dict[str, Any]] = []

    for idx, lesson in enumerate(lessons):
        content = str(lesson.get("content", "")).strip()
        if not content:
            rejected_entries.append({
                "index": idx,
                "reason": "empty_content",
                "overlap_score": 0.0,
                "lesson_score": 0.0,
            })
            continue

        overlap_score = _lesson_overlap_score(question, content)
        lesson_score = _lesson_quality_score(lesson)
        rejection_reasons: List[str] = []

        if overlap_score < cfg.overlap_min:
            rejection_reasons.append("low_overlap")
        if lesson_score < cfg.lesson_score_min:
            rejection_reasons.append("low_quality")

        if rejection_reasons:
            rejected_entries.append({
                "index": idx,
                "reason": ",".join(rejection_reasons),
                "overlap_score": overlap_score,
                "lesson_score": lesson_score,
            })
            continue

        lesson_copy = dict(lesson)
        lesson_copy["quality_gate"] = {
            "overlap_score": overlap_score,
            "lesson_score": lesson_score,
        }
        accepted_candidates.append((lesson_score, overlap_score, lesson_copy))

    accepted_candidates.sort(key = lambda row: (row[0], row[1]), reverse = True)
    accepted_lessons = [row[2] for row in accepted_candidates[:cfg.max_accepted_lessons]]

    accepted_scores = [row[0] for row in accepted_candidates[:cfg.max_accepted_lessons]]
    accepted_quality_avg = sum(accepted_scores) / len(accepted_scores) if accepted_scores else 0.0
    output_score = 1.0 if output_valid else 0.0
    gate_score = 0.5 * output_score + 0.5 * accepted_quality_avg
    should_apply_update = bool(accepted_lessons) and gate_score >= cfg.gate_score_min

    rejection_counts: Dict[str, int] = {
        "empty_content": 0,
        "low_overlap": 0,
        "low_quality": 0,
    }
    for entry in rejected_entries:
        reasons = str(entry.get("reason", "")).split(",")
        for reason in reasons:
            if reason in rejection_counts:
                rejection_counts[reason] += 1

    diagnostics = {
        "config": {
            "gate_score_min": cfg.gate_score_min,
            "lesson_score_min": cfg.lesson_score_min,
            "overlap_min": cfg.overlap_min,
            "max_accepted_lessons": cfg.max_accepted_lessons,
        },
        "output_valid": output_valid,
        "output_score": output_score,
        "accepted_quality_avg": accepted_quality_avg,
        "gate_score": gate_score,
        "should_apply_update": should_apply_update,
        "num_lessons_input": len(lessons),
        "num_lessons_accepted": len(accepted_lessons),
        "num_lessons_rejected": len(rejected_entries),
        "rejection_counts": rejection_counts,
        "rejected_examples": rejected_entries[:5],
    }
    return {"accepted_lessons": accepted_lessons, "diagnostics": diagnostics}


class Reflector:
    """
    Reflector: Analyzes execution traces and extracts concrete lessons.
    Enhancement 5: Supports optional rubric_feedback for precise lesson extraction.
    """

    def __init__(self, llm):
        self.llm = llm

    def reflect(
        self,
        trace: ExecutionTrace,
        max_refinement_rounds: int = 3,
        rubric_feedback: Optional[List[Tuple[str, bool]]] = None,
    ) -> List[Dict[str, Any]]:
        prompt = REFLECTOR_PROMPT.format(
            trace=trace.format_trace(),
            question=trace.question,
            ground_truth=trace.ground_truth or "Not available",
            model_answer=trace.model_answer,
            success="Success" if trace.success else "Failed",
        )

        if rubric_feedback:
            rubric_section = "\n\nEvaluation rubric results:\n"
            for rubric, passed in rubric_feedback:
                rubric_section += f"  {'PASS' if passed else 'FAIL'}: {rubric}\n"
            prompt += rubric_section

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a curation assistant. Output ONLY valid JSON that matches the provided schema. "
                    "Never include explanations, markdown fences, or additional text."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        for round_num in range(max_refinement_rounds):
            try:
                response = self.llm.chat(messages, temperature=0.3, max_tokens=2000)
                content = response["choices"][0]["message"]["content"]
                lessons_data = self._parse_json_response(content)

                if lessons_data and "lessons" in lessons_data:
                    return lessons_data["lessons"]

                if round_num < max_refinement_rounds - 1:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "Your response was not valid JSON. Please output ONLY valid JSON with the required structure."
                    })
            except Exception as e:
                if round_num == max_refinement_rounds - 1:
                    return []

        return []

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        try:
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            return json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except Exception:
                    pass
            return None


class Curator:
    """Curator: Synthesizes lessons into structured delta updates."""

    def __init__(self, llm, memory: ACEMemory):
        self.llm = llm
        self.memory = memory

    @staticmethod
    def _infer_memory_attributes(content: str, tags: List[str]) -> Tuple[str, Optional[str]]:
        lower = content.lower()
        if any(keyword in lower for keyword in ["step", "goal", "convert", "next", "state", "progress", "now do", "procedure", "workflow", "sequence"]):
            memory_type = "procedural"
        elif any(keyword in lower for keyword in ["user", "prefers", "request", "frustration", "asked", "likes", "history", "experience", "observed"]):
            memory_type = "episodic"
        elif any(keyword in lower for keyword in ["misconception", "error", "mistake", "struggle", "issue", "confused"]):
            memory_type = "episodic"
        elif any(tag.lower() == "state" for tag in tags):
            memory_type = "procedural"
        else:
            memory_type = "semantic"
        return memory_type, None

    @staticmethod
    def _normalise_tags(tags: List[str]) -> List[str]:
        normalised: List[str] = []
        seen = set()
        for tag in tags:
            if not tag:
                continue
            slug = re.sub(r"[^a-z0-9]+", "_", tag.strip().lower()).strip("_")
            if not slug:
                continue
            if slug in {"semantic", "episodic", "procedural"}:
                continue
            if slug not in seen:
                normalised.append(slug)
                seen.add(slug)
        return normalised

    def _derive_supporting_bullets(
        self,
        content: str,
        normalized_tags: List[str],
        learner_id: Optional[str],
        topic: Optional[str],
        facets: Optional[Dict[str, Any]] = None,
    ) -> List[Bullet]:
        return []

    def _lessons_to_delta(
        self,
        lessons: List[Dict[str, Any]],
        learner_id: Optional[str] = None,
        topic: Optional[str] = None,
        facets: Optional[Dict[str, Any]] = None,
    ) -> DeltaUpdate:
        delta = DeltaUpdate()
        similarity_threshold = float(os.getenv("ACE_CURATOR_SIMILARITY", "0.9"))
        for idx, lesson in enumerate(lessons, 1):
            content = (lesson.get("content") or "").strip()
            if not content:
                continue
            existing_bullet, similarity = self.memory.find_similar_bullet(
                content,
                learner_id=learner_id,
                topic=topic,
                threshold=similarity_threshold,
                return_score=True,
            )
            if existing_bullet:
                entry = delta.update_bullets.setdefault(existing_bullet.id, {"helpful": 0, "harmful": 0})
                entry["helpful"] += 1
                continue

            tags = lesson.get("tags") or []
            ltype = lesson.get("type")
            if ltype:
                tags = list(dict.fromkeys(tags + [ltype]))
            if not tags:
                tags = ["lesson"]
            memory_type, concept = self._infer_memory_attributes(content, tags)
            normalized_tags = self._normalise_tags(tags)

            bullet = Bullet(
                id="",
                content=content,
                tags=normalized_tags,
                helpful_count=1,
                learner_id=learner_id,
                topic=topic,
                concept=concept,
                memory_type=memory_type,
            )
            delta.new_bullets.append(bullet)

            supporting = self._derive_supporting_bullets(content, normalized_tags, learner_id, topic, facets)
            for supplemental in supporting:
                existing_support, score = self.memory.find_similar_bullet(
                    supplemental.content,
                    learner_id=learner_id,
                    topic=topic,
                    threshold=float(os.getenv("ACE_CURATOR_SUPPORT_SIMILARITY", "0.95")),
                    return_score=True,
                )
                if existing_support:
                    entry = delta.update_bullets.setdefault(existing_support.id, {"helpful": 0, "harmful": 0})
                    entry["helpful"] += 1
                else:
                    delta.new_bullets.append(supplemental)

        delta.metadata = {
            "reasoning": "heuristic_lessons_to_bullets",
            "num_lessons": len(lessons),
        }
        if learner_id:
            delta.metadata["learner_id"] = learner_id
        if topic:
            delta.metadata["topic"] = topic
        return delta

    def curate(
        self,
        lessons: List[Dict[str, Any]],
        query: str,
        learner_id: Optional[str] = None,
        topic: Optional[str] = None,
        facets: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> DeltaUpdate:
        if not lessons:
            return DeltaUpdate()

        use_llm = os.getenv("ACE_CURATOR_USE_LLM", "false").lower() in {"1", "true", "yes"}

        relevant_bullets = self.memory.retrieve_relevant_bullets(
            query, top_k=10, learner_id=learner_id, topic=topic, facets=facets,
        )
        current_bullets_str = "\n".join(
            f"ID: {b.id}\n{b.format_for_prompt()}\nType: {b.memory_type} | Learner: {b.learner_id} | Topic: {b.topic}"
            for b in relevant_bullets
        ) if relevant_bullets else "No existing bullets"

        prompt = CURATOR_PROMPT.format(
            lessons=json.dumps(lessons, indent=2),
            current_bullets=current_bullets_str,
        )

        if not use_llm:
            delta = self._lessons_to_delta(lessons, learner_id=learner_id, topic=topic, facets=facets)
            delta.metadata.update({
                "reasoning": "heuristic_lessons_to_bullets",
                "prompt": prompt,
                "learner_id": learner_id,
                "topic": topic,
            })
            return delta

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a curation assistant. Output ONLY valid JSON that matches the provided schema. "
                    "Never include explanations, markdown fences, or additional text."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        max_rounds = 3
        delta_data = None
        content = ""

        for round_idx in range(max_rounds):
            response = self.llm.chat(
                messages, temperature=0.2, max_tokens=2000, response_mime_type="application/json",
            )
            content = response["choices"][0]["message"]["content"]
            delta_data = self._parse_json_response(content)
            if delta_data:
                break
            if round_idx < max_rounds - 1:
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "Your previous response was not valid JSON. Please respond with ONLY the JSON object matching the required schema.",
                })

        if not delta_data:
            fallback = self._lessons_to_delta(lessons, learner_id=learner_id, topic=topic, facets=facets)
            fallback.metadata.update({
                "reasoning": "fallback_from_unparsed_curator",
                "num_lessons": len(lessons),
                "raw_response": content.strip(),
                "learner_id": learner_id,
                "topic": topic,
            })
            return fallback

        delta = DeltaUpdate()
        for bullet_data in delta_data.get("new_bullets", []):
            helpful = bullet_data.get("helpful")
            harmful = bullet_data.get("harmful")
            tags = bullet_data.get("tags", [])
            bullet = Bullet(
                id="",
                content=bullet_data["content"],
                tags=tags,
                helpful_count=int(helpful) if helpful is not None else 1,
                harmful_count=int(harmful) if harmful is not None else 0,
                learner_id=bullet_data.get("learner_id") or learner_id,
                topic=bullet_data.get("topic") or topic,
                concept=bullet_data.get("concept"),
                memory_type=bullet_data.get("memory_type"),
            )
            if bullet.memory_type not in {"semantic", "episodic", "procedural"}:
                inferred_type, inferred_concept = self._infer_memory_attributes(bullet.content, tags)
                bullet.memory_type = inferred_type
                if not bullet.concept and inferred_concept:
                    bullet.concept = inferred_concept
            delta.new_bullets.append(bullet)

        delta.update_bullets = delta_data.get("update_bullets", {})
        delta.remove_bullets = set(delta_data.get("remove_bullets", []))
        delta.metadata = {
            "reasoning": delta_data.get("reasoning", ""),
            "num_lessons": len(lessons),
            "raw_response": content.strip(),
            "learner_id": learner_id,
            "topic": topic,
        }
        return delta

    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        try:
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            return json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except Exception:
                    pass
            return None


class ACEPipeline:
    """Complete ACE pipeline: coordinates Generator, Reflector, and Curator."""

    def __init__(self, llm, memory: ACEMemory):
        self.llm = llm
        self.memory = memory
        self.reflector = Reflector(llm)
        self.curator = Curator(llm, memory)

    @staticmethod
    def _fallback_lessons(trace: ExecutionTrace) -> List[Dict[str, Any]]:
        lessons: List[Dict[str, Any]] = []
        messages = trace.trace_messages or []
        user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
        last_user = (user_messages[-1] if user_messages else trace.question) or ""
        lower_last = last_user.lower()

        if any(token in lower_last for token in ["hard", "don't know", "do not know", "confused"]):
            lessons.append({
                "content": "When a task seems difficult, restate the core question in simpler terms and identify the first concrete sub-step.",
                "type": "success",
                "tags": ["empathy", "scaffolding", "fallback"],
            })
        lessons.append({
            "content": "Break complex problems into smaller sub-problems and verify each step before proceeding to the next.",
            "type": "success",
            "tags": ["problem_decomposition", "verification", "fallback"],
        })
        return lessons

    def process_execution(
        self,
        trace: ExecutionTrace,
        apply_update: bool = True,
        rubric_feedback: Optional[List[Tuple[str, bool]]] = None,
    ) -> Optional[DeltaUpdate]:
        lessons = self.reflector.reflect(trace, rubric_feedback=rubric_feedback)

        if not lessons:
            lessons = self._fallback_lessons(trace)

        metadata = trace.metadata or {}
        scratch_state = metadata.get("scratch") if isinstance(metadata, dict) else {}
        if not isinstance(scratch_state, dict):
            scratch_state = {}
        learner_id = metadata.get("learner_id") or scratch_state.get("learner_id")
        topic = scratch_state.get("topic")

        facets = scratch_state.get("ace_retrieval_facets") if isinstance(scratch_state, dict) else None
        quality_gate_eval = apply_quality_gate(
            question = trace.question,
            model_answer = trace.model_answer,
            lessons = lessons,
            config = QualityGateConfig.from_env(),
        )
        accepted_lessons = quality_gate_eval.get("accepted_lessons", [])
        quality_gate = quality_gate_eval.get("diagnostics", {})

        if not accepted_lessons:
            skipped = DeltaUpdate()
            skipped.metadata = {
                "reasoning": "quality_gate_rejected_all_lessons",
                "num_lessons": len(lessons),
                "learner_id": learner_id,
                "topic": topic,
                "quality_gate": quality_gate,
            }
            return skipped

        if not quality_gate.get("should_apply_update", False):
            skipped = DeltaUpdate()
            skipped.metadata = {
                "reasoning": "quality_gate_blocked_update",
                "num_lessons": len(lessons),
                "learner_id": learner_id,
                "topic": topic,
                "quality_gate": quality_gate,
            }
            return skipped

        delta = self.curator.curate(
            accepted_lessons,
            trace.question,
            learner_id = learner_id,
            topic = topic,
            facets = facets,
        )
        delta.metadata["quality_gate"] = quality_gate
        delta.metadata["num_lessons_before_gate"] = len(lessons)
        delta.metadata["num_lessons_after_gate"] = len(accepted_lessons)

        if topic and "topic" not in delta.metadata:
            delta.metadata["topic"] = topic
        if learner_id and "learner_id" not in delta.metadata:
            delta.metadata["learner_id"] = learner_id

        if apply_update:
            self.memory.apply_delta(delta)
        else:
            delta.metadata["update_skipped"] = True

        return delta

    def get_enriched_prompt(
        self,
        question: str,
        base_prompt: str,
        top_k: int = 10,
        learner_id: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> str:
        context = self.memory.format_context(
            question, top_k=top_k, learner_id=learner_id, topic=topic,
        )
        if context:
            return f"{base_prompt}\n\n{context}\n\nQuestion: {question}"
        else:
            return f"{base_prompt}\n\nQuestion: {question}"
