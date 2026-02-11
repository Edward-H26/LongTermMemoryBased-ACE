# LTMBSE ACE Algorithm: Technical Documentation

## 1. Memory Model

### Bullet Structure

Each memory bullet is a structured unit containing 23 fields:

| Field | Type | Description |
|---|---|---|
| `id` | str | Unique identifier (MD5 hash of normalized content) |
| `content` | str | The strategy, lesson, or domain concept |
| `helpful_count` | int | Times marked as helpful |
| `harmful_count` | int | Times marked as harmful |
| `created_at` | str | ISO timestamp of creation |
| `last_used` | str | ISO timestamp of last retrieval |
| `tags` | List[str] | Categorization tags |
| `semantic_strength` | float | Semantic memory component weight |
| `episodic_strength` | float | Episodic memory component weight |
| `procedural_strength` | float | Procedural memory component weight |
| `semantic_access_index` | int | Access counter for semantic decay |
| `episodic_access_index` | int | Access counter for episodic decay |
| `procedural_access_index` | int | Access counter for procedural decay |
| `learner_id` | str | Learner identifier for personalization |
| `topic` | str | Inferred topic |
| `concept` | str | Extracted concept |
| `memory_type` | str | Primary type: semantic, episodic, or procedural |
| `ttl_days` | int | Time-to-live for expiring bullets |
| `content_hash` | str | SHA-256 hash for deduplication |
| `context_scope_id` | str | Context scope for memory isolation (Enhancement 1) |

### Three-Component Strength Model

Each bullet carries three independent strength values corresponding to human memory systems:

- **Semantic memory** (decay rate: 0.01): General domain knowledge and facts. Decays slowly, representing stable knowledge.
- **Episodic memory** (decay rate: 0.05): Specific experiences and observations. Decays moderately, representing fading personal experiences.
- **Procedural memory** (decay rate: 0.002): Step-by-step procedures and workflows. Decays very slowly, representing well-practiced skills.

Only the primary memory type carries active strength. The others are zeroed. This ensures clean separation between knowledge types.

### Access-Clock Exponential Decay

Strength decays based on an access counter (not wall-clock time):

```
component_score = strength * (1 - decay_rate) ^ (access_clock - last_access_index)
```

Every retrieval advances the global `access_clock` by 1. Bullets that are frequently retrieved maintain high scores; unused bullets gradually fade. The total bullet score is the sum of all three component scores.

## 2. Retrieval Scoring

### Configurable Weights (Enhancement 3)

The retrieval formula combines four signals with configurable weights:

```
combined_score = W_relevance * relevance
               + W_strength * normalized_strength
               + W_type * type_priority
               + bonus
```

**Default weights** (original algorithm):
- `ACE_WEIGHT_RELEVANCE = 0.25`
- `ACE_WEIGHT_STRENGTH = 0.55`
- `ACE_WEIGHT_TYPE = 0.20`

**CL-bench weights** (Enhancement 3):
- `ACE_WEIGHT_RELEVANCE = 0.55`
- `ACE_WEIGHT_STRENGTH = 0.25`
- `ACE_WEIGHT_TYPE = 0.20`

The CL-bench configuration prioritizes relevance because each context introduces completely novel knowledge, making historical strength less meaningful.

### Signal Definitions

- **Relevance**: Jaccard similarity between query terms and bullet content words.
- **Normalized Strength**: Decay-adjusted score divided by baseline strength.
- **Type Priority**: `procedural=1.0`, `episodic=0.7`, `semantic=0.4`.
- **Bonus**: Additional scoring for visual needs (+0.2), persona match (+0.1).

## 3. Delta Update Lifecycle

### Apply Delta

When the Curator produces a `DeltaUpdate`, the memory applies it in three steps:

1. **New bullets**: Each new bullet is merged or added via `_merge_or_add_bullet()`. If a similar bullet exists (Jaccard >= 0.9), metadata is merged instead of creating a duplicate.
2. **Update bullets**: Existing bullets receive helpful/harmful count adjustments.
3. **Remove bullets**: Marked bullets are removed from all indexes.

### Grow-and-Refine

After every delta application:
1. **Deduplication**: Pairwise Jaccard similarity check. Bullets exceeding the threshold (0.85) are merged, keeping the one with the higher helpful-harmful delta.
2. **Pruning**: If total bullets exceed `max_bullets` (100), the lowest-scored bullets are removed.

## 4. Three-Role Pipeline

### Reflector

Analyzes execution traces and extracts concrete lessons. Takes the full execution trace (question, model answer, ground truth, tool calls) and produces a JSON array of lessons, each with content, type (success/failure/domain/tool), and tags.

**Enhancement 5**: Accepts optional `rubric_feedback` parameter. When per-rubric satisfaction data is available from a prior evaluation pass, the Reflector appends PASS/FAIL status for each rubric to its analysis prompt, enabling more precise lesson extraction.

### Curator

Synthesizes lessons into delta updates. Operates in two modes:
- **Heuristic mode** (default, no LLM calls): Directly maps lessons to new bullets using keyword-based memory type inference and similarity-based deduplication.
- **LLM mode**: Uses an LLM to produce structured JSON delta updates with reasoning.

### ACEPipeline

Coordinates the full learning loop: Reflector -> Curator -> Memory.apply_delta(). Includes fallback lesson generation when the Reflector fails to extract lessons.

## 5. LangGraph Integration

### Graph Topology

```
START -> router -> planner -> solver -> critic -> ace_learning -> END
```

### Node Responsibilities

- **Router**: Rule-based routing to CoT/ToT/ReAct based on query keywords. Retrieves initial bullets from ACE memory.
- **Planner**: Sets solver parameters (breadth, depth, max_turns, temperature).
- **Solver**: Executes the selected reasoning strategy with ACE-enriched prompts.
- **Critic**: Cleans and extracts the final answer from solver output.
- **ACE Learning**: Creates execution trace, runs Reflector + Curator, applies delta to memory.

### Bullet Injection Point (Enhancement 2)

Original: Prepends bullets to the system message (before context).

Enhanced (`ACE_INJECTION_MODE=post_context`): Inserts bullets immediately before the last user message (after all context). This preserves model attention on the novel CL-bench context. Configurable via environment variable for A/B testing.

## 6. CL-bench Enhancements

### Enhancement 1: Context-Scoped Memory

**Problem**: CL-bench has 500 distinct contexts (board games, legal systems, medical protocols). Global memory causes cross-domain interference.

**Solution**: `context_scope_id` field on Bullet + cache key = `learner_id:context_scope_id`. Bullets from one context do not appear in retrieval for a different context. Tasks within the same context (avg 3.8 per context, 51.1% sequential) share memory for intra-context learning.

### Enhancement 2: Post-Context Bullet Injection

**Problem**: GPT-5.1 ignores context 55.3% of the time. Prepending ACE bullets before the context further diverts attention from the novel knowledge.

**Solution**: Inject bullets AFTER all context messages, immediately before the task question. The model processes the full novel context first, then sees the ACE strategies as supplementary reminders.

### Enhancement 3: Relevance-Dominant Retrieval

**Problem**: The original formula weights strength at 0.55, causing frequently reinforced bullets from unrelated domains to dominate over relevant ones.

**Solution**: Flip weights to `relevance=0.55, strength=0.25` for CL-bench. Each context introduces novel knowledge, so query relevance is more meaningful than historical usage frequency.

### Enhancement 4: Meta-Strategy Seed Bullets

**Problem**: New contexts start with empty memory, providing no guidance on common failure modes.

**Solution**: Pre-seed three procedural bullets when memory is initialized:
1. "Re-read all constraints, rules, and procedures in the context" (targets Context Ignored 55.3%)
2. "Follow the exact output format specified" (targets Format Error 35.3%)
3. "Do not rely on pre-trained knowledge when context provides explicit rules" (targets Context Misused 61.5%)

### Enhancement 5: Rubric-Informed Reflection

**Problem**: The Reflector extracts generic lessons without knowing which specific requirements were missed.

**Solution**: After an evaluation pass produces per-rubric satisfaction status, feed it back to the Reflector. This enables focused bullets like "ensure all API parameters from the documentation are included" rather than vague "pay attention to details."

## 7. Error Analysis Methodology

### How Table 2 is Calculated

Per the CL-bench paper (Section 5, Table 3):

1. Each rubric covers a dimension: factual correctness, computational accuracy, judgment correctness, procedural correctness, content completeness, format compliance.
2. When a task scores 0, the specific failed rubrics reveal the failure mode.
3. A secondary LLM classification pass categorizes each failed rubric into an error type.
4. Error types are NOT mutually exclusive. One task can have multiple types.

**Error Types**:
- **Context Ignored**: Model did not reference information explicitly stated in context
- **Context Misused**: Model incorrectly applied contextual knowledge
- **Format Error**: Model violated explicit formatting instructions
- **Refusal**: Model output is empty or model claims insufficient information

**Aggregation**:
- Context Ignored (%) = tasks with at least one CONTEXT_IGNORED rubric / total tasks
- Context Misused (%) = tasks with at least one CONTEXT_MISUSED rubric / total tasks
- Format Error (%) = tasks with at least one FORMAT_ERROR rubric / total tasks
- Refusal (%) = tasks classified as REFUSAL / total tasks

Row totals exceed 100% because a single task can exhibit multiple error types.
