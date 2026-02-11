# Long-Term Memory Based Self-Evolving Algorithm (LTMBSE ACE)

A standalone implementation of the Agentic Context Engineering (ACE) memory system integrated with a LangGraph multi-strategy reasoning agent. The system evolves its knowledge through structured memory bullets, enabling the agent to learn from past interactions and apply accumulated strategies to future tasks.

This project includes a complete benchmarking pipeline using [CL-bench](https://github.com/Tencent-Hunyuan/CL-bench) (Tencent-Hunyuan) to evaluate context learning performance, comparing baseline GPT-5.1 (High) against GPT-5.1 (High) enhanced with ACE memory.

## Architecture

The ACE pipeline follows a five-node LangGraph workflow:

```
START -> Router -> Planner -> Solver (CoT/ToT/ReAct) -> Critic -> ACE Learning -> END
```

**Memory System**: Structured bullets with three-component strength (semantic, episodic, procedural), access-clock decay, and faceted retrieval scoring with configurable weights.

**Learning Loop**: After each execution, the Reflector extracts lessons from the trace, the Curator synthesizes delta updates, and the memory applies incremental changes (new/update/remove bullets).

**CL-bench Enhancements**: Five targeted modifications improve performance on context learning tasks:
1. Context-scoped memory isolation per context_id
2. Post-context bullet injection (preserves attention on novel knowledge)
3. Relevance-dominant retrieval scoring
4. Meta-strategy seed bullets targeting dominant failure modes
5. Rubric-informed reflection for precise lesson extraction

## Prerequisites

- Python 3.10+
- Neo4j AuraDB instance (free tier available at [neo4j.com/cloud/aura-free](https://neo4j.com/cloud/aura-free/))
- Google Gemini API key (for ACE Reflector)
- OpenAI API key (for GPT-5.1 solver and CL-bench evaluation)

## Installation

```bash
git clone https://github.com/Edward-H26/LongTermMemoryBasedSelfEvolvingAlgorithm.git
cd LongTermMemoryBasedSelfEvolvingAlgorithm

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys and Neo4j credentials
```

## Configuration

All configuration is via environment variables (set in `.env`):

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `gemini` | Backend: `gemini` or `openai` |
| `GEMINI_API_KEY` | | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-3.0-flash` | Gemini model for ACE Reflector |
| `OPENAI_API_KEY` | | OpenAI API key for GPT-5.1 |
| `OPENAI_MODEL` | `gpt-5.1` | OpenAI model for solver/benchmark |
| `NEO4J_URI` | | Neo4j connection URI |
| `NEO4J_USERNAME` | | Neo4j username |
| `NEO4J_PASSWORD` | | Neo4j password |
| `ACE_INJECTION_MODE` | `post_context` | Bullet injection: `post_context` or `pre_context` |
| `ACE_WEIGHT_RELEVANCE` | `0.25` | Retrieval relevance weight (0.55 for CL-bench) |
| `ACE_WEIGHT_STRENGTH` | `0.55` | Retrieval strength weight (0.25 for CL-bench) |
| `ACE_WEIGHT_TYPE` | `0.20` | Retrieval type priority weight |
| `ACE_SEED_META_STRATEGIES` | `true` | Pre-seed meta-strategy bullets |
| `ACE_LLM_TEMPERATURE` | `0.2` | Temperature for ACE pipeline LLM calls |
| `ACE_CURATOR_USE_LLM` | `false` | Use LLM for curation (false = heuristic) |
| `ACE_CURATOR_SIMILARITY` | `0.9` | Similarity threshold for bullet deduplication |
| `ACE_QG_GATE_SCORE_MIN` | `0.65` | Minimum task gate score to apply online memory update |
| `ACE_QG_LESSON_SCORE_MIN` | `0.60` | Minimum per-lesson quality score for acceptance |
| `ACE_QG_OVERLAP_MIN` | `0.10` | Minimum question-lesson overlap score for acceptance |
| `ACE_QG_MAX_ACCEPTED_LESSONS` | `2` | Maximum accepted lessons per task before curation |

## Usage

### Interactive CLI

```bash
python cli.py                              # Interactive REPL
python cli.py --query "What is 2+2?"       # Single query
python cli.py --learner my_user_id         # Specific learner context
python cli.py --context-scope my_context   # Context-scoped memory
```

### Programmatic Entrypoint

```bash
echo '{"messages":[{"role":"user","content":"Explain quantum entanglement"}]}' | python run_agent.py
```

### Memory Analysis

```bash
python analyze_memory.py                       # Analyze default user
python analyze_memory.py --learner my_user_id  # Specific learner
python analyze_memory.py search "strategy"     # Search bullets
python analyze_memory.py export                # Export to file
python analyze_memory.py interactive           # Interactive mode
```

## CL-bench Benchmarking (V3 End to End)

The v3 pipeline is deterministic and quality-gated. Baseline and ACE runs use the same seeded subset via a shared manifest. By default, both scripts clear existing results and wipe the Neo4j database before starting. Use `--no-clear-results` and `--no-clear-db` to resume interrupted runs.

### One-command parallel run (recommended)

```bash
python -m benchmark.run_v3 \
    --manifest benchmark/results/subset_manifest_v3_seed42_n200.json \
    --max-samples 200 \
    --seed 42
```

This clears v3 result files and the Neo4j database, then runs baseline and ACE inference in parallel. Use `--no-clear-results` and `--no-clear-db` to resume.

### Step 1: Run baseline v3 (seed 42, 200 samples)

```bash
python -m benchmark.infer_baseline_v3 \
    --max-samples 200 \
    --seed 42 \
    --manifest benchmark/results/subset_manifest_v3_seed42_n200.json \
    --model gpt-5.1 \
    --output benchmark/results/baseline_v3.jsonl
```

By default, `--clear-results` deletes the output file before starting. Use `--no-clear-results` to resume.

### Step 2: Run ACE direct v3 with balanced quality gate

```bash
python -m benchmark.infer_ace_direct_v3 \
    --max-samples 200 \
    --seed 42 \
    --manifest benchmark/results/subset_manifest_v3_seed42_n200.json \
    --qg-gate-score-min 0.65 \
    --qg-lesson-score-min 0.60 \
    --qg-overlap-min 0.10 \
    --qg-max-accepted-lessons 2 \
    --output benchmark/results/ace_v3.jsonl
```

By default, `--clear-results` deletes output and metrics files, and `--clear-db` wipes all Neo4j nodes and relationships. Use `--no-clear-results` and `--no-clear-db` to resume.

### Step 3: Evaluate both outputs

```bash
python -m benchmark.eval \
    --input benchmark/results/baseline_v3.jsonl \
    --output benchmark/results/baseline_v3_graded.jsonl \
    --judge-model gpt-5.1

python -m benchmark.eval \
    --input benchmark/results/ace_v3.jsonl \
    --output benchmark/results/ace_v3_graded.jsonl \
    --judge-model gpt-5.1
```

### Step 4: Run error analysis

```bash
python -m benchmark.error_analysis \
    --input benchmark/results/baseline_v3_graded.jsonl \
    --output benchmark/results/baseline_v3_graded_errors.jsonl

python -m benchmark.error_analysis \
    --input benchmark/results/ace_v3_graded.jsonl \
    --output benchmark/results/ace_v3_graded_errors.jsonl
```

### Step 5: Generate v3 comparison report

```bash
python -m benchmark.compare \
    --baseline benchmark/results/baseline_v3_graded.jsonl \
    --ace benchmark/results/ace_v3_graded.jsonl \
    --baseline-errors benchmark/results/baseline_v3_graded_errors.jsonl \
    --ace-errors benchmark/results/ace_v3_graded_errors.jsonl \
    --output benchmark/results/comparison_report_v3.md \
    --title-label V3
```

### Required v3 artifacts

- `benchmark/results/subset_manifest_v3_seed42_n200.json`
- `benchmark/results/baseline_v3.jsonl`
- `benchmark/results/ace_v3.jsonl`
- `benchmark/results/ace_v3_metrics.json`
- `benchmark/results/baseline_v3_graded.jsonl`
- `benchmark/results/ace_v3_graded.jsonl`
- `benchmark/results/baseline_v3_graded_errors.jsonl`
- `benchmark/results/ace_v3_graded_errors.jsonl`
- `benchmark/results/comparison_report_v3.md`
- `benchmark/results/comparison_report_v3.json`

### Reproducibility checks

1. Same manifest check:
   Baseline v3 and ACE v3 should process the same ordered `task_id` list from `subset_manifest_v3_seed42_n200.json`.
2. Seed sensitivity check:
   Running with `--seed 43` and a different manifest path should produce a different task set.
3. Report integrity check:
   `benchmark.compare` will fail if baseline and ACE graded files do not contain the same `task_id` set.

## Project Structure

```
LongTermMemoryBasedSelfEvolvingAlgorithm/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── ace_memory.py            # Core: Bullet, DeltaUpdate, ACEMemory
│   ├── ace_components.py        # Reflector, Curator, ACEPipeline
│   ├── storage.py               # Neo4jMemoryStore
│   ├── llm.py                   # Multi-backend LLM with token tracking
│   ├── tools.py                 # Calculator, Search, Neo4j QA
│   ├── solvers.py               # CoT, ToT, ReAct solvers
│   ├── agent.py                 # LangGraph ACE agent
│   └── prompts/                 # System prompts
├── cli.py                       # Interactive CLI
├── run_agent.py                 # Stdin JSON entrypoint
├── analyze_memory.py            # Memory analysis tool
├── benchmark/
│   ├── infer_baseline.py        # CL-bench baseline inference
│   ├── infer_ace.py             # CL-bench ACE inference
│   ├── infer_baseline_v3.py     # CL-bench baseline inference (deterministic subset)
│   ├── infer_ace_direct_v3.py   # CL-bench ACE direct inference (quality-gated)
│   ├── run_v3.py                # Run baseline and ACE v3 in parallel (clears results and DB)
│   ├── sampling.py              # Seeded subset sampling + manifest reuse
│   ├── eval.py                  # Rubric-based evaluation
│   ├── error_analysis.py        # Error type classification
│   ├── compare.py               # Side-by-side comparison
│   ├── metrics.py               # Token/latency collection
│   └── results/                 # Output directory
└── docs/
    └── ALGORITHM.md             # Detailed algorithm documentation
```

## References

- CL-bench: [github.com/Tencent-Hunyuan/CL-bench](https://github.com/Tencent-Hunyuan/CL-bench)
- CL-bench dataset: [huggingface.co/datasets/tencent/CL-bench](https://huggingface.co/datasets/tencent/CL-bench)
- CL-bench paper: [arxiv.org/abs/2602.03587](https://arxiv.org/abs/2602.03587)
