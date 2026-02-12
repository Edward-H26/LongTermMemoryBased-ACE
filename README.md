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
| `GEMINI_MODEL` | `gemini-3-flash-preview` | Gemini model for ACE Reflector |
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
| `ACE_MEMORY_SCOPE_MODE` | `hybrid` | V4 memory mode: `hybrid`, `local`, or `global` |
| `ACE_GLOBAL_GATE_SCORE_MIN` | `0.80` | Minimum gate score required before global memory update |
| `ACE_LOCAL_TOP_K` | `3` | Number of local-context bullets retrieved in V4 |
| `ACE_GLOBAL_TOP_K` | `2` | Number of global bullets retrieved in V4 |
| `ACE_CONTEXT_WORKERS` | `6` | V4 context-level parallel worker count |
| `ACE_STEP_SCORING_MODE` | `near_full` | Process scoring mode: `off`, `near_full`, or `full` |
| `ACE_STEP_SCORER_MODEL` | `gpt-5.1` | Verifier model for intermediate step scoring |
| `ACE_STEP_SCORE_WORKERS` | `8` | Worker count for parallel step scoring |
| `ACE_STEP_SCORE_MIN` | `0.45` | ToT pruning threshold for mean intermediate step score |
| `ACE_MAX_COMPLETION_TOKENS` | `8192` | Default max completion tokens for V4 inference |
| `ACE_EMPTY_OUTPUT_RETRY_MAX_TOKENS` | `16384` | Retry cap when completion was empty at token limit |
| `ACE_NEO4J_RETRY_MAX` | `2` | Neo4j operation retries after first attempt |
| `ACE_NEO4J_RETRY_BACKOFF_SEC` | `1.0` | Exponential backoff base seconds for Neo4j retries |
| `ACE_NEO4J_RECONNECT_ON_SESSION_EXPIRED` | `true` | Reconnect Neo4j driver when `SessionExpired` occurs |
| `ACE_V5_RESUME_FROM_PROGRESS` | `true` | Default for `infer_ace_direct_v5 --resume-from-progress` |
| `ACE_V5_FINALIZE_ORDER` | `true` | Default for `infer_ace_direct_v5 --finalize-order` |
| `ACE_V5_PROGRESS_PATH` | unset | Optional default path for ACE v5 progress journal |
| `OPENAI_ADMIN_API_KEY` | unset | OpenAI admin key used for strict billed-cost reconciliation |
| `OPENAI_COST_PROJECT_ID` | unset | OpenAI project scope used for billed-cost reconciliation |
| `BENCHMARK_COST_MODE` | `dual_source` | Cost accounting mode for v5 reports (`legacy` or `dual_source`) |
| `BENCHMARK_BILLING_POLICY` | `strict` | Billing reconciliation policy (`strict` or `off`) |

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

## CL-bench Benchmarking (V5 Recommended)

The v5 pipeline keeps v4 model behavior and adds reliability features for long runs:

- Neo4j retry + reconnect handling for `SessionExpired`/driver failures.
- Context-parallel ACE with per-task durable progress journaling.
- Resume from output + progress journal without reprocessing completed tasks.
- Deterministic finalize-order output reconstruction from manifest order.
- Parallel post-pipeline stages with side-specific retries.
- Full-pipeline actual metered cost reporting across inference, ACE auxiliary calls, eval, and error analysis.
- Strict OpenAI billed-cost reconciliation against run window metadata.

### One-command v5 run

```bash
python -m benchmark.run_v5 \
    --manifest benchmark/results/v5/subset_manifest_v5_seed42_n200.json \
    --max-samples 200 \
    --seed 42 \
    --sampling-strategy context_dense \
    --memory-scope hybrid \
    --with-report
```

### V5 preflight (static, smoke, or both)

```bash
python -m benchmark.run_v5 \
    --preflight-mode static \
    --manifest benchmark/results/v5/subset_manifest_v5_seed42_n200.json \
    --max-samples 200 \
    --seed 42 \
    --sampling-strategy context_dense \
    --memory-scope hybrid \
    --preflight-report benchmark/results/v5/preflight_v5.json
```

### `run_v5` parameters

| Parameter | Default | Description |
|---|---|---|
| `--manifest` | `benchmark/results/v5/subset_manifest_v5_seed42_n200.json` | Deterministic subset manifest path shared by baseline and ACE. |
| `--max-samples` | `200` | Number of tasks to run. Must be `> 0`. |
| `--seed` | `42` | Seed used when creating a new manifest. |
| `--sampling-strategy` | `context_dense` | `task_random`, `context_dense`, or `context_dense_stratified`. |
| `--memory-scope` | `hybrid` | ACE retrieval scope: `hybrid`, `local`, or `global`. |
| `--output-dir` | `benchmark/results/v5` | V5 artifact directory. |
| `--cost-mode` | `dual_source` | Cost accounting mode passed to post-pipeline compare. |
| `--billing-policy` | `strict` | Billing reconciliation policy for compare (`strict` hard-fails on missing billed data). |
| `--openai-admin-key-env` | `OPENAI_ADMIN_API_KEY` | Env key name for admin billing API token lookup. |
| `--openai-project-id-env` | `OPENAI_COST_PROJECT_ID` | Env key name for OpenAI project id lookup. |
| `--clear-results` | `True` | Remove existing v5 outputs before running. |
| `--clear-db` | `True` | Clear Neo4j before ACE inference. |
| `--with-report` | `True` | Run post-inference pipeline automatically. |
| `--preflight-mode` | `off` | `off`, `static`, `smoke`, or `both`. |
| `--smoke-samples` | `5` | Smoke subset size (`3`, `4`, or `5`). |
| `--smoke-output-dir` | `benchmark/results/v5/smoke_v5` | Smoke run artifact directory. |
| `--estimate-source` | `auto` | `auto`, `v5`, `v4`, `v3`, or `heuristic` (`auto` prefers v4 reports, then v5, then v3, then heuristic). |
| `--preflight-report` | `benchmark/results/v5/preflight_v5.json` | Preflight report path. |
| `--sanitize-after-run` | `False` | Run `benchmark.sanitize_results` after the pipeline finishes. |
| `--sanitize-in-place` | `False` | Rewrite JSONL artifacts in-place during sanitize stage. |
| `--sanitize-mode` | `warn` | Sanitizer mode: `warn` or `strict`. |
| `--sanitize-report` | `benchmark/results/v5/sanitize_report_v5.json` | Sanitizer report path. |

`strict` billing reconciliation prerequisites:

- `OPENAI_ADMIN_API_KEY` must be set.
- `OPENAI_COST_PROJECT_ID` must be set.
- `benchmark/results/v5/run_v5_meta.json` must contain start and end timestamps.

```bash
python -m benchmark.run_v5 \
    --preflight-mode smoke \
    --smoke-samples 5 \
    --seed 42 \
    --sampling-strategy context_dense \
    --memory-scope hybrid \
    --smoke-output-dir benchmark/results/v5/smoke_v5_seed42_n5 \
    --preflight-report benchmark/results/v5/preflight_v5.json
```

```bash
python -m benchmark.run_v5 \
    --preflight-mode both \
    --smoke-samples 5 \
    --manifest benchmark/results/v5/subset_manifest_v5_seed42_n200.json \
    --max-samples 200 \
    --seed 42 \
    --sampling-strategy context_dense \
    --memory-scope hybrid \
    --smoke-output-dir benchmark/results/v5/smoke_v5_seed42_n5 \
    --preflight-report benchmark/results/v5/preflight_v5.json
```

`preflight_v5` estimation uses empirical auxiliary usage from recent `ace_v*_metrics.json` artifacts when available and falls back to explicit defaults otherwise. Smoke scaling prefers full-pipeline measured cost from `comparison_report_v5.json.full_pipeline_cost_metered`.

### V5 ACE resume controls

`infer_ace_direct_v5` adds durability/resume flags:

- `--progress-path` default `<output>.progress.jsonl`
- `--resume-from-progress` default `true`
- `--finalize-order` default `true`

Durability artifacts:

- `benchmark/results/v5/ace_v5.jsonl.progress.jsonl`
- `benchmark/results/v5/ace_v5.jsonl.complete.json`

### V5 sanitize workflow

Sanitize existing result files in-place:

```bash
python -m benchmark.sanitize_results \
    --input-root benchmark/results \
    --version all \
    --in-place \
    --report-path benchmark/results/sanitize_report_inplace.json \
    --mode warn
```

Run sanitize automatically at the end of a v5 run:

```bash
python -m benchmark.run_v5 \
    --manifest benchmark/results/v5/subset_manifest_v5_seed42_n200.json \
    --max-samples 200 \
    --seed 42 \
    --sampling-strategy context_dense \
    --memory-scope hybrid \
    --with-report \
    --sanitize-after-run \
    --sanitize-in-place \
    --sanitize-mode warn \
    --sanitize-report benchmark/results/v5/sanitize_report_v5.json
```

### V5 required artifacts

- `benchmark/results/v5/subset_manifest_v5_seed42_n200.json`
- `benchmark/results/v5/baseline_v5.jsonl`
- `benchmark/results/v5/ace_v5.jsonl`
- `benchmark/results/v5/ace_v5_metrics.json`
- `benchmark/results/v5/baseline_v5_graded.jsonl`
- `benchmark/results/v5/ace_v5_graded.jsonl`
- `benchmark/results/v5/baseline_v5_graded_eval_metrics.json`
- `benchmark/results/v5/ace_v5_graded_eval_metrics.json`
- `benchmark/results/v5/baseline_v5_graded_errors.jsonl`
- `benchmark/results/v5/ace_v5_graded_errors.jsonl`
- `benchmark/results/v5/baseline_v5_graded_errors_error_metrics.json`
- `benchmark/results/v5/ace_v5_graded_errors_error_metrics.json`
- `benchmark/results/v5/comparison_report_v5.md`
- `benchmark/results/v5/comparison_report_v5.json`
- `benchmark/results/v5/ace_v5.jsonl.progress.jsonl`
- `benchmark/results/v5/ace_v5.jsonl.complete.json`
- `benchmark/results/v5/run_v5_meta.json`

When publishing to GitHub, use the allowlist above and avoid publishing raw JSONL artifacts.

### Backfill cost reports for existing v4 and v5 artifacts

```bash
python -m benchmark.backfill_cost_v5 --version v4
python -m benchmark.backfill_cost_v5 --version v5
```

Backfill outputs are non-destructive by default:

- `benchmark/results/v4/cost_backfill_v4.md`
- `benchmark/results/v4/cost_backfill_v4.json`
- `benchmark/results/v5/cost_backfill_v5.md`
- `benchmark/results/v5/cost_backfill_v5.json`

Result layout and schema details:

- `docs/RESULTS_STRUCTURE.md`

## CL-bench Benchmarking (V4 Reference)

The v4 pipeline adds memory-scope switching (`hybrid` default), context-level parallelism, capped-output retry, and step-level process scoring diagnostics.

### Preflight before full run (recommended)

`run_v4` now supports dual preflight modes:

- `static`: config and artifact checks plus cost/time estimate, no benchmark API calls.
- `smoke`: mini live end-to-end run on `3` to `5` tasks.
- `both`: run static first, then smoke.

Static and smoke results are written to `benchmark/results/v4/preflight_v4.json` by default.

```bash
python -m benchmark.run_v4 \
    --preflight-mode static \
    --manifest benchmark/results/v4/subset_manifest_v4_seed42_n200.json \
    --max-samples 200 \
    --seed 42 \
    --sampling-strategy context_dense \
    --memory-scope hybrid \
    --preflight-report benchmark/results/v4/preflight_v4.json
```

```bash
python -m benchmark.run_v4 \
    --preflight-mode smoke \
    --smoke-samples 5 \
    --seed 42 \
    --sampling-strategy context_dense \
    --memory-scope hybrid \
    --smoke-output-dir benchmark/results/v4/smoke_v4_seed42_n5 \
    --preflight-report benchmark/results/v4/preflight_v4.json
```

```bash
python -m benchmark.run_v4 \
    --preflight-mode both \
    --smoke-samples 5 \
    --manifest benchmark/results/v4/subset_manifest_v4_seed42_n200.json \
    --max-samples 200 \
    --seed 42 \
    --sampling-strategy context_dense \
    --memory-scope hybrid \
    --smoke-output-dir benchmark/results/v4/smoke_v4_seed42_n5 \
    --preflight-report benchmark/results/v4/preflight_v4.json
```

Smoke mode requires live model APIs and CL-bench dataset availability (Hub access or local cache).

### One-command full run

```bash
python -m benchmark.run_v4 \
    --manifest benchmark/results/v4/subset_manifest_v4_seed42_n200.json \
    --max-samples 200 \
    --seed 42 \
    --memory-scope hybrid \
    --with-report
```

### `run_v4` parameters

| Parameter | Default | Description |
|---|---|---|
| `--manifest` | `benchmark/results/v4/subset_manifest_v4_seed42_n200.json` | Subset manifest path used by baseline and ACE. |
| `--max-samples` | `200` | Target number of tasks for the full run. Must be `> 0`. |
| `--seed` | `42` | Seed used when creating a new manifest. |
| `--sampling-strategy` | `context_dense` | Sampling strategy: `task_random` or `context_dense`. |
| `--memory-scope` | `hybrid` | ACE memory retrieval scope: `hybrid`, `local`, or `global`. |
| `--output-dir` | `benchmark/results/v4` | Directory for full-run artifacts. |
| `--clear-results` | `True` | Remove existing v4 artifacts before a full run. |
| `--clear-db` | `True` | Wipe Neo4j before ACE inference in full run. |
| `--with-report` | `True` | Run eval, error analysis, and compare after inference. |
| `--preflight-mode` | `off` | Preflight mode: `off`, `static`, `smoke`, `both`. |
| `--smoke-samples` | `5` | Smoke sample count. Allowed values: `3`, `4`, `5`. |
| `--smoke-output-dir` | `benchmark/results/v4/smoke_v4` | Artifact directory for smoke run outputs. |
| `--estimate-source` | `auto` | Estimation source priority: `auto`, `v4`, `v3`, `heuristic`. |
| `--preflight-report` | `benchmark/results/v4/preflight_v4.json` | JSON file with preflight checks, estimates, and smoke measurements. |

Required v4 artifacts:

- `benchmark/results/v4/subset_manifest_v4_seed42_n200.json`
- `benchmark/results/v4/baseline_v4.jsonl`
- `benchmark/results/v4/ace_v4.jsonl`
- `benchmark/results/v4/ace_v4_metrics.json`
- `benchmark/results/v4/baseline_v4_graded.jsonl`
- `benchmark/results/v4/ace_v4_graded.jsonl`
- `benchmark/results/v4/baseline_v4_graded_errors.jsonl`
- `benchmark/results/v4/ace_v4_graded_errors.jsonl`
- `benchmark/results/v4/comparison_report_v4.md`
- `benchmark/results/v4/comparison_report_v4.json`

Optional preflight artifacts:

- `benchmark/results/v4/preflight_v4.json`
- `benchmark/results/v4/smoke_v4/` (or custom `--smoke-output-dir`) including smoke `baseline_v4.jsonl`, `ace_v4.jsonl`, graded files, and report.

If `.env` was created before V5, append missing keys from `.env.example` without changing existing values.

## CL-bench Benchmarking (V3 End to End)

The v3 pipeline is deterministic and quality-gated. Baseline and ACE runs use the same seeded subset via a shared manifest. By default, baseline v3 clears only its output file, while ACE direct v3 clears output files and wipes Neo4j. Use `--no-clear-results` and `--no-clear-db` on ACE direct runs to resume interrupted execution.

### One-command parallel run (recommended)

```bash
python -m benchmark.run_v3 \
    --manifest benchmark/results/v3/subset_manifest_v3_seed42_n200.json \
    --max-samples 200 \
    --seed 42
```

This clears v3 result files and the Neo4j database, runs baseline and ACE inference in parallel, verifies 200/200 completion, then runs eval, error analysis, and generates `comparison_report_v3.md`.

#### run_v3 parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--manifest` | `benchmark/results/v3/subset_manifest_v3_seed42_n200.json` | Path to the subset manifest JSON (task_ids, seed, max_samples). |
| `--max-samples` | `200` | Number of tasks to run per pipeline. |
| `--seed` | `42` | Random seed for reproducible subset selection (used when creating manifest). |
| `--output-dir` | `benchmark/results/v3` | Directory for all v3 output files. |
| `--clear-results` | `True` | Delete all v3 result files (inference, graded, errors, reports) before starting. Use `--no-clear-results` to resume. |
| `--clear-db` | `True` | Wipe all Neo4j nodes and relationships before starting. Use `--no-clear-db` to resume. |
| `--with-report` | `True` | After inference completes, run eval, error analysis, and generate comparison_report_v3.md. Use `--no-with-report` to run inference only. |

- `--no-clear-results`: Use when resuming after an interrupted run.
- `--no-clear-db`: Use when resuming; keeps existing ACE memory.
- `--no-with-report`: Use when you only need inference outputs and will run eval/compare separately.

`benchmark.complete_v3_pipeline.py` currently validates against a fixed target of 200 samples. For non-200 subset runs, use `--no-with-report` and run Steps 3-5 manually.

### Step 1: Run baseline v3 (seed 42, 200 samples)

```bash
python -m benchmark.infer_baseline_v3 \
    --max-samples 200 \
    --seed 42 \
    --manifest benchmark/results/v3/subset_manifest_v3_seed42_n200.json \
    --model gpt-5.1 \
    --output benchmark/results/v3/baseline_v3.jsonl
```

By default, `--clear-results` deletes the output file before starting. Use `--no-clear-results` to resume.

### Step 2: Run ACE direct v3 with balanced quality gate

```bash
python -m benchmark.infer_ace_direct_v3 \
    --max-samples 200 \
    --seed 42 \
    --manifest benchmark/results/v3/subset_manifest_v3_seed42_n200.json \
    --qg-gate-score-min 0.65 \
    --qg-lesson-score-min 0.60 \
    --qg-overlap-min 0.10 \
    --qg-max-accepted-lessons 2 \
    --output benchmark/results/v3/ace_v3.jsonl
```

By default, `--clear-results` deletes output and metrics files, and `--clear-db` wipes all Neo4j nodes and relationships. Use `--no-clear-results` and `--no-clear-db` to resume.

### Step 3: Evaluate both outputs

```bash
python -m benchmark.eval \
    --input benchmark/results/v3/baseline_v3.jsonl \
    --output benchmark/results/v3/baseline_v3_graded.jsonl \
    --judge-model gpt-5.1

python -m benchmark.eval \
    --input benchmark/results/v3/ace_v3.jsonl \
    --output benchmark/results/v3/ace_v3_graded.jsonl \
    --judge-model gpt-5.1
```

### Step 4: Run error analysis

```bash
python -m benchmark.error_analysis \
    --input benchmark/results/v3/baseline_v3_graded.jsonl \
    --output benchmark/results/v3/baseline_v3_graded_errors.jsonl

python -m benchmark.error_analysis \
    --input benchmark/results/v3/ace_v3_graded.jsonl \
    --output benchmark/results/v3/ace_v3_graded_errors.jsonl
```

### Step 5: Generate v3 comparison report

```bash
python -m benchmark.compare \
    --baseline benchmark/results/v3/baseline_v3_graded.jsonl \
    --ace benchmark/results/v3/ace_v3_graded.jsonl \
    --baseline-errors benchmark/results/v3/baseline_v3_graded_errors.jsonl \
    --ace-errors benchmark/results/v3/ace_v3_graded_errors.jsonl \
    --output benchmark/results/v3/comparison_report_v3.md \
    --title-label V3
```

### Required v3 artifacts

- `benchmark/results/v3/subset_manifest_v3_seed42_n200.json`
- `benchmark/results/v3/baseline_v3.jsonl`
- `benchmark/results/v3/ace_v3.jsonl`
- `benchmark/results/v3/ace_v3_metrics.json`
- `benchmark/results/v3/baseline_v3_graded.jsonl`
- `benchmark/results/v3/ace_v3_graded.jsonl`
- `benchmark/results/v3/baseline_v3_graded_errors.jsonl`
- `benchmark/results/v3/ace_v3_graded_errors.jsonl`
- `benchmark/results/v3/comparison_report_v3.md`
- `benchmark/results/v3/comparison_report_v3.json`

### Reproducibility checks

1. Same manifest check:
   Baseline v3 and ACE v3 should process the same ordered `task_id` list from `subset_manifest_v3_seed42_n200.json`.
2. Seed sensitivity check:
   Running with `--seed 43` and a different manifest path should produce a different task set.
3. Report integrity check:
   `benchmark.compare` will fail if baseline and ACE graded files do not contain the same `task_id` set.

## Documentation

- `docs/ALGORITHM.md`: Runtime and benchmark algorithm specification, including V5 reliability flow and diagnostics.
- `docs/SETUP.md`: Full environment setup and end-to-end benchmark execution for V5, V4, and V3.
- `docs/RESULTS_STRUCTURE.md`: Canonical versioned artifact tree and schema conventions.
- `docs/CL_BENCH_CALCULATION_DETAILS.md`: Upstream CL-bench metric and scoring calculations.
- `docs/COMPARISON_REPORT_V2_CALCULATION_DETAILS.md`: Local V2 comparison report calculation details.

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
│   ├── step_scoring.py          # Intermediate process scoring for CoT/ToT/ReAct
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
│   ├── infer_baseline_v4.py     # CL-bench baseline inference (v4 capped-output retry)
│   ├── infer_ace_direct_v4.py   # CL-bench ACE direct inference (v4 memory scopes + step scoring)
│   ├── infer_baseline_v5.py     # CL-bench baseline inference (v5 versioned artifacts)
│   ├── infer_ace_direct_v5.py   # CL-bench ACE direct inference (v5 durable resume + fail-soft memory)
│   ├── run_v3.py                # Run baseline and ACE v3 in parallel (clears results and DB)
│   ├── run_v4.py                # Run baseline and ACE v4 in parallel (with optional report)
│   ├── run_v5.py                # Run baseline and ACE v5 in parallel (with reliability features)
│   ├── preflight_v4.py          # V4 static checks, estimates, and mini smoke orchestration
│   ├── preflight_v5.py          # V5 static checks, estimates, and mini smoke orchestration
│   ├── backfill_cost_v5.py      # Supplemental full-cost backfill for existing v4/v5 outputs
│   ├── complete_v4_pipeline.py  # V4 post-inference eval/error/compare pipeline
│   ├── complete_v5_pipeline.py  # V5 post-inference eval/error/compare pipeline with side retries
│   ├── sampling.py              # Seeded subset sampling + manifest reuse
│   ├── eval.py                  # Rubric-based evaluation
│   ├── error_analysis.py        # Error type classification
│   ├── compare.py               # Side-by-side comparison
│   ├── costing.py               # Centralized token-cost and billed reconciliation utilities
│   ├── metrics.py               # Token/latency collection
│   └── results/                 # Output directory
│       ├── v1/                  # V1 benchmark results
│       ├── v2/                  # V2 benchmark results
│       ├── v3/                  # V3 benchmark results
│       ├── v4/                  # V4 benchmark results
│       └── v5/                  # V5 benchmark results
└── docs/
    ├── ALGORITHM.md                             # Runtime and benchmark algorithm documentation
    ├── SETUP.md                                 # Environment setup and benchmark workflow
    ├── RESULTS_STRUCTURE.md                     # Canonical result layout and schema details
    ├── CL_BENCH_CALCULATION_DETAILS.md          # Upstream CL-bench calculation details
    └── COMPARISON_REPORT_V2_CALCULATION_DETAILS.md # Local v2 report calculation details
```

## References

- CL-bench: [github.com/Tencent-Hunyuan/CL-bench](https://github.com/Tencent-Hunyuan/CL-bench)
- CL-bench dataset: [huggingface.co/datasets/tencent/CL-bench](https://huggingface.co/datasets/tencent/CL-bench)
- CL-bench paper: [arxiv.org/abs/2602.03587](https://arxiv.org/abs/2602.03587)
