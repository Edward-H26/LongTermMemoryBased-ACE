# Environment Setup Guide

This guide walks through every environment variable used by the LTMBSE ACE project, how to obtain the required credentials, and what each tuning parameter controls.

## Quick Start (Minimum Required)

To run the basic ACE agent via the CLI, you need credentials from two services:

| Variable | Service | Required For |
|---|---|---|
| `GEMINI_API_KEY` | Google AI Studio | ACE Reflector LLM calls |
| `NEO4J_URI` | Neo4j AuraDB | Memory persistence |
| `NEO4J_USERNAME` | Neo4j AuraDB | Memory persistence |
| `NEO4J_PASSWORD` | Neo4j AuraDB | Memory persistence |

To also run the CL-bench benchmark pipeline, you need one additional credential:

| Variable | Service | Required For |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI Platform | GPT-5.1 solver and evaluation judge |

Once you have these, copy the template and fill in your values:

```bash
cp .env.example .env
```

---

## Service Setup Guides

### 1. Google Gemini API (Required)

The Gemini API powers the ACE Reflector, which extracts lessons from execution traces. `gemini-3-flash-preview` is the current default model in this repository.

**How to get your API key:**

1. Go to [Google AI Studio API Keys](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Select an existing Google Cloud project, or let it create one for you
5. Copy the generated API key string

**Set in your `.env`:**

```env
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-3-flash-preview
```

**Pricing:** Free tier available with generous rate limits. No credit card required to start. See [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing) for current limits.

**Verify it works:**

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key=$GEMINI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"Say hello"}]}]}'
```

A successful response returns JSON with a `candidates` array.

---

### 2. Neo4j AuraDB (Required)

Neo4j stores the ACE memory bullets as JSON blobs attached to user nodes. The free tier is sufficient for development and benchmarking.

**How to create a free instance:**

1. Go to [Neo4j Aura Console](https://console.neo4j.io/?product=aura-db)
2. Sign in or create a Neo4j account
3. Click **"New Instance"**
4. Select **"Create Free instance"**
5. **Important:** Copy the generated password immediately. It is shown only once. You can also download the credentials as a `.txt` file.
6. Tick the confirmation checkbox and click **"Continue"**
7. Wait for the instance to finish provisioning (takes about 1 minute)
8. Once running, copy the **Connection URI** from the instance card (it looks like `neo4j+s://abcdef12.databases.neo4j.io`)

**Set in your `.env`:**

```env
NEO4J_URI=neo4j+s://abcdef12.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-generated-password
```

**Free tier limitations:**
- One free instance per account
- 200,000 nodes and 400,000 relationships
- Instance pauses after 3 days with no write queries
- Paused instances are deleted after 30 days (no recovery)

**Keep your instance alive:** The ACE agent writes to Neo4j on every execution (saving memory bullets), so regular usage prevents auto-pause. For infrequent use, visit the Aura Console and click "Resume" on your paused instance before running the agent.

**V3 benchmark `--clear-db` (default: on):** `infer_ace_direct_v3.py` and `run_v3.py` wipe all nodes and relationships in the database before starting. Use a Neo4j instance dedicated to benchmarking, or pass `--no-clear-db` to resume without wiping.

**Verify it works:**

```bash
python -c "
from neo4j import GraphDatabase
import os
driver = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD')))
with driver.session() as session:
    result = session.run('RETURN 1 AS n')
    print('Connected successfully:', result.single()['n'])
driver.close()
"
```

---

### 3. OpenAI API (Required for Benchmarking)

The OpenAI API is used for the GPT-5.1 (High) solver in the CL-bench benchmark and for the GPT-5.1 evaluation judge. It is not needed if you only use the CLI with the Gemini backend.

**How to get your API key:**

1. Go to [OpenAI Platform API Keys](https://platform.openai.com/api-keys)
2. Sign in or create an OpenAI account
3. Click **"Create new secret key"**
4. Give it a name (e.g., "LTMBSE ACE Benchmark")
5. Copy the key immediately. It is shown only once.
6. Add a payment method under Billing if you do not already have one

**Set in your `.env`:**

```env
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-5.1
```

**Pricing (GPT-5.1):**
- Input: $1.25 per 1M tokens
- Output: $10.00 per 1M tokens
- Estimated cost for 500-sample CL-bench run: ~$32 total

**Optional: Custom base URL** (for HuggingFace Inference Endpoints or other OpenAI-compatible APIs):

```env
OPENAI_API_BASE=https://your-endpoint.endpoints.huggingface.cloud/v1
```

**Verify it works:**

```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | head -c 200
```

A successful response returns a JSON list of available models.

---

### 4. Google Custom Search (Optional)

Enables the `google_search` tool in the ReAct solver for web search queries.

**How to set it up:**

1. Go to [Programmable Search Engine Control Panel](https://programmablesearchengine.google.com/controlpanel/create)
2. Sign in with your Google account
3. Name your search engine (e.g., "ACE Web Search")
4. Under "What to search", select **"Search the entire web"**
5. Click **"Create"**
6. On the next page, copy the **Search Engine ID** (a string like `a1b2c3d4e5f6g7h8i`)
7. Go to [Google Cloud Console](https://console.cloud.google.com/apis/library/customsearch.googleapis.com) and enable the **Custom Search API** for your project

**Set in your `.env`:**

```env
GOOGLE_CSE_ID=a1b2c3d4e5f6g7h8i
```

The search API authenticates using your existing `GEMINI_API_KEY` (same Google Cloud project).

**Note:** The Custom Search JSON API is scheduled to close to new customers on January 1, 2027. Existing users can continue until migration to Vertex AI Search.

---

### 5. Tavily Search (Optional)

Enables the `deep_research` tool in the ReAct solver for AI-powered web research with source aggregation.

**How to get your API key:**

1. Go to [Tavily Platform](https://app.tavily.com)
2. Sign in or create an account (Google, GitHub, or email)
3. Your API key is displayed on the dashboard after login
4. Copy the key (format: `tvly-...`)

**Set in your `.env`:**

```env
TAVILY_API_KEY=tvly-...your-key-here
```

**Pricing:** 1,000 free API credits per month, no credit card required. Paid plans available for higher volume.

---

## Complete Environment Variable Reference

### LLM Configuration

| Variable | Default | Required | Description |
|---|---|---|---|
| `LLM_BACKEND` | `gemini` | No | Which LLM backend to use: `gemini` or `openai` |
| `GEMINI_API_KEY` | (none) | **Yes** | Google Gemini API key from AI Studio |
| `GEMINI_MODEL` | `gemini-3-flash-preview` | No | Gemini model name for the ACE Reflector |
| `OPENAI_API_KEY` | (none) | For benchmark | OpenAI API key from platform.openai.com |
| `OPENAI_MODEL` | `gpt-5.1` | No | OpenAI model name for solver and benchmark |
| `OPENAI_API_BASE` | (none) | No | Custom API base URL for OpenAI-compatible endpoints (HuggingFace, vLLM, etc.) |
| `OPENAI_ADMIN_API_KEY` | (none) | For strict cost reconciliation | OpenAI admin key for organization usage and cost endpoints |
| `OPENAI_COST_PROJECT_ID` | (none) | For strict cost reconciliation | OpenAI project id used to scope billed-cost queries |

### Neo4j Database

| Variable | Default | Required | Description |
|---|---|---|---|
| `NEO4J_URI` | (none) | **Yes** | Neo4j connection URI (e.g., `neo4j+s://xxx.databases.neo4j.io`) |
| `NEO4J_USERNAME` | (none) | **Yes** | Neo4j username (typically `neo4j` for AuraDB) |
| `NEO4J_PASSWORD` | (none) | **Yes** | Neo4j password (generated during instance creation) |
| `NEO4J_DATABASE` | (none) | No | Neo4j database name (for multi-database setups, leave unset for default) |

### ACE Algorithm Tuning

| Variable | Default | Required | Description |
|---|---|---|---|
| `ACE_LLM_TEMPERATURE` | `0.2` | No | Temperature for ACE pipeline LLM calls (Reflector/Curator) |
| `ACE_CURATOR_SIMILARITY` | `0.9` | No | Jaccard similarity threshold for merging new bullets with existing ones |
| `ACE_CURATOR_SUPPORT_SIMILARITY` | `0.95` | No | Stricter threshold for merging supporting/supplemental bullets |
| `ACE_CURATOR_USE_LLM` | `false` | No | Use LLM for curation (`true`) or heuristic mode (`false`). Heuristic is faster and cheaper. |
| `ACE_MEMORY_BASE_STRENGTH` | `100.0` | No | Baseline strength value for new bullets. Higher values make bullets persist longer before decay removes them. |
| `ACE_MEMORY_SIMILARITY_MERGE` | `0.9` | No | Similarity threshold for automatically merging incoming bullets with existing ones during delta application |

### ACE CL-bench Enhancements

| Variable | Default | Required | Description |
|---|---|---|---|
| `ACE_INJECTION_MODE` | `post_context` | No | Where to inject ACE bullets: `post_context` (after context, before task) or `pre_context` (before system message, original behavior) |
| `ACE_WEIGHT_RELEVANCE` | `0.25` | No | Weight for query relevance in retrieval scoring. Set to `0.55` for CL-bench. |
| `ACE_WEIGHT_STRENGTH` | `0.55` | No | Weight for historical strength in retrieval scoring. Set to `0.25` for CL-bench. |
| `ACE_WEIGHT_TYPE` | `0.20` | No | Weight for memory type priority in retrieval scoring |
| `ACE_SEED_META_STRATEGIES` | `true` | No | Pre-seed new memory contexts with meta-strategy bullets targeting common failure modes |

### ACE V3 Quality Gate Defaults

| Variable | Default | Required | Description |
|---|---|---|---|
| `ACE_QG_GATE_SCORE_MIN` | `0.65` | No | Minimum task-level gate score required to apply memory updates |
| `ACE_QG_LESSON_SCORE_MIN` | `0.60` | No | Minimum per-lesson quality score for lesson acceptance |
| `ACE_QG_OVERLAP_MIN` | `0.10` | No | Minimum token-overlap score between question and lesson |
| `ACE_QG_MAX_ACCEPTED_LESSONS` | `2` | No | Maximum accepted lessons per task after sorting by quality and overlap |

### ACE V4 Memory and Process Scoring Defaults

| Variable | Default | Required | Description |
|---|---|---|---|
| `ACE_MEMORY_SCOPE_MODE` | `hybrid` | No | Memory retrieval scope: `hybrid`, `local`, or `global` |
| `ACE_GLOBAL_GATE_SCORE_MIN` | `0.80` | No | Minimum gate score required before writing accepted lessons into global memory |
| `ACE_LOCAL_TOP_K` | `3` | No | Number of context-local bullets retrieved in v4 |
| `ACE_GLOBAL_TOP_K` | `2` | No | Number of global bullets retrieved in v4 |
| `ACE_CONTEXT_WORKERS` | `6` | No | Parallel worker count for context-level ACE v4 inference |
| `ACE_STEP_SCORING_MODE` | `near_full` | No | Step scoring mode: `off`, `near_full`, or `full` |
| `ACE_STEP_SCORER_MODEL` | `gpt-5.1` | No | LLM verifier model for intermediate step scoring |
| `ACE_STEP_SCORE_WORKERS` | `8` | No | Worker count used for per-step verifier calls |
| `ACE_STEP_SCORE_MIN` | `0.45` | No | ToT branch pruning threshold based on mean step score |
| `ACE_MAX_COMPLETION_TOKENS` | `8192` | No | Default completion cap for v4 inference calls |
| `ACE_EMPTY_OUTPUT_RETRY_MAX_TOKENS` | `16384` | No | Retry completion cap when first output is empty at token limit |

### ACE V5 Reliability and Durability Defaults

| Variable | Default | Required | Description |
|---|---|---|---|
| `ACE_NEO4J_RETRY_MAX` | `2` | No | Number of retries for Neo4j load/save after first attempt |
| `ACE_NEO4J_RETRY_BACKOFF_SEC` | `1.0` | No | Exponential backoff base seconds for Neo4j retries |
| `ACE_NEO4J_RECONNECT_ON_SESSION_EXPIRED` | `true` | No | Reset and reconnect Neo4j driver when `SessionExpired` is detected |
| `ACE_V5_RESUME_FROM_PROGRESS` | `true` | No | Default for `infer_ace_direct_v5 --resume-from-progress` |
| `ACE_V5_FINALIZE_ORDER` | `true` | No | Default for `infer_ace_direct_v5 --finalize-order` |
| `ACE_V5_PROGRESS_PATH` | (unset) | No | Optional override for ACE v5 durable journal path |

### Benchmark Cost Reporting Defaults

| Variable | Default | Required | Description |
|---|---|---|---|
| `BENCHMARK_COST_MODE` | `dual_source` | No | Cost accounting mode for compare (`legacy` or `dual_source`) |
| `BENCHMARK_BILLING_POLICY` | `strict` | No | Billing reconciliation policy (`strict` hard-fails or `off`) |

### Optional Search Tools

| Variable | Default | Required | Description |
|---|---|---|---|
| `GOOGLE_CSE_ID` | (none) | No | Google Custom Search Engine ID for the `google_search` tool |
| `TAVILY_API_KEY` | (none) | No | Tavily API key for the `deep_research` tool |

### Analysis and Debugging

| Variable | Default | Required | Description |
|---|---|---|---|
| `ACE_ANALYZE_LEARNER_ID` | `default_user` | No | Default learner ID used by `analyze_memory.py` when `--learner` is not specified |

### Legacy Compatibility

These variables are supported as fallbacks for compatibility with the original Noodeia project. You do not need to set them if you use the standard `NEO4J_*` variables above.

| Variable | Fallback For | Description |
|---|---|---|
| `NEXT_PUBLIC_NEO4J_URI` | `NEO4J_URI` | Used if `NEO4J_URI` is not set |
| `NEXT_PUBLIC_NEO4J_USERNAME` | `NEO4J_USERNAME` | Used if `NEO4J_USERNAME` is not set |
| `NEXT_PUBLIC_NEO4J_PASSWORD` | `NEO4J_PASSWORD` | Used if `NEO4J_PASSWORD` is not set |

---

## Configuration Profiles

### Profile: Local Development (CLI only)

Minimal setup for interactive use. Uses Gemini as the sole LLM backend.

```env
LLM_BACKEND=gemini
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-3-flash-preview

NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

ACE_LLM_TEMPERATURE=0.2
ACE_CURATOR_USE_LLM=false
ACE_SEED_META_STRATEGIES=true
```

### Profile: CL-bench Benchmarking

Full setup for running the baseline vs. ACE comparison on CL-bench.

```env
LLM_BACKEND=openai
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-3-flash-preview
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-5.1

NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

ACE_LLM_TEMPERATURE=0.2
ACE_CURATOR_USE_LLM=false
ACE_INJECTION_MODE=post_context
ACE_WEIGHT_RELEVANCE=0.55
ACE_WEIGHT_STRENGTH=0.25
ACE_WEIGHT_TYPE=0.20
ACE_SEED_META_STRATEGIES=true
ACE_QG_GATE_SCORE_MIN=0.65
ACE_QG_LESSON_SCORE_MIN=0.60
ACE_QG_OVERLAP_MIN=0.10
ACE_QG_MAX_ACCEPTED_LESSONS=2
ACE_MEMORY_SCOPE_MODE=hybrid
ACE_GLOBAL_GATE_SCORE_MIN=0.80
ACE_LOCAL_TOP_K=3
ACE_GLOBAL_TOP_K=2
ACE_CONTEXT_WORKERS=6
ACE_STEP_SCORING_MODE=near_full
ACE_STEP_SCORER_MODEL=gpt-5.1
ACE_STEP_SCORE_WORKERS=8
ACE_STEP_SCORE_MIN=0.45
ACE_MAX_COMPLETION_TOKENS=8192
ACE_EMPTY_OUTPUT_RETRY_MAX_TOKENS=16384
```

### Profile: HuggingFace Inference Endpoint

For benchmarking with open-source models deployed on HuggingFace.

```env
LLM_BACKEND=openai
OPENAI_API_KEY=hf_your-token
OPENAI_API_BASE=https://your-endpoint.endpoints.huggingface.cloud/v1
OPENAI_MODEL=meta-llama/Llama-3.1-8B-Instruct

GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-3-flash-preview

NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

---

## V5 End-to-End Benchmarking (Recommended)

### Recommended one-command run

Use `run_v5` for durable resume, context-parallel ACE inference, and side-retry post-pipeline execution.

```bash
python -m benchmark.run_v5 \
  --manifest benchmark/results/v5/subset_manifest_v5_seed42_n200.json \
  --max-samples 200 \
  --seed 42 \
  --sampling-strategy context_dense \
  --memory-scope hybrid
```

Current defaults in `benchmark/run_v5.py`:

- `--sampling-strategy`: `context_dense`
- `--memory-scope`: `hybrid`
- `--clear-results`: true
- `--clear-db`: true
- `--with-report`: true
- `--preflight-mode`: `off`
- `--smoke-samples`: `5` (allowed values: `3`, `4`, `5`)
- `--smoke-output-dir`: `benchmark/results/v5/smoke_v5`
- `--estimate-source`: `auto` (`v5`, `v4`, `v3`, `heuristic` supported). In `auto`, estimate source priority is `v4 -> v5 -> v3 -> heuristic`.
- `--preflight-report`: `benchmark/results/v5/preflight_v5.json`
- `--cost-mode`: `dual_source`
- `--billing-policy`: `strict`
- `--openai-admin-key-env`: `OPENAI_ADMIN_API_KEY`
- `--openai-project-id-env`: `OPENAI_COST_PROJECT_ID`
- `--sanitize-after-run`: false
- `--sanitize-in-place`: false
- `--sanitize-mode`: `warn`
- `--sanitize-report`: `benchmark/results/v5/sanitize_report_v5.json`

With defaults, `run_v5`:

1. Clears existing v5 artifacts.
2. Optionally clears Neo4j.
3. Runs `infer_baseline_v5` and `infer_ace_direct_v5` in parallel.
4. Uses ACE per-task durable progress journal for crash-safe resume.
5. Writes `benchmark/results/v5/run_v5_meta.json` with phase timestamps.
6. Runs `benchmark.complete_v5_pipeline`, which performs parallel eval, parallel error analysis, per-side retries, parity validation, full-pipeline metered cost reporting, and strict billed-cost reconciliation.
7. Optionally runs `benchmark.sanitize_results` for post-run JSONL sanitization.

Strict billing prerequisites for default v5 flow:

1. `OPENAI_ADMIN_API_KEY` is set.
2. `OPENAI_COST_PROJECT_ID` is set.
3. `benchmark/results/v5/run_v5_meta.json` exists and contains usable start/end timestamps.

### V5 dual preflight

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

`preflight_v5` estimate behavior:

1. Step-scoring token and latency assumptions are derived from recent `ace_v*_metrics.json` artifacts when present.
2. If empirical artifacts are missing, fallback constants are used and recorded in assumptions.
3. Smoke scaling prefers full-pipeline measured cost from `comparison_report_v5.json.full_pipeline_cost_metered.combined_total_cost_usd`.

### V5 manual sequence

```bash
python -m benchmark.infer_baseline_v5 \
  --max-samples 200 \
  --seed 42 \
  --manifest benchmark/results/v5/subset_manifest_v5_seed42_n200.json \
  --sampling-strategy context_dense \
  --output benchmark/results/v5/baseline_v5.jsonl

python -m benchmark.infer_ace_direct_v5 \
  --max-samples 200 \
  --seed 42 \
  --manifest benchmark/results/v5/subset_manifest_v5_seed42_n200.json \
  --sampling-strategy context_dense \
  --memory-scope hybrid \
  --progress-path benchmark/results/v5/ace_v5.jsonl.progress.jsonl \
  --resume-from-progress \
  --finalize-order \
  --output benchmark/results/v5/ace_v5.jsonl

python -m benchmark.complete_v5_pipeline \
  --output-dir benchmark/results/v5 \
  --max-samples 200 \
  --judge-model gpt-5.1 \
  --stage-retry-max 1 \
  --cost-mode dual_source \
  --billing-policy strict \
  --run-meta benchmark/results/v5/run_v5_meta.json
```

### Sanitize existing JSONL in-place

```bash
python -m benchmark.sanitize_results \
  --input-root benchmark/results \
  --version all \
  --in-place \
  --report-path benchmark/results/sanitize_report_inplace.json \
  --mode warn
```

### Enable post-run sanitization in `run_v5`

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

### Required v5 artifacts

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
- `benchmark/results/v5/preflight_v5.json` (when preflight is used)

Publish only allowlisted artifacts to GitHub, keep raw JSONL and progress journals local.

Backfill full-cost reports for existing runs:

```bash
python -m benchmark.backfill_cost_v5 --version v4
python -m benchmark.backfill_cost_v5 --version v5
```

See `docs/RESULTS_STRUCTURE.md` for canonical artifact schema details.

## V4 End-to-End Benchmarking (Reference)

### Recommended one-command run

Use `run_v4` when you want deterministic subset selection, parallel baseline and ACE inference, v4 memory-scope switching, and automatic post-inference evaluation/report generation.

```bash
python -m benchmark.run_v4 \
  --manifest benchmark/results/v4/subset_manifest_v4_seed42_n200.json \
  --max-samples 200 \
  --seed 42 \
  --memory-scope hybrid
```

Current defaults in `benchmark/run_v4.py`:

- `--sampling-strategy`: `context_dense`
- `--memory-scope`: `hybrid`
- `--clear-results`: true
- `--clear-db`: true
- `--with-report`: true
- `--preflight-mode`: `off`
- `--smoke-samples`: `5` (allowed values: `3`, `4`, `5`)
- `--smoke-output-dir`: `benchmark/results/v4/smoke_v4`
- `--estimate-source`: `auto`
- `--preflight-report`: `benchmark/results/v4/preflight_v4.json`

With defaults, the command:

1. Clears existing v4 artifacts.
2. Clears Neo4j.
3. Runs `infer_baseline_v4` and `infer_ace_direct_v4` in parallel.
4. Verifies output line counts against `--max-samples`.
5. Runs `benchmark.complete_v4_pipeline --output-dir ... --skip-wait --max-samples ...` for eval, error analysis, and comparison report generation.

### Dual preflight mode (`static + estimate` and `mini smoke`)

Use preflight when you want to validate setup and estimate budget before committing to a full 200-task run.

#### Stage A: static checks + estimate (no benchmark API calls)

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

#### Stage B: mini live smoke (3 to 5 tasks, full post-pipeline)

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

#### Run both stages in one command

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

`preflight_v4.json` includes:

- blocking issues and warnings
- source-backed low/base/high cost and wall-time estimates
- smoke measurements (runtime, inference cost) when smoke runs
- scaled full-run projection from smoke (`target_samples / smoke_samples`)

Smoke mode requires API connectivity and CL-bench dataset access (Hub online or existing local cache).

If `status` is `failed`, fix `blocking_issues` before running the full benchmark.

### Manual sequence (explicit control)

```bash
python -m benchmark.infer_baseline_v4 \
  --max-samples 200 \
  --seed 42 \
  --manifest benchmark/results/v4/subset_manifest_v4_seed42_n200.json \
  --sampling-strategy context_dense \
  --max-completion-tokens 8192 \
  --empty-output-retry-max-tokens 16384 \
  --output benchmark/results/v4/baseline_v4.jsonl

python -m benchmark.infer_ace_direct_v4 \
  --max-samples 200 \
  --seed 42 \
  --manifest benchmark/results/v4/subset_manifest_v4_seed42_n200.json \
  --sampling-strategy context_dense \
  --memory-scope hybrid \
  --local-top-k 3 \
  --global-top-k 2 \
  --context-workers 6 \
  --step-scoring-mode near_full \
  --step-scorer-model gpt-5.1 \
  --step-score-workers 8 \
  --step-score-min 0.45 \
  --qg-gate-score-min 0.65 \
  --qg-lesson-score-min 0.60 \
  --qg-overlap-min 0.10 \
  --qg-max-accepted-lessons 2 \
  --max-completion-tokens 8192 \
  --empty-output-retry-max-tokens 16384 \
  --output benchmark/results/v4/ace_v4.jsonl

python -m benchmark.complete_v4_pipeline \
  --max-samples 200 \
  --judge-model gpt-5.1
```

### Required v4 artifacts

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
- `benchmark/results/v4/preflight_v4.json` (when preflight is used)

### `.env` sync behavior for new V4 keys

If your local `.env` was created before V5, append missing keys from `.env.example` and keep existing values unchanged.

Example key check:

```bash
comm -13 <(rg '^[A-Z0-9_]+=.*' .env -o | sed 's/=.*//' | sort) \
         <(rg '^[A-Z0-9_]+=.*' .env.example -o | sed 's/=.*//' | sort)
```

## V3 End-to-End Benchmarking

### Recommended one-command run

Use `run_v3` when you want deterministic subset selection, parallel baseline and ACE inference, and automatic post-inference evaluation/report generation.

```bash
python -m benchmark.run_v3 \
  --manifest benchmark/results/v3/subset_manifest_v3_seed42_n200.json \
  --max-samples 200 \
  --seed 42
```

Current defaults in `benchmark/run_v3.py`:

- `--clear-results`: true
- `--clear-db`: true
- `--with-report`: true

With defaults, the command:

1. Clears existing v3 artifacts.
2. Clears Neo4j.
3. Runs `infer_baseline_v3` and `infer_ace_direct_v3` in parallel.
4. Verifies output line counts.
5. Runs `benchmark.complete_v3_pipeline --skip-wait`, which performs eval, error analysis, and comparison report generation.

### Important constraint for non-200 runs

`benchmark.complete_v3_pipeline.py` currently uses `TARGET = 200`.
If `--max-samples` is not 200, run:

- `python -m benchmark.run_v3 ... --no-with-report`

Then run the eval, error analysis, and compare commands manually.

### Manual sequence (explicit control)

```bash
python -m benchmark.infer_baseline_v3 \
  --max-samples 200 \
  --seed 42 \
  --manifest benchmark/results/v3/subset_manifest_v3_seed42_n200.json \
  --output benchmark/results/v3/baseline_v3.jsonl

python -m benchmark.infer_ace_direct_v3 \
  --max-samples 200 \
  --seed 42 \
  --manifest benchmark/results/v3/subset_manifest_v3_seed42_n200.json \
  --qg-gate-score-min 0.65 \
  --qg-lesson-score-min 0.60 \
  --qg-overlap-min 0.10 \
  --qg-max-accepted-lessons 2 \
  --output benchmark/results/v3/ace_v3.jsonl

python -m benchmark.eval \
  --input benchmark/results/v3/baseline_v3.jsonl \
  --output benchmark/results/v3/baseline_v3_graded.jsonl \
  --judge-model gpt-5.1

python -m benchmark.eval \
  --input benchmark/results/v3/ace_v3.jsonl \
  --output benchmark/results/v3/ace_v3_graded.jsonl \
  --judge-model gpt-5.1

python -m benchmark.error_analysis \
  --input benchmark/results/v3/baseline_v3_graded.jsonl \
  --output benchmark/results/v3/baseline_v3_graded_errors.jsonl

python -m benchmark.error_analysis \
  --input benchmark/results/v3/ace_v3_graded.jsonl \
  --output benchmark/results/v3/ace_v3_graded_errors.jsonl

python -m benchmark.compare \
  --baseline benchmark/results/v3/baseline_v3_graded.jsonl \
  --ace benchmark/results/v3/ace_v3_graded.jsonl \
  --baseline-errors benchmark/results/v3/baseline_v3_graded_errors.jsonl \
  --ace-errors benchmark/results/v3/ace_v3_graded_errors.jsonl \
  --output benchmark/results/v3/comparison_report_v3.md \
  --title-label V3
```

### Reproducibility checks

1. Baseline and ACE should share the same manifest path.
2. Baseline and ACE graded files must have identical task-id sets.
3. `benchmark.compare` fails fast if task-id sets differ.
4. For resumed runs, verify there are no duplicate `task_id` rows before grading.

---

## Troubleshooting

**"GEMINI_API_KEY not configured" error:**
Ensure your `.env` file is in the project root directory and that `python-dotenv` is installed (`pip install python-dotenv`). All entry points load `.env` automatically via `load_dotenv()`.

**"Neo4j credentials are not configured" error:**
Check that `NEO4J_URI` starts with `neo4j+s://` (not `bolt://` or `neo4j://`). AuraDB requires TLS encryption.

**Neo4j instance is paused:**
Visit [console.neo4j.io](https://console.neo4j.io), find your instance, and click "Resume". Free instances pause after 3 days of no write activity.

**"OPENAI_API_KEY not set" when running benchmarks:**
The benchmark scripts (`run_v3.py`, `infer_baseline_v3.py`, `infer_ace_direct_v3.py`, `eval.py`, `error_analysis.py`) require OpenAI credentials either directly or in downstream stages. Set `OPENAI_API_KEY` in your `.env`.

**`run_v3` fails on non-200 subset with report enabled:**
`complete_v3_pipeline.py` currently checks for 200 lines. Use `--no-with-report` for non-200 runs, then run post-steps manually.

**`run_v4` reports empty model outputs on difficult tasks:**
V4 automatically retries when the first response is empty and completion hit token cap. You can further increase `ACE_MAX_COMPLETION_TOKENS` and `ACE_EMPTY_OUTPUT_RETRY_MAX_TOKENS`, or pass `--max-completion-tokens` and `--empty-output-retry-max-tokens` directly.

**Google Search returns "GOOGLE_CSE_ID not configured":**
This is expected if you have not set up a Programmable Search Engine. The `google_search` tool is optional. The agent will still work with CoT and ToT modes.

**Tavily returns "tavily-python not installed":**
Install the optional dependency: `pip install tavily-python`. The `deep_research` tool requires both the package and a valid `TAVILY_API_KEY`.
