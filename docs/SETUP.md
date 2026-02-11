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

The Gemini API powers the ACE Reflector, which extracts lessons from execution traces. Gemini 3.0 Flash is the default model.

**How to get your API key:**

1. Go to [Google AI Studio API Keys](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Select an existing Google Cloud project, or let it create one for you
5. Copy the generated API key string

**Set in your `.env`:**

```env
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_MODEL=gemini-3.0-flash
```

**Pricing:** Free tier available with generous rate limits. No credit card required to start. See [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing) for current limits.

**Verify it works:**

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.0-flash:generateContent?key=$GEMINI_API_KEY" \
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

**V3 benchmark `--clear-db` (default: on):** The v3 inference script and `run_v3.py` wipe all nodes and relationships in the database before starting. Use a Neo4j instance dedicated to benchmarking, or pass `--no-clear-db` to resume without wiping.

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
| `GEMINI_MODEL` | `gemini-3.0-flash` | No | Gemini model name for the ACE Reflector |
| `OPENAI_API_KEY` | (none) | For benchmark | OpenAI API key from platform.openai.com |
| `OPENAI_MODEL` | `gpt-5.1` | No | OpenAI model name for solver and benchmark |
| `OPENAI_API_BASE` | (none) | No | Custom API base URL for OpenAI-compatible endpoints (HuggingFace, vLLM, etc.) |

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
GEMINI_MODEL=gemini-3.0-flash

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
GEMINI_MODEL=gemini-3.0-flash
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
```

### Profile: HuggingFace Inference Endpoint

For benchmarking with open-source models deployed on HuggingFace.

```env
LLM_BACKEND=openai
OPENAI_API_KEY=hf_your-token
OPENAI_API_BASE=https://your-endpoint.endpoints.huggingface.cloud/v1
OPENAI_MODEL=meta-llama/Llama-3.1-8B-Instruct

GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-3.0-flash

NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
```

---

## Troubleshooting

**"GEMINI_API_KEY not configured" error:**
Ensure your `.env` file is in the project root directory and that `python-dotenv` is installed (`pip install python-dotenv`). All entry points load `.env` automatically via `load_dotenv()`.

**"Neo4j credentials are not configured" error:**
Check that `NEO4J_URI` starts with `neo4j+s://` (not `bolt://` or `neo4j://`). AuraDB requires TLS encryption.

**Neo4j instance is paused:**
Visit [console.neo4j.io](https://console.neo4j.io), find your instance, and click "Resume". Free instances pause after 3 days of no write activity.

**"OPENAI_API_KEY not set" when running benchmarks:**
The benchmark scripts (`infer_baseline.py`, `eval.py`, `error_analysis.py`) require an OpenAI API key even when the main CLI uses Gemini. Set `OPENAI_API_KEY` in your `.env`.

**Google Search returns "GOOGLE_CSE_ID not configured":**
This is expected if you have not set up a Programmable Search Engine. The `google_search` tool is optional. The agent will still work with CoT and ToT modes.

**Tavily returns "tavily-python not installed":**
Install the optional dependency: `pip install tavily-python`. The `deep_research` tool requires both the package and a valid `TAVILY_API_KEY`.
