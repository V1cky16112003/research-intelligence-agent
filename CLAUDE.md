# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Lint
ruff check . --ignore E501,E402

# Run all tests (no Docker required — conftest stubs heavy deps)
pytest tests/ -v --tb=short

# Run a single test file
pytest tests/test_agent.py -v

# Run RAGAS evaluation against live DB
python -m eval.run_ragas --ci --limit 10

# Stage 1: Load ArXiv papers into Neon (requires dataset + DATABASE_URL)
DATABASE_URL="..." PYTHONPATH=. python3 -m ingestion.loader \
  --file /path/to/arxiv-metadata-oai-snapshot.json \
  --limit 50000 --categories cs.LG,cs.AI,cs.CL,cs.CV

# Stage 2: Embed chunks — run on Kaggle/Colab GPU (see kaggle_ingestion.ipynb)
# Neon free tier is 512 MB; embed ~10k papers to stay within limit
DATABASE_URL="..." PYTHONPATH=. python3 -m ingestion.pipeline --limit 10000 --batch-size 200

# Local dev with Docker
docker-compose up --build
```

## Architecture

The app is a FastAPI service (port 7860) deployed on Hugging Face Spaces (Docker) with a React frontend on Vercel.

**Request flow:** React UI → `POST /chat` → `run_agent()` → LangGraph graph → tool calls → Neon Postgres / DuckDuckGo → audit log write.

### LangGraph Agent (`agent/`)

Four-node state machine: **Planner → Executor → Critic → Reporter**

- `state.py` — `AgentState` TypedDict; the graph's shared state schema
- `nodes.py` — one async function per node; gateway is fetched from the module-level registry (not stored in state)
- `registry.py` — module-level singleton (`set_gateway` / `get_gateway`); avoids LangGraph stripping non-schema state keys
- `graph.py` — wires the graph, `init_graph()` sets up `AsyncPostgresSaver` at startup, falls back to in-memory if no DB
- `tools.py` — `TOOL_DISPATCH` dict mapping tool name → async function: `rag_retrieval`, `sql_analytics`, `web_search`
- `gateway.py` — `LLMGateway`: Groq (Llama 3.3 70B) primary, Gemini 2.5 Flash fallback on 429; wraps Upstash Redis cache
- `redis_client.py` — thin Upstash REST client (no persistent TCP connection)

The Critic node returns `RETRY` or `PASS`; the graph loops back to Executor up to `MAX_RETRIES = 3` times before forcing Reporter.

### API (`app/main.py`)

`Settings` (pydantic-settings) reads from `.env`. Redis URL is assembled at runtime from either `REDIS_URL` or `UPSTASH_REDIS_REST_URL` + `UPSTASH_REDIS_REST_TOKEN`. Every `/chat` request writes a row to `query_audit_log` (latency, tokens, tools called, retrieved chunk IDs).

### Database (`db/`)

Neon Postgres (free tier: 512 MB) with pgvector. Three tables: `papers`, `chunks` (768-dim HNSW index, nomic-embed-text-v2-moe), `query_audit_log`. `connection.py` owns the `AsyncConnectionPool`; `queries.py` has analytics window functions (papers by month, p95 latency, etc.).

**pgvector note:** always use `register_vector_async(conn)` (not `register_vector`) with psycopg3 async connections.

### Ingestion (`ingestion/`)

Two-stage pipeline:
1. `loader.py` — reads ArXiv JSONL snapshot, filters by category, writes to `papers` table (50k papers loaded)
2. `pipeline.py` — batched architecture: buffers N papers, embeds all chunks at once via `sentence-transformers` (nomic-embed-text-v2-moe, 768-dim), bulk-inserts into `chunks` with pgvector

**Cloud embedding:** `kaggle_ingestion.ipynb` is a self-contained notebook for running Stage 2 on Kaggle/Colab free T4 GPU (~7 min for 10k papers). The M1 Mac is too slow for the MoE model without megablocks. Currently ~10k papers are embedded due to Neon's 512 MB free tier limit.

**Connector:** `ArxivAbstractConnector` streams papers from the `papers` table (includes `id` in SELECT to avoid redundant per-paper lookups).

### Evaluation (`eval/`)

`golden_set.json` — 20 Q/A pairs. `run_ragas.py` runs RAGAS (Gemini-as-judge) against live DB; CI gate thresholds: faithfulness ≥ 0.80, answer_relevancy ≥ 0.75, context_precision ≥ 0.70. Metrics logged to DagsHub MLflow.

### Tests (`tests/`)

`conftest.py` stubs `psycopg`, `psycopg_pool`, `pgvector`, `torch`, and `sentence_transformers` so the full test suite runs locally without Docker. 22 tests, 1 skipped (live DB). `PYTHONPATH=.` is required (set in CI env).

### Frontend (`frontend/`)

React 18 + Vite. `VITE_API_URL` env var points to the HF Space backend. Deployed to Vercel with root directory set to `frontend/`.

## Key Environment Variables

```
DATABASE_URL              # Neon postgres connection string
UPSTASH_REDIS_REST_URL    # Upstash REST endpoint
UPSTASH_REDIS_REST_TOKEN  # Upstash auth token
GROQ_API_KEY              # Primary LLM
GEMINI_API_KEY            # Fallback LLM + RAGAS judge
DAGSHUB_TOKEN             # MLflow tracking
DAGSHUB_REPO              # username/reponame for MLflow
HF_SPACE_URL              # GitHub secret for CI keep-alive ping
```

## CI (`.github/workflows/`)

Three jobs: `lint-and-test` (ruff + pytest, no secrets needed), `ragas-quality-gate` (runs against live Neon DB, requires all secrets), `keep-alive` (pings `HF_SPACE_URL/health` on push to main).
