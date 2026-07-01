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

# Apply hybrid search schema migration (tsvector + GIN index for BM25)
psql $DATABASE_URL -f db/migrations/001_contextual_retrieval.sql

# Sync papers (authors, categories) into the Neo4j knowledge graph
NEO4J_URI="..." NEO4J_USER="..." NEO4J_PASSWORD="..." DATABASE_URL="..." PYTHONPATH=. python3 -m graph.graph_sync --limit 10000

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
- `tools.py` — `TOOL_DISPATCH` dict mapping tool name → async function: `rag_retrieval`, `sql_analytics`, `web_search`, `graph_query`. `rag_retrieval_tool` runs hybrid search (vector + BM25 RRF, 16 candidates) then LLM reranks down to top 8. `graph_query_tool` answers relational questions (co-authorship, shared subfields) via fixed parameterized Cypher templates against Neo4j — never LLM-generated Cypher.
- `gateway.py` — `LLMGateway`: Groq (Llama 3.3 70B) primary → NVIDIA NIM (Llama 3.1 70B) → Gemini 2.5 Flash, cascading fallback on 429/5xx; wraps Upstash Redis cache
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

### Knowledge Graph (`graph/`)

Additive layer over the existing vector/BM25 hybrid retrieval — not a replacement.
`neo4j_client.py` owns a singleton async Neo4j driver (same lazy-singleton pattern
as `db/connection.py`'s pool). `graph_sync.py` reads `papers` (arxiv_id, title,
authors, categories) and MERGEs `(:Paper)-[:AUTHORED_BY]->(:Author)` and
`(:Paper)-[:HAS_CATEGORY]->(:Category)` into Neo4j AuraDB (free tier);
idempotent, safe to re-run after new ingestion batches. No citation graph —
the ArXiv metadata snapshot has no reliable reference data.

The `graph_query` agent tool answers relational questions ("what else has this
author published," "who are their co-authors," "what's in this category") via
a small fixed set of parameterized Cypher templates in `agent/tools.py` — not
LLM-generated Cypher, so a malformed or unbounded query can never reach the
database. The Planner node picks `graph_query` vs `rag_retrieval` the same way
it already picks among the other three tools.

### Evaluation (`eval/`)

`golden_set.json` — 20 Q/A pairs. `run_ragas.py` runs RAGAS (Gemini-as-judge) against live DB; CI gate thresholds: faithfulness ≥ 0.80, answer_relevancy ≥ 0.75, context_precision ≥ 0.70. Metrics logged to DagsHub MLflow.

### Tests (`tests/`)

`conftest.py` stubs `psycopg`, `psycopg_pool`, `pgvector`, `torch`, and `sentence_transformers` so the full test suite runs locally without Docker. 52 tests, 0 skipped. `PYTHONPATH=.` is required (set in CI env). `test_contextual_retrieval.py` covers embed prefix logic, reranker ordering/fallback, and BM25 query sanitization. `test_graph.py` covers the Neo4j driver singleton and graph sync idempotency. `test_gateway.py` covers the 3-tier Groq → NVIDIA NIM → Gemini fallback chain.

### Frontend (`frontend/`)

React 18 + Vite. `VITE_API_URL` env var points to the HF Space backend. Deployed to Vercel with root directory set to `frontend/`.

## Key Environment Variables

```
DATABASE_URL              # Neon postgres connection string
UPSTASH_REDIS_REST_URL    # Upstash REST endpoint
UPSTASH_REDIS_REST_TOKEN  # Upstash auth token
GROQ_API_KEY              # Primary LLM
NVIDIA_NIM_API_KEY        # Second-tier LLM fallback
GEMINI_API_KEY            # Third-tier LLM fallback + RAGAS judge
NEO4J_URI                 # Neo4j AuraDB connection URI (graph_query tool)
NEO4J_USER                # Neo4j AuraDB username
NEO4J_PASSWORD            # Neo4j AuraDB password
DAGSHUB_TOKEN             # MLflow tracking
DAGSHUB_REPO              # username/reponame for MLflow
HF_SPACE_URL              # GitHub secret for CI keep-alive ping
```

## CI (`.github/workflows/`)

Three jobs: `lint-and-test` (ruff + pytest, no secrets needed), `ragas-quality-gate` (runs against live Neon DB, requires all secrets), `keep-alive` (pings `HF_SPACE_URL/health` on push to main).
