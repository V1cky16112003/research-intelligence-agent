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
python -m eval.run_ragas --ci --limit 6

# Stage 1: Load ArXiv papers into Neon (requires dataset + DATABASE_URL)
DATABASE_URL="..." PYTHONPATH=. python3 -m ingestion.loader \
  --file /path/to/arxiv-metadata-oai-snapshot.json \
  --limit 50000 --categories cs.LG,cs.AI,cs.CL,cs.CV

# Stage 2: Embed chunks — run on Kaggle/Colab GPU (see kaggle_ingestion.ipynb)
# Neon free tier is 512 MB; embed ~10k papers to stay within limit
DATABASE_URL="..." PYTHONPATH=. python3 -m ingestion.pipeline --limit 10000 --batch-size 200

# Apply hybrid search schema migration (tsvector + GIN index for BM25)
psql $DATABASE_URL -f db/migrations/001_contextual_retrieval.sql

# Apply the LLM cost/latency observability migration (llm_call_log table + views)
psql $DATABASE_URL -f db/migrations/002_llm_call_log.sql

# Shrink embeddings from fp32 to fp16 — the fix for the Neon 512 MB ceiling.
# Takes an ACCESS EXCLUSIVE lock on chunks for ~1 min; see docs/storage.md.
psql $DATABASE_URL -f db/migrations/003_halfvec_embeddings.sql

# ...or apply any migration without a local psql client (uses the app's psycopg):
PYTHONPATH=. python3 -m db.apply_migration db/migrations/003_halfvec_embeddings.sql --dry-run
DATABASE_URL="..." PYTHONPATH=. python3 -m db.apply_migration db/migrations/003_halfvec_embeddings.sql

# Repair published_at on already-loaded papers (dry-run by default; --apply to write).
# The original loader wrote `update_date` into `published_at`; see docs below.
PYTHONPATH=. python3 -m ingestion.backfill_dates \
  --file dataset/arxiv-metadata-oai-snapshot.json --apply

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

- `state.py` — `AgentState` TypedDict; the graph's shared state schema. `retrieved_chunks` and `sql_results` carry the RAG and SQL tool output; `aux_results` (`[{tool, summary, results}]`) carries everything else. That field exists because `graph_query` and `web_search` were previously *write-only*: the executor logged them to `tool_results` for audit, then returned a state update naming only chunks and SQL rows, so `_build_context` never saw them and the reporter answered "the provided context does not contain any information" on a tool call that had in fact succeeded. Any new tool that returns neither chunks nor SQL rows must route into `aux_results`, which `_build_context` renders (capped at `MAX_AUX_ROWS = 25`) for both the reporter and the critic
- `nodes.py` — one async function per node; gateway is fetched from the module-level registry (not stored in state)
- `registry.py` — module-level singleton (`set_gateway` / `get_gateway`); avoids LangGraph stripping non-schema state keys
- `graph.py` — wires the graph, `init_graph()` sets up `AsyncPostgresSaver` at startup, falls back to in-memory if no DB
- `tools.py` — `TOOL_DISPATCH` dict mapping tool name → async function: `rag_retrieval`, `sql_analytics`, `web_search`, `graph_query`. `sql_analytics_tool` dispatches on `query_type` — one of `corpus_stats`, `papers_by_category`, `papers_by_year`, `papers_by_month`, `top_authors`, `query_volume`, `provider_latency`, `experiments`, `cost_by_node`, `retry_overhead` — plus a `_QUERY_TYPE_ALIASES` map that absorbs the planner's near-misses (`total_papers` → `corpus_stats`). Aggregate questions ("how many papers?") route to `corpus_stats`, which returns one row. `rag_retrieval_tool` runs hybrid search (vector + BM25 RRF, 16 candidates) then LLM reranks down to top 8. `graph_query_tool` answers relational questions (co-authorship, shared subfields) via fixed parameterized Cypher templates against Neo4j — never LLM-generated Cypher — and falls back to equivalent Postgres queries when Neo4j is unset, unreachable, or returns nothing (see Knowledge Graph below).
- `gateway.py` — `LLMGateway`: Groq (Llama 3.3 70B) primary → NVIDIA NIM (Llama 3.1 70B) → Gemini 2.5 Flash, cascading fallback on 429/5xx; wraps Upstash Redis cache
- `redis_client.py` — thin Upstash REST client (no persistent TCP connection)

The Critic node returns `RETRY` or `PASS`; the graph loops back to Executor up to `MAX_RETRIES = 3` times before forcing Reporter.

### API (`app/main.py`)

`Settings` (pydantic-settings) reads from `.env`. Redis URL is assembled at runtime from either `REDIS_URL` or `UPSTASH_REDIS_REST_URL` + `UPSTASH_REDIS_REST_TOKEN`. `/chat` is rate-limited (`CHAT_RATE_LIMIT_PER_MINUTE`, default 20) with a CORS allowlist (`ALLOWED_ORIGINS`); see `docs/security.md`. Every `/chat` request writes a row to `query_audit_log` (latency, tokens, tools called, retrieved chunk IDs) plus one `llm_call_log` row per LLM call (node, provider, cost, latency, retry flag — see `docs/cost_latency_observability.md`). `GET /analytics/cost` exposes the aggregated cost/latency-by-node and retry-overhead dashboard data.

### Database (`db/`)

Neon Postgres (free tier: 512 MB) with pgvector. Three tables: `papers`, `chunks` (768-dim HNSW index, nomic-embed-text-v2-moe), `query_audit_log`. `connection.py` owns the `AsyncConnectionPool`; `queries.py` holds the analytics layer; `apply_migration.py` runs a `.sql` file statement-by-statement in autocommit (needed because `VACUUM` and `CREATE INDEX CONCURRENTLY` can't run inside a transaction block) for machines with no `psql`.

**Storage (the 512 MB ceiling):** the database sat at 483 MB with 40k embedded chunks — a bulk `UPDATE` had already failed with `DiskFull` once. `db/migrations/003_halfvec_embeddings.sql` converts `chunks.embedding` from `vector(768)` (3080 bytes) to `halfvec(768)` (1544 bytes), halving both the stored payload and the HNSW index. **Applied 2026-09-08: 483 MB → 341 MB** (HNSW 156 → 78 MB, TOAST ~164 → 100 MB), headroom 29 MB → 171 MB, with no retrieval loss — 25 probe vectors returned identical top-10 neighbours before and after. Note the embeddings stay in TOAST: `TOAST_TUPLE_THRESHOLD` applies to the whole row, and with a ~901-byte `content` the row still exceeds 2 KB, so the heap barely moved (56 → 57 MB). Statement order is load-bearing — the `DROP INDEX` comes first because it frees the headroom the table rewrite needs. No deploy window is required: pgvector registers `vector → halfvec` as an *implicit* cast, so code still binding `%s::vector` keeps working against the converted column. Full rationale, rejected alternatives, rollback, and verification: `docs/storage.md`.

**Analytics design rules** (`db/queries.py`): aggregate in Postgres, never in the LLM. Every analytics function returns `(rows, summary)` where `summary` carries the pre-computed totals, and rows are capped at `MAX_ANALYTICS_ROWS = 200` — the reporter is handed the answer, not asked to sum a truncated result set. Day-window parameters go through `_clamp_days()` (1..3650). `_rows_to_dicts` is `async`: `AsyncCursor.fetchall()` returns a coroutine and must be awaited.

**pgvector note:** always use `register_vector_async(conn)` (not `register_vector`) with psycopg3 async connections.

### Ingestion (`ingestion/`)

Two-stage pipeline:
1. `loader.py` — reads ArXiv JSONL snapshot, filters by category, writes to `papers` table (50k papers loaded)

**Date semantics (important):** `published_at` comes from `versions[0].created` (the RFC 2822 timestamp arXiv received v1), falling back to the `YYMM` prefix of the arXiv ID. It is *not* `update_date` — that field is the day the OAI metadata record was last touched, which is what `updated_at` stores. Loading `published_at` from `update_date` put 27.7% of the corpus in the wrong year and stamped 4,286 papers with 2019-2026, years in which this corpus (IDs 0704-1805) published nothing. `ingestion/backfill_dates.py` repairs rows loaded before the fix; `tests/test_publication_dates.py` guards it.
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

**Postgres fallback (`db/queries.py`: `papers_by_author`, `coauthors`,
`papers_in_category`).** Neo4j is the primary backend but not a hard dependency:
the AuraDB free instance is deleted after inactivity, and when that happened the
whole tool went down. `graph_query_tool` now tries Neo4j only when `NEO4J_URI`
is set, and falls back to Postgres on exception *or on an empty result*. Empty
counts as a fallback because the Cypher templates match `{name: $value}`
exactly, while `papers.authors` stores names surname-first with affiliations
("Bengio Yoshua Universite de Montreal") — so a natural-order name returned zero
rows silently rather than erroring. The Postgres path splits the name into
tokens and requires all of them (`a ILIKE ALL(%s)` over `UNNEST(p.authors)`),
which makes matching order-insensitive. Both `papers.authors` and
`papers.categories` are `TEXT[]`, so this needs no new tables.

### Evaluation (`eval/`)

`golden_set.json` — 20 Q/A pairs. `run_ragas.py` runs RAGAS against live DB; CI gate thresholds: faithfulness ≥ 0.72, answer_relevancy ≥ 0.75, context_precision ≥ 0.70. `check_thresholds()` treats NaN scores as a hard failure, not a silent pass. The judge LLM (`openai/gpt-oss-120b`, chat completions only) runs on Groq, not NVIDIA NIM — NIM's free-tier queue proved too slow/unreliable for CI (individual judge calls observed taking 30s–15min, occasionally failing outright), while Groq answers generation calls in this same pipeline in under a second. The judge model is deliberately different from answer generation's `llama-3.3-70b-versatile`, since Groq tracks daily token quota per model — judging keeps its own independent 100K TPD budget instead of competing with generation for the same one. Smaller judge models were rejected earlier (on NIM): 8B reproducibly echoes the JSON schema instead of a filled instance under `instructor`'s structured-output prompting. Embeddings run on Gemini (`gemini-embedding-001` via its OpenAI-compatible endpoint, rate-limited to 60 rpm client-side). They used to go through NIM, but every NIM embedding model — `nv-embedqa-e5-v5` included — now returns `410 Gone`, which surfaced as `answer_relevancy: nan` and a failed gate rather than an obvious outage. Groq still has no embeddings endpoint. Metrics logged to DagsHub MLflow.

Note: running `run_ragas.py` locally on Python 3.14 exits 1 *after* printing a passing gate — `nest_asyncio` raises `RuntimeError: Timeout should be used inside a task` during asyncio teardown. CI pins Python 3.11 and is unaffected; read the gate verdict, not just the exit code, when running locally.

### Tests (`tests/`)

`conftest.py` stubs `psycopg`, `psycopg_pool`, `pgvector`, `torch`, and `sentence_transformers` so the full test suite runs locally without Docker. 132 tests, 0 skipped. `PYTHONPATH=.` is required (set in CI env). `test_contextual_retrieval.py` covers embed prefix logic, reranker ordering/fallback, and BM25 query sanitization. `test_graph.py` covers the Neo4j driver singleton, graph sync idempotency, author-name tokenization, and the Neo4j → Postgres fallback paths. `test_gateway.py` covers the 3-tier Groq → NVIDIA NIM → Gemini fallback chain. `test_agent.py` covers `aux_results` routing and context rendering. `test_migration_runner.py` covers the migration SQL splitter (dollar-quoted bodies, string literals, statement ordering in 003).

### Frontend (`frontend/`)

React 18 + Vite. `VITE_API_URL` env var points to the HF Space backend. Deployed to Vercel with root directory set to `frontend/`.

## Key Environment Variables

```
DATABASE_URL              # Neon postgres connection string
UPSTASH_REDIS_REST_URL    # Upstash REST endpoint
UPSTASH_REDIS_REST_TOKEN  # Upstash auth token
GROQ_API_KEY              # Primary LLM
NVIDIA_NIM_API_KEY        # Second-tier LLM fallback + RAGAS judge (chat + embeddings)
GEMINI_API_KEY            # Third-tier LLM fallback
NEO4J_URI                 # Neo4j AuraDB connection URI (graph_query tool)
NEO4J_USER                # Neo4j AuraDB username
NEO4J_PASSWORD            # Neo4j AuraDB password
DAGSHUB_TOKEN             # MLflow tracking
DAGSHUB_REPO              # username/reponame for MLflow
HF_SPACE_URL              # GitHub secret for CI keep-alive ping
```

## CI (`.github/workflows/`)

Three jobs: `lint-and-test` (ruff + pytest, no secrets needed), `ragas-quality-gate` (runs against live Neon DB, requires all secrets), `keep-alive` (pings `HF_SPACE_URL/health` on push to main).
