# Design: Drop LLM Blurbs, Add NVIDIA NIM Provider, Add GraphRAG Layer

Date: 2026-07-01
Status: Approved for planning

## Motivation

Two free-tier bottlenecks are blocking smooth production operation:
- Groq (100k tokens/day) and Gemini (5 RPM) are exhausted by Stage 2b contextual
  blurb generation, which the Anthropic Contextual Retrieval paper describes as
  an optional enhancement (retrieval works without it, at slightly worse recall).
- The agent's chat-time LLM calls have only two fallback tiers (Groq → Gemini),
  both of which can be simultaneously rate-limited under real traffic.

Separately, the ArXiv corpus has relational structure (co-authorship, shared
subfields) that flat vector/BM25 retrieval cannot answer well ("what else has
this author written," "what's related in the same subfield"). A knowledge graph
layer is a natural, additive extension — not a replacement for existing hybrid
retrieval, which already handles single-hop/detail questions well.

## 1. Drop LLM-generated contextual blurbs

- Remove the LLM call from `ingestion/context_generator.py`. Stage 2b
  (`--contextual` flag on `ingestion/pipeline.py`) is removed entirely, along
  with the delete+reinsert re-embedding workflow it drove.
- The `chunks.context` column and `content_tsv` generated column stay in the
  schema (nullable, harmless) — no migration needed. Nothing populates
  `context` going forward.
- `search_similar_chunks_hybrid()` is unaffected: it already fuses on `content_tsv`,
  which is `context + content`, and already tolerates a null/absent context
  column via its existing fallback path.
- Rate-limit-aware retry/session-quota-tracking code that existed solely to
  make Stage 2b survive Groq/Gemini exhaustion is removed as dead code.

**Not doing:** repopulating existing `context` values, keeping Stage 2b behind
a flag for later use. If contextual blurbs are wanted again later, it's a new
feature decision, not a dormant code path to maintain now.

## 2. Add NVIDIA NIM as a third LLM gateway provider

- `agent/gateway.py`'s `LLMGateway` becomes a three-tier chain:
  **Groq (primary) → NVIDIA NIM (2nd fallback) → Gemini (3rd fallback)**.
- New constructor param `nvidia_api_key`; new `AsyncOpenAI` client pointed at
  `https://integrate.api.nvidia.com/v1`.
- `NIM_MODEL = "meta/llama-3.1-70b-instruct"` — confirmed OpenAI-style tool-calling
  support, comparable quality/size to the existing Groq primary model, so
  behavior stays consistent across the fallback chain.
- The existing retry loop (`RETRY_DELAYS`, 429/5xx handling, jitter) extends to
  three providers instead of two; same exponential-backoff-then-fallback shape.
- Cache key logic, tool-calling pass-through, and provider tagging on responses
  (`"provider": "groq" | "nvidia_nim" | "gemini"`) extend uniformly — no
  special-casing for NIM beyond base URL/model/key.
- `Settings` (`app/main.py`) gains `NVIDIA_NIM_API_KEY` env var, wired the same
  way `GROQ_API_KEY`/`GEMINI_API_KEY` already are.

**Not doing:** load-balancing/round-robin across all three proactively — this
stays a pure ordered-fallback chain, consistent with current design.

## 3. GraphRAG: Neo4j AuraDB layer for authors/categories

### Scope
Graph relationships are limited to data already present in the `papers` table:
- `(:Paper)-[:AUTHORED_BY]->(:Author)`
- `(:Paper)-[:HAS_CATEGORY]->(:Category)`

No citation graph — the ArXiv metadata snapshot has no reliable reference data,
and LLM-based citation extraction would reintroduce the API budget pressure
this whole change is trying to remove.

### Components
- `graph/neo4j_client.py` — thin async wrapper around the official `neo4j`
  Python driver; owns connection lifecycle (mirrors `db/connection.py`'s pool
  pattern for consistency).
- `graph/graph_sync.py` — re-runnable sync script that reads `papers` (arxiv_id,
  title, authors, categories) and MERGEs `Paper`/`Author`/`Category` nodes and
  their relationships into Neo4j. Idempotent (MERGE, not CREATE) so it can be
  re-run after new papers are ingested without duplicating nodes.
- New `graph_query` tool in `agent/tools.py`, registered in `TOOL_DISPATCH`
  alongside `rag_retrieval`, `sql_analytics`, `web_search`. Takes a natural-language
  question and maps it to one of a small, fixed set of **parameterized Cypher
  templates** (e.g. "papers by author X," "papers in category Y," "co-authors of
  author X") — not LLM-generated Cypher. This is a deliberate safety/predictability
  choice: fixed templates can't hallucinate a malformed or unbounded query.
- The existing Planner node decides when to call `graph_query` vs `rag_retrieval`,
  the same way it already chooses among the current three tools — no separate
  routing layer.
- `Settings` gains `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` for AuraDB free tier.

### Data flow
`graph_sync.py` runs after ingestion (Stage 1/2), populating Neo4j from Postgres.
At query time: Planner → (optionally) `graph_query` tool → Cypher template →
Neo4j → paper metadata (arxiv_id, title, authors, categories) → Reporter node
synthesizes alongside any `rag_retrieval` results, same as today's multi-tool flow.

### Testing
- `graph_sync.py`: unit test MERGE idempotency (run twice, assert no duplicate nodes).
- `graph_query` tool: unit test each Cypher template against a stubbed Neo4j
  driver (matching the existing `conftest.py` pattern of stubbing heavy
  dependencies — `neo4j` driver gets stubbed the same way `psycopg`/`pgvector` are).
- No live Neo4j connection required for the test suite, consistent with the
  rest of the project's no-Docker-required test philosophy.

**Not doing:** citation extraction, LLM-generated Cypher, replacing vector/BM25
retrieval, graph visualization UI (out of scope for this change).

## Summary of file-level changes

| File | Change |
|---|---|
| `ingestion/context_generator.py` | Remove LLM call / Stage 2b logic |
| `ingestion/pipeline.py` | Remove `--contextual` flag and delete+reinsert flow |
| `agent/gateway.py` | Add NIM as 2nd fallback tier |
| `app/main.py` (`Settings`) | Add `NVIDIA_NIM_API_KEY`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` |
| `graph/neo4j_client.py` | New — Neo4j async driver wrapper |
| `graph/graph_sync.py` | New — Postgres → Neo4j sync script |
| `agent/tools.py` | Add `graph_query` tool + `TOOL_DISPATCH` entry |
| `tests/` | New tests for graph sync + graph_query; remove/update Stage 2b tests |
| `CLAUDE.md` | Update to reflect removed Stage 2b, new NIM tier, new graph layer |
