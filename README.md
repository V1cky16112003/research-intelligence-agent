---
title: Research Intelligence Agent
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# Research Intelligence Agent

An autonomous research assistant for ArXiv ML papers. Ask a question in natural language and get a cited, grounded answer — backed by hybrid semantic+keyword search over a 50K-paper corpus, live SQL analytics, a knowledge graph of authors/categories, and web search fallback for anything out of corpus.

**Live demo:** `https://<your-hf-space>.hf.space` · **UI:** `https://<your-vercel>.vercel.app`

---

## What It Does

- **Hybrid retrieval** — pgvector cosine search + BM25 full-text, fused with RRF, then LLM-reranked to the top 8 chunks
- **SQL analytics** — "how many LLM papers per month in 2023?" runs a real aggregate Postgres query (window functions, percentiles), not an LLM guess
- **Knowledge graph** — co-authorship and category relationships via Neo4j (fixed Cypher templates only, never LLM-generated), with an automatic Postgres fallback if Neo4j is unreachable
- **Web search fallback** via DuckDuckGo for questions outside the corpus
- **Cited answers** — every claim traces back to a paper (arxiv_id, title, authors)
- **Self-correcting agent loop** — a Critic node checks the Reporter's draft against retrieved context and can force up to 3 retries
- **Resilient LLM gateway** — Groq (Llama 3.3 70B) → NVIDIA NIM → Gemini 2.5 Flash, cascading fallback on rate limits/errors, with an Upstash Redis response cache
- **Cost & latency observability** — every LLM call is logged (node, provider, tokens, cost, latency, retries) and exposed via a dashboard endpoint
- **RAGAS quality gate in CI** — a PR that degrades retrieval or answer quality fails the build

---

## Architecture

```
React UI (Vercel)
      │
      ▼
FastAPI /chat  (rate-limited, CORS-locked — HF Spaces Docker, port 7860)
      │
      ▼
LangGraph state machine
┌────────────────────────────────────────────────┐
│  Planner → Executor → Critic → Reporter         │
│                │           │                     │
│          [tool calls]  PASS / RETRY (≤3)        │
└────────────────────────────────────────────────┘
      │            │            │            │
  RAG tool      SQL tool    Graph tool    Web search
 (pgvector +   (aggregate    (Neo4j, w/    (DuckDuckGo)
  BM25 + RRF    queries in   Postgres
  + rerank)     Postgres)    fallback)

Neon Postgres (pgvector, halfvec embeddings)
Neo4j AuraDB (co-authorship / category graph)
Upstash Redis (LLM response cache)
AsyncPostgresSaver (LangGraph checkpoints — multi-turn memory)
```

Every `/chat` request writes an audit row (latency, tokens, tools called, chunk IDs) plus one cost/latency row per LLM call.

---

## Tech Stack

| Role | Tool |
|------|------|
| Agent orchestration | LangGraph — Planner → Executor → Critic → Reporter |
| Primary LLM | Groq (Llama 3.3 70B) |
| Fallback LLMs | NVIDIA NIM → Gemini 2.5 Flash |
| Embeddings | nomic-embed-text-v2-moe (768-dim) |
| Vector + SQL store | Postgres (Neon) with pgvector (halfvec) |
| Knowledge graph | Neo4j AuraDB, Postgres fallback |
| Cache | Upstash Redis (REST, no persistent connection) |
| Eval | RAGAS + DagsHub MLflow |
| API host | Hugging Face Spaces (Docker) |
| UI | React 18 + Vite on Vercel |

---

## Quickstart (local with Docker)

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
cp .env.example .env
# Fill in GROQ_API_KEY and DATABASE_URL at minimum
docker-compose up --build

curl http://localhost:7860/health

# Ingest an ArXiv corpus (e.g. the Cornell ArXiv Kaggle dataset):
docker-compose exec app python -m ingestion.loader \
  --file /path/to/arxiv-metadata-oai-snapshot.json \
  --limit 50000 --categories cs.LG,cs.AI,cs.CL,cs.CV

docker-compose exec app python -m ingestion.pipeline --limit 10000

curl -X POST http://localhost:7860/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings on attention in transformers?"}'
```

See `CLAUDE.md` for the full command reference (migrations, backfills, graph sync, evaluation).

---

## Project Structure

```
app/            FastAPI app + settings
agent/          LangGraph nodes, graph, tools, LLM gateway, state
db/             Postgres connection pool, analytics queries, migrations
graph/          Neo4j client + knowledge graph sync
ingestion/      ArXiv loader, embedding pipeline, date backfill, rebalance
eval/           RAGAS golden set + evaluation runner
docs/           Storage/security architecture notes
tests/          Unit tests (no Docker required — deps are stubbed)
frontend/       React chat UI (Vite)
.github/        CI workflows
```

---

## Testing & CI

```bash
ruff check . --ignore E501,E402
pytest tests/ -v --tb=short
```

CI runs three jobs on every push/PR to `main`: lint + test (no secrets needed), a RAGAS quality gate against the live database, and a keep-alive ping to prevent Space cold starts.

---

## Contributing

Issues and pull requests are welcome. Please run the lint and test commands above before opening a PR.

## License

MIT — see [LICENSE](LICENSE).
