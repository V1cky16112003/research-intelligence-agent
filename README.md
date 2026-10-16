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

Ask a question about machine-learning research in plain English and get a cited answer. The agent decides whether to search paper text, run a real SQL aggregate, walk the author graph, or search the web, then checks its own draft against what it retrieved before answering.

The corpus is 50K ArXiv ML papers (2007-2018). The whole stack runs on free tiers.

**Live:** [UI](https://frontend-vert-eight-61.vercel.app/) (sign-in required) · [API](https://v1cky1611-research-agent.hf.space/ready)

---

## What you can ask

| Kind of question | Example | What answers it |
|---|---|---|
| Meaning / findings | "How do GANs suffer from mode collapse, and what fixes were proposed?" | Hybrid search over paper text, reranked |
| Counts and trends | "How many cs.LG papers were published per month in 2017?" | Aggregate SQL in Postgres, not an LLM guess |
| Relationships | "Who are Yoshua Bengio's most frequent co-authors?" | Neo4j graph, with a Postgres fallback |
| Outside the corpus | Anything the papers can't answer | DuckDuckGo web search |

Every answer carries its sources (arXiv ID, title, authors), and SQL answers show the underlying table.

---

## The chat UI

- **History sidebar.** Conversations are grouped Pinned / Today / Yesterday / Previous 7 days / Older. Search, rename, pin and delete them. Each conversation keeps its own agent session, so a follow-up continues the right thread.
- **Usage left.** A gauge shows how many of your daily questions remain and when the window resets. It reads a read-only `GET /usage`, so checking it never costs a question.
- **Live run trace.** While the agent works you see Planner → Executor → Critic → Reporter fill in, then the answer with source cards.
- **Shortcuts.** `Cmd/Ctrl+K` searches chats, `Cmd/Ctrl+Shift+O` starts a new one. On phones the sidebar is a drawer.

History is stored in your browser, per user. Clearing site data clears it. Design notes and what was left out: [`docs/ui-plan.md`](docs/ui-plan.md).

---

## How it works

```
React UI (Vercel)
      │  Supabase sign-in (GitHub OAuth / email link)
      ▼
FastAPI  /chat  /chat/stream  /usage      (Hugging Face Spaces, Docker, port 7860)
  per-user rate limit · daily quota · admission control · CORS allowlist
      │
      ▼
LangGraph:   Planner → Executor → Critic → Reporter
                         │           │
                   tool calls    PASS or RETRY (up to 3)
      ┌──────────┬───────┴───┬─────────────┐
   rag_retrieval  sql_analytics  graph_query   web_search
   pgvector+BM25  aggregates in  Neo4j, Postgres  DuckDuckGo
   +RRF+rerank    Postgres       fallback

Neon Postgres (pgvector, halfvec) · Neo4j AuraDB · Upstash Redis · LangGraph checkpoints
```

- **Retrieval.** pgvector cosine search and BM25 are fused with reciprocal rank fusion (16 candidates), then an LLM reranks down to the top 8.
- **Analytics are computed, not generated.** Aggregates run in Postgres and arrive pre-summed, so the model reports a number instead of adding up a truncated table.
- **Graph queries use fixed Cypher templates.** The model never writes Cypher. If Neo4j is unset, unreachable or returns nothing, the same question runs against Postgres.
- **The Critic loop.** The Critic can send the Executor back for more evidence up to three times. It skips the loop when the answer rests on SQL or graph rows, where re-running retrieval can't help.
- **LLM gateway.** Groq (`openai/gpt-oss-120b`) → NVIDIA NIM (`openai/gpt-oss-20b`) → Gemini 2.5 Flash. A 429 fails over immediately instead of waiting out a backoff. Responses are cached in Upstash Redis.
- **Observability.** Every request writes an audit row (latency, tokens, tools, chunk IDs) and every LLM call writes a cost/latency row, exposed at `GET /analytics/cost`. OpenTelemetry traces export when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.

### Tech stack

| Role | Tool |
|---|---|
| Agent | LangGraph |
| LLMs | Groq → NVIDIA NIM → Gemini 2.5 Flash |
| Embeddings | nomic-embed-text-v2-moe (768-dim) |
| Vector + SQL | Neon Postgres, pgvector (halfvec) |
| Graph | Neo4j AuraDB |
| Cache, rate limits | Upstash Redis (REST) |
| Auth | Supabase (JWT verified in the API) |
| Evaluation | RAGAS, DagsHub MLflow |
| API host | Hugging Face Spaces (Docker) |
| UI | React 18 + Vite on Vercel |

### Scope and limits

- SQL analytics and the graph cover all 50K loaded papers. Semantic search covers the embedded subset, which is smaller because Neon's free tier caps storage at 512 MB. See [`docs/storage.md`](docs/storage.md).
- Free-tier LLM quotas are the real throughput ceiling. The API admits 4 concurrent agent runs and rejects the rest with `503` plus `Retry-After`, because more only turns into provider `429`s.
- Each signed-in user gets 100 questions per 24 hours and 20 requests per minute (both configurable).

---

## Run it locally

You need Docker, a Postgres database with pgvector (a free Neon project works) and a Groq API key.

```bash
git clone https://github.com/V1cky16112003/research-intelligence-agent.git
cd research-intelligence-agent
cp .env.example .env          # set GROQ_API_KEY and DATABASE_URL at minimum
docker-compose up --build

curl http://localhost:7860/health
curl http://localhost:7860/ready
```

Auth is off until you set `SUPABASE_URL`, so a local API accepts requests without a token:

```bash
curl -X POST http://localhost:7860/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings on attention in transformers?"}'
```

**Load data** (the [Cornell ArXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) on Kaggle):

```bash
# Stage 1: papers into Postgres
docker-compose exec app python -m ingestion.loader \
  --file /path/to/arxiv-metadata-oai-snapshot.json \
  --limit 50000 --categories cs.LG,cs.AI,cs.CL,cs.CV

# Stage 2: embed chunks (slow on CPU; kaggle_ingestion.ipynb runs it on a free GPU)
docker-compose exec app python -m ingestion.pipeline --limit 10000
```

**Run the UI:**

```bash
cd frontend && npm ci
VITE_API_URL=http://localhost:7860 npm run dev
```

The full command reference (migrations, date backfill, graph sync, evaluation) is in [`CLAUDE.md`](CLAUDE.md).

---

## Testing

```bash
ruff check . --ignore E501,E402
PYTHONPATH=. pytest tests/ -v --tb=short     # no Docker, heavy deps are stubbed

cd frontend
npm run test:unit                            # conversation store
npm run test:e2e                             # Playwright, API mocked
```

CI (`.github/workflows/`) runs on every push and PR:

| Job | What it checks |
|---|---|
| Lint & Unit Tests | `ruff` and `pytest` |
| Dependency Audit | `pip-audit`, `npm audit`, frontend build |
| Frontend E2E | Playwright on desktop and mobile viewports, plus a Supabase-auth build |
| RAGAS Quality Gate | Faithfulness, answer relevancy and context precision must each score at least 0.80 on a sample of the golden set, against the live database |

Scheduled jobs poll `/ready` every 6 hours (this also keeps the free Space awake), probe every pinned LLM model daily because providers retire free models silently, and take an encrypted nightly database backup.

---

## Project layout

```
app/          FastAPI app, settings, Supabase JWT verification
agent/        LangGraph nodes, tools, LLM gateway, Redis client, tracing
db/           Connection pool, analytics queries, migrations
graph/        Neo4j client and knowledge-graph sync
ingestion/    ArXiv loader, embedding pipeline, date backfill
eval/         RAGAS golden set and runner
frontend/     React chat UI and Playwright tests
docs/         operations, security, storage, cost/latency, UI plan
tests/        Backend tests
.github/      CI, scheduled checks, nightly backup
```

Operations runbook (deploys, rollback, staging, observability): [`docs/operations.md`](docs/operations.md). Security model: [`docs/security.md`](docs/security.md).

---

## Contributing

Issues and pull requests are welcome. Run the lint and test commands above first.

## License

MIT. See [LICENSE](LICENSE).
