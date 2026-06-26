# Autonomous Research Intelligence Agent

An AI-powered research assistant for ArXiv ML papers. Ask questions in natural language — get cited, grounded answers backed by a 50K-paper corpus, with SQL analytics and live web search fallback.

**Live demo:** `https://<your-hf-space>.hf.space` · **UI:** `https://<your-vercel>.vercel.app`

---

## What It Does

- **Semantic search** over ArXiv abstracts (cs.LG, cs.AI, cs.CL, cs.CV) via pgvector
- **SQL analytics** — "how many LLM papers per month in 2023?" returns real window-function query results
- **Web search fallback** via DuckDuckGo for out-of-corpus questions
- **Cited answers** — every claim traced back to a paper (arxiv_id + title + authors)
- **Resilient LLM** — Groq Llama 3.3 70B primary, Gemini 2.5 Flash fallback on rate-limit
- **RAGAS quality gate** — CI fails if faithfulness < 0.8 or answer relevancy < 0.75

---

## Architecture

```
User → React UI (Vercel)
         │
         ▼
     FastAPI /chat  (HF Spaces Docker, port 7860)
         │
         ▼
  LangGraph State Machine
  ┌──────────────────────────────────────────┐
  │  Planner  →  Executor  →  Critic         │
  │                │            │            │
  │          [tool calls]   PASS/RETRY       │
  │                │            │            │
  │           ←────┘  (≤3 retries)          │
  │                ↓                         │
  │            Reporter  →  cited answer     │
  └──────────────────────────────────────────┘
         │              │              │
    RAG tool       SQL tool      Web search
   (pgvector)  (window funcs)  (DuckDuckGo)
         │
    Neon Postgres (pgvector 0.8.0)
    nomic-embed-text-v2 (768-dim, CPU)
    Upstash Redis (LLM response cache)
    PostgresSaver (LangGraph checkpoints)
```

---

## CV Gap Coverage

| Gap | How This Project Addresses It |
|-----|-------------------------------|
| **Gap 1 — RAG** | ArXiv abstract corpus → nomic-embed-v2 chunks → pgvector HNSW cosine search → cited retrieval tool inside LangGraph |
| **Gap 2 — MLOps** | RAGAS golden set (20 Q/A) · Gemini-as-judge CI gate · thresholds: faithfulness ≥0.8 / answer_relevancy ≥0.75 / context_precision ≥0.7 · DagsHub MLflow param+metric logging |
| **Gap 3 — SQL** | Neon Postgres schema with `papers`, `chunks`, `query_audit_log` · HNSW + btree indexes · window-function analytics (ROW_NUMBER per category, rolling 7-day volume, PERCENTILE_CONT p95 latency) · every `/chat` request writes an audit row |

---

## Tech Stack

| Role | Tool |
|------|------|
| Agent | LangGraph 1.2.6 — Planner→Executor→Critic→Reporter |
| Primary LLM | Groq — Llama 3.3 70B (70 RPM free tier) |
| Fallback LLM | Gemini 2.5 Flash (exponential backoff on 429) |
| Embeddings | nomic-embed-text-v2 (137M params, 768-dim, CPU) |
| Vector + SQL | pgvector 0.8.0 on Neon Postgres (HNSW index) |
| Cache | Upstash Redis (HTTP REST, no persistent connection) |
| Eval | RAGAS 0.4.3 + DagsHub MLflow 3.x |
| API host | Hugging Face Spaces (Docker, 16GB RAM) |
| UI | React 18 + Vite on Vercel |

---

## Quickstart (local with Docker)

```bash
git clone <repo>
cd "RAG project"
cp .env.example .env
# Fill in GROQ_API_KEY and GEMINI_API_KEY at minimum
docker-compose up --build

# In another terminal:
curl http://localhost:7860/health
# {"status":"ok","version":"1.0.0"}

# Ingest ArXiv corpus (download Kaggle dataset first):
# https://www.kaggle.com/datasets/Cornell-University/arxiv
docker-compose exec app python -m ingestion.loader \
  --file /path/to/arxiv-metadata-oai-snapshot.json \
  --limit 50000 --categories cs.LG,cs.AI,cs.CL,cs.CV

docker-compose exec app python -m ingestion.pipeline --limit 50000

# Ask a question:
curl -X POST http://localhost:7860/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings on attention in transformers?"}'
```

---

## Deploy to Hugging Face Spaces

1. Create a new Space → **Docker** runtime → **16GB** hardware
2. Push this repo to the Space's git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/<space-name>
   git push space main
   ```
3. Add these **Space secrets** (Settings → Repository secrets):
   ```
   DATABASE_URL          # Neon connection string
   UPSTASH_REDIS_REST_URL
   UPSTASH_REDIS_REST_TOKEN
   GROQ_API_KEY
   GEMINI_API_KEY
   DAGSHUB_TOKEN
   DAGSHUB_REPO          # username/reponame
   ```
4. Space builds automatically. First cold start downloads nomic-embed-v2 (~500MB) — takes ~2 min.

---

## Deploy React UI to Vercel

```bash
cd frontend
npm install
# Test locally against your Space:
VITE_API_URL=https://<username>-<space-name>.hf.space npm run dev
```

1. Push `frontend/` to GitHub (or the same repo)
2. Import in Vercel → set **Root Directory** to `frontend`
3. Add environment variable: `VITE_API_URL=https://<username>-<space-name>.hf.space`
4. Deploy

---

## GitHub Actions CI

Three jobs on every push/PR to `main`:

| Job | What it does |
|-----|-------------|
| `lint-and-test` | ruff + pytest (22 tests, runs without Docker via conftest stubs) |
| `ragas-quality-gate` | Runs RAGAS on 10 golden questions against live DB; exits 1 if thresholds breach |
| `keep-alive` | Pings `HF_SPACE_URL/health` to prevent cold start on next user |

Add these **GitHub secrets** (Settings → Secrets → Actions):
```
DATABASE_URL, GROQ_API_KEY, GEMINI_API_KEY,
UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN,
DAGSHUB_TOKEN, DAGSHUB_REPO, HF_SPACE_URL
```

---

## Project Structure

```
app/            FastAPI app + settings
agent/          LangGraph nodes, graph, tools, gateway, state
db/             Postgres connection pool, schema, analytics queries
ingestion/      ArXiv loader, SourceConnector, embed pipeline
eval/           RAGAS golden set + evaluation runner
monitoring/     Evidently drift (stretch)
tests/          22 unit tests (no Docker required)
frontend/       React chat UI (Vite)
.github/        CI workflow
Dockerfile      HF Spaces production image
docker-compose  Local dev (Postgres + Redis + app)
```

---

## RAGAS Quality Gate

The CI gate (`eval/golden_set.json`, `eval/run_ragas.py`) runs 10 of 20 golden Q/A pairs on every PR:

```
Thresholds:  faithfulness ≥ 0.80
             answer_relevancy ≥ 0.75
             context_precision ≥ 0.70
```

Results are logged to DagsHub MLflow under experiment `rag-eval`. A PR that degrades retrieval quality fails the build before merge.
