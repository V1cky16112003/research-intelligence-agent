# Autonomous Research Intelligence Agent — Production-Ready Implementation Plan (2026)

## TL;DR
- Build it on a **fully free, externally-distributed topology**: FastAPI + LangGraph agent on **Hugging Face Spaces (Docker, free CPU, no credit card, 48h sleep window)**, **Neon or Supabase Postgres with pgvector** (one DB for both the RAG and SQL gaps), **Upstash serverless Redis**, **Groq** (fastest free inference) as primary LLM with **Cerebras** and **Gemini Flash** as fallbacks, **DagsHub** free hosted MLflow + DVC, a **RAGAS** CI quality gate, and **Evidently** embedding-drift monitoring.
- **Consolidate the vector store into pgvector** rather than adding a dedicated vector DB — it closes Gap 1 (RAG) and Gap 3 (SQL) in one database, is genuinely free, and the deliberate-architecture-tradeoff narrative is more impressive in interviews than bolting on Pinecone/Qdrant.
- **Keep LangGraph** — it remains the strongest 2026 CV signal for explainable, state-machine agents (LangGraph 1.0 went GA on Oct 22, 2025, the first stable major release in the durable-agent space, latest PyPI release langgraph 1.2.6) while you name-drop PydanticAI/OpenAI Agents SDK to show landscape awareness.

## Key Findings

### Free LLM inference — speed/limits/quality (mid-2026)
- **Groq (primary)** runs open-weight models on custom LPU silicon at roughly 300–1,000 tokens/sec — the fastest free inference available. Free tier is no-credit-card, rate-limited rather than token-capped: ~30 RPM, and for `llama-3.3-70b-versatile` around 1,000 requests/day, 12K tokens/min, 100K tokens/day; `llama-3.1-8b-instant` is far more permissive (up to ~14,400 RPD). OpenAI-compatible endpoint (`api.groq.com/openai/v1`), so it's a one-line drop-in. Llama 3.3 70B has solid tool/function-calling, ideal for the Planner/Critic nodes. (Governance note: per Groq's Dec 24, 2025 newsroom post, NVIDIA took a non-exclusive license and "Jonathan Ross, Groq's Founder, Sunny Madra, Groq's President, and other members of the Groq team will join Nvidia… Groq will continue to operate as an independent company with Simon Edwards stepping into the role of Chief Executive Officer. GroqCloud will continue to operate without interruption." CNBC reported the deal at ~$20B. Treat any "~90% of engineering left" claim as unverified — sources only confirm leadership/core engineering moved.)
- **Cerebras (fallback / high-volume)** offers the most generous free daily volume — **1,000,000 tokens/day**, 30 RPM, no credit card. Cerebras' own press release (verified by Artificial Analysis) states: "Cerebras achieves over 2,600 tokens per second on Llama 4 Scout – 19x faster than the fastest GPU solutions… the leading GPU solution delivers 137 tokens per second." Free tier has an 8,192-token default context (expandable to 128K on request) and includes Llama 3.3 70B, Qwen3 32B/235B and GPT-OSS 120B. Best for batch jobs like embedding-eval and nightly RAGAS runs. OpenAI-compatible (`api.cerebras.ai/v1`).
- **Google Gemini Flash (tool-calling / long-context fallback)**: after the April 1 2026 tightening, the free tier is **Flash/Flash-Lite only** (Pro models moved to paid). Gemini 3 Flash free tier ≈ 10 RPM / 250K TPM / 1,500 RPD with a 1M-token context window; Flash-Lite gives 15 RPM. Strong native tool-calling and multimodal — the best free fallback when a tool call needs a huge context or stricter function-calling.
- **Recommended combination for an agentic loop making many calls**: Groq Llama 3.3 70B primary (speed), automatic fallback to Cerebras (when Groq 429s / for batch eval), and Gemini 3 Flash for long-context tool calls. Wrap all three behind a single OpenAI-compatible client with exponential backoff on HTTP 429.

### Free embeddings
- **BGE-M3** (BAAI, MIT) is the production open-source default — dense + sparse + multi-vector hybrid retrieval in one model, 100+ languages, 8,192-token context, ~63 MTEB. **Qwen3-Embedding-8B** "ranks No.1 in the MTEB multilingual leaderboard (as of June 5, 2025, score 70.58)" per the Qwen technical report (arXiv:2506.05176), but by the March 2026 snapshot it had been surpassed (NVIDIA's Llama-Embed-Nemotron-8B leads multilingual MTEB, and BGE-en-ICL reaches 71.24) and it needs more VRAM than a free CPU tier comfortably allows. **nomic-embed-text-v1.5** (Apache-2.0, 768-dim, 8,192 context) is the best lightweight option.
- **Recommendation for scientific/technical text on free compute**: run **BAAI/bge-small-en-v1.5** (384-dim) or **nomic-embed-text-v1.5** locally via `sentence-transformers` on the HF Space's CPU. bge-small keeps the pgvector HNSW index small and query latency low; nomic gives a longer context for whole-section chunks. The free **Gemini embedding** API is the offload option if CPU embedding latency becomes the bottleneck. This directly leverages the candidate's existing HuggingFace/PyTorch knowledge.

### Free vector database — the pgvector consolidation call
- With HNSW (since pgvector 0.5.0) **pgvector matches or beats dedicated vector DBs below ~10M vectors**. Supabase's pgvector v0.5.0 HNSW blog reports: "With higher accuracy@10 of 0.99 HNSW even outperforms qdrant on equivalent compute resources… HNSW demonstrated over six times better performance while maintaining the same level of accuracy." Above ~5–10M vectors latency climbs and dedicated engines pull ahead: per Kalvium Labs (2026) citing Qdrant's published benchmarks, "Qdrant outperforms all four on throughput for self-hosted setups: ~850 QPS at p95 ~8ms on 1M vectors," while "at 5M vectors, pgvector p95 latency climbs to 80-140ms depending on ef_search." Qdrant also does in-graph metadata filtering vs pgvector's post-filter.
- **Decision: use pgvector.** For an ArXiv-ML demo corpus (tens of thousands of chunks) it's free, removes a service, lets you `JOIN` embeddings against relational metadata in one SQL query, and consolidates Gap 1 + Gap 3. The CV-impressive move is the *judgment*: "I chose pgvector deliberately; here's the recall/latency tradeoff vs Qdrant and the >10M-vector point where I'd migrate." Optionally run Qdrant in docker-compose locally to demonstrate breadth. Qdrant Cloud's 1GB-free-forever tier is the named fallback.

### Free PostgreSQL hosting (must support pgvector + survive a live demo)
- **Neon**: serverless, separated compute/storage, **scale-to-zero after 5 min idle with sub-1s resume** (cold start ~0.4–0.75s), pgvector supported, instant DB branching (great for per-PR CI test databases), free tier ~0.5GB with a $5 spend cap. Acquired by Databricks (May 2025), runs independently.
- **Supabase**: 500MB DB, **best-documented pgvector**, Supavisor connection pooling, 2 free projects; free projects **pause after ~7 days of inactivity** (needs a keep-alive ping).
- **Koyeb** Postgres: pgvector + 40 extensions, auto-sleeps after 5 min. **Render** free Postgres expires after 30 days (avoid for persistence). ElephantSQL has wound down its free offering. CockroachDB serverless free exists but isn't pgvector.
- **Recommendation: Neon** — scale-to-zero with sub-second resume fits a free live demo, and **branch-per-PR is a genuine MLOps talking point** that ties the SQL layer into the CI gate. Supabase is the equal-footing alternative if you prefer always-warm-ish behavior + richer pgvector docs.

### Free live app/API hosting + the multi-service problem
- **Hugging Face Spaces (Docker SDK)** is the best host for the FastAPI+LangGraph app: free CPU (2 vCPU / ~16GB RAM), **no credit card**, and per HF docs it "will go to sleep if inactive for more than a set time (currently, 48 hours)" — the most forgiving free window. `/tmp` is the only writable path; expose ports 80/443/8080. The 16GB RAM comfortably runs local embeddings.
- **Render** free web service: no card, 750 instance-hrs/mo, but **sleeps after 15 min with a 30–60s cold start** (Render: "Free web services spin down after 15 minutes of inactivity and restart on the next request, with spin-up taking about one minute"). **Koyeb** free instance: 512MB/0.1vCPU (Frankfurt/DC only), scales to zero after ~1 hr, usually no card. **Google Cloud Run** has a generous never-expiring free tier (180,000 vCPU-seconds, 360,000 GiB-seconds, 2M requests/month, scale-to-zero) but **requires a credit card** to activate billing. **Fly.io removed its free tier** — "There is no 'free account/free tier' on Fly.io. We do have a Free Trial program" (2 VM-hrs/7 days), then paid (~$2–5/mo minimum).
- **Solving the multi-service problem the realistic way**: don't try to co-host Postgres+Redis+vectorDB inside one free container. Use **managed free SaaS for state** and host only the stateless app: FastAPI+LangGraph on **HF Spaces**, Postgres+pgvector on **Neon**, Redis on **Upstash** (HTTP REST, works from any host), MLflow on **DagsHub**, optional React frontend on **Vercel**. This is the topology real teams use and demonstrates 12-factor thinking.
- **Cold-start mitigation**: HF's 48h window means demos rarely sleep; add a **GitHub Actions scheduled `curl /health` ping** (or UptimeRobot) every few hours to keep it warm, and show a loading state in the React UI.

### Free Redis / caching
- **Upstash serverless Redis (free)**: 256MB, **500K commands/month**, 10GB bandwidth, no credit card, scales to zero, and crucially exposes an **HTTP/REST API** so it works from the HF Space without persistent TCP. (Upstash moved off the old 10K-commands/day cap to the 500K/month allowance on March 12, 2025.) Use it for LLM-response caching, embedding cache, and rate-limit counters. Redis Cloud free is only 30MB; Upstash wins.

### MLflow / experiment tracking — free and live
- **DagsHub** gives every repo a **free hosted MLflow server** (`https://dagshub.com/<user>/<repo>.mlflow`) with team access control + a full MLflow UI, **plus DVC data versioning with 10GB free storage**, and now supports **MLflow 3.x** (model-first `LoggedModel`, GenAI tracing). Free for teams ≤3. This is the most CV-impressive free option because it consolidates code + data + experiments + model registry in one place. The candidate's existing W&B knowledge transfers conceptually; logging to a hosted MLflow server is the marketable, framework-agnostic MLOps signal.

### LangGraph — still the right 2026 choice
- LangGraph 1.0 went GA on Oct 22, 2025 — per LangChain's announcement, "the first stable major release in the durable agent framework space," with "90M monthly downloads and powering production applications at Uber, JP Morgan, Blackrock, Cisco, and more" (the README also names Klarna, Replit, Elastic). Production patterns are now first-class: typed shared state, **checkpointers (`SqliteSaver`, `PostgresSaver`)**, `interrupt()` human-in-the-loop, streaming, node caching, deferred nodes, and pre/post-model hooks. It's the most-adopted multi-agent framework by search volume (~27,100/mo).
- **Honest take vs newer frameworks**: **PydanticAI** (stable v1.0, April 2026) is lighter and type-safe but stateless-by-default with no native checkpointing; **OpenAI Agents SDK** is model-locked to OpenAI (disqualifying for a free open-weight stack); **CrewAI** is role-based with no built-in checkpointing. For an *explainable state-machine agent* you can whiteboard in an interview, LangGraph signals the most. **Keep LangGraph**, use the **`PostgresSaver` checkpointer on the same Neon DB** (another consolidation win), and mention PydanticAI as the framework you'd reach for on a pure-extraction microservice.

### RAGAS / RAG evaluation — current state
- The **faithfulness / answer-relevancy / context-precision / context-recall** metrics remain the standard; common production thresholds are faithfulness ≥0.8, context precision ≥0.7–0.8, answer relevancy ≥0.75 (regulated domains push ≥0.9). **RAGAS** is the dataset-level metric + synthetic-golden-set tool; **DeepEval** is pytest-native and the natural CI blocker; TruLens/Phoenix/Langfuse cover dashboards/traces.
- **Recommendation**: generate a golden Q/A set with RAGAS, run RAGAS in the **GitHub Actions quality gate** to block PRs that drop below thresholds, and log every metric to DagsHub-MLflow. Critically, **use a different judge LLM (Gemini Flash) than the generator (Groq Llama)** to avoid score inflation, and **pin the judge model version** so historical comparisons stay valid. Optionally wrap RAGAS metrics in DeepEval's pytest assertions for the cleanest CI ergonomics (`deepeval test run`).

### Evidently / drift monitoring — current state
- **Evidently** (Apache-2.0, v0.7.x line, March 2026) ships 20+ statistical tests/distance metrics (KS, PSI, Wasserstein, Jensen-Shannon, chi-squared) and **first-class embedding-drift detection** (Euclidean/cosine distance, model-based classifier, share-of-drifted-components, UMAP visualization), with HTML/JSON/dict output and MLflow/Grafana integration. NannyML is better for performance estimation under delayed labels; Arize Phoenix for LLM traces.
- **Best free approach**: set the **reference distribution = ingestion-time embedding sample**, the **current = rolling sample of live query embeddings**, run Evidently on a schedule (GitHub Actions cron), and alarm when the share of drifted components crosses threshold — surfacing "users are now asking about topics the corpus doesn't cover."

### ArXiv ingestion + PDF parsing
- **Metadata**: the **Kaggle arXiv metadata dataset** (CC0, JSONL, 1.7M+ papers, regularly refreshed) is the best bulk source → load into Postgres. Use the **arXiv API / OAI-PMH** (via `export.arxiv.org`) for incremental updates; full-text PDFs are on S3 (requester-pays) or the free `gs://arxiv-dataset` GCS bucket.
- **PDF parsing for equations + structure**: **Docling (IBM, Apache-2.0)** and **Marker** lead in 2026 for scientific layout, converting PDF→structured Markdown with section/heading recovery; **PyMuPDF** is the fast baseline; LlamaParse's free tier is limited. **Recommendation: Docling** for PDF→Markdown (preserves section hierarchy and math), with PyMuPDF as fallback.
- **Source-agnostic note**: hide ingestion behind a `SourceConnector` interface (`fetch_metadata()`, `fetch_fulltext()`, `to_canonical_doc()`); the ArXiv connector is one implementation, so adding PubMed/Semantic Scholar later is a new class, not a rewrite.

### Chunking strategy for scientific papers
- The Vecta/FloTorch February 2026 benchmark of 7 strategies across 50 academic papers ranked "recursive 512-token splitting first at 69% accuracy, while semantic chunking landed at 54% after producing fragments averaging just 43 tokens." There's a **context cliff around 2,500 tokens**, and sentence chunking ≈ semantic up to ~5,000 tokens at a fraction of the cost.
- **Recommendation**: **structure-aware first, recursive second** — use Docling's recovered headings to split on section boundaries (Abstract / Intro / Methods / Results / etc.), then **recursive character split at 512 tokens with 10–20% overlap** *within* each section, attaching metadata (`paper_id`, `arxiv_category`, `section_title`, `chunk_index`). This beats naive semantic chunking on academic text, is far cheaper, and the section metadata powers SQL analytics.

## Details

### A) Finalized opinionated tech stack table

| Component | Chosen free tool (2026) | Why it's the best/fastest free option | CV signal |
|---|---|---|---|
| Agent orchestration | **LangGraph 1.x** (`PostgresSaver` checkpointer) | Stable GA, explicit state-machine, checkpointing + HITL + streaming first-class; most-adopted multi-agent framework | "I build explainable, durable state-machine agents, not prompt spaghetti" |
| Primary LLM | **Groq — Llama 3.3 70B** | Fastest free inference (300–1,000 tok/s LPU); good tool-calling; OpenAI-compatible | Knows latency matters for agent loops |
| Fallback LLM (volume) | **Cerebras — Llama 4 Scout / Qwen3** | 1M free tokens/day, 2,600+ tok/s; ideal for batch eval | Designs multi-provider fallback |
| Fallback LLM (context/tools) | **Gemini 3 Flash** | 1M context, strong tool-calling, free | Right-tools-for-the-job judgment |
| Embeddings | **BAAI/bge-small-en-v1.5** (local) / nomic-embed fallback | Top-tier quality at 384-dim, free on CPU, small HNSW index | HuggingFace/PyTorch transfer |
| Vector store | **pgvector (HNSW)** in Postgres | Free, ≤10M-vector parity with dedicated DBs, consolidates with SQL | Architectural-tradeoff reasoning |
| Relational DB + audit log | **Neon Postgres** (pgvector) | Serverless scale-to-zero, sub-1s resume, branch-per-PR, free | SQL + serverless ops |
| Cache / rate-limit | **Upstash Redis** | 500K cmds/mo free, HTTP REST works from any host, no card | Caching + cost discipline |
| App/API host | **Hugging Face Spaces (Docker)** | No card, 48h sleep window, ~16GB RAM, ports 80/443/8080 | Docker/HF/FastAPI transfer |
| Frontend | **Vercel** (React) | Free static/SSR hosting | React transfer |
| Experiment tracking | **DagsHub hosted MLflow + DVC** | Free MLflow 3.x server + 10GB data versioning per repo | End-to-end MLOps |
| RAG evaluation gate | **RAGAS** (+ optional DeepEval pytest) | Standard metrics, synthetic golden sets, CI-friendly | Automated quality gates |
| Drift monitoring | **Evidently v0.7.x** | First-class embedding drift, Apache-2.0, MLflow/Grafana hooks | Production ML monitoring |
| PDF parsing | **Docling (IBM)** + PyMuPDF fallback | Best scientific structure/equation extraction, free | Data-engineering rigor |
| Metadata ingestion | **Kaggle arXiv dataset** + arXiv API | CC0 bulk + incremental | Pragmatic data sourcing |
| CI/CD | **GitHub Actions** | Free for public repos; runs gate + drift cron | DevOps |
| Containerization | **Docker + docker-compose** | Local parity with HF Space | Reproducibility |

### B) System architecture & free live-deployment topology
- **Client (React on Vercel)** → HTTPS → **FastAPI on HF Spaces (Docker)** exposing `/chat`, `/health`, `/metrics`.
- FastAPI hosts the **LangGraph agent** (Planner → Executor → Critic → Reporter state machine) and three tools: **SQL analytics tool** (→ Neon Postgres), **RAG retrieval tool** (→ pgvector in the same Neon DB), **web search tool** (free DuckDuckGo/SearXNG or Tavily free tier).
- **State & memory**: LangGraph checkpoints via `PostgresSaver` on Neon; **Upstash Redis** caches LLM responses, embeddings, and holds rate-limit counters.
- **LLM gateway** inside the app: OpenAI-compatible client routing Groq→Cerebras→Gemini with backoff.
- **MLOps plane (offline + CI)**: GitHub Actions runs ingestion, the **RAGAS gate**, and the **Evidently drift cron**; metrics/artifacts → **DagsHub MLflow + DVC**.
- Everything that holds state is an external free managed service, so the only thing that can "sleep" is the stateless app container (48h window + keep-alive ping).

### C) Week-by-week plan (4–5 weeks @ 2–3 hrs/day) — schema FIRST
- **Week 1 — Data foundation & SQL (Gap 3 first).** Design the **PostgreSQL schema before any ML code**: `papers` (arxiv_id PK, title, authors[], categories[], abstract, published_at, updated_at), `chunks` (id, paper_id FK, section_title, chunk_index, content, token_count, `embedding vector(384)`), `query_audit_log` (id, ts, user_query, route, tools_called[], latency_ms, tokens_in, tokens_out, llm_provider, retrieved_chunk_ids[], faithfulness_score), and an `experiments` view. Add an HNSW index on `chunks.embedding`, btree indexes, and write the **analytical queries with window functions** (e.g., `ROW_NUMBER() OVER (PARTITION BY category ORDER BY published_at)`, rolling 7-day query volume, p95 latency per provider via `PERCENTILE_CONT`). Load Kaggle metadata → `papers`. Stand up the FastAPI skeleton + Docker + Neon connection + Upstash.
- **Week 2 — Ingestion & RAG core (Gap 1).** Docling PDF→Markdown; structure-aware + recursive-512 chunking; bge-small embeddings; bulk insert into `chunks`; build the **RAG retrieval tool** (vector + metadata filter SQL). Stand up DagsHub MLflow; log the first ingestion run + corpus stats. Wire query audit logging on every request.
- **Week 3 — LangGraph agent.** Implement Planner/Executor/Critic/Reporter nodes, the state schema, the tool registry, conditional edges, retries, and `PostgresSaver` checkpointing. Add the SQL analytics tool and web search tool. Implement the LLM gateway with provider fallback.
- **Week 4 — MLOps & quality gates (Gap 2).** Build the RAGAS golden set + the GitHub Actions **RAGAS quality gate** (Gemini judge, pinned), the Evidently embedding-drift cron, model/prompt versioning in MLflow, and full CI (lint, pytest, build, gate). Use a branch-per-PR Neon DB for tests.
- **Week 5 — Polish & ship.** React UI, deploy to HF Spaces, keep-alive ping, README with architecture diagram + live demo URL + GIF, tests to ~70% coverage, and recorded interview talking points.

### D) Data pipeline
ArXiv API/Kaggle → `SourceConnector.fetch_metadata()` → **Postgres `papers`** → `fetch_fulltext()` (PDF) → **Docling** PDF→Markdown → **structure-aware split** on sections → **recursive 512-token, 10–20% overlap** within sections → **bge-small embeddings** (batched on CPU) → insert `chunks` with metadata → **HNSW index** → retrieval tool queries `ORDER BY embedding <=> :q LIMIT k` with category/section filters. Fully source-agnostic via the connector interface.

### E) Agent architecture (LangGraph)
- **State schema (`TypedDict`)**: `user_query`, `plan: list[step]`, `current_step`, `tool_calls: list`, `retrieved_context: list[chunk]`, `draft_answer`, `critique`, `retry_count`, `final_report`, `audit: dict`.
- **Nodes**: **PlannerNode** (LLM decomposes the query into steps + picks tools via function-calling), **ExecutorNode** (runs the chosen tool — SQL / RAG / web — and appends results to state), **CriticNode** (LLM grades grounding/coverage; if insufficient and `retry_count<N`, routes back to Planner/Executor), **ReporterNode** (synthesizes the cited final report).
- **Tool selection**: the Planner emits a structured tool choice; conditional edges route on it. Factual-corpus questions → RAG tool; "how many / trend / top-N" questions → SQL analytics tool; out-of-corpus/recency → web search.
- **Retry/error handling**: per-tool try/except → on tool failure, the Critic sees an error sentinel and re-plans with an alternate tool (e.g., RAG empty → web search); LLM 429 → gateway fallback Groq→Cerebras→Gemini with exponential backoff; the `PostgresSaver` checkpoint lets a crashed run resume; a hard cap on retries prevents loops.

### F) MLOps layer wiring
- **Experiment tracking**: each ingestion/eval run logs params (embedding model, chunk size, overlap, k), metrics (RAGAS scores, retrieval latency), and artifacts (golden set, config) to **DagsHub MLflow**; DVC versions the corpus + embeddings.
- **RAGAS CI gate**: GitHub Actions on PR → spins a Neon branch DB → runs RAGAS over the golden set with **Gemini judge (pinned)** → **fails the build** if faithfulness/context-precision/answer-relevancy drop below thresholds → posts scores to MLflow. Mirrors unit-test discipline for RAG quality.
- **Drift monitoring**: a scheduled Action samples live query embeddings from `query_audit_log`, runs **Evidently** against the ingestion-time reference, writes the HTML/JSON report as a build artifact + to MLflow, and alarms on drift.

### G) Repo, README, testing, deployment
- **Structure**: `app/` (FastAPI, routes), `agent/` (graph, nodes, tools, state), `ingestion/` (connectors, parsing, chunking, embeddings), `db/` (schema.sql, migrations, queries), `eval/` (ragas, golden set), `monitoring/` (evidently), `tests/`, `.github/workflows/`, `Dockerfile`, `docker-compose.yml` (Postgres+Redis+app for local parity), `README.md`.
- **README requirements**: one-line value prop, **live demo URL**, architecture diagram, the three CV gaps mapped to components, local-setup, env vars, the RAGAS gate explanation, and a demo GIF.
- **Testing**: pytest unit tests (chunking, tool routing, SQL queries), an integration test of the full graph on a fixture corpus, RAGAS eval as a gated test, ~70% coverage target.
- **Deployment**: Dockerfile (uvicorn on 7860, caches to `/tmp`, non-root user) → push to the HF Space; secrets in HF Space settings; GitHub Actions builds/tests on PR and can push to the Space on merge.

### H) Exact CV bullet(s)
- *"Built and deployed a live, fully-serverless Autonomous Research Intelligence Agent (LangGraph state-machine: Planner/Executor/Critic/Reporter) over an ArXiv-ML corpus, with multi-provider free LLM routing (Groq/Cerebras/Gemini), pgvector RAG, and a PostgreSQL analytics layer — shipped free on Hugging Face Spaces."*
- *"Engineered a production MLOps pipeline: DagsHub-hosted MLflow experiment tracking + DVC, an automated RAGAS quality gate in GitHub Actions that blocks regressions, and Evidently embedding-drift monitoring."*
- *"Designed a normalized PostgreSQL schema with pgvector and window-function analytics over a query audit log, consolidating vector search and relational analytics in one database."*

### I) Interview talking points enabled
- **Tool-failure handling**: how the Critic re-plans on empty retrieval and how the LLM gateway fails over on 429s with backoff and checkpoint-resume.
- **Chunking choice**: why structure-aware + recursive-512 beats pure semantic on academic papers (the ~43-token fragment failure mode and the ~2,500-token context cliff).
- **Quality gate mechanics**: golden set, judge-vs-generator separation to avoid score inflation, pinned judge version, threshold-blocking in CI.
- **pgvector vs dedicated DB**: the 0.99 accuracy@10 parity below ~10M vectors and the 5M-vector / 80–140ms p95 migration trigger.
- **Drift**: reference-vs-current embedding distributions and what "share of drifted components" means for a RAG corpus.
- **Framework judgment**: why LangGraph's checkpointing/HITL fit an explainable agent vs PydanticAI/OpenAI Agents SDK.
- **Free-tier engineering**: the stateless-app + managed-state topology and cold-start mitigation.

## Recommendations
1. **Start with the schema and SQL (Week 1).** It de-risks the hardest-to-change layer and front-loads Gap 3.
2. **Commit to pgvector + Neon + HF Spaces + Upstash + DagsHub now**; don't shop tiers mid-build.
3. **Build the LLM gateway with fallback before the agent** so rate limits never block development.
4. **Gate from day one**: even a 20-question RAGAS golden set in CI is a strong signal.
5. **Thresholds that change the plan**: if the corpus exceeds ~5–10M vectors or p95 vector latency exceeds ~150ms, migrate to Qdrant; if Groq 429s dominate development, promote Cerebras to primary; if HF Spaces RAM is tight during embedding, offload to the Gemini embedding API.

## Caveats
- Free-tier limits change frequently — Gemini cut its free tier twice (Dec 2025, April 2026) and Fly.io removed its free tier entirely; re-verify quotas before launch.
- Groq/Gemini per-model RPD/TPD figures vary across sources and by account/region; treat the numbers as current-snapshot, not contractual.
- Supabase pauses free projects after ~7 days idle and Neon free is ~0.5GB with a $5 cap — size the demo corpus accordingly and add keep-alives.
- HF Spaces cold-start *duration* isn't officially published (only the 48h sleep window is); expect tens of seconds on wake.
- MTEB leadership rotates fast (Qwen3-Embedding-8B led in mid-2025 but was surpassed by early 2026) — pick by your own retrieval test, not the leaderboard headline.
- Vendor benchmarks (chunking, vector DB) are directional, not independently confirmed; validate on your own corpus.