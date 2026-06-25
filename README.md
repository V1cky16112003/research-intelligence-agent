# Autonomous Research Intelligence Agent

An AI-powered research assistant for ArXiv ML papers — LangGraph state-machine agent, pgvector RAG, and SQL analytics over a 50K-paper corpus.

## Tech Stack
- **Agent:** LangGraph 1.2.6 (Planner→Executor→Critic→Reporter)
- **LLM:** Groq Llama 3.3 70B → Gemini 2.5 Flash fallback
- **Embeddings:** nomic-embed-text-v2 (768-dim, local CPU)
- **Vector + SQL DB:** pgvector 0.8.0 on Neon Postgres
- **Cache:** Upstash Redis
- **Deploy:** Hugging Face Spaces (API) + Vercel (React UI)

## Quickstart
```bash
cp .env.example .env
# Fill in GROQ_API_KEY, GEMINI_API_KEY at minimum
docker-compose up
curl http://localhost:7860/health
```

## Architecture
*Coming soon — see project idea.md*
