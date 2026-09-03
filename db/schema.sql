-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Papers table
CREATE TABLE IF NOT EXISTS papers (
    id           BIGSERIAL PRIMARY KEY,
    arxiv_id     TEXT UNIQUE NOT NULL,
    title        TEXT NOT NULL,
    authors      TEXT[] NOT NULL DEFAULT '{}',
    categories   TEXT[] NOT NULL DEFAULT '{}',
    abstract     TEXT,
    published_at TIMESTAMPTZ,
    updated_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Chunks table (768-dim for nomic-embed-text-v2)
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    paper_id      BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_title TEXT NOT NULL DEFAULT 'abstract',
    chunk_index   INT  NOT NULL DEFAULT 0,
    content       TEXT NOT NULL,
    context       TEXT,                          -- LLM-generated situating blurb (NULL = not contextualised)
    content_tsv   TSVECTOR GENERATED ALWAYS AS (
                      to_tsvector('english', coalesce(context, '') || ' ' || content)
                  ) STORED,                      -- for BM25 full-text search
    token_count   INT,
    embedding     vector(768),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Query audit log
CREATE TABLE IF NOT EXISTS query_audit_log (
    id                  BIGSERIAL PRIMARY KEY,
    session_id          TEXT,
    ts                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_query          TEXT NOT NULL,
    route               TEXT,
    tools_called        TEXT[]   DEFAULT '{}',
    latency_ms          INT,
    tokens_in           INT,
    tokens_out          INT,
    llm_provider        TEXT,
    retrieved_chunk_ids BIGINT[] DEFAULT '{}',
    faithfulness_score  FLOAT,
    answer_relevancy    FLOAT
);

-- HNSW index (pgvector 0.8.0)
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for BM25 full-text search
CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin ON chunks USING gin(content_tsv);

-- Supporting btree/GIN indexes
CREATE INDEX IF NOT EXISTS papers_categories_gin ON papers USING GIN (categories);
CREATE INDEX IF NOT EXISTS papers_published_at_idx ON papers (published_at DESC);
CREATE INDEX IF NOT EXISTS chunks_paper_id_idx ON chunks (paper_id);
CREATE INDEX IF NOT EXISTS audit_ts_idx ON query_audit_log (ts DESC);
CREATE INDEX IF NOT EXISTS audit_session_idx ON query_audit_log (session_id);

-- Per-LLM-call cost/latency log (see db/migrations/002_llm_call_log.sql for notes)
CREATE TABLE IF NOT EXISTS llm_call_log (
    id           BIGSERIAL PRIMARY KEY,
    session_id   TEXT,
    ts           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    node         TEXT NOT NULL,
    provider     TEXT,
    model        TEXT,
    tokens_in    INT,
    tokens_out   INT,
    cost_usd     NUMERIC(12, 8),
    latency_ms   INT,
    is_retry     BOOLEAN NOT NULL DEFAULT FALSE,
    cached       BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS llm_call_log_ts_idx ON llm_call_log (ts DESC);
CREATE INDEX IF NOT EXISTS llm_call_log_session_idx ON llm_call_log (session_id);
CREATE INDEX IF NOT EXISTS llm_call_log_node_idx ON llm_call_log (node);

-- Analytics view
CREATE OR REPLACE VIEW experiments AS
SELECT
    DATE_TRUNC('day', ts)                                          AS day,
    llm_provider,
    route,
    COUNT(*)                                                       AS query_count,
    AVG(latency_ms)                                                AS avg_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)      AS p95_latency_ms,
    AVG(faithfulness_score)                                        AS avg_faithfulness,
    AVG(answer_relevancy)                                          AS avg_relevancy
FROM query_audit_log
GROUP BY 1, 2, 3;

-- Cost/latency by day, node, provider — the main cost/latency dashboard aggregate.
CREATE OR REPLACE VIEW llm_cost_latency AS
SELECT
    DATE_TRUNC('day', ts)                                     AS day,
    node,
    provider,
    COUNT(*)                                                  AS call_count,
    SUM(tokens_in)                                            AS tokens_in,
    SUM(tokens_out)                                           AS tokens_out,
    SUM(cost_usd)                                             AS total_cost_usd,
    AVG(latency_ms)                                           AS avg_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms
FROM llm_call_log
WHERE NOT cached
GROUP BY 1, 2, 3;

-- Retry overhead as its own line item — cost/latency attributable to
-- Critic-triggered retry passes, isolated from "happy path" node totals.
CREATE OR REPLACE VIEW llm_retry_overhead AS
SELECT
    DATE_TRUNC('day', ts)                AS day,
    is_retry,
    COUNT(*)                             AS call_count,
    SUM(cost_usd)                        AS total_cost_usd,
    SUM(latency_ms)                      AS total_latency_ms,
    AVG(latency_ms)                      AS avg_latency_ms
FROM llm_call_log
WHERE NOT cached
GROUP BY 1, 2;
