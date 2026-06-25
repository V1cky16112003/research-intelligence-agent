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

-- Supporting btree/GIN indexes
CREATE INDEX IF NOT EXISTS papers_categories_gin ON papers USING GIN (categories);
CREATE INDEX IF NOT EXISTS papers_published_at_idx ON papers (published_at DESC);
CREATE INDEX IF NOT EXISTS chunks_paper_id_idx ON chunks (paper_id);
CREATE INDEX IF NOT EXISTS audit_ts_idx ON query_audit_log (ts DESC);
CREATE INDEX IF NOT EXISTS audit_session_idx ON query_audit_log (session_id);

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
