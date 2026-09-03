-- Migration 002: LLM call-level cost/latency observability
-- Adds a per-LLM-call log (node, provider, tokens, estimated cost, latency,
-- retry attribution) plus two aggregation views built on top of it.
-- Safe to run multiple times (IF NOT EXISTS guards throughout).
--
-- Apply with:
--   psql $DATABASE_URL -f db/migrations/002_llm_call_log.sql

CREATE TABLE IF NOT EXISTS llm_call_log (
    id           BIGSERIAL PRIMARY KEY,
    session_id   TEXT,
    ts           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    node         TEXT NOT NULL,          -- planner | executor | critic | reporter
    provider     TEXT,                    -- groq | nvidia_nim | gemini
    model        TEXT,
    tokens_in    INT,
    tokens_out   INT,
    cost_usd     NUMERIC(12, 8),
    latency_ms   INT,
    is_retry     BOOLEAN NOT NULL DEFAULT FALSE,  -- true if issued during a Critic-triggered retry pass
    cached       BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS llm_call_log_ts_idx ON llm_call_log (ts DESC);
CREATE INDEX IF NOT EXISTS llm_call_log_session_idx ON llm_call_log (session_id);
CREATE INDEX IF NOT EXISTS llm_call_log_node_idx ON llm_call_log (node);

-- Cost/latency by day, node, provider — the main dashboard aggregate.
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
