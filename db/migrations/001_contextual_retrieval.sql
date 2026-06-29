-- Migration 001: Contextual Retrieval
-- Adds LLM context blurb column and tsvector column for BM25 hybrid search.
-- Safe to run multiple times (IF NOT EXISTS / IF NOT EXISTS guards).
--
-- Apply with:
--   psql $DATABASE_URL -f db/migrations/001_contextual_retrieval.sql
--
-- After applying, re-embed chunks on Kaggle/Colab:
--   python -m ingestion.pipeline --limit 10000 --batch-size 200

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS context TEXT;

-- Generated column: concatenates context blurb + raw content for full-text search.
-- Postgres requires dropping and re-adding generated columns if they don't exist,
-- so we guard with a DO block.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'content_tsv'
    ) THEN
        ALTER TABLE chunks ADD COLUMN content_tsv TSVECTOR
            GENERATED ALWAYS AS (
                to_tsvector('english', coalesce(context, '') || ' ' || content)
            ) STORED;
    END IF;
END
$$;

CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_content_tsv_gin
    ON chunks USING gin(content_tsv);
