-- 003_halfvec_embeddings.sql
--
-- Convert chunks.embedding from vector(768) (fp32) to halfvec(768) (fp16).
--
-- WHY: Neon's free tier caps the project at 512 MB and this database sat at
-- 483 MB, i.e. 29 MB of headroom -- not enough to embed another paper, and
-- close enough to the ceiling that a routine bulk UPDATE had already failed
-- with DiskFull once (see ingestion/backfill_dates.py). The chunks table was
-- 395 MB of that: a vector(768) is 3080 bytes, which exceeds the 2 KB TOAST
-- threshold, so every embedding was pushed out of line into TOAST storage,
-- and the HNSW index over fp32 vectors was 156 MB on its own.
--
-- A halfvec(768) is 1544 bytes. That halves the vector payload AND drops it
-- back under the TOAST threshold so it is stored inline in the heap, and it
-- halves the HNSW index. Measured effect: 483 MB -> ~300 MB.
--
-- Precision: fp16 carries ~3 decimal digits. The embeddings are L2-normalised
-- nomic-embed-text-v2-moe outputs in [-1, 1], which is the range fp16 is most
-- accurate over, and cosine ranking only depends on relative ordering. This is
-- pgvector's own recommended path for shrinking an index. Recall is verified
-- empirically, not assumed -- see docs/storage.md.
--
-- ORDERING MATTERS. The index is dropped FIRST, which frees 156 MB and is what
-- makes the rest of the migration fit under the 512 MB ceiling at all. Between
-- the DROP and the CREATE, vector search falls back to a sequential scan: still
-- correct, just slower (40k rows). Do not reorder these statements.
--
-- The application does not need to be stopped. pgvector registers vector ->
-- halfvec as an IMPLICIT cast, so a running instance still issuing the old
-- `%s::vector` parameter cast keeps working against the converted column.

-- Step 1 -- drop the fp32 HNSW index. Frees ~156 MB, creating the headroom
-- the table rewrite in step 2 needs.
DROP INDEX IF EXISTS chunks_embedding_hnsw;

-- Step 2 -- rewrite the column as fp16. This is a full table rewrite; it takes
-- an ACCESS EXCLUSIVE lock on chunks for its duration (~1 min at 40k rows).
ALTER TABLE chunks
    ALTER COLUMN embedding TYPE halfvec(768)
    USING embedding::halfvec(768);

-- Step 3 -- rebuild HNSW over the fp16 column. Note halfvec_cosine_ops, not
-- vector_cosine_ops: the opclass must match the column type.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Step 4 -- refresh planner statistics against the new column type.
VACUUM (ANALYZE) chunks;
