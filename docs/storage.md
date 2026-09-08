# Storage: living inside Neon's 512 MB free tier

## The problem

The database sat at **483 MB of a 512 MB ceiling** — 29 MB of headroom, on a
corpus that was only 80% embedded (40,000 of 50,000 papers had chunks).

That is not merely "nearly full". It had already caused a production failure:
the `published_at` backfill hit `DiskFull` partway through a bulk `UPDATE`,
because an update writes new row versions before the old ones can be vacuumed.
Any operation that temporarily doubles a chunk of the table — a backfill, a
re-index, a restore — had no room to run. Embedding the remaining 10,000 papers
was out of the question.

### Where the space went

| Relation | Total | Notes |
|---|---:|---|
| `chunks` | 395 MB | 56 MB heap · 175 MB indexes · ~164 MB TOAST |
| `papers` | 78 MB | 50,000 rows, metadata only |
| everything else | ~2 MB | audit log, LLM call log, LangGraph checkpoints |

Inside `chunks`, the embeddings were the whole story:

| Component | Size |
|---|---:|
| `chunks_embedding_hnsw` (HNSW over fp32) | 156 MB |
| embedding payload (TOAST) | ~123 MB |
| `chunks_content_tsv_gin` (BM25) | 17 MB |
| content + metadata | ~56 MB |

Two earlier ideas were investigated and rejected on evidence:

- **Prune redundant chunks.** There are none. The table holds 40,001 rows for
  40,000 papers — one chunk per paper, each the sole retrieval representation of
  that paper. Deleting chunks removes papers from the index; it does not reclaim
  slack. (`context` is 100% NULL and every `section_title` is `'abstract'`, so
  there is no redundancy hiding there either.)
- **Recency-stratified rebalance** (`ingestion/rebalance.py`). Its premise was an
  artefact of the `published_at` corruption: it exists to re-weight the corpus
  toward post-2021 papers, and once the dates were repaired the corpus turned out
  to end at 2018-05-16 with **zero** post-2021 papers. The tool now refuses to
  run without `--force`.

## The fix: fp32 → fp16 vectors

`db/migrations/003_halfvec_embeddings.sql` converts `chunks.embedding` from
`vector(768)` to `halfvec(768)`.

A `vector(768)` occupies 3080 bytes; a `halfvec(768)` occupies 1544. Both the
stored payload and the HNSW index built over it halve.

The embeddings stay in TOAST either way. An earlier draft of this document
claimed fp16 would bring them back inline because 1544 bytes is under the 2 KB
threshold — that is wrong, and the applied migration confirms it: the `chunks`
heap was 56 MB before and 57 MB after. `TOAST_TUPLE_THRESHOLD` applies to the
whole row, not to one attribute. The average row here is a 901-byte `content`
plus the embedding, so at 1544 bytes the row is still ~2.5 KB and Postgres
still pushes its widest attribute out of line. The win is that the out-of-line
payload is half the size (164 MB → 100 MB), not that it moved.

Measured, on the live database (2026-09-08):

| | Before | After |
|---|---:|---:|
| Database total | 483 MB | **341 MB** |
| `chunks` total | 395 MB | 252 MB |
| — heap | 56 MB | 57 MB |
| — TOAST | ~164 MB | 100 MB |
| — HNSW index | 156 MB | 78 MB |
| Headroom under 512 MB | 29 MB | **171 MB** |

The migration took 57 seconds end to end: 0.0s to drop the index, 16.3s for the
column rewrite, 40.2s to rebuild HNSW, 0.4s to vacuum.

### Why this is safe

fp16 carries about three decimal digits of precision. The embeddings are
L2-normalised `nomic-embed-text-v2-moe` outputs, so every component lies in
[-1, 1] — the range fp16 represents most accurately — and cosine ranking depends
only on the *relative order* of scores, not their absolute values. On top of
that, `rag_retrieval` over-fetches 16 candidates and LLM-reranks down to 8, so
the pipeline already tolerates small perturbations in ANN ordering.

This is pgvector's own recommended approach for shrinking an index, but it was
verified rather than assumed: top-10 neighbour sets for 25 fixed probe vectors
were captured before the migration and re-measured after (see "Verification").

**Result: no measurable retrieval loss.** Mean overlap@10 was 1.000, minimum
overlap@10 was 1.000, and the exact nearest neighbour was preserved for 25 of
25 probes. Every probe returned an identical top-10 to the fp32 index.

### Why the statement order matters

The index is dropped **first**. That frees 156 MB, and it is the only reason the
rest of the migration fits under the ceiling at all — a table rewrite needs room
for both the old and new copies at once. Between the `DROP` and the `CREATE`,
vector search degrades to a sequential scan over 40k rows: still correct, just
slower. Do not reorder the statements.

The application does not need to be stopped. pgvector registers `vector ->
halfvec` as an **implicit** cast, so an instance still running the old
`%s::vector` parameter cast keeps working against the converted column.

## Running it

```bash
psql $DATABASE_URL -f db/migrations/003_halfvec_embeddings.sql
```

If there is no local `psql` client, use the in-repo runner instead — it executes
the same file statement-by-statement in autocommit, which is required here
because the final `VACUUM` cannot run inside a transaction block:

```bash
# print the statements without touching the database
PYTHONPATH=. python3 -m db.apply_migration db/migrations/003_halfvec_embeddings.sql --dry-run

# apply (DATABASE_URL from the environment or .env)
DATABASE_URL="postgresql://..." PYTHONPATH=. python3 -m db.apply_migration \
  db/migrations/003_halfvec_embeddings.sql
```

Roughly a minute at 40k rows, most of it the HNSW rebuild. `ALTER TABLE` holds
an `ACCESS EXCLUSIVE` lock on `chunks` for the rewrite.

**Rollback.** The conversion is reversible in the same shape — `ALTER TABLE
chunks ALTER COLUMN embedding TYPE vector(768)`, then recreate the index with
`vector_cosine_ops`. It is lossy, though: values that went through fp16 do not
regain the precision they had before, so a rollback restores the type and the
disk usage, not the original numbers.

## Verification

```bash
# column type and index opclass
psql $DATABASE_URL -c "\d+ chunks"

# the index is actually being used (expect an Index Scan, not a Seq Scan)
psql $DATABASE_URL -c "EXPLAIN SELECT id FROM chunks ORDER BY embedding <=> (SELECT embedding FROM chunks LIMIT 1) LIMIT 10"
```

Confirmed after the applied migration: the column reports `halfvec`, and the
plan is `Index Scan using chunks_embedding_hnsw`, not a sequential scan.
All four agent tool paths (`rag_retrieval` ×2, `sql_analytics`, `graph_query`)
were re-run end to end against the converted column with **no code change** —
the implicit `vector -> halfvec` cast means `%s::vector` bindings still work.

## What the headroom is for

171 MB free is enough to embed the remaining 10,000 papers (bringing the corpus
to full coverage at roughly 420–450 MB) and still leave room for a bulk `UPDATE`
to run without hitting `DiskFull` again. That is a tighter margin than the
earlier ~210 MB estimate, because the embeddings stayed in TOAST rather than
moving inline — plan the next ingestion batch against 171 MB, not 210 MB.

If the corpus later grows past that, the next levers in order of preference are:
binary quantization for the index only (keeping halfvec for reranking), dropping
`chunks.content` in favour of joining `papers.abstract` at query time, and
finally a paid Neon tier.
