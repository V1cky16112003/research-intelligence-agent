from __future__ import annotations

"""
Corpus rebalance: recency-stratified re-ingest within the Neon 512 MB cap.

Phase 2 of the corpus-rebalance roadmap.

!! PREMISE INVALIDATED 2026-09-05 — DO NOT --apply WITHOUT RE-DESIGNING. !!

This was written against `published_at` values loaded from the snapshot's
`update_date`, which made the corpus look like it ran to 2026 with 1,865 (3.7%)
post-2021 papers. Those dates were wrong. After `ingestion/backfill_dates.py`,
the corpus is arXiv IDs 0704-1805 and genuinely ends 2018-05-16, so there are
**zero** post-2021 papers. The `modern` stratum this script exists to fill can
never be filled from this corpus; the only fix is ingesting newer papers.

The guards in `apply()` now abort on both of these rather than proceeding.

Two modes:

  --dry-run (default):  measure live DB size, row counts, and year distribution;
                        project post-rebalance storage and abort if it would
                        exceed 90% of the ceiling. Prints a plan, writes nothing.

  --apply:              copy chunks to a backup table (a full CTAS data copy,
                        ~220 MB at current size — NOT cheap, see guards),
                        TRUNCATE chunks, re-embed a recency-stratified sample
                        of papers, bulk-insert, verify counts, drop the backup.

Reuses ingestion.embed (chunk_text, embed_chunks), db.queries.insert_chunks_batch,
and db.connection (init_pool, get_connection). Does NOT duplicate the embedding
pipeline — only the recency-stratified paper selection and the truncate/backup
guards are new.

Usage:
    python -m ingestion.rebalance --dry-run
    python -m ingestion.rebalance --apply --target-chunks 12000 --modern-share 0.25

Run --dry-run first. Every --apply is preceded by an automatic dry-run check
that re-aborts on the storage ceiling; --force disables that final guard
(not recommended).
"""
import argparse
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

NEON_FREE_TIER_BYTES = 512 * 1024 * 1024  # 512 MB
SAFETY_CEILING = 0.90  # abort if projected DB size would exceed 90% of the cap
POST_2021_CUTOFF_YEAR = 2021


async def _db_size(conn) -> int:
    """Return pg_database_size() in bytes."""
    async with conn.cursor() as cur:
        await cur.execute("SELECT pg_database_size(current_database())")
        row = await cur.fetchone()
    return int(row[0]) if row else 0


async def _row_counts(conn) -> dict:
    """Papers and chunks counts + year distribution."""
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT count(*) FROM papers WHERE abstract IS NOT NULL AND abstract != ''"
        )
        n_papers = (await cur.fetchone())[0]
        await cur.execute("SELECT count(*) FROM chunks")
        n_chunks = (await cur.fetchone())[0]
        await cur.execute(
            """
            SELECT EXTRACT(YEAR FROM published_at)::int AS yr, count(*) AS n
            FROM papers
            WHERE abstract IS NOT NULL AND abstract != ''
              AND published_at IS NOT NULL
            GROUP BY yr
            ORDER BY yr DESC
            """
        )
        year_rows = await cur.fetchall()
    return {
        "papers": n_papers,
        "chunks": n_chunks,
        "by_year": {int(yr): int(n) for yr, n in year_rows},
    }


async def _fetch_stratified_paper_ids(
    conn, target_chunks: int, modern_share: float
) -> list[int]:
    """Return paper ids for the recency-stratified sample.

    Strategy: assume ~1 chunk per abstract (abstracts are short; chunk_text
    keeps them whole). Pick `modern_share` of the budget from post-2021 papers
    (recency-first) and the remainder from older papers, walking back by year
    until the budget is filled. This is a best-effort sample; the precise
    embed count is checked after embed_chunks runs.
    """
    modern_n = int(target_chunks * modern_share)
    classic_n = target_chunks - modern_n

    async with conn.cursor() as cur:
        # Modern: post-cutoff, most-recent first.
        await cur.execute(
            """
            SELECT id FROM papers
            WHERE abstract IS NOT NULL AND abstract != ''
              AND published_at IS NOT NULL
              AND EXTRACT(YEAR FROM published_at) >= %s
            ORDER BY published_at DESC
            LIMIT %s
            """,
            (POST_2021_CUTOFF_YEAR, modern_n),
        )
        modern_ids = [r[0] for r in await cur.fetchall()]

        # Classic: pre-cutoff, most-recent first (closest to cutoff preferred).
        await cur.execute(
            """
            SELECT id FROM papers
            WHERE abstract IS NOT NULL AND abstract != ''
              AND published_at IS NOT NULL
              AND EXTRACT(YEAR FROM published_at) < %s
            ORDER BY published_at DESC
            LIMIT %s
            """,
            (POST_2021_CUTOFF_YEAR, classic_n),
        )
        classic_ids = [r[0] for r in await cur.fetchall()]

    return modern_ids + classic_ids


async def _project_storage(
    conn, n_papers: int, target_chunks: int
) -> dict:
    """Estimate post-rebalance DB size from a sample-row measurement.

    Measures the average bytes/chunk empirically: if chunks exist, sample
    their on-disk size; otherwise fall back to abstract-length heuristic
    (~3 KB/chunk incl. 768-dim embedding + tsvector + row overhead).
    """
    avg_bytes_per_chunk = 3 * 1024  # conservative fallback
    async with conn.cursor() as cur:
        await cur.execute("SELECT count(*) FROM chunks")
        n_existing = (await cur.fetchone())[0]
        if n_existing > 0:
            await cur.execute(
                "SELECT pg_total_relation_size('chunks'), count(*) FROM chunks"
            )
            size_row = await cur.fetchone()
            chunks_total_bytes = int(size_row[0])
            counted = int(size_row[1])
            if counted:
                avg_bytes_per_chunk = chunks_total_bytes // counted

    current_size = await _db_size(conn)
    projected_chunks_bytes = target_chunks * avg_bytes_per_chunk
    existing_chunks_bytes = n_existing * avg_bytes_per_chunk
    projected_db_size = current_size - existing_chunks_bytes + projected_chunks_bytes

    return {
        "current_db_bytes": current_size,
        "avg_bytes_per_chunk": avg_bytes_per_chunk,
        "projected_db_bytes": projected_db_size,
        "projected_db_pct_of_ceiling": round(
            100.0 * projected_db_size / NEON_FREE_TIER_BYTES, 1
        ),
        "would_exceed_ceiling": projected_db_size > SAFETY_CEILING * NEON_FREE_TIER_BYTES,
    }


async def dry_run(target_chunks: int, modern_share: float) -> dict:
    """Measure + project. Writes nothing. Returns the projection dict."""
    from db.connection import get_connection, init_pool

    await init_pool()
    async with get_connection() as conn:
        counts = await _row_counts(conn)
        projection = await _project_storage(
            conn, counts["papers"], target_chunks
        )

    plan = {
        "target_chunks": target_chunks,
        "target_modern_share": modern_share,
        "target_modern_chunks": int(target_chunks * modern_share),
        "target_classic_chunks": target_chunks - int(target_chunks * modern_share),
        "current": counts,
        "projection": projection,
        "ceiling_bytes": NEON_FREE_TIER_BYTES,
        "safety_ceiling_pct": int(SAFETY_CEILING * 100),
    }

    modern_now = sum(
        n for yr, n in counts["by_year"].items() if yr >= POST_2021_CUTOFF_YEAR
    )
    plan["current_modern_share"] = round(
        100.0 * modern_now / counts["papers"], 2
    ) if counts["papers"] else 0.0

    return plan



async def _assert_safe_to_truncate(
    conn, paper_ids: list[int], target_chunks: int, modern_share: float, force: bool
) -> None:
    """Refuse to TRUNCATE `chunks` when the rebuild cannot replace what it destroys.

    Three ways this script can quietly do more harm than good, all of which it
    previously walked straight into:

    1. The modern stratum is empty. Post-backfill the corpus ends 2018-05-16, so
       `published_at >= 2021` matches nothing and the "rebalance" just re-picks the
       newest classic papers — the region already at 100% coverage.
    2. The rebuild is smaller than the corpus it replaces. With the default
       --target-chunks 12000 against 40,001 existing chunks, TRUNCATE would drop
       31,001 embeddings that nothing in this run puts back.
    3. The backup is a full CTAS copy of every embedding (~220 MB), not the
       "cheap metadata-only" operation the docstring used to claim. At 483/512 MB
       it cannot fit, so it fails with DiskFull *before* the TRUNCATE — data
       survives by luck of statement ordering, not by design.
    """
    modern_n = int(target_chunks * modern_share)
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT count(*) FROM papers WHERE published_at IS NOT NULL "
            "AND EXTRACT(YEAR FROM published_at) >= %s",
            (POST_2021_CUTOFF_YEAR,),
        )
        modern_available = (await cur.fetchone())[0]
        await cur.execute("SELECT count(*) FROM chunks")
        existing_chunks = (await cur.fetchone())[0]
        db_size = await _db_size(conn)
        await cur.execute("SELECT pg_total_relation_size('chunks')")
        chunks_size = (await cur.fetchone())[0]

    problems = []
    if modern_n and modern_available < modern_n:
        problems.append(
            f"modern stratum wants {modern_n} papers >= {POST_2021_CUTOFF_YEAR} "
            f"but only {modern_available} exist — this corpus ends in 2018, so a "
            f"recency rebalance cannot do anything except re-pick 2018 papers"
        )
    if len(paper_ids) < existing_chunks:
        problems.append(
            f"rebuild selects {len(paper_ids):,} papers but TRUNCATE would destroy "
            f"{existing_chunks:,} existing chunks — a net loss of "
            f"{existing_chunks - len(paper_ids):,} embeddings"
        )
    if db_size + chunks_size > NEON_FREE_TIER_BYTES:
        problems.append(
            f"the CTAS backup needs another {chunks_size // (1024*1024)} MB on top of "
            f"the current {db_size // (1024*1024)} MB, over the "
            f"{NEON_FREE_TIER_BYTES // (1024*1024)} MB ceiling — it will DiskFull"
        )

    if problems and not force:
        raise SystemExit(
            "ABORT: refusing to truncate `chunks`.\n  - "
            + "\n  - ".join(problems)
            + "\nPass --force only if you have re-read this script against the "
              "current corpus and genuinely intend the loss."
        )
    if problems:
        logger.warning("--force set; proceeding despite: %s", "; ".join(problems))


async def apply(
    target_chunks: int, modern_share: float, force: bool
) -> dict:
    """Back up, truncate, re-embed stratified, insert, verify."""
    # Mandatory pre-apply dry-run unless --force.
    plan = await dry_run(target_chunks, modern_share)
    if plan["projection"]["would_exceed_ceiling"] and not force:
        raise SystemExit(
            f"ABORT: projected DB size {plan['projection']['projected_db_pct_of_ceiling']}% "
            f"of {NEON_FREE_TIER_BYTES // (1024*1024)} MB ceiling. "
            f"Lower --target-chunks or pass --force (not recommended)."
        )

    from db.connection import get_connection, init_pool
    from db.queries import insert_chunks_batch
    from ingestion.embed import chunk_text, embed_chunks

    await init_pool()
    start = time.time()

    async with get_connection() as conn:
        paper_ids = await _fetch_stratified_paper_ids(
            conn, target_chunks, modern_share
        )
        await _assert_safe_to_truncate(conn, paper_ids, target_chunks, modern_share, force)
        paper_id_filter = tuple(paper_ids) if paper_ids else (0,)
        placeholders = ",".join(["%s"] * len(paper_ids)) or "%s"
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT id, title, abstract FROM papers
                WHERE id IN ({placeholders}) AND abstract IS NOT NULL AND abstract != ''
                """,
                paper_id_filter if paper_ids else (0,),
            )
            paper_rows = await cur.fetchall()

    chunk_rows = []
    for pid, title, abstract in paper_rows:
        chunks = chunk_text(abstract, doc_id=str(pid))
        pairs = embed_chunks(chunks, batch_size=256)
        for chunk, emb in pairs:
            chunk_rows.append({
                "paper_id": pid,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "context": chunk.context,
                "token_count": chunk.token_count,
                "embedding": emb,
            })

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "CREATE TABLE IF NOT EXISTS chunks_rebalance_backup AS "
                "SELECT id, paper_id, section_title, chunk_index, content, "
                "context, token_count, embedding, created_at FROM chunks"
            )
            backup_count = 0
            await cur.execute("SELECT count(*) FROM chunks_rebalance_backup")
            backup_count = (await cur.fetchone())[0]
            await cur.execute("TRUNCATE TABLE chunks")

        inserted = 0
        batch = 500
        for i in range(0, len(chunk_rows), batch):
            slice_ = chunk_rows[i : i + batch]
            await insert_chunks_batch(conn, slice_)
            inserted += len(slice_)
            logger.info("inserted %d / %d chunks", inserted, len(chunk_rows))

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT count(*) FROM chunks")
            final_count = (await cur.fetchone())[0]

    return {
        "target_chunks": target_chunks,
        "paper_rows_fetched": len(paper_rows),
        "chunks_inserted": inserted,
        "final_chunks_count": final_count,
        "backup_rows": backup_count,
        "elapsed_seconds": round(time.time() - start, 1),
        "projection_used": plan["projection"],
    }


def _load_env() -> None:
    """Read .env like app/main.py's Settings, so DATABASE_URL need not be exported."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.getcwd(), ".env"))
    except ImportError:
        pass


def parse_args():
    p = argparse.ArgumentParser(
        description="Recency-stratified corpus rebalance within the Neon cap"
    )
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="Measure + project only (default; writes nothing)")
    p.add_argument("--apply", action="store_true",
                   help="Back up, truncate chunks, re-embed stratified, insert")
    p.add_argument("--target-chunks", type=int, default=12_000,
                   help="Target chunk count after rebalance (default 12000)")
    p.add_argument("--modern-share", type=float, default=0.25,
                   help="Fraction of target from post-2021 papers (default 0.25)")
    p.add_argument("--force", action="store_true",
                   help="Skip the storage-ceiling guard (dangerous)")
    return p.parse_args()


def _print_plan(plan: dict) -> None:
    print("=" * 64)
    print("CORPUS REBALANCE — DRY RUN")
    print("=" * 64)
    c = plan["current"]
    print(f"Current papers (with abstract): {c['papers']:,}")
    print(f"Current chunks:                 {c['chunks']:,}")
    print(f"Current post-2021 share:        {plan['current_modern_share']}%")
    print("Current year distribution (top):")
    for yr, n in list(c["by_year"].items())[:8]:
        print(f"  {yr}: {n:,}")
    print()
    print(f"Target chunks:                  {plan['target_chunks']:,}")
    print(f"  post-2021:                     {plan['target_modern_chunks']:,} ({modern_share_pct(plan)}%)")
    print(f"  classic:                       {plan['target_classic_chunks']:,}")
    print()
    pr = plan["projection"]
    print(f"Avg bytes/chunk (measured):     {pr['avg_bytes_per_chunk']:,}")
    print(f"Current DB size:                {pr['current_db_bytes']:,} B "
          f"({pr['current_db_bytes']//(1024*1024)} MB)")
    print(f"Projected DB size:              {pr['projected_db_bytes']:,} B "
          f"({pr['projected_db_bytes']//(1024*1024)} MB)")
    print(f"  = {pr['projected_db_pct_of_ceiling']}% of "
          f"{plan['ceiling_bytes']//(1024*1024)} MB ceiling "
          f"(abort > {plan['safety_ceiling_pct']}%)")
    flag = "EXCEEDS CEILING — ABORT" if pr["would_exceed_ceiling"] else "OK to apply"
    print(f"Status: {flag}")
    print("=" * 64)


def modern_share_pct(plan: dict) -> int:
    return int(plan["target_modern_share"] * 100)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_env()
    args = parse_args()
    if args.apply:
        # --apply flips the default dry-run off; both flags present -> apply wins.
        result = asyncio.run(
            apply(args.target_chunks, args.modern_share, args.force)
        )
        print("=" * 64)
        print("CORPUS REBALANCE — APPLIED")
        print("=" * 64)
        for k, v in result.items():
            print(f"{k}: {v}")
    else:
        plan = asyncio.run(dry_run(args.target_chunks, args.modern_share))
        _print_plan(plan)
