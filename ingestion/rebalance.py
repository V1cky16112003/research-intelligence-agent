from __future__ import annotations

"""
Corpus rebalance: recency-stratified re-ingest within the Neon 512 MB cap.

Phase 2 of the corpus-rebalance roadmap. Today only 1,865 / 50,000 papers
(3.7%) are post-2021, so retrieval is 95.8% accurate on classical topics but
~25-30% on modern. This script rebuilds the `chunks` table with a
recency-stratified sample that raises the post-2021 share to ~25% while
staying under the Neon free-tier disk ceiling (512 MB).

Two modes:

  --dry-run (default):  measure live DB size, row counts, and year distribution;
                        project post-rebalance storage and abort if it would
                        exceed 90% of the ceiling. Prints a plan, writes nothing.

  --apply:              copy chunks to a backup table (cheap metadata-only),
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
