from __future__ import annotations

"""
Backfill `papers.published_at` with true v1 submission dates.

The original loader stored `update_date` (the day the OAI *metadata record* was
last touched) in `published_at`. That is not a publication date: for 27.7% of the
loaded corpus it lands in a different year than the paper actually appeared, and
4,286 papers ended up stamped 2019-2026 — years in which this corpus (arXiv IDs
0704-1805) published nothing. Every temporal analytic read those phantom months.

This re-derives the real date from `versions[0].created` in the snapshot and
rewrites `published_at`, moving `update_date` into `updated_at` where it belongs.

Dry run (default) — prints the before/after year histogram and changes nothing:

    DATABASE_URL="..." PYTHONPATH=. python3 -m ingestion.backfill_dates \\
        --file dataset/arxiv-metadata-oai-snapshot.json

Apply:

    ... --file dataset/arxiv-metadata-oai-snapshot.json --apply

Idempotent and re-runnable. Recoverable without the snapshot: the pre-backfill
value of `published_at` is preserved in `updated_at`, which the old loader set
to the identical `update_date` value.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

from ingestion.loader import parse_published_at, parse_updated_at

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill true publication dates")
    parser.add_argument("--file", required=True, help="Path to arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    parser.add_argument("--batch-size", type=int, default=5000, help="Update batch size")
    parser.add_argument("--db-url", help="Postgres URL (defaults to DATABASE_URL)")
    return parser.parse_args()



async def _vacuum_papers() -> None:
    """Reclaim the batch's dead tuples so the next batch reuses the pages.

    VACUUM cannot run inside a transaction block, so it needs its own autocommit
    connection rather than the one running the batched UPDATE.
    """
    from db.connection import get_connection
    try:
        async with get_connection() as conn:
            await conn.set_autocommit(True)
            async with conn.cursor() as cur:
                await cur.execute("VACUUM papers")
            await conn.set_autocommit(False)
    except Exception as e:  # non-fatal: worst case the next batch extends the file
        logging.getLogger(__name__).warning("VACUUM failed: %s", e)


async def main() -> None:
    args = parse_args()
    # Read .env the way app/main.py's Settings does, so the operator does not have
    # to put the connection string (password and all) on the command line.
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.getcwd(), ".env"))
    except ImportError:
        pass
    db_url = args.db_url or os.getenv("DATABASE_URL", "")
    if not db_url:
        print("ERROR: DATABASE_URL not set and --db-url not provided", file=sys.stderr)
        sys.exit(1)

    from db.connection import close_pool, get_connection, init_pool

    await init_pool(database_url=db_url)

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT arxiv_id, published_at FROM papers")
            current = {row[0]: row[1] for row in await cur.fetchall()}
    print(f"{len(current)} papers in the database")

    # Single streaming pass; stop as soon as every stored paper has been matched.
    fixes: list[tuple[str, object, object]] = []
    before, after = Counter(), Counter()
    unmatched = set(current)
    scanned = 0

    with open(Path(args.file), encoding="utf-8") as f:
        for line in f:
            if not unmatched:
                break
            scanned += 1
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            arxiv_id = rec.get("id")
            if arxiv_id not in unmatched:
                continue
            unmatched.discard(arxiv_id)

            published_at = parse_published_at(rec)
            if published_at is None:
                continue
            old = current[arxiv_id]
            before[old.year if old else None] += 1
            after[published_at.year] += 1
            if old is None or old.date() != published_at.date():
                fixes.append((arxiv_id, published_at, parse_updated_at(rec)))

    print(f"scanned {scanned:,} snapshot lines; {len(unmatched)} stored papers not found in snapshot")
    print(f"{len(fixes):,} of {len(current):,} rows have the wrong published_at "
          f"({100 * len(fixes) / max(len(current), 1):.1f}%)")

    years = sorted({y for y in (set(before) | set(after)) if y is not None})
    print(f"\n{'year':>6} {'STORED (update_date)':>22} {'TRUE (v1 submission)':>22}")
    for y in years:
        print(f"{y:>6} {before.get(y, 0):>22} {after.get(y, 0):>22}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
        await close_pool()
        return

    # Neon free tier is a hard 512 MB *project* ceiling, and Postgres MVCC writes a
    # new row version for every updated row before the old one can be reclaimed. A
    # single 50k-row UPDATE therefore needs ~65 MB of headroom that does not exist —
    # it dies with DiskFull partway through and rolls back everything.
    #
    # So: small batches, commit each one, and VACUUM between them. The commit ends the
    # transaction so the previous batch's dead tuples become reclaimable, and the
    # VACUUM marks those pages reusable, so the next batch writes into them instead of
    # extending the file. Peak extra space is one batch, not the whole table.
    #
    # Re-running is safe and resumes automatically: `fixes` is computed by comparing
    # stored dates against the snapshot, so rows already corrected are simply absent.
    updated = 0
    async with get_connection() as conn:
        for start in range(0, len(fixes), args.batch_size):
            batch = fixes[start:start + args.batch_size]
            ids = [f[0] for f in batch]
            pubs = [f[1] for f in batch]
            upds = [f[2] for f in batch]
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE papers p SET published_at = d.pa, updated_at = d.ua "
                    "FROM (SELECT * FROM unnest(%s::text[], %s::timestamptz[], "
                    "%s::timestamptz[]) AS t(arxiv_id, pa, ua)) d "
                    "WHERE p.arxiv_id = d.arxiv_id",
                    (ids, pubs, upds),
                )
                updated += cur.rowcount
            await conn.commit()
            print(f"  updated {updated:,}/{len(fixes):,}", flush=True)
            await _vacuum_papers()

    print(f"  {updated:,} rows updated")
    print(f"\nDone. Rewrote published_at for {len(fixes):,} papers.")
    await close_pool()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
