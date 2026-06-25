from __future__ import annotations
"""
ArXiv metadata loader.

Reads the Kaggle arXiv metadata JSONL snapshot
(arxiv-metadata-oai-snapshot.json) and bulk-inserts into the papers table.

Download from: https://www.kaggle.com/datasets/Cornell-University/arxiv
(~4GB, 2.4M papers as of 2026)

Usage:
    python -m ingestion.loader \\
        --file /path/to/arxiv-metadata-oai-snapshot.json \\
        --limit 50000 \\
        --categories cs.LG,cs.AI,stat.ML \\
        --batch-size 500
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load ArXiv metadata into Postgres")
    parser.add_argument("--file", required=True, help="Path to arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--limit", type=int, default=50_000, help="Max papers to load (default 50000)")
    parser.add_argument("--categories", help="Comma-separated category filter, e.g. cs.LG,cs.AI")
    parser.add_argument("--batch-size", type=int, default=500, help="Insert batch size (default 500)")
    parser.add_argument("--db-url", help="Postgres URL (defaults to DATABASE_URL env var)")
    return parser.parse_args()


def parse_record(line: str) -> Optional[dict]:
    """Parse a single JSONL line into a paper dict. Returns None on error."""
    try:
        rec = json.loads(line.strip())
        # Parse authors
        if rec.get("authors_parsed"):
            authors = [" ".join(filter(None, parts)).strip() for parts in rec["authors_parsed"]]
        else:
            authors = [a.strip() for a in rec.get("authors", "").split(",") if a.strip()]
        # Parse categories
        categories = rec.get("categories", "").split()
        # Parse date
        published_at = None
        if rec.get("update_date"):
            try:
                published_at = datetime.strptime(rec["update_date"], "%Y-%m-%d")
            except ValueError:
                pass
        return {
            "arxiv_id": rec["id"],
            "title": rec.get("title", "").replace("\n", " ").strip(),
            "authors": authors,
            "categories": categories,
            "abstract": rec.get("abstract", "").strip(),
            "published_at": published_at,
            "updated_at": published_at,
        }
    except Exception as e:
        logger.debug("Skipping malformed record: %s", e)
        return None


async def main() -> None:
    args = parse_args()

    # Setup DB
    db_url = args.db_url or os.getenv("DATABASE_URL", "")
    if not db_url:
        print("ERROR: DATABASE_URL not set and --db-url not provided", file=sys.stderr)
        sys.exit(1)

    from db.connection import init_pool, get_connection
    from db.queries import insert_paper

    await init_pool(database_url=db_url)

    filter_categories = set(args.categories.split(",")) if args.categories else None
    total = 0
    batch: list[dict] = []
    file_path = Path(args.file)

    print(f"Loading from {file_path}...")
    if filter_categories:
        print(f"Filtering to categories: {filter_categories}")
    print(f"Limit: {args.limit} papers, batch size: {args.batch_size}")

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if total >= args.limit:
                break
            if not line.strip():
                continue

            paper = parse_record(line)
            if paper is None:
                continue

            # Category filter
            if filter_categories and not filter_categories.intersection(set(paper["categories"])):
                continue

            batch.append(paper)

            if len(batch) >= args.batch_size:
                async with get_connection() as conn:
                    for p in batch:
                        try:
                            await insert_paper(conn, p)
                            total += 1
                        except Exception as e:
                            logger.warning("Failed to insert paper %s: %s", p.get("arxiv_id"), e)
                    await conn.commit()
                batch = []
                if total % 1000 == 0:
                    print(f"Loaded {total} papers...")

    # Insert remaining batch
    if batch:
        async with get_connection() as conn:
            for p in batch:
                try:
                    await insert_paper(conn, p)
                    total += 1
                except Exception as e:
                    logger.warning("Failed to insert paper %s: %s", p.get("arxiv_id"), e)
            await conn.commit()

    print(f"Done. Loaded {total} papers from {file_path.name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
