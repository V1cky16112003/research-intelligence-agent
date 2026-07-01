from __future__ import annotations
"""
Sync papers (authors, categories) from Postgres into Neo4j AuraDB.

Builds:
  (:Paper {arxiv_id, title})-[:AUTHORED_BY]->(:Author {name})
  (:Paper {arxiv_id, title})-[:HAS_CATEGORY]->(:Category {name})

Idempotent: uses MERGE so re-running after new papers are ingested does not
duplicate nodes or relationships. Safe to run repeatedly (e.g. after each
ingestion batch, or on a schedule).

Usage:
    DATABASE_URL=... NEO4J_URI=... NEO4J_USER=... NEO4J_PASSWORD=... \
        python -m graph.graph_sync --limit 10000
"""
import argparse
import asyncio
import logging

logger = logging.getLogger(__name__)

_SYNC_CYPHER = """
MERGE (p:Paper {arxiv_id: $arxiv_id})
SET p.title = $title
WITH p
UNWIND $authors AS author_name
MERGE (a:Author {name: author_name})
MERGE (p)-[:AUTHORED_BY]->(a)
WITH p
UNWIND $categories AS category_name
MERGE (c:Category {name: category_name})
MERGE (p)-[:HAS_CATEGORY]->(c)
"""


async def sync_papers_to_graph(driver, papers: list[dict]) -> int:
    """
    Sync a list of paper dicts into Neo4j.

    Args:
        driver: A Neo4j AsyncDriver (from graph.neo4j_client.get_driver()).
        papers: List of dicts with keys: arxiv_id, title, authors, categories.

    Returns:
        Number of papers synced.
    """
    count = 0
    async with driver.session() as session:
        for paper in papers:
            await session.run(
                _SYNC_CYPHER,
                {
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors") or [],
                    "categories": paper.get("categories") or [],
                },
            )
            count += 1
    return count


async def run_sync(limit: int = 50_000) -> dict:
    """Fetch papers from Postgres and sync them into Neo4j."""
    from db.connection import init_pool, get_connection
    from graph.neo4j_client import get_driver

    await init_pool()
    driver = get_driver()

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT arxiv_id, title, authors, categories FROM papers LIMIT %s",
                (limit,),
            )
            col_names = [d[0] for d in cur.description]
            rows = await cur.fetchall()
            papers = [dict(zip(col_names, row)) for row in rows]

    count = await sync_papers_to_graph(driver, papers)
    logger.info("Synced %d papers to Neo4j", count)
    return {"synced": count}


def parse_args():
    p = argparse.ArgumentParser(description="Sync papers table into Neo4j graph")
    p.add_argument("--limit", type=int, default=50_000)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    asyncio.run(run_sync(limit=args.limit))
