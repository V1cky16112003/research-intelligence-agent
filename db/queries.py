from __future__ import annotations
"""
Analytical SQL queries and CRUD operations.
All functions accept a psycopg v3 AsyncConnection.
"""
import logging
from typing import Any

import psycopg

logger = logging.getLogger(__name__)


def _rows_to_dicts(cursor: psycopg.AsyncCursor) -> list[dict[str, Any]]:
    """Convert cursor rows to list of dicts using column names from description."""
    if cursor.description is None:
        return []
    col_names = [d[0] for d in cursor.description]
    return [dict(zip(col_names, row)) for row in cursor.fetchall()]


async def insert_paper(conn: psycopg.AsyncConnection, paper: dict[str, Any]) -> int:
    """Upsert a paper record. Returns the paper id."""
    sql = """
        INSERT INTO papers (arxiv_id, title, authors, categories, abstract, published_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (arxiv_id) DO UPDATE SET
            title      = EXCLUDED.title,
            authors    = EXCLUDED.authors,
            categories = EXCLUDED.categories,
            abstract   = EXCLUDED.abstract,
            updated_at = EXCLUDED.updated_at
        RETURNING id
    """
    async with conn.cursor() as cur:
        await cur.execute(
            sql,
            (
                paper["arxiv_id"],
                paper["title"],
                paper["authors"],
                paper["categories"],
                paper.get("abstract"),
                paper.get("published_at"),
                paper.get("updated_at"),
            ),
        )
        row = await cur.fetchone()
        return row[0]


async def _register_vector(conn: psycopg.AsyncConnection) -> None:
    from pgvector.psycopg import register_vector_async
    await register_vector_async(conn)


async def insert_chunks_batch(conn: psycopg.AsyncConnection, chunks: list[dict[str, Any]]) -> None:
    """Bulk insert chunks with embeddings using executemany."""
    await _register_vector(conn)

    sql = """
        INSERT INTO chunks (paper_id, section_title, chunk_index, content, token_count, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    params = [
        (
            c["paper_id"],
            c.get("section_title", "abstract"),
            c.get("chunk_index", 0),
            c["content"],
            c.get("token_count"),
            c.get("embedding"),
        )
        for c in chunks
    ]
    async with conn.cursor() as cur:
        await cur.executemany(sql, params)
    logger.debug("Inserted %d chunks", len(chunks))


async def search_similar_chunks(
    conn: psycopg.AsyncConnection,
    query_embedding: list[float],
    k: int = 10,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Cosine similarity search with optional category filter."""
    await _register_vector(conn)

    if categories:
        sql = """
            SELECT
                c.id,
                c.content,
                c.paper_id,
                c.section_title,
                c.chunk_index,
                1 - (c.embedding <=> %s::vector) AS similarity_score,
                p.arxiv_id,
                p.title,
                p.authors,
                p.categories
            FROM chunks c
            JOIN papers p ON c.paper_id = p.id
            WHERE p.categories && %s::text[]
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        params = (query_embedding, categories, query_embedding, k)
    else:
        sql = """
            SELECT
                c.id,
                c.content,
                c.paper_id,
                c.section_title,
                c.chunk_index,
                1 - (c.embedding <=> %s::vector) AS similarity_score,
                p.arxiv_id,
                p.title,
                p.authors,
                p.categories
            FROM chunks c
            JOIN papers p ON c.paper_id = p.id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        params = (query_embedding, query_embedding, k)

    async with conn.cursor() as cur:
        await cur.execute(sql, params)
        col_names = [d[0] for d in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(col_names, row)) for row in rows]


async def log_query(conn: psycopg.AsyncConnection, **kwargs: Any) -> None:
    """Insert a query audit log row."""
    sql = """
        INSERT INTO query_audit_log (
            session_id, user_query, route, tools_called,
            latency_ms, tokens_in, tokens_out, llm_provider,
            retrieved_chunk_ids, faithfulness_score, answer_relevancy
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    async with conn.cursor() as cur:
        await cur.execute(
            sql,
            (
                kwargs.get("session_id"),
                kwargs.get("user_query", ""),
                kwargs.get("route"),
                kwargs.get("tools_called") or [],
                kwargs.get("latency_ms"),
                kwargs.get("tokens_in"),
                kwargs.get("tokens_out"),
                kwargs.get("llm_provider"),
                kwargs.get("retrieved_chunk_ids") or [],
                kwargs.get("faithfulness_score"),
                kwargs.get("answer_relevancy"),
            ),
        )
    logger.debug("Logged query for session %s", kwargs.get("session_id"))


async def papers_per_category_per_month(conn: psycopg.AsyncConnection) -> list[dict[str, Any]]:
    """Window function: paper count per (category, month) with recency rank."""
    sql = """
        SELECT
            cat,
            DATE_TRUNC('month', published_at) AS month,
            COUNT(*) AS paper_count,
            ROW_NUMBER() OVER (PARTITION BY cat ORDER BY DATE_TRUNC('month', published_at) DESC) AS recency_rank
        FROM papers, UNNEST(categories) AS cat
        WHERE published_at IS NOT NULL
        GROUP BY cat, month
        ORDER BY cat, month DESC
        LIMIT 500
    """
    async with conn.cursor() as cur:
        await cur.execute(sql)
        col_names = [d[0] for d in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(col_names, row)) for row in rows]


async def rolling_query_volume(conn: psycopg.AsyncConnection, days: int = 7) -> list[dict[str, Any]]:
    """Rolling N-day query volume from audit log."""
    sql = """
        WITH date_series AS (
            SELECT generate_series(
                CURRENT_DATE - (%s - 1) * INTERVAL '1 day',
                CURRENT_DATE,
                INTERVAL '1 day'
            )::date AS day
        )
        SELECT
            d.day,
            COUNT(q.id) AS query_count
        FROM date_series d
        LEFT JOIN query_audit_log q ON DATE_TRUNC('day', q.ts)::date = d.day
        GROUP BY d.day
        ORDER BY d.day
    """
    async with conn.cursor() as cur:
        await cur.execute(sql, (days,))
        col_names = [d[0] for d in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(col_names, row)) for row in rows]


async def provider_p95_latency(conn: psycopg.AsyncConnection) -> list[dict[str, Any]]:
    """P95 latency per LLM provider from audit log."""
    sql = """
        SELECT
            llm_provider,
            COUNT(*) AS query_count,
            ROUND(AVG(latency_ms)::numeric, 1) AS avg_latency_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
            MIN(latency_ms) AS min_latency_ms,
            MAX(latency_ms) AS max_latency_ms
        FROM query_audit_log
        WHERE latency_ms IS NOT NULL
        GROUP BY llm_provider
    """
    async with conn.cursor() as cur:
        await cur.execute(sql)
        col_names = [d[0] for d in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(col_names, row)) for row in rows]


async def get_experiments_summary(conn: psycopg.AsyncConnection) -> list[dict[str, Any]]:
    """SELECT * FROM experiments view, last 30 days."""
    sql = """
        SELECT *
        FROM experiments
        WHERE day >= NOW() - INTERVAL '30 days'
        ORDER BY day DESC, llm_provider, route
    """
    async with conn.cursor() as cur:
        await cur.execute(sql)
        col_names = [d[0] for d in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(col_names, row)) for row in rows]
