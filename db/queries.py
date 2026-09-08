from __future__ import annotations

"""
Analytical SQL queries and CRUD operations.
All functions accept a psycopg v3 AsyncConnection.
"""
import logging
import re
from typing import Any

import psycopg

logger = logging.getLogger(__name__)


async def _rows_to_dicts(cursor: psycopg.AsyncCursor) -> list[dict[str, Any]]:
    """Convert cursor rows to list of dicts using column names from description.

    `await`s fetchall(): on an AsyncCursor it returns a coroutine, so the previous
    synchronous version raised "'coroutine' object is not iterable" for every caller.
    It had no callers, so nothing surfaced it until the analytics rebuild used it.
    """
    if cursor.description is None:
        return []
    col_names = [d[0] for d in cursor.description]
    rows = await cursor.fetchall()
    return [dict(zip(col_names, row)) for row in rows]


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
        INSERT INTO chunks (paper_id, section_title, chunk_index, content, context, token_count, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    params = [
        (
            c["paper_id"],
            c.get("section_title", "abstract"),
            c.get("chunk_index", 0),
            c["content"],
            c.get("context") or None,  # store NULL rather than empty string
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


def _to_tsquery_safe(query: str) -> str:
    """Convert a free-text query to a safe plainto_tsquery-style string.

    Strips punctuation so the string is safe to pass to plainto_tsquery().
    We use plainto_tsquery (not to_tsquery) in the SQL, so the string
    doesn't need special operators — just cleaned words.
    """
    words = re.sub(r"[^\w\s]", "", query).split()
    return " ".join(words)


async def search_similar_chunks_hybrid(
    conn: psycopg.AsyncConnection,
    query_embedding: list[float],
    query_text: str,
    k: int = 8,
    categories: list[str] | None = None,
    rrf_k: int = 60,
    candidate_multiplier: int = 2,
) -> list[dict[str, Any]]:
    """Hybrid retrieval: dense vector (HNSW) + BM25 (tsvector) fused with RRF.

    Fetches k * candidate_multiplier candidates from each leg, then merges
    via Reciprocal Rank Fusion and returns the top k results.

    Falls back to pure vector search if content_tsv column is absent
    (i.e. migration 001 hasn't been applied yet).
    """
    await _register_vector(conn)

    candidates = k * candidate_multiplier
    bm25_text = _to_tsquery_safe(query_text)

    # Check whether the content_tsv column exists (graceful degradation)
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='chunks' AND column_name='content_tsv' LIMIT 1"
        )
        has_tsv = (await cur.fetchone()) is not None

    if not has_tsv or not bm25_text.strip():
        logger.debug("Falling back to pure vector search (no tsvector column or empty query)")
        return await search_similar_chunks(conn, query_embedding, k=k, categories=categories)

    # Build params list dynamically to handle optional category filter
    # Legs: vector (2 uses of embedding + optional cat), bm25 (optional cat), final LIMIT
    if categories:
        sql = """
            WITH vector_ranked AS (
                SELECT c.id,
                       ROW_NUMBER() OVER (ORDER BY c.embedding <=> %s::vector) AS rank
                FROM chunks c
                JOIN papers p ON c.paper_id = p.id
                WHERE p.categories && %s::text[]
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            ),
            bm25_ranked AS (
                SELECT c.id,
                       ROW_NUMBER() OVER (ORDER BY ts_rank(c.content_tsv, q) DESC) AS rank
                FROM chunks c
                JOIN papers p ON c.paper_id = p.id,
                     plainto_tsquery('english', %s) q
                WHERE c.content_tsv @@ q
                  AND p.categories && %s::text[]
                ORDER BY ts_rank(c.content_tsv, q) DESC
                LIMIT %s
            ),
            rrf AS (
                SELECT id, SUM(1.0 / (%s + rank)) AS score
                FROM (
                    SELECT id, rank FROM vector_ranked
                    UNION ALL
                    SELECT id, rank FROM bm25_ranked
                ) combined
                GROUP BY id
                ORDER BY score DESC
                LIMIT %s
            )
            SELECT c.id, c.content, c.context, c.paper_id, c.section_title, c.chunk_index,
                   rrf.score AS similarity_score,
                   p.arxiv_id, p.title, p.authors, p.categories
            FROM rrf
            JOIN chunks c ON rrf.id = c.id
            JOIN papers p ON c.paper_id = p.id
            ORDER BY rrf.score DESC
        """
        params = (
            query_embedding, categories, query_embedding, candidates,  # vector leg
            bm25_text, categories, candidates,                          # bm25 leg
            rrf_k, k,                                                   # rrf + final limit
        )
    else:
        sql = """
            WITH vector_ranked AS (
                SELECT c.id,
                       ROW_NUMBER() OVER (ORDER BY c.embedding <=> %s::vector) AS rank
                FROM chunks c
                JOIN papers p ON c.paper_id = p.id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            ),
            bm25_ranked AS (
                SELECT c.id,
                       ROW_NUMBER() OVER (ORDER BY ts_rank(c.content_tsv, q) DESC) AS rank
                FROM chunks c
                JOIN papers p ON c.paper_id = p.id,
                     plainto_tsquery('english', %s) q
                WHERE c.content_tsv @@ q
                ORDER BY ts_rank(c.content_tsv, q) DESC
                LIMIT %s
            ),
            rrf AS (
                SELECT id, SUM(1.0 / (%s + rank)) AS score
                FROM (
                    SELECT id, rank FROM vector_ranked
                    UNION ALL
                    SELECT id, rank FROM bm25_ranked
                ) combined
                GROUP BY id
                ORDER BY score DESC
                LIMIT %s
            )
            SELECT c.id, c.content, c.context, c.paper_id, c.section_title, c.chunk_index,
                   rrf.score AS similarity_score,
                   p.arxiv_id, p.title, p.authors, p.categories
            FROM rrf
            JOIN chunks c ON rrf.id = c.id
            JOIN papers p ON c.paper_id = p.id
            ORDER BY rrf.score DESC
        """
        params = (
            query_embedding, query_embedding, candidates,  # vector leg
            bm25_text, candidates,                          # bm25 leg
            rrf_k, k,                                       # rrf + final limit
        )

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


async def log_llm_calls(conn: psycopg.AsyncConnection, session_id: str, calls: list[dict[str, Any]]) -> None:
    """Bulk insert per-LLM-call cost/latency records for one /chat request."""
    if not calls:
        return
    sql = """
        INSERT INTO llm_call_log (
            session_id, node, provider, model, tokens_in, tokens_out,
            cost_usd, latency_ms, is_retry, cached
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = [
        (
            session_id,
            c.get("node", "unknown"),
            c.get("provider"),
            c.get("model"),
            c.get("tokens_in", 0),
            c.get("tokens_out", 0),
            c.get("cost_usd", 0.0),
            c.get("latency_ms", 0),
            c.get("is_retry", False),
            c.get("cached", False),
        )
        for c in calls
    ]
    async with conn.cursor() as cur:
        await cur.executemany(sql, params)
    logger.debug("Logged %d llm_call_log rows for session %s", len(calls), session_id)


# Caller-supplied day windows reach here from a public query string
# (/analytics/cost?days=). Unclamped, a large value makes Postgres raise
# "interval out of range" and the endpoint 500s.
MAX_WINDOW_DAYS = 3650


def _clamp_days(days: int, default: int = 30) -> int:
    try:
        return max(1, min(int(days), MAX_WINDOW_DAYS))
    except (TypeError, ValueError):
        return default


async def llm_cost_by_node(conn: psycopg.AsyncConnection, days: int = 30) -> list[dict[str, Any]]:
    """SELECT * FROM llm_cost_latency view, last N days."""
    days = _clamp_days(days)
    sql = """
        SELECT *
        FROM llm_cost_latency
        WHERE day >= NOW() - (%s || ' days')::interval
        ORDER BY day DESC, node, provider
    """
    async with conn.cursor() as cur:
        await cur.execute(sql, (days,))
        col_names = [d[0] for d in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(col_names, row)) for row in rows]


async def retry_overhead_summary(conn: psycopg.AsyncConnection, days: int = 30) -> list[dict[str, Any]]:
    """SELECT * FROM llm_retry_overhead view, last N days."""
    days = _clamp_days(days)
    sql = """
        SELECT *
        FROM llm_retry_overhead
        WHERE day >= NOW() - (%s || ' days')::interval
        ORDER BY day DESC, is_retry
    """
    async with conn.cursor() as cur:
        await cur.execute(sql, (days,))
        col_names = [d[0] for d in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(col_names, row)) for row in rows]


# ---------------------------------------------------------------------------
# Corpus analytics
#
# Design rule, learned the hard way: **every function here returns a complete,
# pre-aggregated answer plus an explicit summary — never a truncated sample the
# LLM is expected to add up.**
#
# The predecessor of these functions returned the 500 highest-recency
# (category, month) rows out of 8001 and let the reporter sum them. Two failures
# compounded. The 500-row window covered 4 of the corpus's 134 months, so it
# omitted 90% of the papers; and LLMs cannot reliably sum 500 numbers anyway.
# Asked "how many papers are in the corpus?", the agent answered 3,098 against a
# true 50,000, burned all three critic retries arguing with itself about the sum,
# and shipped the wrong number with full confidence. Neither the audit log nor
# the test suite could see anything wrong: the tool "ran" and "returned rows".
#
# So: aggregate in Postgres, which can count; bound every result set to something
# an LLM can actually read; and when a bound does bite, say so numerically in the
# summary rather than silently dropping the tail.
# ---------------------------------------------------------------------------

# Hard ceiling on rows returned to the agent, so a mis-parameterised call degrades
# into a visibly-capped answer instead of a 15k-token context dump.
MAX_ANALYTICS_ROWS = 200


async def _fetch(conn: psycopg.AsyncConnection, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    async with conn.cursor() as cur:
        await cur.execute(sql, params)
        return await _rows_to_dicts(cur)


async def corpus_stats(conn: psycopg.AsyncConnection) -> dict[str, Any]:
    """One row describing the whole corpus — the answer to "how many papers?".

    This is the query the agent had no way to ask. Counting is Postgres's job.
    """
    sql = """
        SELECT
            (SELECT COUNT(*) FROM papers)                                        AS total_papers,
            (SELECT COUNT(*) FROM chunks)                                        AS total_chunks,
            (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL)            AS embedded_chunks,
            (SELECT COUNT(DISTINCT paper_id) FROM chunks)                        AS papers_with_chunks,
            (SELECT COUNT(DISTINCT c) FROM papers, UNNEST(categories) c)         AS distinct_categories,
            (SELECT COUNT(DISTINCT a) FROM papers, UNNEST(authors) a)            AS distinct_authors,
            (SELECT MIN(published_at)::date FROM papers)                         AS earliest_publication,
            (SELECT MAX(published_at)::date FROM papers)                         AS latest_publication,
            (SELECT COUNT(*) FROM papers WHERE published_at IS NULL)             AS papers_missing_date
    """
    rows = await _fetch(conn, sql)
    return rows[0] if rows else {}


async def papers_by_category(
    conn: psycopg.AsyncConnection,
    year: int | None = None,
    limit: int = 25,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Paper counts per ArXiv category, largest first.

    A paper cross-listed in three categories counts once in each, so the per-category
    counts sum to more than the paper total. The summary reports both, because an LLM
    handed only the rows will otherwise add them up and over-report the corpus size.
    """
    limit = max(1, min(int(limit), MAX_ANALYTICS_ROWS))
    where = "WHERE published_at IS NOT NULL"
    params: list[Any] = []
    if year:
        where += " AND EXTRACT(YEAR FROM published_at) = %s"
        params.append(year)

    totals = await _fetch(conn, f"""
        SELECT COUNT(DISTINCT c) AS distinct_categories,
               COUNT(*)          AS total_category_assignments
        FROM papers, UNNEST(categories) c
        {where}
    """, tuple(params))
    paper_total = await _fetch(conn, f"""
        SELECT COUNT(*) AS total_papers FROM papers {where}
    """, tuple(params))

    rows = await _fetch(conn, f"""
        SELECT c AS category,
               COUNT(*)                  AS paper_count,
               MIN(published_at)::date   AS first_paper,
               MAX(published_at)::date   AS latest_paper
        FROM papers, UNNEST(categories) c
        {where}
        GROUP BY c
        ORDER BY paper_count DESC, category
        LIMIT %s
    """, (*params, limit))

    summary = dict(totals[0]) if totals else {}
    summary.update(paper_total[0] if paper_total else {})
    summary.update({
        "filter_year": year,
        "categories_shown": len(rows),
        "note": ("Papers are cross-listed, so per-category counts sum to more than "
                 "total_papers. Use total_papers for 'how many papers'."),
    })
    if summary.get("distinct_categories", 0) > len(rows):
        summary["truncated"] = (
            f"showing the {len(rows)} largest of {summary['distinct_categories']} categories"
        )
    return rows, summary


async def papers_by_year(
    conn: psycopg.AsyncConnection,
    category: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Publication counts per year — complete, never truncated (~20 rows)."""
    source = "FROM papers" if not category else "FROM papers, UNNEST(categories) c"
    where = "WHERE published_at IS NOT NULL"
    params: list[Any] = []
    if category:
        where += " AND c = %s"
        params.append(category)

    rows = await _fetch(conn, f"""
        SELECT EXTRACT(YEAR FROM published_at)::int AS year,
               COUNT(*) AS paper_count
        {source}
        {where}
        GROUP BY year
        ORDER BY year
    """, tuple(params))
    return rows, {
        "filter_category": category,
        "total_papers": sum(r["paper_count"] for r in rows),
        "years_covered": len(rows),
    }


async def papers_by_month(
    conn: psycopg.AsyncConnection,
    category: str | None = None,
    year: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Monthly publication counts, aggregated across categories unless one is named.

    Deliberately *not* per (category, month): that cross product is 8001 rows for
    this corpus and cannot be returned whole, which is what forced the old
    truncate-and-hope behaviour. One row per month is ~134 rows — complete, and
    small enough to read.
    """
    source = "FROM papers" if not category else "FROM papers, UNNEST(categories) c"
    where = "WHERE published_at IS NOT NULL"
    params: list[Any] = []
    if category:
        where += " AND c = %s"
        params.append(category)
    if year:
        where += " AND EXTRACT(YEAR FROM published_at) = %s"
        params.append(year)

    rows = await _fetch(conn, f"""
        SELECT DATE_TRUNC('month', published_at)::date AS month,
               COUNT(*) AS paper_count
        {source}
        {where}
        GROUP BY month
        ORDER BY month
        LIMIT %s
    """, (*params, MAX_ANALYTICS_ROWS + 1))

    truncated = len(rows) > MAX_ANALYTICS_ROWS
    rows = rows[:MAX_ANALYTICS_ROWS]
    summary = {
        "filter_category": category,
        "filter_year": year,
        "total_papers": sum(r["paper_count"] for r in rows),
        "months_covered": len(rows),
        "first_month": rows[0]["month"] if rows else None,
        "last_month": rows[-1]["month"] if rows else None,
    }
    if truncated:
        summary["truncated"] = (
            f"capped at {MAX_ANALYTICS_ROWS} months — narrow with a year filter"
        )
    return rows, summary


async def top_authors(
    conn: psycopg.AsyncConnection,
    category: str | None = None,
    limit: int = 25,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Most prolific authors in the corpus, optionally within one category."""
    limit = max(1, min(int(limit), MAX_ANALYTICS_ROWS))
    where = ""
    params: list[Any] = []
    if category:
        where = "WHERE p.categories && ARRAY[%s]::text[]"
        params.append(category)

    rows = await _fetch(conn, f"""
        SELECT a AS author,
               COUNT(*)                    AS paper_count,
               MIN(p.published_at)::date   AS first_paper,
               MAX(p.published_at)::date   AS latest_paper
        FROM papers p, UNNEST(p.authors) a
        {where}
        GROUP BY a
        ORDER BY paper_count DESC, author
        LIMIT %s
    """, (*params, limit))
    return rows, {"filter_category": category, "authors_shown": len(rows)}


async def rolling_query_volume(conn: psycopg.AsyncConnection, days: int = 7) -> list[dict[str, Any]]:
    """Rolling N-day query volume from audit log."""
    days = _clamp_days(days, default=7)
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


async def provider_p95_latency(
    conn: psycopg.AsyncConnection, days: int = 30
) -> list[dict[str, Any]]:
    """P95 latency per LLM provider from the audit log, last N days.

    Windowed rather than all-time: an unbounded average silently mixes in latencies
    from provider tiers and model IDs that are no longer in the cascade, so "how
    fast is the system" answers with the history of a system that no longer exists.
    """
    days = _clamp_days(days)
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
          AND ts >= NOW() - (%s || ' days')::interval
        GROUP BY llm_provider
        ORDER BY query_count DESC
    """
    return await _fetch(conn, sql, (days,))


async def get_experiments_summary(
    conn: psycopg.AsyncConnection, days: int = 30
) -> list[dict[str, Any]]:
    """SELECT * FROM experiments view, last N days."""
    days = _clamp_days(days)
    sql = """
        SELECT *
        FROM experiments
        WHERE day >= NOW() - (%s || ' days')::interval
        ORDER BY day DESC, llm_provider, route
        LIMIT %s
    """
    return await _fetch(conn, sql, (days, MAX_ANALYTICS_ROWS))


# ---------------------------------------------------------------------------
# Relational ("knowledge graph") queries, answered from Postgres.
#
# The graph_query tool was originally Neo4j-only. Neo4j AuraDB's free tier
# deletes an instance after a stretch of inactivity, and it did: the configured
# host stopped resolving (NXDOMAIN), so graph_query returned an error string for
# every call and the reporter dutifully said it had no information. One dead
# free-tier instance took out a quarter of the agent's tool surface.
#
# Everything those Cypher templates traverse — authorship and category
# membership — is already in `papers` as TEXT[] columns. So the same three
# questions are answerable here, with no second datastore in the request path.
# Neo4j remains the primary (see agent/tools.py); this is the fallback that
# keeps the feature working when it is not there.
#
# Author-name matching is token-AND, not equality. Names in the ArXiv snapshot
# are stored surname-first with the affiliation glued on -- "Bengio Yoshua",
# "Bengio Yoshua Universite de Montreal", "Hinton Geoffrey E." -- while the
# planner passes whatever the user typed, usually "Yoshua Bengio". The Cypher
# templates matched on {name: $value} exactly, so they returned zero rows for
# any natural-order name; that bug was invisible while the database was up
# because an empty result is indistinguishable from "no such author".
# ---------------------------------------------------------------------------

GRAPH_QUERY_TYPES = ("papers_by_author", "papers_by_category", "coauthors")


def _author_patterns(name: str) -> list[str]:
    """Split an author name into ILIKE patterns, one per meaningful token.

    A stored name must contain *every* token to match, which makes the lookup
    order-insensitive ("Yoshua Bengio" == "Bengio Yoshua") and tolerant of
    trailing affiliations, while still keeping "Bengio Samy" out of the results
    for "Yoshua Bengio".
    """
    tokens = [t for t in re.split(r"[^\w'-]+", name or "") if len(t) > 1]
    if not tokens:
        # Single-initial or punctuation-only input: fall back to the raw string
        # so the query stays well-formed and simply matches little.
        stripped = (name or "").strip()
        return [f"%{stripped}%"] if stripped else ["%"]
    return [f"%{t}%" for t in tokens]


async def author_variants(
    conn: psycopg.AsyncConnection, name: str, limit: int = 10
) -> list[str]:
    """The stored spellings of `name` — surfaced so an empty result is legible."""
    rows = await _fetch(conn, """
        SELECT DISTINCT a AS author
        FROM papers, UNNEST(authors) a
        WHERE a ILIKE ALL(%s)
        ORDER BY author
        LIMIT %s
    """, (_author_patterns(name), max(1, min(int(limit), 50))))
    return [r["author"] for r in rows]


async def papers_by_author(
    conn: psycopg.AsyncConnection, name: str, limit: int = 20
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Papers in the corpus written by `name`, most recent first."""
    limit = max(1, min(int(limit), MAX_ANALYTICS_ROWS))
    patterns = _author_patterns(name)

    total = await _fetch(conn, """
        SELECT COUNT(*) AS total
        FROM papers p
        WHERE EXISTS (SELECT 1 FROM UNNEST(p.authors) a WHERE a ILIKE ALL(%s))
    """, (patterns,))
    rows = await _fetch(conn, """
        SELECT p.arxiv_id, p.title, p.published_at::date AS published_at, p.categories
        FROM papers p
        WHERE EXISTS (SELECT 1 FROM UNNEST(p.authors) a WHERE a ILIKE ALL(%s))
        ORDER BY p.published_at DESC NULLS LAST, p.arxiv_id
        LIMIT %s
    """, (patterns, limit))

    matched = total[0]["total"] if total else 0
    return rows, {
        "author": name,
        "total_papers": matched,
        "papers_shown": len(rows),
        "truncated": matched > len(rows),
        "matched_name_variants": await author_variants(conn, name),
        "source": "postgres",
    }


async def coauthors(
    conn: psycopg.AsyncConnection, name: str, limit: int = 20
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Everyone who has shared a paper with `name`, most frequent collaborator first."""
    limit = max(1, min(int(limit), MAX_ANALYTICS_ROWS))
    patterns = _author_patterns(name)

    rows = await _fetch(conn, """
        WITH matched AS (
            SELECT p.id, p.authors
            FROM papers p
            WHERE EXISTS (SELECT 1 FROM UNNEST(p.authors) a WHERE a ILIKE ALL(%s))
        )
        SELECT co AS coauthor, COUNT(*) AS shared_papers
        FROM matched, UNNEST(matched.authors) co
        WHERE NOT (co ILIKE ALL(%s))
        GROUP BY co
        ORDER BY shared_papers DESC, coauthor
        LIMIT %s
    """, (patterns, patterns, limit))

    return rows, {
        "author": name,
        "coauthors_shown": len(rows),
        "matched_name_variants": await author_variants(conn, name),
        "source": "postgres",
    }


async def papers_in_category(
    conn: psycopg.AsyncConnection, category: str, limit: int = 20
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Papers filed under an ArXiv category code, most recent first.

    Uses `&&` against papers_categories_gin rather than the ILIKE matching the
    author queries need: category codes are controlled vocabulary ("cs.LG"), so
    exact array overlap is both correct and index-backed here.
    """
    limit = max(1, min(int(limit), MAX_ANALYTICS_ROWS))
    total = await _fetch(conn, """
        SELECT COUNT(*) AS total FROM papers WHERE categories && ARRAY[%s]::text[]
    """, (category,))
    rows = await _fetch(conn, """
        SELECT arxiv_id, title, published_at::date AS published_at, authors
        FROM papers
        WHERE categories && ARRAY[%s]::text[]
        ORDER BY published_at DESC NULLS LAST, arxiv_id
        LIMIT %s
    """, (category, limit))

    matched = total[0]["total"] if total else 0
    return rows, {
        "category": category,
        "total_papers": matched,
        "papers_shown": len(rows),
        "truncated": matched > len(rows),
        "source": "postgres",
    }
