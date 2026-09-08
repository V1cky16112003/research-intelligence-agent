from __future__ import annotations

"""
Agent tools: RAG retrieval, SQL analytics, web search.
Each tool is an async function that accepts a string input and returns a string result.
"""
import asyncio
import json
import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)


async def rag_retrieval_tool(query: str, categories: str | None = None) -> str:
    """
    Retrieve relevant paper chunks using hybrid search (dense vector + BM25) and LLM reranking.

    Pipeline:
      1. Embed query with nomic-embed (search_query: prefix)
      2. Run hybrid RRF search: HNSW cosine + tsvector BM25 (falls back to pure vector if
         content_tsv column not present, i.e. migration 001 not yet applied)
      3. Rerank top-16 candidates with LLM gateway → return top 8

    Args:
        query: The search query
        categories: Optional comma-separated ArXiv category filter (e.g. "cs.LG,cs.AI")

    Returns:
        JSON string with list of retrieved chunks and their metadata.
    """
    from agent.registry import get_gateway
    from agent.reranker import rerank
    from db.connection import get_connection
    from db.queries import search_similar_chunks_hybrid
    from ingestion.embed import embed_query

    category_list = [c.strip() for c in categories.split(",")] if categories else None

    try:
        query_embedding = embed_query(query)
        async with get_connection() as conn:
            candidates = await search_similar_chunks_hybrid(
                conn,
                query_embedding=query_embedding,
                query_text=query,
                k=16,           # fetch 16 for reranker to choose from
                categories=category_list,
            )

        gateway = get_gateway()
        results = await rerank(gateway, query=query, candidates=candidates, top_k=8)

        return json.dumps({
            "tool": "rag_retrieval",
            "query": query,
            "results": results,
            "count": len(results),
        }, default=str)
    except Exception as e:
        logger.error("RAG retrieval failed: %s", e)
        return json.dumps({"tool": "rag_retrieval", "error": str(e), "results": []})


SQL_QUERY_TYPES = (
    "corpus_stats",
    "papers_by_category",
    "papers_by_year",
    "papers_by_month",
    "top_authors",
    "query_volume",
    "provider_latency",
    "experiments",
    "cost_by_node",
    "retry_overhead",
)

# Aliases for query types the planner keeps reaching for that don't exist. It is an
# LLM reading a prompt, not a schema, so it invents near-misses; mapping them beats
# returning "Unknown query_type" and letting the reporter conclude the corpus is empty.
_QUERY_TYPE_ALIASES = {
    "paper_count": "corpus_stats",
    "count_papers": "corpus_stats",
    "total_papers": "corpus_stats",
    "corpus_size": "corpus_stats",
    "stats": "corpus_stats",
    "categories": "papers_by_category",
    "category_counts": "papers_by_category",
    "papers_per_category": "papers_by_category",
    "papers_by_author": "top_authors",
    "authors": "top_authors",
    "publication_trend": "papers_by_month",
    "trends": "papers_by_month",
    "papers_per_month": "papers_by_month",
    "papers_per_year": "papers_by_year",
}


async def sql_analytics_tool(
    query_type: str,
    category: str | None = None,
    year: int | str | None = None,
    limit: int | str | None = None,
    **_ignored: object,
) -> str:
    """
    Run SQL analytics over the papers corpus and the operational logs.

    Every query type returns a *complete, pre-aggregated* answer plus a `summary`
    object carrying the totals. That is the whole design point. The previous version
    returned the 500 most-recent (category, month) rows out of 8001 and expected the
    reporter LLM to add them up; asked "how many papers are in the corpus?" the agent
    answered 3,098 against a true 50,000. Postgres counts; the LLM narrates.

    Args:
        query_type: One of SQL_QUERY_TYPES (near-miss names are aliased, not rejected).
        category: Optional ArXiv category filter, e.g. 'cs.LG'.
        year: Optional publication-year filter, e.g. 2015.
        limit: Optional row cap for the ranked lists (papers_by_category, top_authors).
        **_ignored: Swallows argument names the planner invented. The planner is an
            LLM, so it routinely emits plausible-but-undeclared kwargs (observed live:
            {"query_type": "papers_by_month", "category": "cs.LG", "agg": "sum"}).
            Without this, such a call raised TypeError, which the executor turned into
            {"error": ..., "results": []} — the tool reported as "called" while
            silently returning zero rows. Degrading to the valid subset of filters is
            far better than answering nothing.

    Returns:
        JSON string: {"tool", "query_type", "summary", "results", "count"}.
    """
    from db import queries
    from db.connection import get_connection

    if _ignored:
        logger.info("sql_analytics ignoring unsupported planner args: %s", sorted(_ignored))

    requested = (query_type or "").strip()
    query_type = _QUERY_TYPE_ALIASES.get(requested, requested)
    if query_type != requested:
        logger.info("sql_analytics mapped query_type %r -> %r", requested, query_type)

    # The planner emits numbers as ints or strings ("2023"); normalize, and drop
    # anything non-numeric rather than letting it reach the query layer.
    def _as_int(value: object, name: str) -> int | None:
        if value is None:
            return None
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            logger.info("sql_analytics ignoring non-numeric %s: %r", name, value)
            return None

    year = _as_int(year, "year")
    row_limit = _as_int(limit, "limit") or 25

    if query_type not in SQL_QUERY_TYPES:
        return json.dumps({
            "tool": "sql_analytics",
            "error": f"Unknown query_type: {requested}. Valid: {', '.join(SQL_QUERY_TYPES)}",
            "summary": {},
            "results": [],
            "count": 0,
        })

    try:
        summary: dict = {}
        async with get_connection() as conn:
            if query_type == "corpus_stats":
                stats = await queries.corpus_stats(conn)
                results, summary = [stats], stats
            elif query_type == "papers_by_category":
                results, summary = await queries.papers_by_category(conn, year=year, limit=row_limit)
            elif query_type == "papers_by_year":
                results, summary = await queries.papers_by_year(conn, category=category)
            elif query_type == "papers_by_month":
                results, summary = await queries.papers_by_month(conn, category=category, year=year)
            elif query_type == "top_authors":
                results, summary = await queries.top_authors(conn, category=category, limit=row_limit)
            elif query_type == "query_volume":
                results = await queries.rolling_query_volume(conn, days=7)
                summary = {"total_queries": sum(r.get("query_count", 0) for r in results)}
            elif query_type == "provider_latency":
                results = await queries.provider_p95_latency(conn)
            elif query_type == "experiments":
                results = await queries.get_experiments_summary(conn)
            elif query_type == "cost_by_node":
                results = await queries.llm_cost_by_node(conn)
            else:  # retry_overhead — the last member of SQL_QUERY_TYPES
                results = await queries.retry_overhead_summary(conn)

        if not results:
            summary = dict(summary)
            summary["note"] = (
                f"No rows for {query_type}. This means the underlying table is empty "
                "for this window, not that the corpus lacks the data."
            )

        return json.dumps({
            "tool": "sql_analytics",
            "query_type": query_type,
            "summary": summary,
            "results": results,
            "count": len(results),
        }, default=str)
    except Exception as e:
        logger.error("SQL analytics failed: %s", e)
        return json.dumps({
            "tool": "sql_analytics", "error": str(e), "summary": {}, "results": [], "count": 0,
        })


async def web_search_tool(query: str) -> str:
    """
    Search the web using DuckDuckGo for out-of-corpus or recent information.

    Args:
        query: Search query string

    Returns:
        JSON string with search results (title, url, snippet).
    """
    try:
        from duckduckgo_search import DDGS

        def _sync_search() -> list:
            with DDGS() as ddgs:
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                    for r in ddgs.text(query, max_results=5)
                ]

        results = await asyncio.to_thread(_sync_search)
        return json.dumps({
            "tool": "web_search",
            "query": query,
            "results": results,
            "count": len(results),
        })
    except Exception as e:
        logger.error("Web search failed: %s", e)
        return json.dumps({"tool": "web_search", "error": str(e), "results": []})


# Fixed, parameterized Cypher templates — deliberately not LLM-generated, so a
# malformed or unbounded query can never reach the graph database.
_GRAPH_CYPHER_TEMPLATES = {
    "papers_by_author": (
        "MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author {name: $value}) "
        "RETURN p.arxiv_id AS arxiv_id, p.title AS title LIMIT 20"
    ),
    "papers_by_category": (
        "MATCH (p:Paper)-[:HAS_CATEGORY]->(c:Category {name: $value}) "
        "RETURN p.arxiv_id AS arxiv_id, p.title AS title LIMIT 20"
    ),
    "coauthors": (
        "MATCH (:Author {name: $value})<-[:AUTHORED_BY]-(:Paper)-[:AUTHORED_BY]->(a:Author) "
        "WHERE a.name <> $value "
        "RETURN DISTINCT a.name AS name LIMIT 20"
    ),
}


async def _graph_query_neo4j(query_type: str, value: str) -> list[dict]:
    """Run one fixed Cypher template. Raises if the graph is unreachable."""
    from graph.neo4j_client import get_driver

    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(_GRAPH_CYPHER_TEMPLATES[query_type], {"value": value})
        return await result.data()


async def _graph_query_postgres(query_type: str, value: str) -> tuple[list[dict], dict]:
    """Answer the same relational question from the `papers` TEXT[] columns."""
    from db.connection import get_connection
    from db.queries import coauthors, papers_by_author, papers_in_category

    handler = {
        "papers_by_author": papers_by_author,
        "papers_by_category": papers_in_category,
        "coauthors": coauthors,
    }[query_type]

    async with get_connection() as conn:
        return await handler(conn, value)


async def graph_query_tool(query_type: str, value: str) -> str:
    """
    Answer relational questions (co-authorship, shared subfields) about the corpus.

    Tries the Neo4j knowledge graph first and falls back to the equivalent
    Postgres query when the graph is unreachable *or* returns nothing.

    Falling back on an empty result, not just on an exception, is deliberate.
    The Cypher templates match author nodes on `{name: $value}` exactly, but
    ArXiv stores names surname-first with affiliations attached ("Bengio Yoshua
    Universite de Montreal"), so a natural-order name from the planner returns
    zero records rather than an error. Treating empty-from-graph as "ask
    Postgres" turns that silent miss into an answer.

    Args:
        query_type: One of: 'papers_by_author', 'papers_by_category', 'coauthors'
        value: The author name or category code to query for.

    Returns:
        JSON string with list of results and their metadata.
    """
    if query_type not in _GRAPH_CYPHER_TEMPLATES:
        return json.dumps({
            "tool": "graph_query",
            "error": f"Unknown query_type: {query_type}. Valid: {list(_GRAPH_CYPHER_TEMPLATES)}",
            "results": [],
        })

    if os.getenv("NEO4J_URI"):
        try:
            records = await _graph_query_neo4j(query_type, value)
            if records:
                return json.dumps({
                    "tool": "graph_query",
                    "query_type": query_type,
                    "value": value,
                    "results": records,
                    "count": len(records),
                    "summary": {"source": "neo4j"},
                }, default=str)
            logger.info("Neo4j returned no rows for %s=%r; falling back to Postgres",
                        query_type, value)
        except Exception as e:
            logger.warning("Neo4j unavailable (%s); falling back to Postgres", e)

    try:
        rows, summary = await _graph_query_postgres(query_type, value)
        return json.dumps({
            "tool": "graph_query",
            "query_type": query_type,
            "value": value,
            "summary": summary,
            "results": rows,
            "count": len(rows),
        }, default=str)
    except Exception as e:
        logger.error("Graph query failed: %s", e)
        return json.dumps({"tool": "graph_query", "error": str(e), "results": []})


# OpenAI-format tool definitions for the LangGraph Planner
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "rag_retrieval",
            "description": "Search the ArXiv ML paper corpus using semantic similarity. Use for questions about paper content, methods, findings, or authors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Semantic search query"},
                    "categories": {"type": "string", "description": "Optional comma-separated ArXiv categories to filter (e.g. 'cs.LG,cs.AI')"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_analytics",
            "description": "Run SQL analytics over the papers database. Use for counting papers, corpus size, category and author rankings, publication trends, and operational metrics. Returns pre-aggregated totals — read the summary rather than adding up rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": list(SQL_QUERY_TYPES),
                        "description": (
                            "corpus_stats: total papers/chunks/authors/categories and the corpus date range — use for 'how many papers', 'how big is the corpus'. "
                            "papers_by_category: paper counts per ArXiv category, largest first. "
                            "papers_by_year: publication counts per year (optionally one category). "
                            "papers_by_month: monthly publication counts (optionally one category and/or year). "
                            "top_authors: most prolific authors (optionally within one category). "
                            "query_volume: recent daily user-query volume. provider_latency: LLM latency stats. "
                            "experiments: RAGAS eval metrics. cost_by_node: $ cost and latency per agent node per day. "
                            "retry_overhead: cost/latency attributable to Critic-triggered retries vs the happy path."
                        ),
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional ArXiv category filter (e.g. 'cs.LG', 'cs.AI') for papers_by_year, papers_by_month and top_authors. Omit for corpus-wide numbers.",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Optional publication-year filter (e.g. 2015) for papers_by_category and papers_by_month.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional row cap for the ranked lists papers_by_category and top_authors (default 25).",
                    },
                },
                "required": ["query_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for recent or out-of-corpus information. Use when the question is about current events, recent papers not in the corpus, or general knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_query",
            "description": "Query the paper knowledge graph for relational questions: what else an author has written, co-authorship, or papers sharing a subfield/category. Use for questions like 'what else has this author published' or 'who are this author's collaborators', not for content/topic search (use rag_retrieval for that).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["papers_by_author", "papers_by_category", "coauthors"],
                        "description": "papers_by_author: papers written by a given author. papers_by_category: papers in a given ArXiv category. coauthors: other authors who have co-written a paper with the given author.",
                    },
                    "value": {
                        "type": "string",
                        "description": "The author name (for papers_by_author/coauthors) or category code like 'cs.LG' (for papers_by_category)",
                    },
                },
                "required": ["query_type", "value"],
            },
        },
    },
]

# Dispatch map: tool name → async function
TOOL_DISPATCH: dict[str, Callable] = {
    "rag_retrieval": rag_retrieval_tool,
    "sql_analytics": sql_analytics_tool,
    "web_search": web_search_tool,
    "graph_query": graph_query_tool,
}
