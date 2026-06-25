from __future__ import annotations
"""
Agent tools: RAG retrieval, SQL analytics, web search.
Each tool is an async function that accepts a string input and returns a string result.
"""
import json
import logging

logger = logging.getLogger(__name__)


async def rag_retrieval_tool(query: str, categories: str | None = None) -> str:
    """
    Retrieve relevant paper chunks from pgvector using semantic similarity.

    Args:
        query: The search query
        categories: Optional comma-separated ArXiv category filter (e.g. "cs.LG,cs.AI")

    Returns:
        JSON string with list of retrieved chunks and their metadata.
    """
    from ingestion.embed import embed_query
    from db.connection import get_connection
    from db.queries import search_similar_chunks

    category_list = [c.strip() for c in categories.split(",")] if categories else None

    try:
        query_embedding = embed_query(query)
        async with get_connection() as conn:
            results = await search_similar_chunks(
                conn,
                query_embedding=query_embedding,
                k=8,
                categories=category_list,
            )
        return json.dumps({
            "tool": "rag_retrieval",
            "query": query,
            "results": results,
            "count": len(results),
        }, default=str)
    except Exception as e:
        logger.error("RAG retrieval failed: %s", e)
        return json.dumps({"tool": "rag_retrieval", "error": str(e), "results": []})


async def sql_analytics_tool(query_type: str) -> str:
    """
    Run SQL analytics queries over the papers corpus.

    Args:
        query_type: One of: 'papers_by_month', 'query_volume', 'provider_latency', 'experiments'

    Returns:
        JSON string with query results.
    """
    from db.connection import get_connection
    from db import queries

    try:
        async with get_connection() as conn:
            if query_type == "papers_by_month":
                results = await queries.papers_per_category_per_month(conn)
            elif query_type == "query_volume":
                results = await queries.rolling_query_volume(conn, days=7)
            elif query_type == "provider_latency":
                results = await queries.provider_p95_latency(conn)
            elif query_type == "experiments":
                results = await queries.get_experiments_summary(conn)
            else:
                return json.dumps({"error": f"Unknown query_type: {query_type}. Valid: papers_by_month, query_volume, provider_latency, experiments"})

        return json.dumps({
            "tool": "sql_analytics",
            "query_type": query_type,
            "results": results,
            "count": len(results),
        }, default=str)
    except Exception as e:
        logger.error("SQL analytics failed: %s", e)
        return json.dumps({"tool": "sql_analytics", "error": str(e), "results": []})


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
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return json.dumps({
            "tool": "web_search",
            "query": query,
            "results": results,
            "count": len(results),
        })
    except Exception as e:
        logger.error("Web search failed: %s", e)
        return json.dumps({"tool": "web_search", "error": str(e), "results": []})


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
            "description": "Run SQL analytics over the papers database. Use for counting papers, trends, publication stats, or query latency metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["papers_by_month", "query_volume", "provider_latency", "experiments"],
                        "description": "papers_by_month: paper counts by category/month. query_volume: recent query trends. provider_latency: LLM latency stats. experiments: full eval metrics.",
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
]

# Dispatch map: tool name → async function
TOOL_DISPATCH: dict[str, callable] = {
    "rag_retrieval": rag_retrieval_tool,
    "sql_analytics": sql_analytics_tool,
    "web_search": web_search_tool,
}
