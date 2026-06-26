from __future__ import annotations
"""Tests for agent tools — mocks all external dependencies."""
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from agent.tools import TOOL_DISPATCH, TOOL_DEFINITIONS


@pytest.mark.asyncio
async def test_rag_retrieval_returns_json():
    # Tools use lazy imports inside function bodies — patch at source modules
    mock_results = [{"content": "test chunk", "arxiv_id": "2024.0001", "title": "Test Paper"}]
    with (
        patch("ingestion.embed.embed_query", return_value=[0.1] * 768),
        patch("db.connection.get_connection") as mock_conn_cm,
        patch("db.queries.search_similar_chunks", new_callable=AsyncMock, return_value=mock_results),
    ):
        mock_conn = AsyncMock()
        mock_conn_cm.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn_cm.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await TOOL_DISPATCH["rag_retrieval"]("attention mechanism", categories=None)
        data = json.loads(result)
        assert data["tool"] == "rag_retrieval"
        assert data["count"] == 1


@pytest.mark.asyncio
async def test_rag_retrieval_handles_error():
    with patch("ingestion.embed.embed_query", side_effect=RuntimeError("DB not ready")):
        result = await TOOL_DISPATCH["rag_retrieval"]("test query")
        data = json.loads(result)
        assert "error" in data
        assert data["results"] == []


@pytest.mark.asyncio
async def test_web_search_returns_json():
    pytest.importorskip("duckduckgo_search", reason="duckduckgo_search not installed")
    mock_results = [{"title": "Test", "href": "http://example.com", "body": "snippet"}]
    with patch("duckduckgo_search.DDGS") as mock_ddgs_cls:
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = mock_results
        mock_ddgs_cls.return_value.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = await TOOL_DISPATCH["web_search"]("transformer models 2024")
        data = json.loads(result)
        assert data["tool"] == "web_search"
        assert data["count"] >= 0


def test_tool_definitions_valid():
    """All tool definitions have required OpenAI function-calling fields."""
    assert len(TOOL_DEFINITIONS) == 3
    for td in TOOL_DEFINITIONS:
        assert td["type"] == "function"
        assert "name" in td["function"]
        assert "description" in td["function"]
        assert "parameters" in td["function"]


def test_tool_dispatch_matches_definitions():
    """Every tool definition has a corresponding dispatch entry."""
    for td in TOOL_DEFINITIONS:
        name = td["function"]["name"]
        assert name in TOOL_DISPATCH, f"Tool '{name}' in TOOL_DEFINITIONS but not in TOOL_DISPATCH"
