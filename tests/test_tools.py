from __future__ import annotations
"""Tests for agent tools — mocks all external dependencies."""
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from agent.tools import TOOL_DISPATCH, TOOL_DEFINITIONS


@pytest.mark.asyncio
async def test_rag_retrieval_returns_json():
    # Tools use lazy imports inside function bodies — patch at source modules.
    # Hybrid retrieval calls search_similar_chunks_hybrid + rerank + get_gateway.
    mock_results = [{"content": "test chunk", "arxiv_id": "2024.0001", "title": "Test Paper"}]
    mock_gateway = MagicMock()
    with (
        patch("ingestion.embed.embed_query", return_value=[0.1] * 768),
        patch("db.connection.get_connection") as mock_conn_cm,
        patch("db.queries.search_similar_chunks_hybrid", new_callable=AsyncMock, return_value=mock_results),
        patch("agent.registry.get_gateway", return_value=mock_gateway),
        patch("agent.reranker.rerank", new_callable=AsyncMock, return_value=mock_results),
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
    assert len(TOOL_DEFINITIONS) == 4
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


@pytest.mark.asyncio
async def test_graph_query_papers_by_author():
    """query_type='papers_by_author' must run the AUTHORED_BY Cypher template and return results."""
    from agent.tools import graph_query_tool

    mock_record = {"arxiv_id": "1234.5678", "title": "Attention Is All You Need"}
    mock_result = MagicMock()
    mock_result.data = AsyncMock(return_value=[mock_record])

    mock_session = MagicMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch("graph.neo4j_client.get_driver", return_value=mock_driver):
        result_json = await graph_query_tool(query_type="papers_by_author", value="Ashish Vaswani")

    result = json.loads(result_json)
    assert result["tool"] == "graph_query"
    assert result["count"] == 1
    assert result["results"][0]["arxiv_id"] == "1234.5678"


@pytest.mark.asyncio
async def test_graph_query_papers_by_category():
    """query_type='papers_by_category' must run the HAS_CATEGORY Cypher template."""
    from agent.tools import graph_query_tool

    mock_result = MagicMock()
    mock_result.data = AsyncMock(return_value=[{"arxiv_id": "9999.0001", "title": "A Survey of Y"}])

    mock_session = MagicMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch("graph.neo4j_client.get_driver", return_value=mock_driver):
        result_json = await graph_query_tool(query_type="papers_by_category", value="cs.LG")

    result = json.loads(result_json)
    assert result["count"] == 1
    assert result["results"][0]["title"] == "A Survey of Y"


@pytest.mark.asyncio
async def test_graph_query_coauthors():
    """query_type='coauthors' must run the co-authorship Cypher template."""
    from agent.tools import graph_query_tool

    mock_result = MagicMock()
    mock_result.data = AsyncMock(return_value=[{"name": "Noam Shazeer"}])

    mock_session = MagicMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    with patch("graph.neo4j_client.get_driver", return_value=mock_driver):
        result_json = await graph_query_tool(query_type="coauthors", value="Ashish Vaswani")

    result = json.loads(result_json)
    assert result["count"] == 1
    assert result["results"][0]["name"] == "Noam Shazeer"


@pytest.mark.asyncio
async def test_graph_query_unknown_type_returns_error():
    """An unrecognized query_type must return an error, not raise or run an arbitrary query."""
    from agent.tools import graph_query_tool

    result_json = await graph_query_tool(query_type="delete_everything", value="x")
    result = json.loads(result_json)
    assert "error" in result
    assert result["results"] == []


@pytest.mark.asyncio
async def test_graph_query_driver_error_returns_empty_results():
    """If Neo4j is unreachable, the tool must return a graceful error, not crash the agent."""
    from agent.tools import graph_query_tool

    with patch("graph.neo4j_client.get_driver", side_effect=RuntimeError("connection refused")):
        result_json = await graph_query_tool(query_type="papers_by_author", value="Anyone")

    result = json.loads(result_json)
    assert "error" in result
    assert result["results"] == []
