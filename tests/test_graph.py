from __future__ import annotations

"""Tests for graph/neo4j_client.py and graph/graph_sync.py — no live Neo4j connection."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_get_driver_creates_singleton():
    """get_driver() must create the driver once and reuse it on subsequent calls."""
    from graph import neo4j_client

    neo4j_client._driver = None  # reset singleton for test isolation
    mock_driver = MagicMock()

    with patch("graph.neo4j_client.AsyncGraphDatabase") as mock_gdb:
        mock_gdb.driver.return_value = mock_driver

        d1 = neo4j_client.get_driver(uri="bolt://fake", user="u", password="p")
        d2 = neo4j_client.get_driver(uri="bolt://fake", user="u", password="p")

        assert d1 is d2
        mock_gdb.driver.assert_called_once_with("bolt://fake", auth=("u", "p"))

    neo4j_client._driver = None  # cleanup


@pytest.mark.asyncio
async def test_sync_papers_to_graph_merges_nodes_and_relationships():
    """sync_papers_to_graph must MERGE Paper/Author/Category nodes with correct relationships."""
    from graph.graph_sync import sync_papers_to_graph

    papers = [
        {
            "arxiv_id": "1234.5678",
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "categories": ["cs.LG", "cs.CL"],
        }
    ]

    mock_session = MagicMock()
    mock_session.run = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    count = await sync_papers_to_graph(mock_driver, papers)

    assert count == 1
    # One Cypher call per paper (batched MERGE of paper+authors+categories in one query)
    assert mock_session.run.call_count == 1
    cypher, params = mock_session.run.call_args[0][0], mock_session.run.call_args[0][1]
    assert "MERGE" in cypher
    assert params["arxiv_id"] == "1234.5678"
    assert params["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
    assert params["categories"] == ["cs.LG", "cs.CL"]


@pytest.mark.asyncio
async def test_sync_papers_to_graph_skips_papers_with_no_authors_or_categories():
    """A paper with empty authors/categories lists must still sync the Paper node itself."""
    from graph.graph_sync import sync_papers_to_graph

    papers = [{"arxiv_id": "0000.0001", "title": "Untitled", "authors": [], "categories": []}]

    mock_session = MagicMock()
    mock_session.run = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    count = await sync_papers_to_graph(mock_driver, papers)

    assert count == 1
    mock_session.run.assert_called_once()


@pytest.mark.asyncio
async def test_run_agent_omits_continuity_fields_from_initial_state():
    """run_agent()'s per-invoke state must not reset retrieved_chunks/sql_results/
    citations/previous_user_query — the checkpointer keys off session_id, and
    including these would overwrite last turn's values, breaking multi-turn
    follow-ups (e.g. "summarize that") which rely on the prior turn's context
    and query surviving."""
    import agent.graph as graph_module

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "final_report": "answer", "citations": [], "sql_results": None,
        "tools_called": [], "llm_provider": "groq", "tokens_in": 1, "tokens_out": 1,
    })
    original_graph = graph_module._graph
    graph_module._graph = mock_graph
    try:
        await graph_module.run_agent("summarize that in one sentence", "session-1")
    finally:
        graph_module._graph = original_graph

    sent_state = mock_graph.ainvoke.call_args[0][0]
    for continuity_key in ("retrieved_chunks", "sql_results", "citations", "previous_user_query"):
        assert continuity_key not in sent_state, (
            f"{continuity_key!r} must be omitted so the checkpointer's prior-turn "
            "value survives into this invoke"
        )
    # Per-turn fields must still reset every call.
    assert sent_state["plan"] == []
    assert sent_state["tools_called"] == []
    assert sent_state["tokens_in"] == 0


@pytest.mark.asyncio
async def test_init_graph_enters_the_checkpointer_context_manager(monkeypatch):
    """AsyncPostgresSaver.from_conn_string is an @asynccontextmanager — calling
    .setup() directly on the unentered object (as init_graph() used to) raises
    AttributeError, which was silently caught into checkpointer=None. This meant
    the LangGraph checkpointer was never actually active in production, no
    matter what state run_agent() sent it: multi-turn follow-ups always started
    from a blank slate. init_graph() must __aenter__ the context manager to get
    a real, usable saver, and close_graph() must __aexit__ it at shutdown."""
    import sys
    import types

    from langgraph.checkpoint.base import BaseCheckpointSaver

    import agent.graph as graph_module

    monkeypatch.setenv("DATABASE_URL", "postgresql://fake")

    # spec=BaseCheckpointSaver so build_graph()'s isinstance check (via
    # ensure_valid_checkpointer) accepts it as a real checkpointer.
    mock_saver = MagicMock(spec=BaseCheckpointSaver)
    mock_saver.setup = AsyncMock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_saver)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    # The real langgraph.checkpoint.postgres package imports psycopg internals
    # (Capabilities, Connection, ...) at import time that conftest's bare psycopg
    # stub doesn't provide. init_graph() imports this submodule lazily specifically
    # to avoid that at collection time, so stub the submodule itself here rather
    # than importing the real thing.
    fake_module = types.ModuleType("langgraph.checkpoint.postgres.aio")
    fake_module.AsyncPostgresSaver = MagicMock()
    fake_module.AsyncPostgresSaver.from_conn_string = MagicMock(return_value=mock_cm)
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres.aio", fake_module)

    try:
        await graph_module.init_graph()
        assert graph_module._checkpointer is mock_saver
        mock_saver.setup.assert_awaited_once()

        await graph_module.close_graph()
        mock_cm.__aexit__.assert_awaited_once()
        assert graph_module._checkpointer_cm is None
    finally:
        graph_module._graph = None
        graph_module._checkpointer = None
        graph_module._checkpointer_cm = None
