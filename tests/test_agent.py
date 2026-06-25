from __future__ import annotations
"""Tests for the LangGraph agent graph — all LLM/DB calls mocked."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_planner_creates_plan():
    from agent.nodes import planner_node
    mock_gw = MagicMock()
    mock_gw.chat = AsyncMock(return_value={
        "content": '[{"step": "search papers", "tool": "rag_retrieval", "args": {"query": "transformers"}}]',
        "provider": "groq",
        "tokens_in": 50,
        "tokens_out": 30,
    })
    state = {"user_query": "Tell me about transformers", "session_id": "s1", "_gateway": mock_gw,
             "tokens_in": 0, "tokens_out": 0}
    result = await planner_node(state)
    assert len(result["plan"]) == 1
    assert result["plan"][0]["tool"] == "rag_retrieval"
    assert result["llm_provider"] == "groq"


@pytest.mark.asyncio
async def test_planner_handles_bad_json():
    from agent.nodes import planner_node
    mock_gw = MagicMock()
    mock_gw.chat = AsyncMock(return_value={
        "content": "not valid json at all",
        "provider": "gemini",
        "tokens_in": 10,
        "tokens_out": 5,
    })
    state = {"user_query": "test query", "session_id": "s1", "_gateway": mock_gw,
             "tokens_in": 0, "tokens_out": 0}
    result = await planner_node(state)
    # Falls back to default RAG plan
    assert len(result["plan"]) >= 1
    assert result["plan"][0]["tool"] == "rag_retrieval"


@pytest.mark.asyncio
async def test_executor_calls_tool():
    from agent.nodes import executor_node
    import json
    mock_result = json.dumps({
        "tool": "rag_retrieval",
        "results": [{"content": "chunk", "arxiv_id": "2024.001", "title": "Test"}],
        "count": 1,
    })
    with patch("agent.tools.TOOL_DISPATCH", {"rag_retrieval": AsyncMock(return_value=mock_result)}):
        state = {
            "user_query": "test",
            "plan": [{"step": "search", "tool": "rag_retrieval", "args": {"query": "test"}}],
            "current_step": 0,
            "tools_called": [],
            "retrieved_chunks": [],
            "sql_results": [],
        }
        result = await executor_node(state)
        assert "rag_retrieval" in result["tools_called"]
        assert len(result["retrieved_chunks"]) == 1


@pytest.mark.asyncio
async def test_critic_returns_pass():
    from agent.nodes import critic_node
    mock_gw = MagicMock()
    mock_gw.chat = AsyncMock(return_value={
        "content": '{"verdict": "PASS", "reason": "sufficient context"}',
        "provider": "groq",
        "tokens_in": 20,
        "tokens_out": 10,
    })
    state = {
        "user_query": "test",
        "_gateway": mock_gw,
        "retrieved_chunks": [{"content": "some content"}],
        "sql_results": [],
        "retry_count": 0,
        "tokens_in": 0,
        "tokens_out": 0,
    }
    result = await critic_node(state)
    assert result["_critic_verdict"] == "PASS"
    assert result["retry_count"] == 0


@pytest.mark.asyncio
async def test_reporter_produces_answer():
    from agent.nodes import reporter_node
    mock_gw = MagicMock()
    mock_gw.chat = AsyncMock(return_value={
        "content": "Transformers use self-attention mechanisms. Sources: [2017.1234]",
        "provider": "groq",
        "tokens_in": 100,
        "tokens_out": 80,
    })
    state = {
        "user_query": "What are transformers?",
        "_gateway": mock_gw,
        "retrieved_chunks": [{"content": "attention is all you need", "arxiv_id": "2017.1234", "title": "Attention", "authors": ["Vaswani"]}],
        "sql_results": [],
        "tokens_in": 0,
        "tokens_out": 0,
    }
    result = await reporter_node(state)
    assert result["final_report"] is not None
    assert len(result["citations"]) == 1
    assert result["citations"][0]["arxiv_id"] == "2017.1234"
