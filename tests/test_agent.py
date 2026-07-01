from __future__ import annotations
"""Tests for the LangGraph agent graph — all LLM/DB calls mocked."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_planner_creates_plan():
    from agent.nodes import planner_node
    from agent.registry import set_gateway
    mock_gw = MagicMock()
    mock_gw.chat = AsyncMock(return_value={
        "content": '[{"step": "search papers", "tool": "rag_retrieval", "args": {"query": "transformers"}}]',
        "provider": "groq",
        "tokens_in": 50,
        "tokens_out": 30,
    })
    set_gateway(mock_gw)
    state = {"user_query": "Tell me about transformers", "session_id": "s1",
             "tokens_in": 0, "tokens_out": 0}
    result = await planner_node(state)
    assert len(result["plan"]) == 1
    assert result["plan"][0]["tool"] == "rag_retrieval"
    assert result["llm_provider"] == "groq"


@pytest.mark.asyncio
async def test_planner_handles_bad_json():
    from agent.nodes import planner_node
    from agent.registry import set_gateway
    mock_gw = MagicMock()
    mock_gw.chat = AsyncMock(return_value={
        "content": "not valid json at all",
        "provider": "gemini",
        "tokens_in": 10,
        "tokens_out": 5,
    })
    set_gateway(mock_gw)
    state = {"user_query": "test query", "session_id": "s1",
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
    from agent.registry import set_gateway
    mock_gw = MagicMock()
    mock_gw.chat = AsyncMock(return_value={
        "content": '{"verdict": "PASS", "reason": "sufficient context", "refined_query": null}',
        "provider": "groq",
        "tokens_in": 20,
        "tokens_out": 10,
    })
    set_gateway(mock_gw)
    state = {
        "user_query": "test",
        "draft_answer": "Transformers use self-attention.",
        "retrieved_chunks": [{"content": "some content"}],
        "sql_results": [],
        "retry_count": 0,
        "tokens_in": 0,
        "tokens_out": 0,
    }
    result = await critic_node(state)
    assert result["_critic_verdict"] == "PASS"
    assert result["retry_count"] == 0
    assert result["refined_query"] is None


@pytest.mark.asyncio
async def test_reporter_produces_answer():
    from agent.nodes import reporter_node
    from agent.registry import set_gateway
    mock_gw = MagicMock()
    mock_gw.chat = AsyncMock(return_value={
        "content": "Transformers use self-attention mechanisms. Sources: [2017.1234]",
        "provider": "groq",
        "tokens_in": 100,
        "tokens_out": 80,
    })
    set_gateway(mock_gw)
    state = {
        "user_query": "What are transformers?",
        "retrieved_chunks": [{"content": "attention is all you need", "arxiv_id": "2017.1234", "title": "Attention", "authors": ["Vaswani"]}],
        "sql_results": [],
        "tokens_in": 0,
        "tokens_out": 0,
    }
    result = await reporter_node(state)
    assert result["final_report"] is not None
    assert len(result["citations"]) == 1
    assert result["citations"][0]["arxiv_id"] == "2017.1234"


# ---------------------------------------------------------------------------
# New tests — robustness and real critic loop
# ---------------------------------------------------------------------------

def test_extract_json_bare_object():
    """_extract_json pulls out a bare JSON object."""
    from agent.nodes import _extract_json
    text = '{"verdict": "PASS", "reason": "ok"}'
    assert json.loads(_extract_json(text))["verdict"] == "PASS"


def test_extract_json_fenced():
    """_extract_json handles markdown code fences."""
    from agent.nodes import _extract_json
    text = '```json\n{"verdict": "RETRY", "reason": "vague"}\n```'
    assert json.loads(_extract_json(text))["verdict"] == "RETRY"


def test_extract_json_array():
    """_extract_json handles JSON arrays (planner output)."""
    from agent.nodes import _extract_json
    text = 'Here is the plan:\n```\n[{"step": "search", "tool": "rag_retrieval"}]\n```'
    parsed = json.loads(_extract_json(text))
    assert isinstance(parsed, list)
    assert parsed[0]["tool"] == "rag_retrieval"


@pytest.mark.asyncio
async def test_reporter_sets_draft_answer():
    """Reporter writes draft_answer as well as final_report."""
    from agent.nodes import reporter_node
    from agent.registry import set_gateway
    mock_gw = MagicMock()
    mock_gw.chat = AsyncMock(return_value={
        "content": "This is the draft answer.",
        "provider": "groq",
        "tokens_in": 50,
        "tokens_out": 30,
    })
    set_gateway(mock_gw)
    state = {
        "user_query": "What is BERT?",
        "retrieved_chunks": [],
        "sql_results": [],
        "tokens_in": 0,
        "tokens_out": 0,
    }
    result = await reporter_node(state)
    assert result["draft_answer"] == "This is the draft answer."
    assert result["final_report"] == "This is the draft answer."


@pytest.mark.asyncio
async def test_critic_sees_draft_and_sets_refined_query():
    """Critic RETRY response sets refined_query and increments retry_count."""
    from agent.nodes import critic_node
    from agent.registry import set_gateway
    mock_gw = MagicMock()
    mock_gw.chat = AsyncMock(return_value={
        "content": '{"verdict": "RETRY", "reason": "too vague", "refined_query": "BERT masked language model pretraining"}',
        "provider": "groq",
        "tokens_in": 30,
        "tokens_out": 20,
    })
    set_gateway(mock_gw)
    state = {
        "user_query": "What is BERT?",
        "draft_answer": "BERT is a language model.",
        "retrieved_chunks": [{"content": "some context", "title": "BERT paper", "arxiv_id": "1810.04805"}],
        "sql_results": [],
        "retry_count": 0,
        "tokens_in": 0,
        "tokens_out": 0,
    }
    result = await critic_node(state)
    assert result["_critic_verdict"] == "RETRY"
    assert result["retry_count"] == 1
    assert result["refined_query"] == "BERT masked language model pretraining"


@pytest.mark.asyncio
async def test_critic_retry_exhausted_becomes_pass():
    """Critic RETRY is ignored when retry_count already at MAX_RETRIES."""
    from agent.nodes import critic_node, MAX_RETRIES
    from agent.registry import set_gateway
    mock_gw = MagicMock()
    mock_gw.chat = AsyncMock(return_value={
        "content": '{"verdict": "RETRY", "reason": "still vague", "refined_query": "something"}',
        "provider": "groq",
        "tokens_in": 20,
        "tokens_out": 10,
    })
    set_gateway(mock_gw)
    state = {
        "user_query": "test",
        "draft_answer": "Some answer.",
        "retrieved_chunks": [],
        "sql_results": [],
        "retry_count": MAX_RETRIES,  # already exhausted
        "tokens_in": 0,
        "tokens_out": 0,
    }
    result = await critic_node(state)
    # Verdict is RETRY but retry_count is not incremented and refined_query stays None
    assert result["_critic_verdict"] == "RETRY"
    assert result["retry_count"] == MAX_RETRIES
    assert result["refined_query"] is None


@pytest.mark.asyncio
async def test_executor_runs_all_steps():
    """Executor runs every step in the plan, not just the first."""
    from agent.nodes import executor_node
    rag_result = json.dumps({"tool": "rag_retrieval", "results": [{"content": "chunk", "id": 1}], "count": 1})
    sql_result = json.dumps({"tool": "sql_analytics", "results": [{"count": 42}], "count": 1})
    mock_rag = AsyncMock(return_value=rag_result)
    mock_sql = AsyncMock(return_value=sql_result)
    with patch("agent.tools.TOOL_DISPATCH", {"rag_retrieval": mock_rag, "sql_analytics": mock_sql}):
        state = {
            "user_query": "test",
            "plan": [
                {"step": "retrieve", "tool": "rag_retrieval", "args": {"query": "test"}},
                {"step": "stats", "tool": "sql_analytics", "args": {"query_type": "papers_by_month"}},
            ],
            "current_step": 0,
            "tools_called": [],
            "retrieved_chunks": [],
            "sql_results": [],
            "refined_query": None,
        }
        result = await executor_node(state)
    assert "rag_retrieval" in result["tools_called"]
    assert "sql_analytics" in result["tools_called"]
    assert len(result["retrieved_chunks"]) == 1
    assert len(result["sql_results"]) == 1


@pytest.mark.asyncio
async def test_executor_handles_bad_args():
    """Executor recovers from TypeError (bad LLM-planned args) without raising."""
    from agent.nodes import executor_node
    rag_fallback = json.dumps({"tool": "rag_retrieval", "results": [{"content": "fallback", "id": 99}], "count": 1})

    def bad_tool(**kwargs):
        raise TypeError("unexpected keyword argument 'q'")

    with patch("agent.tools.TOOL_DISPATCH", {"rag_retrieval": AsyncMock(side_effect=TypeError("bad arg"))}):
        with patch("agent.tools.rag_retrieval_tool", AsyncMock(return_value=rag_fallback)):
            state = {
                "user_query": "test fallback",
                "plan": [{"step": "search", "tool": "rag_retrieval", "args": {"q": "wrong key"}}],
                "current_step": 0,
                "tools_called": [],
                "retrieved_chunks": [],
                "sql_results": [],
                "refined_query": None,
            }
            # Should not raise
            result = await executor_node(state)
    assert "rag_retrieval" in result["tools_called"]


@pytest.mark.asyncio
async def test_executor_handles_malformed_tool_json():
    """Executor records an error but does not crash when tool returns invalid JSON."""
    from agent.nodes import executor_node

    async def bad_json_tool(**kwargs):
        return "this is not json {"

    with patch("agent.tools.TOOL_DISPATCH", {"rag_retrieval": bad_json_tool}):
        state = {
            "user_query": "test",
            "plan": [{"step": "search", "tool": "rag_retrieval", "args": {"query": "test"}}],
            "current_step": 0,
            "tools_called": [],
            "retrieved_chunks": [],
            "sql_results": [],
            "refined_query": None,
        }
        result = await executor_node(state)
    # The step recorded an error result and the executor did not raise
    assert len(result["tool_results"]) == 1
    assert "error" in result["tool_results"][0]["result"]


@pytest.mark.asyncio
async def test_executor_retry_path_merges_chunks():
    """Executor retry path merges new chunks into existing, deduped by id."""
    from agent.nodes import executor_node
    new_chunk = {"id": 2, "content": "new chunk", "arxiv_id": "2024.002"}
    rag_result = json.dumps({"tool": "rag_retrieval", "results": [new_chunk], "count": 1})

    with patch("agent.tools.rag_retrieval_tool", AsyncMock(return_value=rag_result)):
        state = {
            "user_query": "test",
            "plan": [],
            "current_step": 0,
            "tools_called": [],
            "retrieved_chunks": [{"id": 1, "content": "existing chunk"}],
            "sql_results": [],
            "refined_query": "BERT pretraining objectives",
        }
        result = await executor_node(state)

    # Both chunks present, refined_query cleared
    assert len(result["retrieved_chunks"]) == 2
    ids = {c["id"] for c in result["retrieved_chunks"]}
    assert ids == {1, 2}
    assert result["refined_query"] is None


def test_cache_key_varies_with_max_tokens():
    """Gateway cache key differs when max_tokens changes."""
    from agent.gateway import LLMGateway
    gw = LLMGateway(groq_api_key="x", nvidia_api_key="z", gemini_api_key="y")
    messages = [{"role": "user", "content": "hello"}]
    key1 = gw._cache_key("model", messages, 0.1, 512, None)
    key2 = gw._cache_key("model", messages, 0.1, 1024, None)
    assert key1 != key2


def test_cache_key_varies_with_tools():
    """Gateway cache key differs when tools list changes."""
    from agent.gateway import LLMGateway
    gw = LLMGateway(groq_api_key="x", nvidia_api_key="z", gemini_api_key="y")
    messages = [{"role": "user", "content": "hello"}]
    key_no_tools = gw._cache_key("model", messages, 0.1, 512, None)
    key_with_tools = gw._cache_key("model", messages, 0.1, 512, [{"type": "function"}])
    assert key_no_tools != key_with_tools
