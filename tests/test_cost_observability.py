from __future__ import annotations

"""Tests for per-node LLM cost/latency instrumentation."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.gateway import LLMGateway, estimate_cost_usd


def test_estimate_cost_usd_known_model():
    cost = estimate_cost_usd("openai/gpt-oss-120b", tokens_in=1_000_000, tokens_out=1_000_000)
    assert cost == pytest.approx(0.15 + 0.75)


def test_estimate_cost_usd_unknown_model_is_zero():
    assert estimate_cost_usd("some-unlisted-model", tokens_in=1000, tokens_out=1000) == 0.0


@pytest.mark.asyncio
async def test_chat_tags_node_and_cost():
    gw = LLMGateway(groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake")
    gw._with_retry = AsyncMock(return_value={
        "content": "hi", "tool_calls": None, "tokens_in": 10, "tokens_out": 5,
    })
    result = await gw.chat([{"role": "user", "content": "hi"}], cache=False, node="planner")
    assert result["node"] == "planner"
    assert result["is_retry"] is False
    assert result["cost_usd"] >= 0.0
    assert result["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_chat_cache_hit_still_tags_node_with_zero_cost():
    redis = MagicMock()
    redis.get = AsyncMock(return_value='{"content": "cached", "provider": "groq", "model": "openai/gpt-oss-120b", "tokens_in": 10, "tokens_out": 5}')
    gw = LLMGateway(groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake", redis_client=redis)
    result = await gw.chat([{"role": "user", "content": "hi"}], node="critic", is_retry=True)
    assert result["cached"] is True
    assert result["node"] == "critic"
    assert result["is_retry"] is True
    assert result["cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_planner_node_emits_llm_calls_record():
    from agent.nodes import planner_node
    from agent.registry import set_gateway
    mock_gw = MagicMock()
    mock_gw.chat = AsyncMock(return_value={
        "content": "[]", "provider": "groq", "model": "openai/gpt-oss-120b",
        "tokens_in": 50, "tokens_out": 10, "cost_usd": 0.0001, "latency_ms": 120,
        "node": "planner", "is_retry": False, "cached": False,
    })
    set_gateway(mock_gw)
    state = {"user_query": "q", "session_id": "s1", "tokens_in": 0, "tokens_out": 0}
    result = await planner_node(state)
    assert result["llm_calls"] == [{
        "node": "planner", "provider": "groq", "model": "openai/gpt-oss-120b",
        "tokens_in": 50, "tokens_out": 10, "cost_usd": 0.0001, "latency_ms": 120,
        "is_retry": False, "cached": False,
    }]
    mock_gw.chat.assert_awaited_once()
    assert mock_gw.chat.await_args.kwargs["node"] == "planner"


@pytest.mark.asyncio
async def test_reporter_and_critic_tag_is_retry_from_state():
    from agent.nodes import critic_node, reporter_node
    from agent.registry import set_gateway
    mock_gw = MagicMock()
    mock_gw.chat = AsyncMock(return_value={
        "content": '{"verdict": "PASS", "reason": "ok", "refined_query": null}',
        "provider": "groq", "model": "openai/gpt-oss-120b",
        "tokens_in": 5, "tokens_out": 5, "cost_usd": 0.0, "latency_ms": 50,
        "node": "critic", "is_retry": True, "cached": False,
    })
    set_gateway(mock_gw)
    state = {
        "user_query": "q", "session_id": "s1", "retry_count": 1,
        "retrieved_chunks": [], "sql_results": [], "draft_answer": "draft",
        "tokens_in": 0, "tokens_out": 0,
    }
    await critic_node(state)
    assert mock_gw.chat.await_args.kwargs["is_retry"] is True

    mock_gw.chat = AsyncMock(return_value={
        "content": "answer", "provider": "groq", "model": "openai/gpt-oss-120b",
        "tokens_in": 5, "tokens_out": 5, "cost_usd": 0.0, "latency_ms": 50,
        "node": "reporter", "is_retry": True, "cached": False,
    })
    await reporter_node(state)
    assert mock_gw.chat.await_args.kwargs["is_retry"] is True


@pytest.mark.asyncio
async def test_log_llm_calls_inserts_rows():
    from db.queries import log_llm_calls
    conn = MagicMock()
    cursor = AsyncMock()
    conn.cursor.return_value.__aenter__ = AsyncMock(return_value=cursor)
    conn.cursor.return_value.__aexit__ = AsyncMock(return_value=False)
    calls = [{
        "node": "planner", "provider": "groq", "model": "openai/gpt-oss-120b",
        "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.001, "latency_ms": 100,
        "is_retry": False, "cached": False,
    }]
    await log_llm_calls(conn, session_id="s1", calls=calls)
    cursor.executemany.assert_awaited_once()


@pytest.mark.asyncio
async def test_log_llm_calls_noop_on_empty_list():
    from db.queries import log_llm_calls
    conn = MagicMock()
    conn.cursor = MagicMock()
    await log_llm_calls(conn, session_id="s1", calls=[])
    conn.cursor.assert_not_called()
