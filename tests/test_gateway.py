from __future__ import annotations
"""Unit tests for LLM gateway — no real API calls."""
import json
from unittest.mock import AsyncMock, MagicMock
import pytest
from agent.gateway import LLMGateway, GatewayExhaustedError
from openai import RateLimitError


def _make_response(content: str = "hello", provider: str = "groq") -> dict:
    return {
        "content": content,
        "tool_calls": None,
        "tokens_in": 10,
        "tokens_out": 5,
    }


@pytest.mark.asyncio
async def test_groq_success():
    """Groq succeeds on first try — returns provider=groq, cached=False."""
    gw = LLMGateway(groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake")
    gw._with_retry = AsyncMock(return_value=_make_response())
    result = await gw.chat([{"role": "user", "content": "hi"}], cache=False)
    assert result["provider"] == "groq"
    assert result["cached"] is False
    assert result["content"] == "hello"


@pytest.mark.asyncio
async def test_groq_and_nim_429_falls_back_to_gemini():
    """Groq and NIM both rate-limited — gateway falls back to Gemini."""
    gw = LLMGateway(groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake")

    call_order = []
    async def mock_retry(client, model, messages, temperature, max_tokens, tools, provider_name):
        call_order.append(provider_name)
        if provider_name in ("groq", "nvidia_nim"):
            raise RateLimitError("rate limited", response=MagicMock(status_code=429), body={})
        return _make_response("gemini answer")

    gw._with_retry = mock_retry
    result = await gw.chat([{"role": "user", "content": "hi"}], cache=False)
    assert result["provider"] == "gemini"
    assert call_order == ["groq", "nvidia_nim", "gemini"]


@pytest.mark.asyncio
async def test_groq_429_falls_back_to_nim():
    """Groq rate-limited, NIM succeeds — gateway stops at NIM, never calls Gemini."""
    gw = LLMGateway(groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake")

    call_order = []
    async def mock_retry(client, model, messages, temperature, max_tokens, tools, provider_name):
        call_order.append(provider_name)
        if provider_name == "groq":
            raise RateLimitError("rate limited", response=MagicMock(status_code=429), body={})
        return _make_response("nim answer")

    gw._with_retry = mock_retry
    result = await gw.chat([{"role": "user", "content": "hi"}], cache=False)
    assert result["provider"] == "nvidia_nim"
    assert result["model"] == LLMGateway.NIM_MODEL
    assert call_order == ["groq", "nvidia_nim"]


@pytest.mark.asyncio
async def test_cache_hit():
    """Cache hit returns cached=True and skips LLM call entirely."""
    cached_payload = json.dumps({
        "content": "cached answer",
        "tool_calls": None,
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "tokens_in": 5,
        "tokens_out": 3,
    })
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value=cached_payload)

    gw = LLMGateway(groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake", redis_client=mock_redis)
    gw._with_retry = AsyncMock(side_effect=AssertionError("Should not call LLM on cache hit"))

    result = await gw.chat([{"role": "user", "content": "hi"}], cache=True)
    assert result["cached"] is True
    assert result["content"] == "cached answer"
    gw._with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_all_three_exhausted():
    """Groq, NIM, and Gemini all fail — GatewayExhaustedError raised."""
    gw = LLMGateway(groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake")

    async def always_fail(client, model, messages, temperature, max_tokens, tools, provider_name):
        raise RateLimitError("rate limited", response=MagicMock(status_code=429), body={})

    gw._with_retry = always_fail
    with pytest.raises(GatewayExhaustedError):
        await gw.chat([{"role": "user", "content": "hi"}], cache=False)


@pytest.mark.asyncio
async def test_redis_client_detection():
    """create_redis_client returns correct type based on URL scheme."""
    from agent.redis_client import create_redis_client, UpstashRedisClient, LocalRedisClient

    upstash = await create_redis_client("https://default:token@my-host.upstash.io")
    assert isinstance(upstash, UpstashRedisClient)

    local = await create_redis_client("redis://localhost:6379")
    assert isinstance(local, LocalRedisClient)

    none_client = await create_redis_client(None)
    assert none_client is None
