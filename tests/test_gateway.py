from __future__ import annotations

"""Unit tests for LLM gateway — no real API calls."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import RateLimitError

from agent.gateway import GatewayExhaustedError, LLMGateway


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


def test_construction_with_blank_nvidia_key():
    """nvidia_api_key defaults to '' and construction succeeds without it set."""
    gw = LLMGateway(groq_api_key="fake", nvidia_api_key="", gemini_api_key="fake")
    assert gw is not None


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
async def test_nim_model_override():
    """nim_model constructor override is used for the NIM fallback call and result."""
    gw = LLMGateway(
        groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake",
        nim_model="meta/llama-3.3-70b-instruct",
    )

    call_order = []
    async def mock_retry(client, model, messages, temperature, max_tokens, tools, provider_name):
        call_order.append((provider_name, model))
        if provider_name == "groq":
            raise RateLimitError("rate limited", response=MagicMock(status_code=429), body={})
        return _make_response("nim answer")

    gw._with_retry = mock_retry
    result = await gw.chat([{"role": "user", "content": "hi"}], cache=False)
    assert result["provider"] == "nvidia_nim"
    assert result["model"] == "meta/llama-3.3-70b-instruct"
    assert call_order == [("groq", "llama-3.3-70b-versatile"), ("nvidia_nim", "meta/llama-3.3-70b-instruct")]


@pytest.mark.asyncio
async def test_gemini_disabled_raises_after_nim_exhausted():
    """enable_gemini=False must skip the Gemini tier entirely — never calling it."""
    gw = LLMGateway(
        groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake",
        enable_gemini=False,
    )

    call_order = []
    async def mock_retry(client, model, messages, temperature, max_tokens, tools, provider_name):
        call_order.append(provider_name)
        raise RateLimitError("rate limited", response=MagicMock(status_code=429), body={})

    gw._with_retry = mock_retry
    with pytest.raises(GatewayExhaustedError):
        await gw.chat([{"role": "user", "content": "hi"}], cache=False)
    assert call_order == ["groq", "nvidia_nim"]  # gemini never attempted


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
async def test_with_retry_fails_fast_on_daily_quota_error():
    """A 'tokens per day' RateLimitError must not be retried — the quota can't clear
    within our [1s, 4s, 16s] backoff window, so retrying just wastes ~21s per call."""
    gw = LLMGateway(groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake")

    daily_quota_error = RateLimitError(
        "Rate limit reached for model `llama-3.3-70b-versatile` in organization "
        "`org_fake` service tier `on_demand` on tokens per day (TPD): Limit 100000, "
        "Used 98963, Requested 1568. Please try again in 7m38.784s.",
        response=MagicMock(status_code=429),
        body={"message": "...", "type": "tokens", "code": "rate_limit_exceeded"},
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=daily_quota_error)

    with pytest.raises(RateLimitError):
        await gw._with_retry(
            mock_client, "llama-3.3-70b-versatile",
            [{"role": "user", "content": "hi"}], 0.1, 512, None, "groq",
        )

    assert mock_client.chat.completions.create.call_count == 1  # no retries attempted


@pytest.mark.asyncio
async def test_redis_client_detection():
    """create_redis_client returns correct type based on URL scheme."""
    from agent.redis_client import LocalRedisClient, UpstashRedisClient, create_redis_client

    upstash = await create_redis_client("https://default:token@my-host.upstash.io")
    assert isinstance(upstash, UpstashRedisClient)

    local = await create_redis_client("redis://localhost:6379")
    assert isinstance(local, LocalRedisClient)

    none_client = await create_redis_client(None)
    assert none_client is None
