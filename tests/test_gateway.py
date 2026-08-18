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


def _make_sdk_response(content: str = "hello", finish_reason: str = "stop") -> MagicMock:
    """Mimics the raw OpenAI SDK response object that _call_provider parses, as opposed
    to _make_response's already-normalized dict returned by _with_retry."""
    resp = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = None
    resp.choices = [choice]
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


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
    assert call_order == [("groq", LLMGateway.GROQ_MODEL), ("nvidia_nim", "meta/llama-3.3-70b-instruct")]


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


def test_groq_model_is_not_a_decommissioned_llama():
    """Regression: Groq retired its Llama chat models, so `llama-3.3-70b-versatile`
    began returning 404 on every call. Because the gateway treats any Groq exception
    as "fall back", the primary tier failed silently 100% of the time and every
    request went to NVIDIA NIM's slow free-tier queue — mean latency 50s, p95 234s,
    and RAGAS CI timeouts. Nothing surfaced this as an error, so pin the expectation."""
    from agent.gateway import LLMGateway

    assert not LLMGateway.GROQ_MODEL.startswith("llama-")
    assert LLMGateway.GROQ_MODEL == "openai/gpt-oss-120b"


@pytest.mark.asyncio
async def test_reasoning_effort_sent_only_to_gpt_oss_models():
    """gpt-oss needs reasoning_effort to stop hidden reasoning eating max_tokens, but
    NIM (meta/*) and Gemini reject the parameter — so it must be scoped by model."""
    from agent.gateway import LLMGateway

    gw = LLMGateway(groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake")
    seen: list[dict] = []

    async def fake_create(**kwargs):
        seen.append(kwargs)
        return _make_sdk_response("ok")

    gw._groq.chat.completions.create = fake_create
    await gw._call_provider(gw._groq, "openai/gpt-oss-120b", [], 0.0, 512, None)
    assert seen[-1]["reasoning_effort"] == "low"

    await gw._call_provider(gw._groq, "meta/llama-3.1-70b-instruct", [], 0.0, 512, None)
    assert "reasoning_effort" not in seen[-1]

    await gw._call_provider(gw._groq, "gemini-2.5-flash", [], 0.0, 512, None)
    assert "reasoning_effort" not in seen[-1]


@pytest.mark.asyncio
async def test_truncated_empty_generation_is_logged(caplog):
    """A reasoning model that burns max_tokens returns finish_reason=length with empty
    content and no exception. That silently broke the reranker on every query, so it
    must at least be visible in the logs rather than passing as a valid empty answer."""
    from agent.gateway import LLMGateway

    gw = LLMGateway(groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake")
    truncated = _make_sdk_response("", finish_reason="length")

    async def fake_create(**kwargs):
        return truncated

    gw._groq.chat.completions.create = fake_create
    with caplog.at_level("WARNING"):
        await gw._call_provider(gw._groq, "openai/gpt-oss-120b", [], 0.0, 128, None)

    assert "empty content" in caplog.text
