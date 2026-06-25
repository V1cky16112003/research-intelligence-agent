from __future__ import annotations
"""
LLM Gateway: Groq (primary) → Gemini 2.5 Flash (fallback).

Features:
- Exponential backoff on 429: delays [1s, 4s, 16s] with ±20% jitter
- Falls back to Gemini when Groq is exhausted
- Redis response cache (TTL 1hr, SHA256 key on model+messages+temperature)
- Provider tagging on every response for audit logging
- OpenAI-format tool calling passed through unchanged to both providers
"""
import asyncio
import hashlib
import json
import logging
import random
from typing import Any

from openai import AsyncOpenAI, RateLimitError, APIStatusError, NotGiven

logger = logging.getLogger(__name__)

NOT_GIVEN = NotGiven()


class GatewayExhaustedError(Exception):
    """Raised when all LLM providers fail after retries."""


class LLMGateway:
    """Routes LLM calls: Groq (primary) → Gemini 2.5 Flash (fallback)."""

    GROQ_MODEL = "llama-3.3-70b-versatile"
    GEMINI_MODEL = "gemini-2.5-flash"
    RETRY_DELAYS = [1.0, 4.0, 16.0]

    def __init__(
        self,
        groq_api_key: str,
        gemini_api_key: str,
        redis_client=None,
    ) -> None:
        self._groq = AsyncOpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._gemini = AsyncOpenAI(
            api_key=gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self._redis = redis_client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str = GROQ_MODEL,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        tools: list[dict] | None = None,
        cache: bool = True,
    ) -> dict[str, Any]:
        """
        Call the LLM with automatic fallback.

        Returns:
            {
                "content": str | None,
                "tool_calls": list | None,
                "provider": "groq" | "gemini",
                "model": str,
                "tokens_in": int,
                "tokens_out": int,
                "cached": bool,
            }
        Raises:
            GatewayExhaustedError: if both Groq and Gemini fail.
        """
        # Cache check
        cache_key = self._cache_key(model, messages, temperature)
        if cache and self._redis:
            try:
                cached = await self._redis.get(cache_key)
                if cached:
                    result = json.loads(cached)
                    result["cached"] = True
                    return result
            except Exception:
                pass  # Cache miss on error — proceed

        # Try Groq
        result = None
        try:
            result = await self._with_retry(
                self._groq, model, messages, temperature, max_tokens, tools, "groq"
            )
            result["provider"] = "groq"
            result["model"] = model
        except Exception as groq_exc:
            logger.warning("Groq exhausted (%s), falling back to Gemini", groq_exc)
            try:
                result = await self._with_retry(
                    self._gemini, self.GEMINI_MODEL, messages, temperature, max_tokens, tools, "gemini"
                )
                result["provider"] = "gemini"
                result["model"] = self.GEMINI_MODEL
            except Exception as gemini_exc:
                raise GatewayExhaustedError(
                    f"Both providers exhausted. Groq: {groq_exc}. Gemini: {gemini_exc}"
                ) from gemini_exc

        result["cached"] = False

        # Store in cache
        if cache and self._redis:
            try:
                await self._redis.set(cache_key, json.dumps(result), ttl=3600)
            except Exception:
                pass  # Don't fail on cache write error

        return result

    def _cache_key(self, model: str, messages: list, temperature: float) -> str:
        """SHA256-based cache key."""
        payload = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            sort_keys=True,
        )
        return "llm:" + hashlib.sha256(payload.encode()).hexdigest()[:32]

    async def _with_retry(
        self,
        client: AsyncOpenAI,
        model: str,
        messages: list,
        temperature: float,
        max_tokens: int,
        tools: list | None,
        provider_name: str,
    ) -> dict[str, Any]:
        """Call a provider with exponential backoff on 429/5xx."""
        last_exc: Exception | None = None
        for i, delay in enumerate(self.RETRY_DELAYS):
            try:
                return await self._call_provider(client, model, messages, temperature, max_tokens, tools)
            except RateLimitError as exc:
                last_exc = exc
                if i < len(self.RETRY_DELAYS) - 1:
                    jitter = delay * random.uniform(-0.2, 0.2)
                    wait = max(0.1, delay + jitter)
                    logger.warning("%s 429, retry %d/%d in %.1fs", provider_name, i + 1, len(self.RETRY_DELAYS), wait)
                    await asyncio.sleep(wait)
            except APIStatusError as exc:
                if exc.status_code >= 500 and i == 0:
                    last_exc = exc
                    logger.warning("%s 5xx (%d), retrying once", provider_name, exc.status_code)
                    await asyncio.sleep(1.0)
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    async def _call_provider(
        self,
        client: AsyncOpenAI,
        model: str,
        messages: list,
        temperature: float,
        max_tokens: int,
        tools: list | None,
    ) -> dict[str, Any]:
        """Single provider call, returns normalized response dict (no provider/cached keys)."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        resp = await client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        return {
            "content": msg.content,
            "tool_calls": tool_calls,
            "tokens_in": resp.usage.prompt_tokens if resp.usage else 0,
            "tokens_out": resp.usage.completion_tokens if resp.usage else 0,
        }
