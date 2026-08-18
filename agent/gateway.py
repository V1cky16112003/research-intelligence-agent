from __future__ import annotations

"""
LLM Gateway: Groq (primary) → NVIDIA NIM (2nd fallback) → Gemini 2.5 Flash (3rd fallback).

Features:
- Exponential backoff on 429/5xx: 4 total attempts with delays [1s, 4s, 16s] + ±20% jitter
- Falls back Groq → NVIDIA NIM → Gemini as each tier is exhausted
- Redis response cache (TTL 1hr, SHA256 key on model+messages+temperature+max_tokens+tools)
- Provider tagging on every response for audit logging
- OpenAI-format tool calling passed through unchanged to all three providers
- `nim_model` and `enable_gemini` are per-instance overrides — callers (e.g. the
  RAGAS CI gate) can pin a different NIM model or disable the Gemini tier
  without changing the defaults used by the production /chat path
"""
import asyncio
import hashlib
import json
import logging
import random
from typing import Any

from openai import APIStatusError, AsyncOpenAI, RateLimitError

logger = logging.getLogger(__name__)


class GatewayExhaustedError(Exception):
    """Raised when all LLM providers fail after retries."""


def _is_daily_quota_error(exc: RateLimitError) -> bool:
    """True if a 429 is a daily token-quota error (e.g. Groq's TPD limit), not a
    transient per-minute rate limit. Daily quotas reset on the order of minutes to
    hours, so retrying within our [1s, 4s, 16s] backoff window can never succeed —
    doing so only wastes ~21s per call before falling back to the next provider."""
    return "per day" in str(exc).lower()


class LLMGateway:
    """Routes LLM calls: Groq (primary) → NVIDIA NIM (fallback) → Gemini 2.5 Flash (fallback)."""

    # Groq decommissioned its Llama chat models: `llama-3.3-70b-versatile` now 404s
    # ("does not exist or you do not have access to it") on every request, so the
    # primary tier failed 100% of the time and every call silently cascaded to NVIDIA
    # NIM's slow free-tier queue — the cause of the 50s mean / 234s p95 latencies in
    # query_audit_log. gpt-oss-120b is the strongest chat model the account can still
    # reach (verified live: 1.5s for a planner call) and, unlike qwen3.6-27b, it keeps
    # chain-of-thought in a separate `reasoning` field instead of emitting <think>
    # blocks into `content`, where they would corrupt the planner/critic JSON parse.
    GROQ_MODEL = "openai/gpt-oss-120b"
    NIM_MODEL = "meta/llama-3.1-70b-instruct"
    GEMINI_MODEL = "gemini-2.5-flash"
    RETRY_DELAYS = [1.0, 4.0, 16.0]

    def __init__(
        self,
        groq_api_key: str,
        gemini_api_key: str,
        nvidia_api_key: str = "",
        redis_client=None,
        nim_model: str = "",
        enable_gemini: bool = True,
    ) -> None:
        self._groq = AsyncOpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._nim = AsyncOpenAI(
            api_key=nvidia_api_key or "unset",
            base_url="https://integrate.api.nvidia.com/v1",
        )
        self._gemini = AsyncOpenAI(
            api_key=gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self._redis = redis_client
        self._nim_model = nim_model or self.NIM_MODEL
        self._enable_gemini = enable_gemini

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
        Call the LLM with automatic 3-tier fallback.

        Returns:
            {
                "content": str | None,
                "tool_calls": list | None,
                "provider": "groq" | "nvidia_nim" | "gemini",
                "model": str,
                "tokens_in": int,
                "tokens_out": int,
                "cached": bool,
            }
        Raises:
            GatewayExhaustedError: if Groq, NVIDIA NIM, and Gemini all fail.
        """
        # Cache check
        cache_key = self._cache_key(model, messages, temperature, max_tokens, tools)
        if cache and self._redis:
            try:
                cached = await self._redis.get(cache_key)
                if cached:
                    result = json.loads(cached)
                    result["cached"] = True
                    return result
            except Exception:
                pass  # Cache miss on error — proceed

        result = None
        groq_exc = nim_exc = gemini_exc = None

        try:
            result = await self._with_retry(
                self._groq, model, messages, temperature, max_tokens, tools, "groq"
            )
            result["provider"] = "groq"
            result["model"] = model
        except Exception as exc:
            groq_exc = exc
            logger.warning("Groq exhausted (%s), falling back to NVIDIA NIM", exc)
            try:
                result = await self._with_retry(
                    self._nim, self._nim_model, messages, temperature, max_tokens, tools, "nvidia_nim"
                )
                result["provider"] = "nvidia_nim"
                result["model"] = self._nim_model
            except Exception as exc2:
                nim_exc = exc2
                if not self._enable_gemini:
                    raise GatewayExhaustedError(
                        f"Groq and NVIDIA NIM exhausted (Gemini fallback disabled). "
                        f"Groq: {groq_exc}. NVIDIA NIM: {nim_exc}."
                    ) from nim_exc
                logger.warning("NVIDIA NIM exhausted (%s), falling back to Gemini", exc2)
                try:
                    result = await self._with_retry(
                        self._gemini, self.GEMINI_MODEL, messages, temperature, max_tokens, tools, "gemini"
                    )
                    result["provider"] = "gemini"
                    result["model"] = self.GEMINI_MODEL
                except Exception as exc3:
                    gemini_exc = exc3
                    raise GatewayExhaustedError(
                        f"All providers exhausted. Groq: {groq_exc}. "
                        f"NVIDIA NIM: {nim_exc}. Gemini: {gemini_exc}"
                    ) from gemini_exc

        result["cached"] = False

        # Store in cache
        if cache and self._redis:
            try:
                await self._redis.set(cache_key, json.dumps(result), ttl=3600)
            except Exception:
                pass  # Don't fail on cache write error

        return result

    def _cache_key(
        self, model: str, messages: list, temperature: float, max_tokens: int, tools: list | None
    ) -> str:
        """SHA256-based cache key — includes max_tokens and tools to avoid collisions."""
        tools_hash = hashlib.sha256(
            json.dumps(tools or [], sort_keys=True).encode()
        ).hexdigest()[:16]
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools_hash": tools_hash,
            },
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
        """Call a provider with exponential backoff on 429/5xx.

        Makes up to len(RETRY_DELAYS)+1 total attempts. Delays [1s, 4s, 16s] ±20% jitter
        are applied between consecutive failed attempts, so all three delays are used.
        """
        last_exc: Exception | None = None
        max_attempts = len(self.RETRY_DELAYS) + 1
        for attempt in range(max_attempts):
            try:
                return await self._call_provider(client, model, messages, temperature, max_tokens, tools)
            except RateLimitError as exc:
                last_exc = exc
                if _is_daily_quota_error(exc):
                    break  # won't clear within our retry window — fail over immediately
            except APIStatusError as exc:
                if exc.status_code >= 500:
                    last_exc = exc
                else:
                    raise

            if attempt < len(self.RETRY_DELAYS):
                delay = self.RETRY_DELAYS[attempt]
                jitter = delay * random.uniform(-0.2, 0.2)
                wait = max(0.1, delay + jitter)
                logger.warning(
                    "%s error, retry %d/%d in %.1fs",
                    provider_name, attempt + 1, max_attempts - 1, wait,
                )
                await asyncio.sleep(wait)

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
