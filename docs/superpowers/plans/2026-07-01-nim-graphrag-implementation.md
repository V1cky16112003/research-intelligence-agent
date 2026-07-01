# NIM Provider, Drop LLM Blurbs, GraphRAG Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the LLM-dependent Stage 2b contextual blurb step, add NVIDIA NIM as a third LLM gateway fallback tier, and add an additive Neo4j GraphRAG layer over author/category relationships.

**Architecture:** `agent/gateway.py`'s `LLMGateway` becomes a 3-tier ordered fallback (Groq → NVIDIA NIM → Gemini). `ingestion/context_generator.py` and the `--contextual` pipeline path are deleted. A new `graph/` package syncs `papers` (authors, categories) into Neo4j AuraDB via idempotent MERGE, and a new `graph_query` agent tool answers relational questions using fixed parameterized Cypher templates (no LLM-generated Cypher), registered in `TOOL_DISPATCH` so the existing Planner node can pick it.

**Tech Stack:** Python 3.11, `openai` SDK (NIM is OpenAI-compatible), `neo4j` Python driver, pytest + pytest-asyncio, existing `conftest.py` dependency-stubbing pattern.

---

## Task 1: Add NVIDIA NIM as third gateway provider

**Files:**
- Modify: `agent/gateway.py`
- Test: `tests/test_gateway.py`

- [ ] **Step 1: Write the failing test for the 3-tier fallback chain**

Add to `tests/test_gateway.py` (after `test_groq_429_falls_back_to_gemini`):

```python
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
async def test_all_three_exhausted():
    """Groq, NIM, and Gemini all fail — GatewayExhaustedError raised."""
    gw = LLMGateway(groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake")

    async def always_fail(client, model, messages, temperature, max_tokens, tools, provider_name):
        raise RateLimitError("rate limited", response=MagicMock(status_code=429), body={})

    gw._with_retry = always_fail
    with pytest.raises(GatewayExhaustedError):
        await gw.chat([{"role": "user", "content": "hi"}], cache=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_gateway.py -v`
Expected: `test_groq_and_nim_429_falls_back_to_gemini`, `test_groq_429_falls_back_to_nim`, `test_all_three_exhausted` FAIL with `TypeError: LLMGateway.__init__() got an unexpected keyword argument 'nvidia_api_key'`

- [ ] **Step 3: Update `LLMGateway` to add the NIM tier**

Replace the whole file `agent/gateway.py` with:

```python
from __future__ import annotations
"""
LLM Gateway: Groq (primary) → NVIDIA NIM (2nd fallback) → Gemini 2.5 Flash (3rd fallback).

Features:
- Exponential backoff on 429/5xx: 4 total attempts with delays [1s, 4s, 16s] + ±20% jitter
- Falls back Groq → NVIDIA NIM → Gemini as each tier is exhausted
- Redis response cache (TTL 1hr, SHA256 key on model+messages+temperature+max_tokens+tools)
- Provider tagging on every response for audit logging
- OpenAI-format tool calling passed through unchanged to all three providers
"""
import asyncio
import hashlib
import json
import logging
import random
from typing import Any

from openai import AsyncOpenAI, RateLimitError, APIStatusError

logger = logging.getLogger(__name__)


class GatewayExhaustedError(Exception):
    """Raised when all LLM providers fail after retries."""


class LLMGateway:
    """Routes LLM calls: Groq (primary) → NVIDIA NIM (fallback) → Gemini 2.5 Flash (fallback)."""

    GROQ_MODEL = "llama-3.3-70b-versatile"
    NIM_MODEL = "meta/llama-3.1-70b-instruct"
    GEMINI_MODEL = "gemini-2.5-flash"
    RETRY_DELAYS = [1.0, 4.0, 16.0]

    def __init__(
        self,
        groq_api_key: str,
        nvidia_api_key: str,
        gemini_api_key: str,
        redis_client=None,
    ) -> None:
        self._groq = AsyncOpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._nim = AsyncOpenAI(
            api_key=nvidia_api_key,
            base_url="https://integrate.api.nvidia.com/v1",
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
                    self._nim, self.NIM_MODEL, messages, temperature, max_tokens, tools, "nvidia_nim"
                )
                result["provider"] = "nvidia_nim"
                result["model"] = self.NIM_MODEL
            except Exception as exc2:
                nim_exc = exc2
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
```

- [ ] **Step 4: Update existing `test_gateway.py` calls that construct `LLMGateway` without `nvidia_api_key`**

The existing tests (`test_groq_success`, `test_groq_429_falls_back_to_gemini`, `test_cache_hit`, `test_both_exhausted`) all call `LLMGateway(groq_api_key="fake", gemini_api_key="fake")`. Update every such call in `tests/test_gateway.py` to `LLMGateway(groq_api_key="fake", nvidia_api_key="fake", gemini_api_key="fake")`.

Also update `test_groq_429_falls_back_to_gemini`'s `mock_retry` to account for the new NIM hop — since it only defines behavior for `provider_name == "groq"` vs else, and NIM will now be tried before Gemini, rename `test_groq_429_falls_back_to_gemini` to `test_groq_and_nim_429_falls_back_to_gemini` and merge it with the version added in Step 1 (delete the duplicate — keep the Step 1 version, remove the old `test_groq_429_falls_back_to_gemini`). Also rename `test_both_exhausted` to `test_all_three_exhausted` and merge with the Step 1 version the same way (delete the older, weaker two-provider version).

- [ ] **Step 5: Run full gateway test suite**

Run: `PYTHONPATH=. pytest tests/test_gateway.py -v`
Expected: All tests PASS (7 tests: `test_groq_success`, `test_groq_and_nim_429_falls_back_to_gemini`, `test_groq_429_falls_back_to_nim`, `test_cache_hit`, `test_all_three_exhausted`, `test_redis_client_detection`)

- [ ] **Step 6: Wire `NVIDIA_NIM_API_KEY` into `Settings` and gateway construction**

In `app/main.py`, modify the `Settings` class (currently at lines ~17-38) to add the new field after `gemini_api_key`:

```python
    groq_api_key: str = ""
    nvidia_nim_api_key: str = ""
    gemini_api_key: str = ""
```

Modify the gateway construction in `lifespan()` (currently):

```python
    gateway = LLMGateway(
        groq_api_key=settings.groq_api_key,
        gemini_api_key=settings.gemini_api_key,
        redis_client=redis_client,
    )
```

to:

```python
    gateway = LLMGateway(
        groq_api_key=settings.groq_api_key,
        nvidia_api_key=settings.nvidia_nim_api_key,
        gemini_api_key=settings.gemini_api_key,
        redis_client=redis_client,
    )
```

- [ ] **Step 7: Run the full test suite to check nothing else broke**

Run: `PYTHONPATH=. pytest tests/ -v --tb=short`
Expected: All tests PASS except pre-existing skip (duckduckgo_search)

- [ ] **Step 8: Commit**

```bash
git add agent/gateway.py tests/test_gateway.py app/main.py
git commit -m "feat: add NVIDIA NIM as third LLM gateway fallback tier"
```

---

## Task 2: Remove LLM-generated contextual blurb step (Stage 2b)

**Files:**
- Delete: `ingestion/context_generator.py`
- Modify: `ingestion/pipeline.py`
- Delete tests covering removed behavior in: `tests/test_contextual_retrieval.py`

- [ ] **Step 1: Delete the context generator module**

```bash
git rm ingestion/context_generator.py
```

- [ ] **Step 2: Remove Stage 2b logic from `ingestion/pipeline.py`**

Replace the whole file `ingestion/pipeline.py` with:

```python
from __future__ import annotations
"""
End-to-end ingestion pipeline.

Usage:
    python -m ingestion.pipeline --limit 50000 --batch-size 500

This script:
1. Streams papers from ArxivAbstractConnector (already in DB via loader.py)
2. Chunks each abstract
3. Embeds chunks in large batches (GPU-efficient)
4. Bulk-inserts into chunks table with embeddings
"""
import argparse
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


async def run_pipeline(
    limit: int = 50_000,
    batch_size: int = 500,
) -> dict:
    """
    Run the full ingestion pipeline.
    Accumulates `batch_size` papers before embedding to maximise GPU utilisation.

    Args:
        limit: Maximum number of papers to process.
        batch_size: Papers buffered before a GPU embed+insert flush.

    Returns:
        Stats dict: {total_docs, total_chunks, elapsed_seconds}
    """
    from db.connection import init_pool, get_connection
    from db.queries import insert_chunks_batch
    from ingestion.connector import ArxivAbstractConnector
    from ingestion.embed import chunk_text, embed_chunks

    await init_pool()

    connector = ArxivAbstractConnector()
    total_docs = 0
    total_chunks = 0
    start = time.time()

    # Buffer: list of (paper_id, title, abstract, [Chunk, ...])
    buffer: list[tuple[int, str, str, list]] = []

    async def flush(buf: list[tuple[int, str, str, list]]) -> int:
        """Embed and insert one buffer of (paper_id, title, abstract, chunks) tuples."""
        if not buf:
            return 0

        # Flatten all chunks
        paper_ids_flat = []
        chunks_flat = []
        for pid, _title, _abstract, chunks in buf:
            for c in chunks:
                paper_ids_flat.append(pid)
                chunks_flat.append(c)

        pairs = embed_chunks(chunks_flat)

        rows = [
            {
                "paper_id": paper_ids_flat[i],
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "context": chunk.context,
                "token_count": chunk.token_count,
                "embedding": emb,
            }
            for i, (chunk, emb) in enumerate(pairs)
        ]

        async with get_connection() as conn:
            await insert_chunks_batch(conn, rows)
            await conn.commit()

        return len(rows)

    async for doc in connector.fetch_documents(limit=limit):
        chunks = chunk_text(doc.content, doc_id=doc.doc_id)
        if not chunks:
            continue

        paper_id = doc.metadata["paper_id"]
        title = doc.metadata.get("title", "")
        abstract = doc.content  # connector yields the abstract as doc.content
        buffer.append((paper_id, title, abstract, chunks))
        total_docs += 1

        if len(buffer) >= batch_size:
            n = await flush(buffer)
            total_chunks += n
            buffer = []
            elapsed = time.time() - start
            rate = total_docs / elapsed * 60
            logger.info(
                "Processed %d docs | %d chunks | %.0f docs/min | %.1fs elapsed",
                total_docs, total_chunks, rate, elapsed,
            )

    # Flush remainder
    n = await flush(buffer)
    total_chunks += n

    elapsed = time.time() - start
    stats = {"total_docs": total_docs, "total_chunks": total_chunks, "elapsed_seconds": round(elapsed, 1)}
    logger.info("Pipeline complete: %s", stats)
    return stats


def parse_args():
    p = argparse.ArgumentParser(description="Run the embedding ingestion pipeline")
    p.add_argument("--limit", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=500)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    asyncio.run(run_pipeline(limit=args.limit, batch_size=args.batch_size))
```

Note: `Chunk.context` field and the `insert_chunks_batch`/schema `context` column stay as-is per the spec — `chunk.context` will just always be `""` (its dataclass default) going forward, which `insert_chunks_batch` already stores as `NULL`.

- [ ] **Step 3: Remove the now-dead context-generation tests from `tests/test_contextual_retrieval.py`**

Remove these two test functions entirely (they test the deleted `ingestion/context_generator.py`):
- `test_generate_context_calls_gateway`
- `test_generate_context_returns_empty_on_gateway_failure`

Also remove the now-unused `_make_gateway` helper's only remaining callers check — `_make_gateway` is still used by the reranker tests later in the file, so **keep** `_make_gateway`. Only delete the two `generate_context` test functions and their leading `# 1. generate_context — calls gateway with correct prompt` section comment block.

Update the file's module docstring at the top from:

```python
"""
Tests for contextual retrieval components:
  - ingestion/context_generator.py
  - ingestion/embed.py (Chunk.context field + embed_chunks prefix)
  - agent/reranker.py
  - db/queries.py (search_similar_chunks_hybrid SQL structure)
"""
```

to:

```python
"""
Tests for hybrid retrieval components:
  - ingestion/embed.py (Chunk.context field + embed_chunks prefix — context is
    always empty now that Stage 2b LLM blurb generation has been removed, but
    embed_chunks must still handle a populated context field for forward
    compatibility with the chunks.context column)
  - agent/reranker.py
  - db/queries.py (search_similar_chunks_hybrid SQL structure)
"""
```

- [ ] **Step 4: Run the updated test file**

Run: `PYTHONPATH=. pytest tests/test_contextual_retrieval.py -v`
Expected: All remaining tests PASS (embed_chunks context tests, reranker tests, `_to_tsquery_safe` test)

- [ ] **Step 5: Search for any other references to the deleted module/flag**

Run: `grep -rn "context_generator\|--contextual\|generate_context\b" --include=*.py --include=*.md --include=*.ipynb .`

Expected output: only matches in `CLAUDE.md` and `kaggle_ingestion.ipynb` (handled in Task 4) — no remaining `.py` references outside what was already removed. If any `.py` file still references `context_generator` or the `contextual=` pipeline arg, fix it now before proceeding.

- [ ] **Step 6: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v --tb=short`
Expected: All tests PASS except pre-existing skip (duckduckgo_search)

- [ ] **Step 7: Commit**

```bash
git add ingestion/pipeline.py tests/test_contextual_retrieval.py
git rm ingestion/context_generator.py
git commit -m "refactor: remove LLM-generated contextual blurb step (Stage 2b)"
```

---

## Task 3: Neo4j client and graph sync script

**Files:**
- Create: `graph/__init__.py`
- Create: `graph/neo4j_client.py`
- Create: `graph/graph_sync.py`
- Test: `tests/test_graph.py`
- Modify: `requirements.txt`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add the `neo4j` driver to requirements**

In `requirements.txt`, add a new line after the `redis>=5.0.8` line:

```
neo4j>=5.24.0
```

- [ ] **Step 2: Stub the `neo4j` package in `conftest.py` so tests run without a live driver**

In `tests/conftest.py`, add after the `torch / sentence_transformers` block (after `_st.SentenceTransformer = MagicMock()`):

```python

# --- neo4j ---
_neo4j = _make_module("neo4j")
_neo4j.AsyncGraphDatabase = MagicMock()
_neo4j.AsyncDriver = MagicMock
```

- [ ] **Step 3: Write the failing test for `neo4j_client.py`**

Create `tests/test_graph.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'graph'`

- [ ] **Step 5: Create `graph/__init__.py`**

```bash
mkdir -p "graph"
```

Create `graph/__init__.py` (empty file):

```python
```

- [ ] **Step 6: Create `graph/neo4j_client.py`**

```python
from __future__ import annotations
"""
Thin async wrapper around the Neo4j Python driver.

Mirrors db/connection.py's singleton-pool pattern: one driver instance per
process, created lazily on first use and reused thereafter.
"""
import logging
import os

from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)

_driver = None


def get_driver(
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
):
    """Return the singleton Neo4j async driver, creating it on first call."""
    global _driver
    if _driver is None:
        resolved_uri = uri or os.getenv("NEO4J_URI", "")
        resolved_user = user or os.getenv("NEO4J_USER", "")
        resolved_password = password or os.getenv("NEO4J_PASSWORD", "")
        if not resolved_uri:
            raise ValueError("NEO4J_URI must be set to use the graph layer")
        _driver = AsyncGraphDatabase.driver(
            resolved_uri, auth=(resolved_user, resolved_password)
        )
        logger.info("Neo4j driver initialized")
    return _driver


async def close_driver() -> None:
    """Close the driver, if open. Call on app shutdown."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
```

- [ ] **Step 7: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_graph.py -v`
Expected: `test_get_driver_creates_singleton` PASSES

- [ ] **Step 8: Write the failing test for graph sync idempotency**

Add to `tests/test_graph.py`:

```python
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
```

- [ ] **Step 9: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'graph.graph_sync'`

- [ ] **Step 10: Create `graph/graph_sync.py`**

```python
from __future__ import annotations
"""
Sync papers (authors, categories) from Postgres into Neo4j AuraDB.

Builds:
  (:Paper {arxiv_id, title})-[:AUTHORED_BY]->(:Author {name})
  (:Paper {arxiv_id, title})-[:HAS_CATEGORY]->(:Category {name})

Idempotent: uses MERGE so re-running after new papers are ingested does not
duplicate nodes or relationships. Safe to run repeatedly (e.g. after each
ingestion batch, or on a schedule).

Usage:
    DATABASE_URL=... NEO4J_URI=... NEO4J_USER=... NEO4J_PASSWORD=... \
        python -m graph.graph_sync --limit 10000
"""
import argparse
import asyncio
import logging

logger = logging.getLogger(__name__)

_SYNC_CYPHER = """
MERGE (p:Paper {arxiv_id: $arxiv_id})
SET p.title = $title
WITH p
UNWIND $authors AS author_name
MERGE (a:Author {name: author_name})
MERGE (p)-[:AUTHORED_BY]->(a)
WITH p
UNWIND $categories AS category_name
MERGE (c:Category {name: category_name})
MERGE (p)-[:HAS_CATEGORY]->(c)
"""


async def sync_papers_to_graph(driver, papers: list[dict]) -> int:
    """
    Sync a list of paper dicts into Neo4j.

    Args:
        driver: A Neo4j AsyncDriver (from graph.neo4j_client.get_driver()).
        papers: List of dicts with keys: arxiv_id, title, authors, categories.

    Returns:
        Number of papers synced.
    """
    count = 0
    async with driver.session() as session:
        for paper in papers:
            await session.run(
                _SYNC_CYPHER,
                {
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors") or [],
                    "categories": paper.get("categories") or [],
                },
            )
            count += 1
    return count


async def run_sync(limit: int = 50_000) -> dict:
    """Fetch papers from Postgres and sync them into Neo4j."""
    from db.connection import init_pool, get_connection
    from graph.neo4j_client import get_driver

    await init_pool()
    driver = get_driver()

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT arxiv_id, title, authors, categories FROM papers LIMIT %s",
                (limit,),
            )
            col_names = [d[0] for d in cur.description]
            rows = await cur.fetchall()
            papers = [dict(zip(col_names, row)) for row in rows]

    count = await sync_papers_to_graph(driver, papers)
    logger.info("Synced %d papers to Neo4j", count)
    return {"synced": count}


def parse_args():
    p = argparse.ArgumentParser(description="Sync papers table into Neo4j graph")
    p.add_argument("--limit", type=int, default=50_000)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    asyncio.run(run_sync(limit=args.limit))
```

- [ ] **Step 11: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_graph.py -v`
Expected: All 3 tests PASS

- [ ] **Step 12: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v --tb=short`
Expected: All tests PASS except pre-existing skip

- [ ] **Step 13: Commit**

```bash
git add graph/ tests/test_graph.py tests/conftest.py requirements.txt
git commit -m "feat: add Neo4j client and Postgres-to-graph sync script"
```

---

## Task 4: `graph_query` agent tool with fixed Cypher templates

**Files:**
- Modify: `agent/tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Read the existing `tests/test_tools.py` to match its mocking conventions**

Run: `sed -n '1,40p' tests/test_tools.py`

(Use this to confirm how `rag_retrieval_tool`/`sql_analytics_tool` tests mock `db.connection.get_connection` — the `graph_query_tool` test below follows the same `patch()` style.)

- [ ] **Step 2: Write the failing test for `graph_query_tool`**

Add to `tests/test_tools.py`:

```python
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
```

Confirm `tests/test_tools.py` already has `import json`, `from unittest.mock import AsyncMock, MagicMock, patch`, and `import pytest` at the top — if any are missing, add them.

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_tools.py -v -k graph_query`
Expected: FAIL with `ImportError: cannot import name 'graph_query_tool' from 'agent.tools'`

- [ ] **Step 4: Add `graph_query_tool` and its Cypher templates to `agent/tools.py`**

In `agent/tools.py`, add after the `web_search_tool` function (before the `# OpenAI-format tool definitions` comment):

```python
# Fixed, parameterized Cypher templates — deliberately not LLM-generated, so a
# malformed or unbounded query can never reach the graph database.
_GRAPH_CYPHER_TEMPLATES = {
    "papers_by_author": (
        "MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author {name: $value}) "
        "RETURN p.arxiv_id AS arxiv_id, p.title AS title LIMIT 20"
    ),
    "papers_by_category": (
        "MATCH (p:Paper)-[:HAS_CATEGORY]->(c:Category {name: $value}) "
        "RETURN p.arxiv_id AS arxiv_id, p.title AS title LIMIT 20"
    ),
    "coauthors": (
        "MATCH (:Author {name: $value})<-[:AUTHORED_BY]-(:Paper)-[:AUTHORED_BY]->(a:Author) "
        "WHERE a.name <> $value "
        "RETURN DISTINCT a.name AS name LIMIT 20"
    ),
}


async def graph_query_tool(query_type: str, value: str) -> str:
    """
    Answer relational questions (co-authorship, shared subfields) using the
    Neo4j knowledge graph built from paper authors/categories.

    Args:
        query_type: One of: 'papers_by_author', 'papers_by_category', 'coauthors'
        value: The author name or category code to query for.

    Returns:
        JSON string with list of results and their metadata.
    """
    from graph.neo4j_client import get_driver

    cypher = _GRAPH_CYPHER_TEMPLATES.get(query_type)
    if cypher is None:
        return json.dumps({
            "tool": "graph_query",
            "error": f"Unknown query_type: {query_type}. Valid: {list(_GRAPH_CYPHER_TEMPLATES)}",
            "results": [],
        })

    try:
        driver = get_driver()
        async with driver.session() as session:
            result = await session.run(cypher, {"value": value})
            records = await result.data()

        return json.dumps({
            "tool": "graph_query",
            "query_type": query_type,
            "value": value,
            "results": records,
            "count": len(records),
        }, default=str)
    except Exception as e:
        logger.error("Graph query failed: %s", e)
        return json.dumps({"tool": "graph_query", "error": str(e), "results": []})
```

- [ ] **Step 5: Register `graph_query` in `TOOL_DEFINITIONS` and `TOOL_DISPATCH`**

In `agent/tools.py`, add a new entry to `TOOL_DEFINITIONS` (the list literal near the bottom of the file), after the `web_search` definition:

```python
    {
        "type": "function",
        "function": {
            "name": "graph_query",
            "description": "Query the paper knowledge graph for relational questions: what else an author has written, co-authorship, or papers sharing a subfield/category. Use for questions like 'what else has this author published' or 'who are this author's collaborators', not for content/topic search (use rag_retrieval for that).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["papers_by_author", "papers_by_category", "coauthors"],
                        "description": "papers_by_author: papers written by a given author. papers_by_category: papers in a given ArXiv category. coauthors: other authors who have co-written a paper with the given author.",
                    },
                    "value": {
                        "type": "string",
                        "description": "The author name (for papers_by_author/coauthors) or category code like 'cs.LG' (for papers_by_category)",
                    },
                },
                "required": ["query_type", "value"],
            },
        },
    },
```

Update `TOOL_DISPATCH` from:

```python
TOOL_DISPATCH: dict[str, Callable] = {
    "rag_retrieval": rag_retrieval_tool,
    "sql_analytics": sql_analytics_tool,
    "web_search": web_search_tool,
}
```

to:

```python
TOOL_DISPATCH: dict[str, Callable] = {
    "rag_retrieval": rag_retrieval_tool,
    "sql_analytics": sql_analytics_tool,
    "web_search": web_search_tool,
    "graph_query": graph_query_tool,
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_tools.py -v -k graph_query`
Expected: All 5 tests PASS

- [ ] **Step 7: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v --tb=short`
Expected: All tests PASS except pre-existing skip

- [ ] **Step 8: Commit**

```bash
git add agent/tools.py tests/test_tools.py
git commit -m "feat: add graph_query agent tool with fixed Cypher templates"
```

---

## Task 5: Wire Neo4j settings and driver lifecycle into the app

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: Add Neo4j settings fields**

In `app/main.py`, extend `Settings` (same block edited in Task 1 Step 6) to add after `embed_dim`:

```python
    embed_model: str = "nomic-ai/nomic-embed-text-v2-moe"
    embed_dim: int = 768
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""
```

- [ ] **Step 2: Initialize the Neo4j driver at startup and close it at shutdown**

In `app/main.py`'s `lifespan()` function, after the `# LangGraph agent` block (`await init_graph()`), add:

```python
    # Neo4j graph driver (optional — graph_query tool degrades gracefully if unset)
    if settings.neo4j_uri:
        from graph.neo4j_client import get_driver
        get_driver(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password)
        logger.info("Neo4j graph driver ready")
    else:
        logger.warning("NEO4J_URI not set — graph_query tool will error gracefully if invoked")
```

Find the shutdown section (after `yield` in `lifespan()`) and add driver cleanup:

```python
    # Shutdown
    if settings.neo4j_uri:
        from graph.neo4j_client import close_driver
        await close_driver()
```

(Add this alongside whatever other shutdown cleanup already exists after `yield` — check the current contents of that section first with `sed -n '/# Shutdown/,$p' app/main.py` before editing, and place the Neo4j cleanup as an additional block rather than replacing existing shutdown code.)

- [ ] **Step 3: Run the full test suite**

Run: `PYTHONPATH=. pytest tests/ -v --tb=short`
Expected: All tests PASS except pre-existing skip

- [ ] **Step 4: Manually verify the app still boots without Neo4j configured**

Run: `PYTHONPATH=. python -c "from app.main import app; print('OK')"`
Expected: Prints `OK` with no exception (Neo4j absence must not crash startup — `neo4j_uri` defaults to `""`, so the driver init is skipped)

- [ ] **Step 5: Commit**

```bash
git add app/main.py
git commit -m "feat: wire Neo4j driver lifecycle and settings into app startup"
```

---

## Task 6: Update documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `kaggle_ingestion.ipynb` (remove contextual retrieval section reference, if present)

- [ ] **Step 1: Update the Commands section in `CLAUDE.md`**

Remove the `--contextual` example command block and the "Apply contextual retrieval schema migration" line's framing as still-relevant-to-run-with-contextual (the migration itself stays — `context`/`content_tsv` columns remain in schema per the spec — but the `--contextual` ingestion example is deleted). Find the block:

```
# Stage 2b (optional): re-embed with LLM-generated contextual blurbs (Anthropic Contextual Retrieval)
# Requires migration below applied first. Groq free tier is 100k tokens/day; Gemini free
# tier is 5 req/min — budget accordingly (see kaggle_ingestion.ipynb rate-limit notes)
GROQ_API_KEY="..." GEMINI_API_KEY="..." DATABASE_URL="..." PYTHONPATH=. python3 -m ingestion.pipeline \
  --limit 10000 --batch-size 200 --contextual

# Apply contextual retrieval schema migration (context column, tsvector, GIN index)
psql $DATABASE_URL -f db/migrations/001_contextual_retrieval.sql
```

Replace with:

```
# Apply hybrid search schema migration (tsvector + GIN index for BM25)
psql $DATABASE_URL -f db/migrations/001_contextual_retrieval.sql

# Sync papers (authors, categories) into the Neo4j knowledge graph
NEO4J_URI="..." NEO4J_USER="..." NEO4J_PASSWORD="..." DATABASE_URL="..." PYTHONPATH=. python3 -m graph.graph_sync --limit 10000
```

- [ ] **Step 2: Update the LangGraph Agent section**

In the `agent/gateway.py` bullet, change:
```
- `gateway.py` — `LLMGateway`: Groq (Llama 3.3 70B) primary, Gemini 2.5 Flash fallback on 429; wraps Upstash Redis cache
```
to:
```
- `gateway.py` — `LLMGateway`: Groq (Llama 3.3 70B) primary → NVIDIA NIM (Llama 3.1 70B) → Gemini 2.5 Flash, cascading fallback on 429/5xx; wraps Upstash Redis cache
```

Add a new bullet after the `tools.py` line describing `graph_query`:
```
- `tools.py` — `TOOL_DISPATCH` dict mapping tool name → async function: `rag_retrieval`, `sql_analytics`, `web_search`, `graph_query`. `rag_retrieval_tool` runs hybrid search (vector + BM25 RRF, 16 candidates) then LLM reranks down to top 8. `graph_query_tool` answers relational questions (co-authorship, shared subfields) via fixed parameterized Cypher templates against Neo4j — never LLM-generated Cypher.
```
(replacing the existing `tools.py` bullet, which currently ends at `rerank down to top 8`.)

- [ ] **Step 3: Add a new "Knowledge Graph (`graph/`)" subsection to the Architecture section**

Add after the "### Ingestion (`ingestion/`)" section and before "### Evaluation (`eval/`)":

```markdown
### Knowledge Graph (`graph/`)

Additive layer over the existing vector/BM25 hybrid retrieval — not a replacement.
`neo4j_client.py` owns a singleton async Neo4j driver (same lazy-singleton pattern
as `db/connection.py`'s pool). `graph_sync.py` reads `papers` (arxiv_id, title,
authors, categories) and MERGEs `(:Paper)-[:AUTHORED_BY]->(:Author)` and
`(:Paper)-[:HAS_CATEGORY]->(:Category)` into Neo4j AuraDB (free tier);
idempotent, safe to re-run after new ingestion batches. No citation graph —
the ArXiv metadata snapshot has no reliable reference data.

The `graph_query` agent tool answers relational questions ("what else has this
author published," "who are their co-authors," "what's in this category") via
a small fixed set of parameterized Cypher templates in `agent/tools.py` — not
LLM-generated Cypher, so a malformed or unbounded query can never reach the
database. The Planner node picks `graph_query` vs `rag_retrieval` the same way
it already picks among the other three tools.
```

- [ ] **Step 4: Remove Stage 2b / contextual re-embed documentation from the Ingestion section**

Find and delete the paragraph starting with `**Contextual re-embed (notebook, \`contextual=True\`):**` in the `### Ingestion (\`ingestion/\`)` section of `CLAUDE.md`.

- [ ] **Step 5: Update the Tests section**

Find:
```
`test_contextual_retrieval.py` covers context generation, embed prefix logic, reranker ordering/fallback, and BM25 query sanitization.
```
Replace with:
```
`test_contextual_retrieval.py` covers embed prefix logic, reranker ordering/fallback, and BM25 query sanitization. `test_graph.py` covers the Neo4j driver singleton and graph sync idempotency. `test_gateway.py` covers the 3-tier Groq → NVIDIA NIM → Gemini fallback chain.
```

- [ ] **Step 6: Update the Key Environment Variables section**

Find:
```
GROQ_API_KEY              # Primary LLM
GEMINI_API_KEY            # Fallback LLM + RAGAS judge
```
Replace with:
```
GROQ_API_KEY              # Primary LLM
NVIDIA_NIM_API_KEY        # Second-tier LLM fallback
GEMINI_API_KEY            # Third-tier LLM fallback + RAGAS judge
NEO4J_URI                 # Neo4j AuraDB connection URI (graph_query tool)
NEO4J_USER                # Neo4j AuraDB username
NEO4J_PASSWORD            # Neo4j AuraDB password
```

- [ ] **Step 7: Check `kaggle_ingestion.ipynb` for stale Stage 2b references**

Run: `grep -n "contextual\|context_generator" kaggle_ingestion.ipynb`

If matches are found, note them for the user — do not edit the notebook automatically in this task (notebooks are hand-curated for Colab/Kaggle and out of scope for this plan's automated edits). Report the line/cell numbers found so the user can decide whether to update the notebook separately.

- [ ] **Step 8: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for NIM fallback tier, dropped Stage 2b, GraphRAG layer"
```

---

## Final Verification

- [ ] **Step 1: Run lint**

Run: `ruff check . --ignore E501,E402`
Expected: No errors (or only pre-existing ones unrelated to this change)

- [ ] **Step 2: Run the full test suite one more time**

Run: `PYTHONPATH=. pytest tests/ -v --tb=short`
Expected: All tests PASS except the pre-existing duckduckgo_search skip

- [ ] **Step 3: Confirm no dangling references to removed code**

Run: `grep -rn "context_generator\|--contextual\b" --include=*.py .`
Expected: No output
