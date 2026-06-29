"""
Tests for contextual retrieval components:
  - ingestion/context_generator.py
  - ingestion/embed.py (Chunk.context field + embed_chunks prefix)
  - agent/reranker.py
  - db/queries.py (search_similar_chunks_hybrid SQL structure)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gateway(response_content: str) -> MagicMock:
    """Return a mock LLMGateway whose chat() returns a given content string."""
    gw = MagicMock()
    gw.chat = AsyncMock(return_value={"content": response_content, "cached": False})
    return gw


def _make_chunk(content: str, context: str = "") -> "Chunk":  # noqa: F821
    from ingestion.embed import Chunk
    return Chunk(
        doc_id="1234",
        section_title="abstract",
        chunk_index=0,
        content=content,
        token_count=len(content.split()),
        context=context,
    )


# ---------------------------------------------------------------------------
# 1. generate_context — calls gateway with correct prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_context_calls_gateway():
    """generate_context must include title, abstract, and chunk in the prompt."""
    from ingestion.context_generator import generate_context

    gateway = _make_gateway("This chunk discusses attention mechanisms.")

    result = await generate_context(
        gateway,
        title="Attention Is All You Need",
        abstract="We propose a new model architecture...",
        chunk="The encoder maps an input sequence of symbol representations...",
    )

    assert result == "This chunk discusses attention mechanisms."
    gateway.chat.assert_called_once()
    call_messages = gateway.chat.call_args[1]["messages"]
    prompt_text = call_messages[0]["content"]
    assert "Attention Is All You Need" in prompt_text
    assert "We propose a new model architecture" in prompt_text
    assert "encoder maps an input sequence" in prompt_text


@pytest.mark.asyncio
async def test_generate_context_returns_empty_on_gateway_failure():
    """generate_context must return '' when the gateway raises, not propagate."""
    from ingestion.context_generator import generate_context

    gateway = MagicMock()
    gateway.chat = AsyncMock(side_effect=RuntimeError("gateway down"))

    result = await generate_context(
        gateway,
        title="Paper",
        abstract="Abstract text.",
        chunk="Chunk text.",
    )

    assert result == ""


# ---------------------------------------------------------------------------
# 2. embed_chunks — context prepended to embedding text
# ---------------------------------------------------------------------------

class _FakeEmbeddings:
    """Mimics the ndarray returned by SentenceTransformer.encode()."""
    def __init__(self, data: list[list[float]]):
        self._data = data
    def tolist(self) -> list[list[float]]:
        return self._data


def test_embed_chunks_uses_context_when_present():
    """When chunk.context is set, the embedded text must start with the blurb."""
    from ingestion.embed import embed_chunks

    mock_model = MagicMock()
    mock_model.encode.return_value = _FakeEmbeddings([[0.1] * 768])

    chunk = _make_chunk(content="We present a new architecture.", context="This paper introduces a transformer.")

    with patch("ingestion.embed.get_model", return_value=mock_model):
        embed_chunks([chunk])

    texts_arg = mock_model.encode.call_args[0][0]
    assert len(texts_arg) == 1
    assert texts_arg[0].startswith("search_document: This paper introduces a transformer.\n\n")
    assert "We present a new architecture." in texts_arg[0]


def test_embed_chunks_no_context_uses_plain_text():
    """When chunk.context is empty, embedding must use the plain content prefix."""
    from ingestion.embed import embed_chunks

    mock_model = MagicMock()
    mock_model.encode.return_value = _FakeEmbeddings([[0.2] * 768])

    chunk = _make_chunk(content="Plain content here.", context="")

    with patch("ingestion.embed.get_model", return_value=mock_model):
        embed_chunks([chunk])

    texts_arg = mock_model.encode.call_args[0][0]
    assert texts_arg[0] == "search_document: Plain content here."


# ---------------------------------------------------------------------------
# 3. rerank — parse valid JSON + fallback on bad JSON
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reranker_reorders_by_llm_indices():
    """rerank must reorder candidates according to the LLM's 1-based index array."""
    from agent.reranker import rerank

    candidates = [
        {"content": "chunk A", "id": 1},
        {"content": "chunk B", "id": 2},
        {"content": "chunk C", "id": 3},
    ]
    gateway = _make_gateway("[2, 3, 1]")

    result = await rerank(gateway, query="test query", candidates=candidates, top_k=3)

    assert [r["id"] for r in result] == [2, 3, 1]


@pytest.mark.asyncio
async def test_reranker_returns_top_k():
    """rerank must truncate to top_k even when more candidates are returned."""
    from agent.reranker import rerank

    candidates = [{"content": f"chunk {i}", "id": i} for i in range(5)]
    gateway = _make_gateway("[3, 1, 5, 2, 4]")

    result = await rerank(gateway, query="test", candidates=candidates, top_k=3)

    assert len(result) == 3
    assert result[0]["id"] == 2  # index 3 → 0-based 2


@pytest.mark.asyncio
async def test_reranker_fallback_on_bad_json():
    """rerank must return original order (up to top_k) when JSON parsing fails."""
    from agent.reranker import rerank

    candidates = [{"content": f"chunk {i}", "id": i} for i in range(4)]
    gateway = _make_gateway("sorry I cannot rank these passages")

    result = await rerank(gateway, query="test", candidates=candidates, top_k=4)

    # Original order preserved
    assert [r["id"] for r in result] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_reranker_fallback_on_gateway_error():
    """rerank must return original order when gateway itself raises."""
    from agent.reranker import rerank

    candidates = [{"content": "chunk A", "id": 0}, {"content": "chunk B", "id": 1}]
    gateway = MagicMock()
    gateway.chat = AsyncMock(side_effect=RuntimeError("network error"))

    result = await rerank(gateway, query="test", candidates=candidates, top_k=2)

    assert [r["id"] for r in result] == [0, 1]


# ---------------------------------------------------------------------------
# 4. _to_tsquery_safe — strips punctuation safely
# ---------------------------------------------------------------------------

def test_to_tsquery_safe_strips_punctuation():
    """_to_tsquery_safe must strip punctuation and return joined words."""
    from db.queries import _to_tsquery_safe

    assert _to_tsquery_safe("attention is all you need!") == "attention is all you need"
    assert _to_tsquery_safe("BERT: pre-training...") == "BERT pretraining"
    assert _to_tsquery_safe("  ") == ""
