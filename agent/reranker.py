from __future__ import annotations
"""
LLM-based reranker for retrieved chunks.

After hybrid retrieval returns N candidates, this module asks the gateway to
rank them by relevance to the original query and returns the top-k in ranked order.
Falls back to the original ordering if the LLM response cannot be parsed.
"""
import json
import logging

logger = logging.getLogger(__name__)

_RERANK_PROMPT = """\
You are a relevance ranking assistant. Given a search query and a list of text passages, \
rank the passages from most to least relevant to the query.

Query: {query}

Passages:
{passages}

Return ONLY a JSON array of 1-based passage indices in ranked order (most relevant first).
Example for 3 passages: [2, 1, 3]
Return nothing else — no explanation, no markdown, just the JSON array."""


async def rerank(
    gateway,
    query: str,
    candidates: list[dict],
    top_k: int = 8,
) -> list[dict]:
    """
    Rerank candidate chunks by relevance to query using the LLM gateway.

    Args:
        gateway: An LLMGateway instance.
        query: The original user query.
        candidates: List of chunk dicts (each must have a "content" key).
        top_k: Number of results to return after reranking.

    Returns:
        Up to top_k chunks in reranked order. Falls back to original order on error.
    """
    if not candidates:
        return candidates

    # Only rerank up to the candidate set; return early if tiny
    if len(candidates) <= 1:
        return candidates[:top_k]

    # Build numbered passage list (first 300 chars each to keep prompt short)
    passage_lines = "\n".join(
        f"[{i + 1}] {c['content'][:300].strip()}"
        for i, c in enumerate(candidates)
    )
    prompt = _RERANK_PROMPT.format(query=query, passages=passage_lines)

    try:
        result = await gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=128,
            cache=True,
        )
        raw = (result.get("content") or "").strip()

        # Parse JSON array of 1-based indices
        indices = json.loads(raw)
        if not isinstance(indices, list):
            raise ValueError(f"Expected list, got {type(indices)}")

        # Deduplicate, clamp to valid range, apply ranking
        seen: set[int] = set()
        reranked: list[dict] = []
        for idx in indices:
            i = int(idx) - 1  # convert to 0-based
            if 0 <= i < len(candidates) and i not in seen:
                seen.add(i)
                reranked.append(candidates[i])

        # Append any candidates the LLM omitted (keeps result count stable)
        for i, c in enumerate(candidates):
            if i not in seen:
                reranked.append(c)

        return reranked[:top_k]

    except Exception as exc:
        logger.warning("Reranking failed (%s), returning original order", exc)
        return candidates[:top_k]
