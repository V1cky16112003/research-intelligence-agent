from __future__ import annotations
"""
Contextual blurb generator for Anthropic-style Contextual Retrieval.

For each chunk, calls the LLM gateway to produce a short situating sentence
that describes where/how this chunk fits within its parent document. This blurb
is prepended to the chunk text before embedding, boosting retrieval recall.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""
import logging

logger = logging.getLogger(__name__)

CONTEXT_MAX_TOKENS = 100  # blurbs are ~50 tokens; 100 gives headroom

_PROMPT_TEMPLATE = """\
<document>
{abstract}
</document>

Here is the chunk to situate:
<chunk>
{chunk}
</chunk>

Write 1-2 sentences that situate this chunk within the paper titled "{title}". \
Focus on what aspect of the paper this chunk covers. Be concise and do not repeat \
the chunk verbatim."""


async def generate_context(
    gateway,
    title: str,
    abstract: str,
    chunk: str,
) -> str:
    """
    Call LLMGateway to generate a ~50-token situating blurb for a chunk.

    Args:
        gateway: An LLMGateway instance (agent.gateway.LLMGateway).
        title: Paper title.
        abstract: Full abstract text (the parent document).
        chunk: The specific chunk text to situate.

    Returns:
        A short string (1-2 sentences) placing the chunk in context.
        Falls back to an empty string if the gateway call fails.
    """
    prompt = _PROMPT_TEMPLATE.format(
        abstract=abstract,
        chunk=chunk,
        title=title,
    )
    try:
        result = await gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=CONTEXT_MAX_TOKENS,
            cache=True,
        )
        blurb = (result.get("content") or "").strip()
        return blurb
    except Exception as exc:
        logger.warning("Context generation failed, skipping blurb: %s", exc)
        return ""
