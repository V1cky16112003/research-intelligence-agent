from __future__ import annotations
"""
End-to-end ingestion pipeline.

Usage:
    python -m ingestion.pipeline --limit 50000 --batch-size 500

    # With contextual retrieval (LLM-generated blurbs):
    GROQ_API_KEY=... GEMINI_API_KEY=... python -m ingestion.pipeline --limit 10000 --contextual

This script:
1. Streams papers from ArxivAbstractConnector (already in DB via loader.py)
2. Optionally generates a contextual blurb per paper via LLMGateway
3. Chunks each abstract
4. Embeds chunks in large batches (GPU-efficient), prepending the blurb when present
5. Bulk-inserts into chunks table with embeddings and context
"""
import argparse
import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)


async def run_pipeline(
    limit: int = 50_000,
    batch_size: int = 500,
    groq_api_key: str | None = None,
    gemini_api_key: str | None = None,
    contextual: bool = False,
) -> dict:
    """
    Run the full ingestion pipeline.
    Accumulates `batch_size` papers before embedding to maximise GPU utilisation.

    Args:
        limit: Maximum number of papers to process.
        batch_size: Papers buffered before a GPU embed+insert flush.
        groq_api_key: Groq API key for context generation (falls back to env var).
        gemini_api_key: Gemini API key (falls back to env var).
        contextual: If True, generate LLM context blurbs for each paper.

    Returns:
        Stats dict: {total_docs, total_chunks, elapsed_seconds}
    """
    from db.connection import init_pool, get_connection
    from db.queries import insert_chunks_batch
    from ingestion.connector import ArxivAbstractConnector
    from ingestion.embed import chunk_text, embed_chunks

    # Resolve API keys from args or environment
    groq_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
    gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")

    # Set up contextual retrieval gateway if requested
    gateway = None
    generate_context_fn = None
    if contextual:
        if not groq_key or not gemini_key:
            raise ValueError(
                "GROQ_API_KEY and GEMINI_API_KEY must be set to use --contextual"
            )
        from agent.gateway import LLMGateway
        from ingestion.context_generator import generate_context
        gateway = LLMGateway(groq_api_key=groq_key, gemini_api_key=gemini_key)
        generate_context_fn = generate_context
        logger.info("Contextual retrieval enabled — will generate LLM blurbs per paper")

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

        # Generate context blurbs (sequential — one LLM call per paper)
        if generate_context_fn and gateway:
            for pid, title, abstract, chunks in buf:
                for chunk in chunks:
                    chunk.context = await generate_context_fn(
                        gateway, title=title, abstract=abstract, chunk=chunk.content
                    )

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
    p.add_argument(
        "--contextual",
        action="store_true",
        default=False,
        help="Generate LLM context blurbs for each chunk (requires GROQ_API_KEY + GEMINI_API_KEY)",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    asyncio.run(run_pipeline(limit=args.limit, batch_size=args.batch_size, contextual=args.contextual))
