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


async def run_pipeline(limit: int = 50_000, batch_size: int = 500) -> dict:
    """
    Run the full ingestion pipeline.
    Accumulates `batch_size` papers before embedding to maximise GPU utilisation.
    Returns stats dict: {total_docs, total_chunks, elapsed_seconds}
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

    # Buffer: list of (paper_id, [Chunk, ...])
    buffer: list[tuple[int, list]] = []

    async def flush(buf: list[tuple[int, list]]) -> int:
        """Embed and insert one buffer of (paper_id, chunks) pairs."""
        if not buf:
            return 0
        # Flatten all chunks while tracking which paper each belongs to
        paper_ids_flat = []
        chunks_flat = []
        for pid, chunks in buf:
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
        buffer.append((paper_id, chunks))
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
