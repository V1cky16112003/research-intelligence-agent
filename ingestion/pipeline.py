from __future__ import annotations
"""
End-to-end ingestion pipeline.

Usage:
    python -m ingestion.pipeline --limit 50000 --batch-size 100

This script:
1. Streams papers from ArxivAbstractConnector (already in DB via loader.py)
2. Chunks each abstract
3. Embeds chunks in batches
4. Bulk-inserts into chunks table with embeddings
"""
import argparse
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


async def run_pipeline(limit: int = 50_000, batch_size: int = 100) -> dict:
    """
    Run the full ingestion pipeline.
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
    pending_chunks = []
    start = time.time()

    async for doc in connector.fetch_documents(limit=limit):
        chunks = chunk_text(doc.content, doc_id=doc.doc_id)
        if not chunks:
            continue

        # Get paper_id from DB
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT id FROM papers WHERE arxiv_id = %s", (doc.doc_id,))
                row = await cur.fetchone()
                if not row:
                    continue
                paper_id = row[0]

        chunk_embedding_pairs = embed_chunks(chunks)

        for chunk, embedding in chunk_embedding_pairs:
            pending_chunks.append({
                "paper_id": paper_id,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "embedding": embedding,
            })

        total_docs += 1

        if len(pending_chunks) >= batch_size:
            async with get_connection() as conn:
                await insert_chunks_batch(conn, pending_chunks)
                await conn.commit()
            total_chunks += len(pending_chunks)
            pending_chunks = []
            if total_docs % 500 == 0:
                elapsed = time.time() - start
                logger.info("Processed %d docs, %d chunks (%.1fs)", total_docs, total_chunks, elapsed)

    # Flush remaining
    if pending_chunks:
        async with get_connection() as conn:
            await insert_chunks_batch(conn, pending_chunks)
            await conn.commit()
        total_chunks += len(pending_chunks)

    elapsed = time.time() - start
    stats = {"total_docs": total_docs, "total_chunks": total_chunks, "elapsed_seconds": round(elapsed, 1)}
    logger.info("Pipeline complete: %s", stats)
    return stats


def parse_args():
    p = argparse.ArgumentParser(description="Run the embedding ingestion pipeline")
    p.add_argument("--limit", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    asyncio.run(run_pipeline(limit=args.limit, batch_size=args.batch_size))
