from __future__ import annotations
"""
Chunking and embedding pipeline.

Model: nomic-ai/nomic-embed-text-v2-moe (768-dim, CPU-friendly, Apache-2.0)
Strategy: recursive character split at 512 tokens with ~10% overlap (50 chars).
For abstracts (~200 words), most will be a single chunk.
"""
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v2-moe")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


@dataclass
class Chunk:
    """A text chunk ready for embedding."""
    doc_id: str
    section_title: str
    chunk_index: int
    content: str
    token_count: int
    context: str = ""  # LLM-generated situating blurb; prepended before embedding


_model = None  # Lazy-loaded


def get_model():
    """Lazy-load the embedding model (downloads on first call, cached in /tmp)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s ...", EMBED_MODEL)
        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
        logger.info("Embedding model loaded (dim=%d)", EMBED_DIM)
    return _model


def chunk_text(text: str, doc_id: str, section_title: str = "abstract") -> list[Chunk]:
    """
    Split text into chunks using recursive character splitting.
    For abstracts (~200 words), usually returns 1 chunk.
    For longer texts, splits at CHUNK_SIZE chars with CHUNK_OVERLAP overlap.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    chunks = []

    # Simple recursive character split
    if len(text) <= CHUNK_SIZE:
        chunks = [text]
    else:
        # Split into overlapping windows
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            # Try to break at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind(". ")
                if last_period > CHUNK_SIZE // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            chunks.append(chunk.strip())
            start = end - CHUNK_OVERLAP
            if start >= len(text):
                break

    return [
        Chunk(
            doc_id=doc_id,
            section_title=section_title,
            chunk_index=i,
            content=c,
            token_count=len(c.split()),  # word-count approximation
        )
        for i, c in enumerate(chunks)
        if c.strip()
    ]


def embed_chunks(chunks: list[Chunk], batch_size: int = 256) -> list[tuple[Chunk, list[float]]]:
    """
    Embed a list of chunks. Returns (chunk, embedding) pairs.
    nomic-embed requires 'search_document: ' prefix for passages.
    """
    if not chunks:
        return []
    model = get_model()
    texts = [
        f"search_document: {c.context}\n\n{c.content}" if c.context else f"search_document: {c.content}"
        for c in chunks
    ]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)
    return list(zip(chunks, embeddings.tolist()))


def embed_query(query: str) -> list[float]:
    """
    Embed a search query.
    nomic-embed requires 'search_query: ' prefix for queries.
    """
    model = get_model()
    embedding = model.encode(
        [f"search_query: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embedding[0].tolist()
