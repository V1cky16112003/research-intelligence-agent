from __future__ import annotations

"""
Source connector interface and ArXiv abstract implementation.

The SourceConnector interface makes ingestion source-agnostic.
ArxivAbstractConnector uses already-loaded papers from the DB
(via ingestion/loader.py) and yields their abstracts as documents.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class Document:
    """Canonical document representation."""
    doc_id: str          # arxiv_id
    title: str
    authors: list[str]
    categories: list[str]
    content: str         # abstract text
    metadata: dict       # arbitrary extra fields


class SourceConnector(ABC):
    """Abstract base for all data sources."""

    @abstractmethod
    async def fetch_documents(self, limit: int = 50_000) -> AsyncIterator[Document]:
        """Yield canonical Documents from the source."""
        ...


class ArxivAbstractConnector(SourceConnector):
    """
    Fetches papers already loaded into the papers table
    and yields their abstracts as Documents.

    This is the v1 connector — no PDF download needed.
    Docling full-PDF connector is a stretch goal.
    """

    async def fetch_documents(self, limit: int = 50_000) -> AsyncIterator[Document]:
        """Stream papers from DB and yield as Documents.

        Skips papers that already have chunks so repeated pipeline runs are
        additive (safe to call again with a higher limit to embed more of
        the corpus) instead of re-embedding and duplicating existing rows.
        """
        from db.connection import get_connection
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT p.id, p.arxiv_id, p.title, p.authors, p.categories, p.abstract
                    FROM papers p
                    WHERE p.abstract IS NOT NULL AND p.abstract != ''
                      AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.paper_id = p.id)
                    ORDER BY p.published_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                async for row in cur:
                    yield Document(
                        doc_id=row[1],
                        title=row[2],
                        authors=row[3] or [],
                        categories=row[4] or [],
                        content=row[5],
                        metadata={"paper_id": row[0]},
                    )
