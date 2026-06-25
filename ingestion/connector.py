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
        """Stream papers from DB and yield as Documents."""
        from db.connection import get_connection
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT arxiv_id, title, authors, categories, abstract
                    FROM papers
                    WHERE abstract IS NOT NULL AND abstract != ''
                    ORDER BY published_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                async for row in cur:
                    yield Document(
                        doc_id=row[0],
                        title=row[1],
                        authors=row[2] or [],
                        categories=row[3] or [],
                        content=row[4],
                        metadata={},
                    )
