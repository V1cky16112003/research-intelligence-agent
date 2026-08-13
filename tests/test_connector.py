"""Tests for ingestion/connector.py — ArxivAbstractConnector.fetch_documents."""
from __future__ import annotations

from unittest.mock import patch

import pytest


class _FakeCursor:
    """Mimics a psycopg3 async cursor: records the executed SQL, async-iterates rows."""

    def __init__(self, rows: list[tuple]):
        self._rows = rows
        self.executed_sql: str | None = None
        self.executed_params: tuple | None = None

    async def execute(self, sql: str, params: tuple | None = None) -> None:
        self.executed_sql = sql
        self.executed_params = params

    def __aiter__(self):
        return self._aiter_rows()

    async def _aiter_rows(self):
        for row in self._rows:
            yield row

    async def __aenter__(self) -> "_FakeCursor":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor

    def cursor(self) -> _FakeCursor:
        return self._cursor


class _FakeConnCM:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor

    async def __aenter__(self) -> _FakeConn:
        return _FakeConn(self._cursor)

    async def __aexit__(self, *exc) -> bool:
        return False


@pytest.mark.asyncio
async def test_fetch_documents_query_excludes_already_embedded_papers():
    """The SELECT must skip papers that already have rows in chunks (NOT EXISTS)."""
    from ingestion.connector import ArxivAbstractConnector

    fake_cur = _FakeCursor(rows=[])
    with patch("db.connection.get_connection", return_value=_FakeConnCM(fake_cur)):
        connector = ArxivAbstractConnector()
        docs = [doc async for doc in connector.fetch_documents(limit=30_000)]

    assert docs == []
    assert "NOT EXISTS" in fake_cur.executed_sql
    assert "chunks" in fake_cur.executed_sql
    assert fake_cur.executed_params == (30_000,)


@pytest.mark.asyncio
async def test_fetch_documents_yields_documents_from_rows():
    from ingestion.connector import ArxivAbstractConnector

    rows = [
        (1, "2024.0001", "Paper Title", ["Author A"], ["cs.LG"], "Abstract text"),
    ]
    fake_cur = _FakeCursor(rows=rows)
    with patch("db.connection.get_connection", return_value=_FakeConnCM(fake_cur)):
        connector = ArxivAbstractConnector()
        docs = [doc async for doc in connector.fetch_documents(limit=10)]

    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == "2024.0001"
    assert doc.title == "Paper Title"
    assert doc.authors == ["Author A"]
    assert doc.categories == ["cs.LG"]
    assert doc.content == "Abstract text"
    assert doc.metadata == {"paper_id": 1}
