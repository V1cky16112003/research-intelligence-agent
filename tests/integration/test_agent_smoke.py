import os

import httpx
import pytest
from pytest_asyncio import fixture

SERVICE_URL = os.getenv("SERVICE_URL", "https://localhost:8000")


@fixture
async def client():
    async with httpx.AsyncClient(base_url=SERVICE_URL, timeout=30.0) as c:
        yield c


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestChat:
    @pytest.mark.asyncio
    async def test_basic_chat(self, client):
        resp = await client.post(
            "/chat",
            json={"query": "What is machine learning?", "session_id": "test-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "citations" in data
        assert "session_id" in data
        assert data["session_id"] == "test-123"
        assert "provider" in data
        assert len(data["answer"]) > 10

    @pytest.mark.asyncio
    async def test_chat_with_sql_tool(self, client):
        resp = await client.post(
            "/chat",
            json={"query": "How many papers in database?", "session_id": "test-sql"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        # SQL tool may or may not be called depending on query

    @pytest.mark.asyncio
    async def test_session_persistence(self, client):
        session_id = "test-session-persist"
        # First query
        r1 = await client.post("/chat", json={"query": "What is deep learning?", "session_id": session_id})
        assert r1.status_code == 200
        # Second query in same session
        r2 = await client.post("/chat", json={"query": "And transformers?", "session_id": session_id})
        assert r2.status_code == 200
        # Both should have same session_id
        assert r1.json()["session_id"] == session_id
        assert r2.json()["session_id"] == session_id


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client):
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_queries" in data
        assert "uptime_seconds" in data
        assert data["total_queries"] >= 0
