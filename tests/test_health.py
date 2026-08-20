from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_lifespan_warms_embedding_model(monkeypatch):
    """The nomic-embed model must load during startup, not lazily on the first
    /chat request — a lazy load raced the agent's retry budget and made the first
    rag_retrieval call after a cold start silently return zero chunks."""
    import app.main as main_module

    # database_url/neo4j_uri off so lifespan skips the real Postgres pool and
    # Neo4j driver (unavailable/stubbed in this test environment) and exercises
    # only the embedding warm-up path under test.
    monkeypatch.setattr(main_module.settings, "database_url", "")
    monkeypatch.setattr(main_module.settings, "neo4j_uri", "")

    with patch("ingestion.embed.get_model") as mock_get_model:
        async with main_module.lifespan(main_module.app):
            pass
        mock_get_model.assert_called_once()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_chat_stub():
    response = client.post("/chat", json={"query": "test query"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "session_id" in data
    assert "citations" in data

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "total_queries" in response.json()
    assert "uptime_seconds" in response.json()
