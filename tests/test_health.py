from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

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
