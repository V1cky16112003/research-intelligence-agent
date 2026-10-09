"""Rate limiting, admission control, readiness and request IDs on the API edge."""
import json
import logging

from fastapi.testclient import TestClient

import app.main as main_module


class _FakeRedis:
    def __init__(self, fail: bool = False):
        self.counts: dict[str, int] = {}
        self.fail = fail

    async def incr_window(self, key, ttl):
        if self.fail:
            raise ConnectionError("upstash down")
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def ping(self):
        if self.fail:
            raise ConnectionError("upstash down")
        return True


def _stub_agent(monkeypatch):
    import agent.graph as graph_module

    async def fake_run_agent(user_query, session_id):
        return {"final_report": "ok", "citations": [], "provider": "groq"}

    monkeypatch.setattr(graph_module, "run_agent", fake_run_agent)
    monkeypatch.setattr(main_module.settings, "database_url", "")


def test_rate_limit_uses_redis_and_returns_429(monkeypatch):
    _stub_agent(monkeypatch)
    redis = _FakeRedis()
    monkeypatch.setattr(main_module, "_redis", redis)
    monkeypatch.setattr(main_module.settings, "chat_rate_limit_per_minute", 2)
    client = TestClient(main_module.app)

    codes = [client.post("/chat", json={"query": "q"}).status_code for _ in range(3)]

    assert codes == [200, 200, 429]
    assert list(redis.counts) and all(k.startswith("rl:chat:") for k in redis.counts)


def test_rate_limit_falls_back_to_local_window_when_redis_errors(monkeypatch):
    _stub_agent(monkeypatch)
    monkeypatch.setattr(main_module, "_redis", _FakeRedis(fail=True))
    monkeypatch.setattr(main_module, "_rate_window", {})
    monkeypatch.setattr(main_module.settings, "chat_rate_limit_per_minute", 1)
    client = TestClient(main_module.app)

    first = client.post("/chat", json={"query": "q"})
    second = client.post("/chat", json={"query": "q"})

    assert first.status_code == 200
    assert second.status_code == 429  # a Redis outage must not remove the brake
    assert second.headers["Retry-After"] == "60"


def test_admission_control_sheds_with_503_and_releases_slot(monkeypatch):
    _stub_agent(monkeypatch)
    monkeypatch.setattr(main_module, "_redis", None)
    monkeypatch.setattr(main_module.settings, "chat_rate_limit_per_minute", 0)
    monkeypatch.setattr(main_module.settings, "max_concurrent_chats", 1)
    client = TestClient(main_module.app)

    monkeypatch.setattr(main_module, "_active_chats", 1)
    busy = client.post("/chat", json={"query": "q"})
    assert busy.status_code == 503 and busy.headers["Retry-After"]

    monkeypatch.setattr(main_module, "_active_chats", 0)
    assert client.post("/chat", json={"query": "q"}).status_code == 200
    assert main_module._active_chats == 0  # slot returned after the run


def test_stream_releases_slot_after_completion(monkeypatch):
    import agent.graph as graph_module

    async def fake_stream_agent(user_query, session_id):
        yield {"type": "result", "data": {"final_report": "x", "citations": []}}

    monkeypatch.setattr(graph_module, "stream_agent", fake_stream_agent)
    monkeypatch.setattr(main_module.settings, "database_url", "")
    monkeypatch.setattr(main_module.settings, "chat_rate_limit_per_minute", 0)
    monkeypatch.setattr(main_module, "_active_chats", 0)

    with TestClient(main_module.app).stream("POST", "/chat/stream", json={"query": "q"}) as res:
        "".join(res.iter_text())

    assert main_module._active_chats == 0


def test_answer_is_capped(monkeypatch):
    monkeypatch.setattr(main_module.settings, "max_answer_chars", 10)
    capped = main_module._cap_answer("x" * 50)
    assert capped.startswith("x" * 10) and "truncated" in capped
    assert main_module._cap_answer("short") == "short"


def test_ready_reports_dependencies(monkeypatch):
    monkeypatch.setattr(main_module.settings, "database_url", "")
    monkeypatch.setattr(main_module, "_redis", _FakeRedis(fail=True))
    res = TestClient(main_module.app).get("/ready")
    # Redis down degrades but does not fail readiness; the DB is what's load-bearing.
    assert res.status_code == 200
    assert res.json()["checks"] == {"database": "disabled", "redis": "error: ConnectionError", "active_chats": "0"}


def test_request_id_is_echoed_and_minted():
    client = TestClient(main_module.app)
    assert client.get("/health", headers={"X-Request-ID": "abc123"}).headers["X-Request-ID"] == "abc123"
    assert client.get("/health").headers["X-Request-ID"]


def test_json_log_line_carries_request_id():
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    record.request_id = "rid-1"
    line = json.loads(main_module._JsonFormatter().format(record))
    assert line["msg"] == "hello world" and line["request_id"] == "rid-1"
