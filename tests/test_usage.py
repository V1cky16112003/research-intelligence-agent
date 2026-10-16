"""GET /usage: read-only daily-question budget."""
import time

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from tests.test_auth import SUPABASE, es256  # noqa: F401


class _Redis:
    def __init__(self, count=None, ttl=3600, fail=False):
        self.count, self._ttl, self.fail = count, ttl, fail
        self.writes = 0

    async def get(self, key):
        if self.fail:
            raise ConnectionError("upstash down")
        return None if self.count is None else str(self.count)

    async def ttl(self, key):
        return self._ttl

    async def incr_window(self, key, ttl):
        self.writes += 1
        return 1


@pytest.fixture
def api(monkeypatch, es256):  # noqa: F811
    monkeypatch.setattr(main_module.settings, "supabase_url", SUPABASE)
    monkeypatch.setattr(main_module.settings, "user_daily_query_limit", 100)
    monkeypatch.setattr(main_module.settings, "chat_rate_limit_per_minute", 20)
    monkeypatch.setattr(main_module, "_redis", None)
    monkeypatch.setattr(main_module, "_rate_window", {})
    return TestClient(main_module.app), {"Authorization": f"Bearer {es256()}"}


def test_reports_usage_from_redis(api, monkeypatch):
    client, hdr = api
    monkeypatch.setattr(main_module, "_redis", _Redis(count=27, ttl=21540))
    assert client.get("/usage", headers=hdr).json() == {
        "enabled": True, "limit": 100, "used": 27, "remaining": 73,
        "resets_in_seconds": 21540, "per_minute_limit": 20,
    }


def test_missing_key_means_unused_full_window(api, monkeypatch):
    client, hdr = api
    monkeypatch.setattr(main_module, "_redis", _Redis(count=None, ttl=-2))
    body = client.get("/usage", headers=hdr).json()
    assert body["used"] == 0 and body["remaining"] == 100 and body["resets_in_seconds"] == 86_400


def test_does_not_consume_quota(api, monkeypatch):
    client, hdr = api
    redis = _Redis(count=5)
    monkeypatch.setattr(main_module, "_redis", redis)
    first = client.get("/usage", headers=hdr).json()
    second = client.get("/usage", headers=hdr).json()
    assert first["used"] == second["used"] == 5
    assert redis.writes == 0 and main_module._rate_window == {}


def test_requires_token_when_auth_enabled(api):
    client, _ = api
    res = client.get("/usage")
    assert res.status_code == 401 and res.headers["WWW-Authenticate"] == "Bearer"


def test_disabled_without_auth_or_with_no_limit(api, monkeypatch):
    client, hdr = api
    monkeypatch.setattr(main_module.settings, "user_daily_query_limit", 0)
    assert client.get("/usage", headers=hdr).json() == {"enabled": False}
    monkeypatch.setattr(main_module.settings, "user_daily_query_limit", 100)
    monkeypatch.setattr(main_module.settings, "supabase_url", "")
    assert client.get("/usage").json() == {"enabled": False}


def test_redis_failure_falls_back_to_in_process_window(api, monkeypatch):
    client, hdr = api
    monkeypatch.setattr(main_module, "_redis", _Redis(fail=True))
    main_module._rate_window["quota:day:user-1"] = (4, time.time() - 600)
    body = client.get("/usage", headers=hdr).json()
    assert body["enabled"] and body["used"] == 4 and body["remaining"] == 96
    assert 86_400 - 700 < body["resets_in_seconds"] <= 86_400 - 590


def test_remaining_clamped_at_zero_when_over_limit(api, monkeypatch):
    client, hdr = api
    monkeypatch.setattr(main_module, "_redis", _Redis(count=101, ttl=10))
    body = client.get("/usage", headers=hdr).json()
    assert body["used"] == 101 and body["remaining"] == 0


def test_cors_allows_get_with_authorization_header(api):
    client, _ = api
    origin = main_module.settings.get_allowed_origins()[0]
    res = client.options("/usage", headers={
        "Origin": origin, "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    })
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == origin
