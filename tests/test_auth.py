"""Supabase JWT verification and the auth gate on /chat."""
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

import app.auth as auth
import app.main as main_module

SUPABASE = "https://proj.supabase.co"
ISS = f"{SUPABASE}/auth/v1"
SECRET = "legacy-hs256-secret-at-least-32-bytes-long!"


def _claims(**over):
    base = {"sub": "user-1", "aud": "authenticated", "iss": ISS,
            "exp": int(time.time()) + 600, "email": "a@b.c"}
    base.update(over)
    return base


@pytest.fixture
def es256(monkeypatch):
    key = ec.generate_private_key(ec.SECP256R1())

    class _Jwks:
        def get_signing_key_from_jwt(self, token):
            return type("K", (), {"key": key.public_key()})()

    monkeypatch.setattr(auth, "_jwks_client", lambda url: _Jwks())
    return lambda **over: jwt.encode(_claims(**over), key, algorithm="ES256", headers={"kid": "k1"})


def test_es256_token_verifies_via_jwks(es256):
    user = auth.verify_token(es256(), SUPABASE)
    assert user.id == "user-1" and user.email == "a@b.c"


def test_hs256_legacy_secret():
    token = jwt.encode(_claims(), SECRET, algorithm="HS256")
    assert auth.verify_token(token, SUPABASE, SECRET).id == "user-1"
    with pytest.raises(auth.AuthError):
        auth.verify_token(token, SUPABASE, "")  # no secret configured


@pytest.mark.parametrize("bad", [
    {"exp": int(time.time()) - 10},
    {"aud": "anon"},
    {"iss": "https://other.supabase.co/auth/v1"},
    {"is_anonymous": True},
])
def test_rejects_bad_claims(es256, bad):
    with pytest.raises(auth.AuthError):
        auth.verify_token(es256(**bad), SUPABASE)


def test_rejects_alg_none():
    token = jwt.encode(_claims(), None, algorithm="none")
    with pytest.raises(auth.AuthError):
        auth.verify_token(token, SUPABASE)


def test_rejects_forged_signature(es256):
    other = ec.generate_private_key(ec.SECP256R1())
    forged = jwt.encode(_claims(), other, algorithm="ES256")
    with pytest.raises(auth.AuthError):
        auth.verify_token(forged, SUPABASE)


@pytest.fixture
def gated(monkeypatch, es256):
    import agent.graph as graph_module

    seen = {}

    async def fake_run_agent(user_query, session_id):
        seen["session_id"] = session_id
        return {"final_report": "ok", "citations": [], "provider": "groq"}

    monkeypatch.setattr(graph_module, "run_agent", fake_run_agent)
    monkeypatch.setattr(main_module.settings, "database_url", "")
    monkeypatch.setattr(main_module.settings, "supabase_url", SUPABASE)
    monkeypatch.setattr(main_module.settings, "chat_rate_limit_per_minute", 0)
    monkeypatch.setattr(main_module, "_redis", None)
    monkeypatch.setattr(main_module, "_rate_window", {})
    monkeypatch.setattr(main_module, "_active_chats", 0)
    return TestClient(main_module.app), es256, seen


def test_chat_requires_token_when_auth_enabled(gated):
    client, _, _ = gated
    for path in ("/chat", "/chat/stream"):
        res = client.post(path, json={"query": "q"})
        assert res.status_code == 401
        assert res.headers["WWW-Authenticate"] == "Bearer"


def test_session_ids_are_scoped_to_the_user(gated):
    client, token, seen = gated
    hdr = {"Authorization": f"Bearer {token()}"}

    # Someone else's session id is not honoured.
    res = client.post("/chat", json={"query": "q", "session_id": "user-2:abc"}, headers=hdr)
    assert res.status_code == 200
    assert seen["session_id"].startswith("user-1:")

    # The user's own session id round-trips.
    own = res.json()["session_id"]
    client.post("/chat", json={"query": "q", "session_id": own}, headers=hdr)
    assert seen["session_id"] == own


def test_daily_quota_is_per_user(gated, monkeypatch):
    client, token, _ = gated
    monkeypatch.setattr(main_module.settings, "user_daily_query_limit", 1)
    a = {"Authorization": f"Bearer {token()}"}
    b = {"Authorization": f"Bearer {token(sub='user-2')}"}
    assert client.post("/chat", json={"query": "q"}, headers=a).status_code == 200
    over = client.post("/chat", json={"query": "q"}, headers=a)
    assert over.status_code == 429 and "today" in over.json()["detail"]
    assert client.post("/chat", json={"query": "q"}, headers=b).status_code == 200


def test_auth_off_without_supabase_url(monkeypatch):
    monkeypatch.setattr(main_module.settings, "supabase_url", "")
    assert main_module._scoped_session_id("abc", None) == "abc"
