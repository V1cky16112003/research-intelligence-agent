"""
Supabase Auth verification for the API.

The frontend signs users in with Supabase (free tier) and sends the session's access
token as `Authorization: Bearer <jwt>`. We verify it locally — no call to Supabase per
request:

- Projects on Supabase's asymmetric signing keys (the default for new projects) are
  verified against the public JWKS at `{SUPABASE_URL}/auth/v1/.well-known/jwks.json`,
  fetched once and cached by PyJWKClient.
- Legacy projects still on the shared HS256 secret set `SUPABASE_JWT_SECRET`.

Auth is on exactly when `SUPABASE_URL` is set, so local dev and the unit tests run
without it. With it on, the user ID replaces X-Forwarded-For as the rate-limit key —
the spoofable-identity gap in docs/security.md.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import jwt
from fastapi import Request

logger = logging.getLogger(__name__)

_ASYMMETRIC_ALGS = ["ES256", "RS256", "EdDSA"]
_jwks_clients: dict[str, jwt.PyJWKClient] = {}


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class User:
    id: str
    email: str | None = None


def _jwks_client(supabase_url: str) -> jwt.PyJWKClient:
    url = f"{supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
    if url not in _jwks_clients:
        _jwks_clients[url] = jwt.PyJWKClient(url, cache_keys=True, lifespan=3600, timeout=5)
    return _jwks_clients[url]


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthError("missing bearer token")
    return token.strip()


def verify_token(token: str, supabase_url: str, jwt_secret: str = "") -> User:
    issuer = f"{supabase_url.rstrip('/')}/auth/v1"
    options = {"require": ["exp", "sub", "aud"]}
    try:
        alg = jwt.get_unverified_header(token).get("alg")
        if alg == "HS256":
            if not jwt_secret:
                raise AuthError("HS256 token but SUPABASE_JWT_SECRET is not set")
            claims = jwt.decode(token, jwt_secret, algorithms=["HS256"],
                                audience="authenticated", issuer=issuer, options=options)
        elif alg in _ASYMMETRIC_ALGS:
            key = _jwks_client(supabase_url).get_signing_key_from_jwt(token)
            claims = jwt.decode(token, key.key, algorithms=_ASYMMETRIC_ALGS,
                                audience="authenticated", issuer=issuer, options=options)
        else:
            raise AuthError(f"unsupported alg {alg!r}")
    except AuthError:
        raise
    except jwt.PyJWTError as e:
        raise AuthError(str(e)) from e
    # Anonymous sign-ins also carry aud=authenticated; they'd let anyone mint
    # unlimited identities and defeat per-user limits.
    if claims.get("is_anonymous"):
        raise AuthError("anonymous sessions are not accepted")
    return User(id=claims["sub"], email=claims.get("email"))


async def authenticate(request: Request, supabase_url: str, jwt_secret: str = "") -> User:
    token = _bearer(request)
    # PyJWKClient fetches the JWKS synchronously on a cache miss; keep it off the loop.
    return await asyncio.to_thread(verify_token, token, supabase_url, jwt_secret)
