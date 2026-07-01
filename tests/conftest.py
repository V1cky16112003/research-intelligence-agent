"""
Stub heavy dependencies that are only available inside Docker
(psycopg, psycopg_pool, sentence_transformers, pgvector, torch).

This lets the test suite run locally without the full dependency stack.
Real integration tests run in CI against docker-compose.
"""
from __future__ import annotations
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock


def _make_module(name: str, **attrs) -> ModuleType:
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# --- psycopg / psycopg_pool ---
_psycopg = _make_module("psycopg")
_psycopg.AsyncConnection = MagicMock()

_psycopg_pool = _make_module("psycopg_pool")
_psycopg_pool.AsyncConnectionPool = MagicMock()

_psycopg_rows = _make_module("psycopg.rows")
_psycopg_rows.dict_row = MagicMock()

# --- pgvector ---
_pgvector = _make_module("pgvector")
_pgvector_psycopg = _make_module("pgvector.psycopg")
_pgvector_psycopg.register_vector = AsyncMock()

# --- torch / sentence_transformers ---
_torch = _make_module("torch")
_st = _make_module("sentence_transformers")
_st.SentenceTransformer = MagicMock()

# --- neo4j ---
_neo4j = _make_module("neo4j")
_neo4j.AsyncGraphDatabase = MagicMock()
_neo4j.AsyncDriver = MagicMock
