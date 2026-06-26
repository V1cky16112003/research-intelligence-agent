from __future__ import annotations
"""
Async PostgreSQL connection pool using psycopg3 + psycopg_pool.
Reads DATABASE_URL from environment.
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import psycopg
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)
_pool: Optional[AsyncConnectionPool] = None


async def init_pool(
    database_url: str | None = None,
    min_size: int = 1,
    max_size: int = 10,
) -> None:
    """Initialize the global connection pool. Call once at app startup."""
    global _pool
    url = database_url or os.getenv("DATABASE_URL", "")
    if not url:
        logger.warning("DATABASE_URL not set — database features disabled")
        return
    _pool = AsyncConnectionPool(
        conninfo=url,
        min_size=min_size,
        max_size=max_size,
        open=False,
        reconnect_timeout=30,
    )
    await _pool.open(wait=True, timeout=30)
    logger.info("Database pool initialized")


async def close_pool() -> None:
    """Close the connection pool. Call at app shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Database pool closed")


@asynccontextmanager
async def get_connection() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """Yield a connection from the pool."""
    if _pool is None:
        raise RuntimeError("Database not configured. Set DATABASE_URL env var and call init_pool().")
    async with _pool.connection(timeout=15) as conn:
        yield conn


async def apply_schema(
    conn: psycopg.AsyncConnection,
    schema_path: str | Path = "db/schema.sql",
) -> None:
    """Apply schema.sql to the database. Safe to run multiple times (idempotent)."""
    sql = Path(schema_path).read_text()
    await conn.execute(sql)
    await conn.commit()
    logger.info("Schema applied from %s", schema_path)
