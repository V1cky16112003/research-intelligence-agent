from __future__ import annotations
"""
Thin async wrapper around the Neo4j Python driver.

Mirrors db/connection.py's singleton-pool pattern: one driver instance per
process, created lazily on first use and reused thereafter.
"""
import logging
import os

from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)

_driver = None


def get_driver(
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
):
    """Return the singleton Neo4j async driver, creating it on first call."""
    global _driver
    if _driver is None:
        resolved_uri = uri or os.getenv("NEO4J_URI", "")
        resolved_user = user or os.getenv("NEO4J_USER", "")
        resolved_password = password or os.getenv("NEO4J_PASSWORD", "")
        if not resolved_uri:
            raise ValueError("NEO4J_URI must be set to use the graph layer")
        _driver = AsyncGraphDatabase.driver(
            resolved_uri, auth=(resolved_user, resolved_password)
        )
        logger.info("Neo4j driver initialized")
    return _driver


async def close_driver() -> None:
    """Close the driver, if open. Call on app shutdown."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
