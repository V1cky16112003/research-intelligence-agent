from __future__ import annotations
"""
Redis client factory.

Auto-detects Upstash (HTTPS REST) vs local Redis (redis://).
Both expose the same async interface.
"""
import logging
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class RedisClient:
    """Abstract base. Use create_redis_client() factory."""

    async def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        raise NotImplementedError


class UpstashRedisClient(RedisClient):
    """Upstash serverless Redis via HTTP REST API."""

    def __init__(self, url: str) -> None:
        parsed = urlparse(url)
        self._token = parsed.password or ""
        self._base_url = f"https://{parsed.hostname}"
        self._headers = {"Authorization": f"Bearer {self._token}"}

    async def get(self, key: str) -> Optional[str]:
        import httpx
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{self._base_url}/get/{key}", headers=self._headers)
            r.raise_for_status()
            data = r.json()
            return data.get("result")  # None if key not found

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        import httpx
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{self._base_url}/set/{key}",
                params={"EX": ttl},
                content=value.encode(),
                headers=self._headers,
            )
            r.raise_for_status()

    async def delete(self, key: str) -> None:
        import httpx
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{self._base_url}/del/{key}", headers=self._headers)
            r.raise_for_status()


class LocalRedisClient(RedisClient):
    """redis-py async client for local/TCP Redis (docker-compose)."""

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis
        self._redis = aioredis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> Optional[str]:
        return await self._redis.get(key)

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        await self._redis.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)


async def create_redis_client(redis_url: str | None) -> Optional[RedisClient]:
    """
    Factory. Returns:
    - None if redis_url is empty/None
    - UpstashRedisClient if URL starts with 'https://'
    - LocalRedisClient otherwise (redis:// scheme)
    """
    if not redis_url:
        logger.warning("REDIS_URL not set — caching disabled")
        return None
    if redis_url.startswith("https://"):
        logger.info("Using Upstash Redis (HTTP REST)")
        return UpstashRedisClient(redis_url)
    logger.info("Using local Redis (redis-py)")
    return LocalRedisClient(redis_url)
