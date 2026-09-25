"""Redis client factory.

Uses a *blocking* connection pool: when all connections are busy, callers wait up to `timeout`
seconds for one to free up. The default non-blocking pool raises "Too many connections"
immediately under bursty load, which would turn a traffic spike into errors.
"""

from __future__ import annotations

from redis.asyncio import BlockingConnectionPool, Redis


def make_redis(url: str, *, max_connections: int = 100, timeout: float = 5.0) -> Redis:
    pool = BlockingConnectionPool.from_url(url, max_connections=max_connections, timeout=timeout)
    return Redis(connection_pool=pool)
