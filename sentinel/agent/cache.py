"""Redis exact-match response cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import redis.asyncio as aioredis

from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer

logger = get_logger(__name__)
tracer = get_tracer("sentinel.cache")


@dataclass(frozen=True)
class CachedResponse:
    """Structure stored inside Redis exact-match cache."""

    answer: str
    sources: list[dict[str, Any]]
    model: str
    provider: str
    tokens_in: int
    tokens_out: int


class ResponseCache:
    """Exact-match Redis response cache to shield quota and eliminate repeat latencies."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 3600,
        enabled: bool = True,
    ) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self._redis: aioredis.Redis | None = None

    def _get_client(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    @staticmethod
    def _compute_key(question: str) -> str:
        """Create a deterministic cache key by normalizing whitespace and casing."""
        normalized = " ".join(question.strip().lower().split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"sentinel:cache:exact:{digest}"

    async def get(self, question: str) -> CachedResponse | None:
        """Lookup answer by exact question match."""
        if not self.enabled:
            return None

        client = self._get_client()
        key = self._compute_key(question)

        with tracer.start_as_current_span("cache.get") as span:
            span.set_attribute("cache.system", "redis")
            span.set_attribute("cache.key", key)

            try:
                raw = await client.get(key)
                if not raw:
                    span.set_attribute("cache.hit", False)
                    return None
                data = json.loads(raw)
                span.set_attribute("cache.hit", True)
                span.set_attribute("cache.model", data.get("model", "unknown"))
                span.set_attribute("cache.provider", data.get("provider", "cache"))
                return CachedResponse(
                    answer=data["answer"],
                    sources=data.get("sources", []),
                    model=data.get("model", "unknown"),
                    provider=data.get("provider", "cache"),
                    tokens_in=data.get("tokens_in", 0),
                    tokens_out=data.get("tokens_out", 0),
                )
            except Exception as exc:
                span.set_attribute("cache.hit", False)
                span.record_exception(exc)
                logger.warning("Redis cache get failed, bypassing cache", extra={"error": str(exc)})
                return None

    async def set(
        self,
        question: str,
        response: CachedResponse,
    ) -> None:
        """Save response to cache with TTL."""
        if not self.enabled:
            return

        client = self._get_client()
        key = self._compute_key(question)

        with tracer.start_as_current_span("cache.set") as span:
            span.set_attribute("cache.system", "redis")
            span.set_attribute("cache.key", key)
            span.set_attribute("cache.ttl_seconds", self.ttl_seconds)

            try:
                payload = json.dumps(asdict(response))
                await client.set(key, payload, ex=self.ttl_seconds)
            except Exception as exc:
                span.record_exception(exc)
                logger.warning("Redis cache set failed", extra={"error": str(exc)})

    async def health_check(self) -> bool:
        """Ping Redis to test connectivity."""
        try:
            client = self._get_client()
            return bool(await client.ping())
        except Exception:
            return False

    async def close(self) -> None:
        """Close connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
