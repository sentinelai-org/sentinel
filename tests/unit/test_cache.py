"""Unit tests for exact-match response cache."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from sentinel.agent.cache import CachedResponse, ResponseCache


def test_compute_key_normalization() -> None:
    """Keys should be case-insensitive and normalize irregular spaces."""
    key1 = ResponseCache._compute_key("What is a Kubernetes Pod?")
    key2 = ResponseCache._compute_key("   what is   a   kubernetes   pod?  ")
    key3 = ResponseCache._compute_key("What is a Service?")

    assert key1 == key2
    assert key1 != key3


@pytest.mark.asyncio
async def test_cache_disabled_bypasses() -> None:
    """When disabled, cache always returns None on get and does not call set."""
    cache = ResponseCache(enabled=False)
    assert await cache.get("hello") is None


@pytest.mark.asyncio
async def test_cache_hit_and_miss() -> None:
    """Test get hit and miss with mocked Redis."""
    cache = ResponseCache(enabled=True)
    mock_redis = AsyncMock()
    cache._redis = mock_redis

    # Miss scenario
    mock_redis.get.return_value = None
    miss_result = await cache.get("What is a Pod?")
    assert miss_result is None

    # Hit scenario
    payload = {
        "answer": "A Pod is the smallest execution unit in Kubernetes.",
        "sources": [{"source": "pods.md"}],
        "model": "gemini-2.5-flash",
        "provider": "gemini",
        "tokens_in": 120,
        "tokens_out": 40,
    }
    mock_redis.get.return_value = json.dumps(payload)
    hit_result = await cache.get("What is a Pod?")

    assert hit_result is not None
    assert hit_result.answer == payload["answer"]
    assert hit_result.provider == "gemini"
    assert hit_result.tokens_in == 120


@pytest.mark.asyncio
async def test_cache_set_stores_with_ttl() -> None:
    """Test storing into cache applies configured TTL."""
    cache = ResponseCache(ttl_seconds=1800, enabled=True)
    mock_redis = AsyncMock()
    cache._redis = mock_redis

    resp = CachedResponse(
        answer="Kubernetes Service provides networking.",
        sources=[],
        model="gemini-2.5-flash",
        provider="gemini",
        tokens_in=50,
        tokens_out=20,
    )
    await cache.set("What is a Service?", resp)

    assert mock_redis.set.called
    args, kwargs = mock_redis.set.call_args
    assert kwargs.get("ex") == 1800


@pytest.mark.asyncio
async def test_cache_gracefully_handles_redis_failure() -> None:
    """Exceptions from Redis do not raise, instead fallback gracefully."""
    cache = ResponseCache(enabled=True)
    mock_redis = AsyncMock()
    mock_redis.get.side_effect = ConnectionError("Redis unreachable")
    cache._redis = mock_redis

    # Should not raise exception
    result = await cache.get("Any question")
    assert result is None
