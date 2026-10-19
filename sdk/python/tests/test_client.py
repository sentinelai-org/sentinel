"""Unit tests for the Sentinel Python SDK."""

import httpx
import pytest
from sentinel_client import (
    AsyncSentinelClient,
    AuthenticationError,
    PayloadTooLargeError,
    RateLimitError,
    SentinelClient,
)

SAMPLE_CHAT_RESPONSE = {
    "answer": "Kubernetes Deployments manage rolling updates incrementally.",
    "sources": [
        {
            "source": "deployments.md",
            "content": "Deployments update pods incrementally.",
            "score": 0.95,
        }
    ],
    "meta": {
        "latency_ms": 120.5,
        "tokens_in": 50,
        "tokens_out": 20,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "cache_hit": False,
        "cost_usd": 0.00001,
        "groundedness_score": 0.92,
        "groundedness_passed": True,
        "guardrail_passed": True,
        "request_id": "req_123",
    },
}


def test_sync_chat_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SentinelClient(api_key="sk-test-key")

    def mock_post(url: str, **kwargs: any) -> httpx.Response:
        assert url == "/v1/chat"
        assert kwargs.get("json", {}).get("question") == "What is a Pod?"
        return httpx.Response(200, json=SAMPLE_CHAT_RESPONSE)

    monkeypatch.setattr(client._client, "post", mock_post)

    resp = client.chat(question="What is a Pod?")
    assert resp.answer == SAMPLE_CHAT_RESPONSE["answer"]
    assert resp.meta.groundedness_passed is True
    assert len(resp.sources) == 1
    assert resp.sources[0].source == "deployments.md"


def test_sync_chat_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SentinelClient(api_key="sk-invalid")

    def mock_post(_url: str, **_kwargs: any) -> httpx.Response:
        return httpx.Response(401, json={"message": "Invalid API key"})

    monkeypatch.setattr(client._client, "post", mock_post)

    with pytest.raises(AuthenticationError) as exc_info:
        client.chat(question="Hello")
    assert "Invalid API key" in str(exc_info.value)
    assert exc_info.value.status_code == 401


def test_sync_chat_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SentinelClient()

    def mock_post(_url: str, **_kwargs: any) -> httpx.Response:
        return httpx.Response(
            429, headers={"Retry-After": "30"}, json={"message": "Rate limit exceeded"}
        )

    monkeypatch.setattr(client._client, "post", mock_post)

    with pytest.raises(RateLimitError) as exc_info:
        client.chat(question="Flood")
    assert exc_info.value.retry_after == 30
    assert exc_info.value.status_code == 429


def test_sync_chat_payload_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    client = SentinelClient()

    def mock_post(_url: str, **_kwargs: any) -> httpx.Response:
        return httpx.Response(413, json={"error": "Payload Too Large"})

    monkeypatch.setattr(client._client, "post", mock_post)

    with pytest.raises(PayloadTooLargeError) as exc_info:
        client.chat(question="Very large")
    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_async_chat_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AsyncSentinelClient(api_key="sk-test-key")

    async def mock_post(url: str, **kwargs: any) -> httpx.Response:
        assert url == "/v1/chat"
        assert kwargs.get("json", {}).get("question") == "What is a Pod?"
        return httpx.Response(200, json=SAMPLE_CHAT_RESPONSE)

    monkeypatch.setattr(client._client, "post", mock_post)

    resp = await client.chat(question="What is a Pod?")
    assert resp.answer == SAMPLE_CHAT_RESPONSE["answer"]
    await client.close()
