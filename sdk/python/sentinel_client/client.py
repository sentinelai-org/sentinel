"""Client implementations for Sentinel Enterprise Gateway."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx

from sentinel_client.exceptions import (
    AuthenticationError,
    PayloadTooLargeError,
    RateLimitError,
    SentinelError,
    ServiceUnavailableError,
)
from sentinel_client.models import (
    ChatResponse,
    FeedbackResponse,
    IngestDocument,
    IngestJobResponse,
    IngestJobStatus,
)


def _handle_response_error(response: httpx.Response) -> None:
    """Raises appropriate SentinelError subclasses based on HTTP status code."""
    if response.is_success:
        return

    try:
        data = response.json()
        message = data.get("message") or data.get("detail") or data.get("error") or response.text
    except Exception:
        data = None
        message = response.text or f"HTTP error {response.status_code}"

    if response.status_code == 401:
        raise AuthenticationError(message, status_code=401, details=data)
    elif response.status_code == 413:
        raise PayloadTooLargeError(message, details=data)
    elif response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        retry_seconds = int(retry_after) if retry_after and retry_after.isdigit() else None
        raise RateLimitError(message, retry_after=retry_seconds, details=data)
    elif response.status_code == 503:
        raise ServiceUnavailableError(message, status_code=503, details=data)
    else:
        raise SentinelError(message, status_code=response.status_code, details=data)


class SentinelClient:
    """Synchronous client for the Sentinel Enterprise AI & RAG Gateway."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "sentinel-client-python/0.1.0",
        }
        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            default_headers.update(headers)

        self._client = httpx.Client(
            base_url=self.base_url, headers=default_headers, timeout=timeout
        )

    def __enter__(self) -> SentinelClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client transport."""
        self._client.close()

    def chat(self, question: str, session_id: str | None = None) -> ChatResponse:
        """Submit a question to the Sentinel RAG Gateway and receive a verified answer."""
        payload: dict[str, Any] = {"question": question}
        if session_id:
            payload["session_id"] = session_id

        res = self._client.post("/v1/chat", json=payload)
        _handle_response_error(res)
        return ChatResponse.model_validate(res.json())

    def chat_stream(
        self, question: str, session_id: str | None = None
    ) -> Generator[dict[str, Any], None, None]:
        """Stream generated answer tokens in real-time via Server-Sent Events (SSE)."""
        payload: dict[str, Any] = {"question": question}
        if session_id:
            payload["session_id"] = session_id

        with self._client.stream("POST", "/v1/chat/stream", json=payload) as response:
            _handle_response_error(response)
            for line in response.iter_lines():
                if line.startswith("data: "):
                    raw_data = line[6:].strip()
                    if raw_data == "[DONE]":
                        break
                    try:
                        yield json.loads(raw_data)
                    except json.JSONDecodeError:
                        pass

    def ingest(self, documents: list[dict[str, Any] | IngestDocument]) -> IngestJobResponse:
        """Enqueue documents for asynchronous background chunking and indexing via Redis Streams."""
        docs_payload = [
            doc.model_dump() if isinstance(doc, IngestDocument) else doc for doc in documents
        ]
        res = self._client.post("/v1/ingest/jobs", json={"documents": docs_payload})
        _handle_response_error(res)
        return IngestJobResponse.model_validate(res.json())

    def get_job_status(self, job_id: str) -> IngestJobStatus:
        """Check progress and status of an asynchronous ingestion job."""
        res = self._client.get(f"/v1/ingest/jobs/{job_id}")
        _handle_response_error(res)
        return IngestJobStatus.model_validate(res.json())

    def submit_feedback(
        self,
        request_id: str,
        rating: int,
        comment: str | None = None,
        correction: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackResponse:
        """Submit user satisfaction rating, comment, or ground-truth correction."""
        payload: dict[str, Any] = {
            "request_id": request_id,
            "rating": rating,
        }
        if comment:
            payload["comment"] = comment
        if correction:
            payload["correction"] = correction
        if metadata:
            payload["metadata"] = metadata

        res = self._client.post("/v1/feedback", json=payload)
        _handle_response_error(res)
        return FeedbackResponse.model_validate(res.json())

    def is_healthy(self) -> bool:
        """Returns True if the Sentinel service process is responding to liveness checks."""
        try:
            res = self._client.get("/health")
            return res.status_code == 200
        except Exception:
            return False

    def is_ready(self) -> bool:
        """Returns True if all downstream services (Qdrant & Redis) are fully connected."""
        try:
            res = self._client.get("/ready")
            return res.status_code == 200
        except Exception:
            return False


class AsyncSentinelClient:
    """Asynchronous client for the Sentinel Enterprise AI & RAG Gateway."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "sentinel-client-python/0.1.0",
        }
        if api_key:
            default_headers["Authorization"] = f"Bearer {api_key}"
        if headers:
            default_headers.update(headers)

        self._client = httpx.AsyncClient(
            base_url=self.base_url, headers=default_headers, timeout=timeout
        )

    async def __aenter__(self) -> AsyncSentinelClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying async HTTP client transport."""
        await self._client.aclose()

    async def chat(self, question: str, session_id: str | None = None) -> ChatResponse:
        """Submit a question to the Sentinel RAG Gateway and receive a verified answer asynchronously."""
        payload: dict[str, Any] = {"question": question}
        if session_id:
            payload["session_id"] = session_id

        res = await self._client.post("/v1/chat", json=payload)
        _handle_response_error(res)
        return ChatResponse.model_validate(res.json())

    async def chat_stream(
        self, question: str, session_id: str | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream generated answer tokens in real-time via Server-Sent Events (SSE)."""
        payload: dict[str, Any] = {"question": question}
        if session_id:
            payload["session_id"] = session_id

        async with self._client.stream("POST", "/v1/chat/stream", json=payload) as response:
            _handle_response_error(response)
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    raw_data = line[6:].strip()
                    if raw_data == "[DONE]":
                        break
                    try:
                        yield json.loads(raw_data)
                    except json.JSONDecodeError:
                        pass

    async def ingest(self, documents: list[dict[str, Any] | IngestDocument]) -> IngestJobResponse:
        """Enqueue documents for asynchronous background chunking and indexing via Redis Streams."""
        docs_payload = [
            doc.model_dump() if isinstance(doc, IngestDocument) else doc for doc in documents
        ]
        res = await self._client.post("/v1/ingest/jobs", json={"documents": docs_payload})
        _handle_response_error(res)
        return IngestJobResponse.model_validate(res.json())

    async def get_job_status(self, job_id: str) -> IngestJobStatus:
        """Check progress and status of an asynchronous ingestion job."""
        res = await self._client.get(f"/v1/ingest/jobs/{job_id}")
        _handle_response_error(res)
        return IngestJobStatus.model_validate(res.json())

    async def submit_feedback(
        self,
        request_id: str,
        rating: int,
        comment: str | None = None,
        correction: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackResponse:
        """Submit user satisfaction rating, comment, or ground-truth correction."""
        payload: dict[str, Any] = {
            "request_id": request_id,
            "rating": rating,
        }
        if comment:
            payload["comment"] = comment
        if correction:
            payload["correction"] = correction
        if metadata:
            payload["metadata"] = metadata

        res = await self._client.post("/v1/feedback", json=payload)
        _handle_response_error(res)
        return FeedbackResponse.model_validate(res.json())

    async def is_healthy(self) -> bool:
        """Returns True if the Sentinel service process is responding to liveness checks."""
        try:
            res = await self._client.get("/health")
            return res.status_code == 200
        except Exception:
            return False

    async def is_ready(self) -> bool:
        """Returns True if all downstream services (Qdrant & Redis) are fully connected."""
        try:
            res = await self._client.get("/ready")
            return res.status_code == 200
        except Exception:
            return False
