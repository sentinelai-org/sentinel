"""Shared pytest fixtures for Sentinel unit and integration tests."""

from __future__ import annotations

from typing import Any

import pytest

from sentinel.agent.cache import CachedResponse, ResponseCache
from sentinel.llm.fake import FakeProvider
from sentinel.rag.chunker import Chunk
from sentinel.rag.vector_store import ScoredDocument, VectorStore
from sentinel.workers.queue import InMemoryJobQueue


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for fast, reliable, zero-dependency testing."""

    def __init__(self) -> None:
        self.points: dict[str, dict[str, Any]] = {}
        self.is_healthy = True

    async def initialize(self) -> None:
        pass

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self.points[chunk.chunk_id] = {
                "chunk": chunk,
                "embedding": embedding,
            }

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 4,
        score_threshold: float = 0.0,
    ) -> list[ScoredDocument]:
        _ = (query_vector, score_threshold)
        # Return all stored points up to top_k with synthetic score 0.85
        results: list[ScoredDocument] = []
        for chunk_id, data in list(self.points.items())[:top_k]:
            chunk = data["chunk"]
            results.append(
                ScoredDocument(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    score=0.85,
                    metadata=chunk.metadata,
                )
            )
        return results

    async def count(self) -> int:
        return len(self.points)

    async def health_check(self) -> bool:
        return self.is_healthy


class InMemoryResponseCache(ResponseCache):
    """In-memory cache subclass avoiding Redis dependencies during tests."""

    def __init__(self) -> None:
        super().__init__(enabled=True)
        self.store: dict[str, CachedResponse] = {}
        self.is_healthy = True

    async def get(self, question: str) -> CachedResponse | None:
        key = self._compute_key(question)
        return self.store.get(key)

    async def set(self, question: str, response: CachedResponse) -> None:
        key = self._compute_key(question)
        self.store[key] = response

    async def health_check(self) -> bool:
        return self.is_healthy

    async def close(self) -> None:
        pass


@pytest.fixture
def in_memory_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def in_memory_cache() -> InMemoryResponseCache:
    return InMemoryResponseCache()


@pytest.fixture
def fake_llm_provider() -> FakeProvider:
    return FakeProvider(
        default_response="Kubernetes Pods encapsulate one or more containers.",
        tokens_in=45,
        tokens_out=18,
        latency_ms=8.0,
        provider_name="fake-gemini",
        model="fake-gemini-2.5",
    )


@pytest.fixture
def in_memory_job_queue() -> InMemoryJobQueue:
    return InMemoryJobQueue()
