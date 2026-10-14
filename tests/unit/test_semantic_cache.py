"""Unit tests for Semantic Vector Cache."""

import math

import pytest

from sentinel.agent.cache import CachedResponse
from sentinel.agent.semantic_cache import SemanticCache
from sentinel.rag.chunker import Chunk
from sentinel.rag.vector_store import ScoredDocument, VectorStore


class CosineVectorStore(VectorStore):
    """Accurate in-memory vector store computing true cosine similarity."""

    def __init__(self) -> None:
        self.items: list[tuple[Chunk, list[float]]] = []

    async def initialize(self) -> None:
        pass

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for c, e in zip(chunks, embeddings, strict=True):
            self.items.append((c, e))

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 4,
        score_threshold: float = 0.0,
    ) -> list[ScoredDocument]:
        scored: list[ScoredDocument] = []

        q_norm = math.sqrt(sum(x * x for x in query_vector)) or 1.0

        for chunk, emb in self.items:
            dot = sum(a * b for a, b in zip(query_vector, emb, strict=True))
            e_norm = math.sqrt(sum(x * x for x in emb)) or 1.0
            cosine = dot / (q_norm * e_norm)

            if cosine >= score_threshold:
                scored.append(
                    ScoredDocument(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        score=cosine,
                        metadata=chunk.metadata,
                    )
                )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    async def count(self) -> int:
        return len(self.items)

    async def health_check(self) -> bool:
        return True


class MockEmbedder:
    """Mock embedder returning deterministic vectors based on keyword presence."""

    def embed_text(self, text: str) -> list[float]:
        t = text.lower()
        if "pod" in t:
            return [1.0, 0.0, 0.0]
        if "service" in t:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


@pytest.mark.asyncio
async def test_semantic_cache_hit_on_similar_query() -> None:
    store = CosineVectorStore()
    embedder = MockEmbedder()
    cache = SemanticCache(
        vector_store=store,
        embedder=embedder,  # type: ignore[arg-type]
        similarity_threshold=0.90,
        enabled=True,
    )

    # Populate cache with "What is a Pod?"
    original_entry = CachedResponse(
        answer="A Pod is the smallest deployable unit in Kubernetes.",
        sources=[{"source": "pods.md", "score": 0.95}],
        model="mock-model",
        provider="mock-provider",
        tokens_in=10,
        tokens_out=20,
    )
    await cache.set("What is a Pod in Kubernetes?", original_entry)

    # Query with a rephrased question: "Can you explain what is a Pod?"
    # Both contain "pod" so MockEmbedder returns [1.0, 0.0, 0.0], cosine = 1.0 >= 0.90
    cached, score = await cache.get("Can you explain what is a Pod?")

    assert cached is not None
    assert cached.answer == original_entry.answer
    assert score >= 0.90
    assert cached.sources[0]["source"] == "pods.md"


@pytest.mark.asyncio
async def test_semantic_cache_miss_on_different_query() -> None:
    store = CosineVectorStore()
    embedder = MockEmbedder()
    cache = SemanticCache(
        vector_store=store,
        embedder=embedder,  # type: ignore[arg-type]
        similarity_threshold=0.90,
        enabled=True,
    )

    original_entry = CachedResponse(
        answer="A Pod is the smallest unit.",
        sources=[],
        model="mock-model",
        provider="mock-provider",
        tokens_in=5,
        tokens_out=5,
    )
    await cache.set("What is a Pod?", original_entry)

    # Query with "What is a Service?" -> vector [0.0, 1.0, 0.0], cosine = 0.0 < 0.90
    cached, score = await cache.get("What is a Service in Kubernetes?")

    assert cached is None
    assert score == 0.0


@pytest.mark.asyncio
async def test_semantic_cache_disabled_bypasses() -> None:
    store = CosineVectorStore()
    embedder = MockEmbedder()
    cache = SemanticCache(
        vector_store=store,
        embedder=embedder,  # type: ignore[arg-type]
        similarity_threshold=0.90,
        enabled=False,
    )

    cached, score = await cache.get("What is a Pod?")
    assert cached is None
    assert score == 0.0
