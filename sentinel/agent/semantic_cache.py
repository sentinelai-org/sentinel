"""Vector-backed semantic cache for near-duplicate and rephrased query answering."""

from __future__ import annotations

import hashlib

from sentinel.agent.cache import CachedResponse
from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer
from sentinel.rag.chunker import Chunk
from sentinel.rag.embedder import LocalEmbedder
from sentinel.rag.vector_store import VectorStore

logger = get_logger(__name__)
tracer = get_tracer("sentinel.cache.semantic")


class SemanticCache:
    """Semantic vector cache matching incoming questions against stored historical responses."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: LocalEmbedder,
        similarity_threshold: float = 0.90,
        enabled: bool = True,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.enabled = enabled

    async def initialize(self) -> None:
        """Ensure the semantic cache vector index is ready."""
        if self.enabled:
            await self.vector_store.initialize()

    async def get(self, question: str) -> tuple[CachedResponse | None, float]:
        """Look up semantically equivalent previous query.

        Returns tuple of (CachedResponse or None, similarity_score).
        """
        if not self.enabled:
            return None, 0.0

        with tracer.start_as_current_span("cache.semantic.lookup") as span:
            span.set_attribute("cache.semantic.question", question)
            try:
                query_vector = self.embedder.embed_text(question)
                results = await self.vector_store.search(
                    query_vector=query_vector,
                    top_k=1,
                    score_threshold=self.similarity_threshold,
                )

                if not results:
                    span.set_attribute("cache.hit", False)
                    return None, 0.0

                top_match = results[0]
                similarity = top_match.score
                meta = top_match.metadata

                span.set_attribute("cache.hit", True)
                span.set_attribute("cache.similarity_score", similarity)
                span.set_attribute("cache.matched_question", meta.get("original_question", ""))

                logger.info(
                    "Semantic cache hit",
                    extra={
                        "incoming_question": question,
                        "matched_question": meta.get("original_question"),
                        "similarity_score": round(similarity, 4),
                    },
                )

                cached = CachedResponse(
                    answer=meta.get("answer", top_match.text),
                    sources=meta.get("sources", []),
                    model=meta.get("model", "semantic-cache"),
                    provider=meta.get("provider", "semantic-cache"),
                    tokens_in=meta.get("tokens_in", 0),
                    tokens_out=meta.get("tokens_out", 0),
                )
                return cached, similarity

            except Exception as exc:
                logger.warning("Semantic cache lookup failed, bypassing", extra={"error": str(exc)})
                span.record_exception(exc)
                return None, 0.0

    async def set(self, question: str, response: CachedResponse) -> None:
        """Store new answer and question vector into the semantic cache."""
        if not self.enabled:
            return

        with tracer.start_as_current_span("cache.semantic.store") as span:
            span.set_attribute("cache.semantic.question", question)
            try:
                vector = self.embedder.embed_text(question)
                chunk_id = hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()[:16]

                chunk = Chunk(
                    chunk_id=f"sem_{chunk_id}",
                    doc_id="semantic_cache",
                    text=response.answer,
                    index=0,
                    metadata={
                        "original_question": question,
                        "answer": response.answer,
                        "sources": response.sources,
                        "model": response.model,
                        "provider": response.provider,
                        "tokens_in": response.tokens_in,
                        "tokens_out": response.tokens_out,
                    },
                )

                await self.vector_store.upsert([chunk], [vector])
                logger.info("Stored response into semantic cache", extra={"question": question})

            except Exception as exc:
                logger.warning("Failed to store into semantic cache", extra={"error": str(exc)})
                span.record_exception(exc)
