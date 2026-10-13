"""Qdrant vector store implementation."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest_models

from sentinel.core.errors import VectorStoreError
from sentinel.core.logging import get_logger
from sentinel.rag.chunker import Chunk
from sentinel.rag.vector_store import ScoredDocument, VectorStore

logger = get_logger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant vector store with AsyncQdrantClient."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "sentinel_docs",
        vector_size: int = 384,
    ) -> None:
        self.url = url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client: AsyncQdrantClient | None = None

    def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(url=self.url)
        return self._client

    async def initialize(self) -> None:
        """Create the collection if it does not already exist."""
        client = self._get_client()
        try:
            collections = await client.get_collections()
            existing_names = [col.name for col in collections.collections]

            if self.collection_name not in existing_names:
                logger.info(
                    "Creating Qdrant collection",
                    extra={"collection": self.collection_name, "vector_size": self.vector_size},
                )
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=self.vector_size,
                        distance=rest_models.Distance.COSINE,
                    ),
                )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize Qdrant collection: {exc}") from exc

    @staticmethod
    def _to_uuid(chunk_id: str) -> str:
        """Convert any chunk identifier into a deterministic valid UUID."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Idempotently upsert chunks into Qdrant."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunks and embeddings length mismatch")

        client = self._get_client()
        points: list[rest_models.PointStruct] = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            point_id = self._to_uuid(chunk.chunk_id)
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "index": chunk.index,
                **chunk.metadata,
            }
            points.append(
                rest_models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        try:
            await client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            logger.info("Upserted points to Qdrant", extra={"count": len(points)})
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert points into Qdrant: {exc}") from exc

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 4,
        score_threshold: float = 0.0,
    ) -> list[ScoredDocument]:
        """Search top_k similar vectors."""
        client = self._get_client()
        try:
            query_res = await client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
            )

            scored_docs: list[ScoredDocument] = []
            for hit in query_res.points:
                payload = hit.payload or {}
                scored_docs.append(
                    ScoredDocument(
                        chunk_id=str(payload.get("chunk_id", hit.id)),
                        text=str(payload.get("text", "")),
                        score=float(hit.score),
                        metadata=payload,
                    )
                )
            return scored_docs
        except Exception as exc:
            raise VectorStoreError(f"Qdrant search error: {exc}") from exc

    async def count(self) -> int:
        """Return total points in the collection."""
        client = self._get_client()
        try:
            info = await client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    async def health_check(self) -> bool:
        """Verify Qdrant is reachable."""
        try:
            client = self._get_client()
            await client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close client connection."""
        if self._client:
            await self._client.close()
            self._client = None
