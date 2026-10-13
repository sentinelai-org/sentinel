"""Abstract interface and models for vector storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from sentinel.rag.chunker import Chunk


@dataclass(frozen=True)
class ScoredDocument:
    """Document snippet retrieved from vector search with similarity score."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create or ensure existence of the index/collection."""
        ...

    @abstractmethod
    async def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert document chunks along with their vector embeddings."""
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 4,
        score_threshold: float = 0.0,
    ) -> list[ScoredDocument]:
        """Search for top_k most similar document chunks."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Count total vectors in the store."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify connection to vector store."""
        ...
