"""Local sentence-transformers embedding provider."""

from __future__ import annotations

from typing import Any

from sentinel.core.logging import get_logger

logger = get_logger(__name__)


class LocalEmbedder:
    """Computes dense vector representations locally using sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._dimension = 384  # bge-small-en-v1.5 output vector size

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> Any:
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            logger.info("Loading local embedding model", extra={"model": self.model_name})
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for a single string."""
        model = self._get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of strings in batch."""
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [vec.tolist() for vec in embeddings]
