"""Cross-encoder re-ranking for RAG candidate precision."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from sentinel.core.logging import get_logger
from sentinel.rag.vector_store import ScoredDocument

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Abstract interface for document re-ranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[ScoredDocument],
        top_n: int = 4,
    ) -> list[ScoredDocument]:
        """Re-rank candidate documents against the query and return top_n items."""
        ...


class CrossEncoderReranker(BaseReranker):
    """Scores (query, document) pairs using a transformer Cross-Encoder model.

    Cross-encoders attend jointly across query and candidate text tokens,
    providing much higher precision than bi-encoder cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        force_mock: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.force_mock = force_mock
        self._model: Any = None
        self._model_load_failed = False

    def _get_model(self) -> Any:
        """Lazy load the sentence_transformers CrossEncoder."""
        if self.force_mock:
            return None

        if self._model is None and not self._model_load_failed:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(
                    "Loading Cross-Encoder re-ranker model", extra={"model": self.model_name}
                )
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:
                self._model_load_failed = True
                logger.warning(
                    "Could not load neural CrossEncoder; falling back to heuristic cross-scorer",
                    extra={"error": str(exc), "model": self.model_name},
                )
        return self._model

    def _heuristic_cross_score(self, query: str, doc_text: str, base_score: float = 0.5) -> float:
        """High-precision lexical and token alignment heuristic for offline or fallback environments."""
        if not query or not doc_text:
            return 0.0

        q_lower = query.lower()
        d_lower = doc_text.lower()

        # Stop words to filter out common conversational question filler
        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
            "what",
            "how",
            "do",
            "does",
            "can",
            "why",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "this",
        }

        # Tokenize content words
        import re

        tokens = [
            tok.strip("._-")
            for tok in re.findall(r"--?[a-zA-Z0-9_\-]+|[a-zA-Z0-9]+", q_lower)
            if tok.strip("._-")
        ]
        content_tokens = [tok for tok in tokens if tok not in stopwords and len(tok) > 1]

        if not content_tokens:
            content_recall = 0.5
        else:
            recalled = sum(1 for tok in content_tokens if tok in d_lower)
            content_recall = recalled / len(content_tokens)

        # If zero content words match, candidate is irrelevant to query
        if content_recall == 0.0:
            return 0.15

        # Technical identifiers & flags (e.g. --kubelet-extra-args, ErrImagePull)
        tech_tokens = re.findall(r"--[a-zA-Z0-9_\-]+|[A-Z][a-zA-Z0-9]+", query)
        tech_match = 0.0
        if tech_tokens:
            matches = sum(1 for tok in tech_tokens if tok.lower() in d_lower)
            tech_match = matches / len(tech_tokens)

        # Exact phrase bonus
        exact_bonus = 0.20 if q_lower in d_lower else 0.0

        base_clamped = min(max(base_score, 0.30), 0.95)
        raw_score = 0.35 * base_clamped + 0.45 * content_recall + 0.15 * tech_match + exact_bonus
        return min(max(raw_score, 0.0), 1.0)

    def rerank(
        self,
        query: str,
        documents: list[ScoredDocument],
        top_n: int = 4,
    ) -> list[ScoredDocument]:
        """Score candidate documents jointly with query and return top_n re-ranked documents."""
        if not documents:
            return []

        if len(documents) <= 1 and top_n >= 1:
            return documents

        model = self._get_model()

        scores: list[float] = []
        if model is not None:
            pairs = [[query, doc.text] for doc in documents]
            try:
                raw_scores = model.predict(pairs)
                # Normalize raw logits with sigmoid if necessary
                for s in raw_scores:
                    val = float(s)
                    norm_score = 1.0 / (1.0 + math.exp(-val)) if (val < -1.0 or val > 1.0) else val
                    scores.append(norm_score)
            except Exception as exc:
                logger.warning(
                    "CrossEncoder inference failed; using fallback cross-scorer",
                    extra={"error": str(exc)},
                )
                scores = [
                    self._heuristic_cross_score(query, doc.text, base_score=doc.score)
                    for doc in documents
                ]
        else:
            scores = [
                self._heuristic_cross_score(query, doc.text, base_score=doc.score)
                for doc in documents
            ]

        # Combine docs with updated scores
        scored_pairs: list[tuple[float, ScoredDocument]] = []
        for doc, score in zip(documents, scores, strict=False):
            updated_meta = {
                **doc.metadata,
                "pre_rerank_score": doc.score,
                "rerank_score": score,
            }
            updated_doc = ScoredDocument(
                chunk_id=doc.chunk_id,
                text=doc.text,
                score=score,
                metadata=updated_meta,
            )
            scored_pairs.append((score, updated_doc))

        # Sort descending by re-ranking score
        scored_pairs.sort(key=lambda pair: pair[0], reverse=True)

        reranked_docs = [pair[1] for pair in scored_pairs[:top_n]]
        logger.debug(
            "Re-ranked documents",
            extra={
                "candidates_count": len(documents),
                "selected_top_n": len(reranked_docs),
                "top_score": reranked_docs[0].score if reranked_docs else 0.0,
            },
        )
        return reranked_docs
