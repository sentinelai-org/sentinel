"""BM25 sparse keyword search and Reciprocal Rank Fusion (RRF) for hybrid retrieval."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from sentinel.core.logging import get_logger
from sentinel.rag.chunker import Chunk
from sentinel.rag.vector_store import ScoredDocument

logger = get_logger(__name__)

# Tokenizer matching technical identifiers, flags, camelCase, snake_case, and standard words
TOKEN_PATTERN = re.compile(r"--?[a-zA-Z0-9_\-\.]+|[a-zA-Z0-9_]+", re.ASCII)


def tokenize(text: str) -> list[str]:
    """Tokenize technical text preserving Kubernetes flags, error codes, and identifiers."""
    if not text:
        return []

    tokens: list[str] = []
    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    for token in raw_tokens:
        clean = token.strip("._-")
        if clean:
            tokens.append(clean)
            # If the token contains hyphens or underscores (e.g. kubelet-extra-args or max_surge),
            # also index sub-tokens to allow partial keyword matching.
            sub_parts = re.split(r"[-_.]+", clean)
            if len(sub_parts) > 1:
                for part in sub_parts:
                    if len(part) > 2 and part != clean:
                        tokens.append(part)
    return tokens


class BM25Index:
    """In-memory BM25Okapi inverted index for keyword-based sparse retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._documents: dict[str, Chunk] = {}
        self._doc_lengths: dict[str, int] = {}
        # Inverted index: term -> {chunk_id: term_frequency}
        self._inverted_index: dict[str, dict[str, int]] = {}
        self._avg_doc_len: float = 0.0

    def __len__(self) -> int:
        return len(self._documents)

    def add_documents(self, chunks: list[Chunk]) -> None:
        """Add document chunks to the BM25 inverted index."""
        if not chunks:
            return

        for chunk in chunks:
            self._documents[chunk.chunk_id] = chunk
            tokens = tokenize(chunk.text)
            doc_len = len(tokens)
            self._doc_lengths[chunk.chunk_id] = doc_len

            # Count term frequencies in this document
            tf_map: dict[str, int] = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            for term, count in tf_map.items():
                if term not in self._inverted_index:
                    self._inverted_index[term] = {}
                self._inverted_index[term][chunk.chunk_id] = count

        # Recalculate average document length
        total_len = sum(self._doc_lengths.values())
        self._avg_doc_len = total_len / len(self._doc_lengths) if self._doc_lengths else 0.0

        logger.debug(
            "Indexed chunks into BM25",
            extra={
                "added_chunks": len(chunks),
                "total_documents": len(self._documents),
                "unique_terms": len(self._inverted_index),
            },
        )

    def search(self, query: str, top_k: int = 20) -> list[ScoredDocument]:
        """Perform BM25Okapi scoring against the index for the given query."""
        if not self._documents or not query:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(self._documents)
        scores: dict[str, float] = {}

        for token in query_tokens:
            postings = self._inverted_index.get(token)
            if not postings:
                continue

            df = len(postings)
            # BM25 standard Robertson-Sparck Jones IDF
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            if idf <= 0.0:
                continue

            for chunk_id, tf in postings.items():
                doc_len = self._doc_lengths.get(chunk_id, 1)
                # BM25 tf scaling
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / (self._avg_doc_len or 1.0))
                )
                term_score = idf * (tf * (self.k1 + 1.0)) / denom
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        if not scores:
            return []

        # Sort descending by BM25 score
        sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results: list[ScoredDocument] = []
        for chunk_id, score in sorted_items:
            chunk = self._documents[chunk_id]
            results.append(
                ScoredDocument(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(score),
                    metadata={**chunk.metadata, "bm25_score": float(score)},
                )
            )

        return results

    def save_to_file(self, file_path: str | Path) -> None:
        """Persist BM25 document chunks and index state to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "documents": [
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": getattr(chunk, "doc_id", chunk.chunk_id),
                    "text": chunk.text,
                    "index": getattr(chunk, "index", 0),
                    "metadata": chunk.metadata,
                }
                for chunk in self._documents.values()
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "Saved BM25 index to disk", extra={"path": str(path), "docs": len(self._documents)}
        )

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> BM25Index:
        """Load and reconstruct BM25 index from a JSON file."""
        path = Path(file_path)
        if not path.is_file():
            logger.warning("BM25 index file not found", extra={"path": str(path)})
            return cls()

        data = json.loads(path.read_text(encoding="utf-8"))
        index = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        chunks = [
            Chunk(
                chunk_id=item["chunk_id"],
                doc_id=item.get("doc_id", item["chunk_id"]),
                text=item["text"],
                index=item.get("index", idx),
                metadata=item.get("metadata", {}),
            )
            for idx, item in enumerate(data.get("documents", []))
        ]

        index.add_documents(chunks)
        logger.info("Loaded BM25 index from disk", extra={"path": str(path), "docs": len(chunks)})
        return index


def reciprocal_rank_fusion(
    dense_results: list[ScoredDocument],
    sparse_results: list[ScoredDocument],
    k: int = 60,
    top_k: int = 20,
) -> list[ScoredDocument]:
    """Combine dense and sparse search rankings using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = sum_{m in {dense, sparse}} 1 / (k + rank_m(d))
    """
    fused_scores: dict[str, float] = {}
    doc_registry: dict[str, ScoredDocument] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}

    dense_scores: dict[str, float] = {}
    sparse_scores: dict[str, float] = {}

    for rank, doc in enumerate(dense_results, start=1):
        chunk_id = doc.chunk_id
        dense_ranks[chunk_id] = rank
        dense_scores[chunk_id] = doc.score
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
        doc_registry[chunk_id] = doc

    for rank, doc in enumerate(sparse_results, start=1):
        chunk_id = doc.chunk_id
        sparse_ranks[chunk_id] = rank
        sparse_scores[chunk_id] = doc.score
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
        if chunk_id not in doc_registry:
            doc_registry[chunk_id] = doc

    # Sort descending by fused RRF score
    sorted_chunks = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    max_possible_rrf = 2.0 / (k + 1)
    fused_documents: list[ScoredDocument] = []
    for chunk_id, rrf_score in sorted_chunks:
        original_doc = doc_registry[chunk_id]
        normalized_rrf = min(rrf_score / max_possible_rrf, 1.0)
        d_score = dense_scores.get(chunk_id)

        # Base score on scale [0.0, 1.0]
        if d_score is not None:
            combined_score = max(d_score, 0.5 * d_score + 0.5 * normalized_rrf)
        else:
            combined_score = normalized_rrf

        merged_metadata = {
            **original_doc.metadata,
            "raw_rrf_score": rrf_score,
            "normalized_rrf_score": normalized_rrf,
            "dense_score": d_score,
            "sparse_score": sparse_scores.get(chunk_id),
            "dense_rank": dense_ranks.get(chunk_id),
            "sparse_rank": sparse_ranks.get(chunk_id),
        }
        fused_documents.append(
            ScoredDocument(
                chunk_id=chunk_id,
                text=original_doc.text,
                score=combined_score,
                metadata=merged_metadata,
            )
        )

    return fused_documents
