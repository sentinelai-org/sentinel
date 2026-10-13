"""Markdown-aware text chunking with configurable size and overlap."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A single chunk of a larger document."""

    chunk_id: str
    doc_id: str
    text: str
    index: int
    metadata: dict[str, Any]


class RecursiveChunker:
    """Splits documents recursively using natural separators."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the list of separators."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        final_chunks: list[str] = []
        current_doc: list[str] = []
        total_len = 0

        for s in splits:
            if not s.strip():
                continue
            piece = (
                s if not current_doc and not separator else (separator + s if current_doc else s)
            )
            piece_len = len(piece)

            if piece_len > self.chunk_size and remaining_separators:
                # Need to subdivide further
                sub_chunks = self._split_text(s, remaining_separators)
                for sub in sub_chunks:
                    if total_len + len(sub) > self.chunk_size and current_doc:
                        combined = "".join(current_doc).strip()
                        if combined:
                            final_chunks.append(combined)
                        # Overlap: keep the tail of current_doc if possible
                        current_doc = []
                        total_len = 0
                    current_doc.append(sub)
                    total_len += len(sub)
            elif total_len + piece_len <= self.chunk_size:
                current_doc.append(piece)
                total_len += piece_len
            else:
                combined = "".join(current_doc).strip()
                if combined:
                    final_chunks.append(combined)
                # Overlap logic
                overlap_text = combined[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                current_doc = [overlap_text + piece] if overlap_text else [piece]
                total_len = sum(len(x) for x in current_doc)

        if current_doc:
            combined = "".join(current_doc).strip()
            if combined:
                final_chunks.append(combined)

        return final_chunks

    def chunk_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split a document into overlapping chunks with deterministic IDs."""
        raw_text = text.strip()
        if not raw_text:
            return []

        base_metadata = metadata or {}
        splits = self._split_text(raw_text, self.separators)

        chunks: list[Chunk] = []
        for idx, split_text in enumerate(splits):
            clean_text = split_text.strip()
            if not clean_text:
                continue

            # Deterministic chunk ID
            id_source = f"{doc_id}:{idx}:{clean_text[:64]}"
            chunk_id = hashlib.sha256(id_source.encode("utf-8")).hexdigest()

            chunk_meta = {**base_metadata, "chunk_index": idx, "doc_id": doc_id}
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=clean_text,
                    index=idx,
                    metadata=chunk_meta,
                )
            )

        return chunks
