"""Markdown document loader for ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinel.core.errors import DocumentProcessingError
from sentinel.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoadedDocument:
    """Represents a loaded raw document."""

    doc_id: str
    text: str
    metadata: dict[str, Any]


class MarkdownLoader:
    """Loads and parses Markdown files from directories or specific files."""

    @staticmethod
    def _extract_title(content: str, default: str) -> str:
        """Extract the first H1 header as title or fallback to default."""
        for line in content.splitlines():
            line_str = line.strip()
            if line_str.startswith("# "):
                return line_str[2:].strip()
        return default

    def load_file(self, file_path: str | Path) -> LoadedDocument:
        """Load a single markdown file."""
        path = Path(file_path)
        if not path.is_file():
            raise DocumentProcessingError(f"File not found: {file_path}")

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            raise DocumentProcessingError(f"Failed to read file {file_path}: {exc}") from exc

        doc_id = path.stem
        title = self._extract_title(content, default=path.stem)
        metadata = {
            "source": path.name,
            "path": str(path.resolve()),
            "title": title,
            "size_bytes": path.stat().st_size,
        }

        return LoadedDocument(doc_id=doc_id, text=content, metadata=metadata)

    def load_directory(self, dir_path: str | Path) -> list[LoadedDocument]:
        """Load all markdown files from a directory."""
        path = Path(dir_path)
        if not path.is_dir():
            raise DocumentProcessingError(f"Directory not found: {dir_path}")

        documents: list[LoadedDocument] = []
        md_files = sorted(list(path.glob("*.md")) + list(path.glob("*.markdown")))

        for md_file in md_files:
            try:
                documents.append(self.load_file(md_file))
            except Exception as exc:
                logger.warning(
                    "Skipping file due to load error",
                    extra={"file": str(md_file), "error": str(exc)},
                )

        return documents
