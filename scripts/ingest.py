"""Idempotent ingestion script for Markdown documentation into Qdrant."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from sentinel.core.config import get_settings
from sentinel.core.logging import get_logger, setup_logging
from sentinel.rag.bm25 import BM25Index
from sentinel.rag.chunker import RecursiveChunker
from sentinel.rag.embedder import LocalEmbedder
from sentinel.rag.loader import MarkdownLoader
from sentinel.rag.qdrant import QdrantVectorStore

logger = get_logger(__name__)


async def run_ingestion(docs_dir: Path, collection_name: str | None = None) -> int:
    """Run end-to-end ingestion pipeline."""
    settings = get_settings()
    target_collection = collection_name or settings.qdrant_collection

    logger.info(
        "Initializing ingestion pipeline",
        extra={"docs_dir": str(docs_dir), "collection": target_collection},
    )

    # 1. Load documents
    loader = MarkdownLoader()
    documents = loader.load_directory(docs_dir)
    if not documents:
        logger.warning("No markdown documents found to ingest", extra={"dir": str(docs_dir)})
        return 0

    logger.info("Loaded documents from disk", extra={"count": len(documents)})

    # 2. Chunk documents
    chunker = RecursiveChunker(chunk_size=600, chunk_overlap=100)
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc_id=doc.doc_id, text=doc.text, metadata=doc.metadata)
        all_chunks.extend(chunks)

    logger.info("Generated document chunks", extra={"total_chunks": len(all_chunks)})

    # 3. Compute local embeddings
    embedder = LocalEmbedder(model_name=settings.embedding_model)
    texts_to_embed = [chunk.text for chunk in all_chunks]
    logger.info(
        "Generating local embeddings with sentence-transformers",
        extra={"model": settings.embedding_model},
    )
    embeddings = embedder.embed_batch(texts_to_embed)

    # 4. Upsert into Qdrant
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=target_collection,
        vector_size=embedder.dimension,
    )
    await vector_store.initialize()
    await vector_store.upsert(all_chunks, embeddings)

    total_count = await vector_store.count()
    await vector_store.close()

    # 5. Build and save BM25 sparse index for Hybrid Search
    bm25_index = BM25Index()
    bm25_index.add_documents(all_chunks)
    bm25_index.save_to_file(settings.bm25_index_path)

    logger.info(
        "Ingestion completed successfully",
        extra={
            "ingested_chunks": len(all_chunks),
            "collection_total": total_count,
            "bm25_index_path": settings.bm25_index_path,
        },
    )
    print(
        f"Ingested {len(all_chunks)} chunks from {len(documents)} documents. "
        f"Total in Qdrant: {total_count}. BM25 sparse index saved to {settings.bm25_index_path}"
    )
    return len(all_chunks)


def main() -> None:
    """CLI entrypoint for ingestion."""
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Ingest Markdown docs into Qdrant for Sentinel")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("./data/docs"),
        help="Path to directory containing Markdown docs",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Target Qdrant collection name (defaults to QDRANT_COLLECTION env)",
    )
    args = parser.parse_args()

    asyncio.run(run_ingestion(args.docs_dir, args.collection))


if __name__ == "__main__":
    main()
