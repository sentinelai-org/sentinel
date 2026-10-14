"""Asynchronous ingestion worker processing document chunks into vector database."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer
from sentinel.rag.chunker import RecursiveChunker
from sentinel.rag.embedder import LocalEmbedder
from sentinel.rag.vector_store import VectorStore
from sentinel.workers.queue import BaseJobQueue, InMemoryJobQueue, RedisJobQueue

logger = get_logger(__name__)
tracer = get_tracer("sentinel.workers.ingest")


class IngestWorker:
    """Worker consuming document ingestion jobs and indexing them into Qdrant."""

    def __init__(
        self,
        queue: BaseJobQueue,
        vector_store: VectorStore,
        embedder: LocalEmbedder,
        chunker: RecursiveChunker | None = None,
        consumer_name: str | None = None,
    ) -> None:
        self.queue = queue
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunker = chunker or RecursiveChunker(chunk_size=600, chunk_overlap=100)
        self.consumer_name = consumer_name or f"worker_{uuid.uuid4().hex[:8]}"
        self._running = False

    async def process_job(self, job_id: str, documents: list[dict[str, Any]]) -> int:
        """Process document batch for a single job: chunk, embed, and upsert."""
        with tracer.start_as_current_span("worker.process_job") as span:
            span.set_attribute("worker.job_id", job_id)
            span.set_attribute("worker.total_documents", len(documents))

            logger.info(
                "Starting background ingestion for job",
                extra={"job_id": job_id, "document_count": len(documents)},
            )
            await self.queue.update_job_status(job_id=job_id, status="processing")

            total_chunks = 0
            try:
                for idx, doc in enumerate(documents, start=1):
                    content = doc.get("content", "")
                    source = doc.get("source", "raw_doc")
                    metadata = dict(doc.get("metadata", {}))
                    metadata.setdefault("source", source)

                    doc_id = f"{source}_{idx}_{uuid.uuid4().hex[:6]}"
                    chunks = self.chunker.chunk_document(
                        doc_id=doc_id,
                        text=content,
                        metadata=metadata,
                    )

                    if chunks:
                        texts_to_embed = [c.text for c in chunks]
                        embeddings = self.embedder.embed_batch(texts_to_embed)
                        await self.vector_store.upsert(chunks, embeddings)
                        total_chunks += len(chunks)

                    await self.queue.update_job_status(
                        job_id=job_id,
                        status="processing",
                        processed_documents=idx,
                    )

                await self.queue.update_job_status(
                    job_id=job_id,
                    status="completed",
                    processed_documents=len(documents),
                )
                span.set_attribute("worker.total_chunks_indexed", total_chunks)
                logger.info(
                    "Completed background ingestion job",
                    extra={"job_id": job_id, "total_chunks": total_chunks},
                )
                return total_chunks

            except Exception as exc:
                logger.error(
                    "Ingestion job failed with error",
                    extra={"job_id": job_id, "error": str(exc)},
                )
                span.record_exception(exc)
                await self.queue.update_job_status(
                    job_id=job_id,
                    status="failed",
                    error=str(exc),
                )
                raise

    async def process_next_message(self, block_ms: int = 1000) -> bool:
        """Read and process next message from queue. Returns True if a job was processed."""
        if isinstance(self.queue, RedisJobQueue):
            client = self.queue._get_client()
            # Read 1 message from stream consumer group
            try:
                streams = {self.queue.stream_name: ">"}
                entries = await client.xreadgroup(
                    groupname=self.queue.consumer_group,
                    consumername=self.consumer_name,
                    streams=streams,
                    count=1,
                    block=block_ms,
                )
            except Exception as exc:
                logger.warning("Error reading from Redis Stream", extra={"error": str(exc)})
                return False

            if not entries:
                return False

            for _stream_name, messages in entries:
                for message_id, data in messages:
                    job_id = data.get("job_id")
                    docs_raw = data.get("documents_json", "[]")
                    documents = json.loads(docs_raw)

                    try:
                        await self.process_job(job_id=job_id, documents=documents)
                    finally:
                        # Acknowledge message in Redis stream
                        await client.xack(
                            self.queue.stream_name, self.queue.consumer_group, message_id
                        )
            return True

        elif isinstance(self.queue, InMemoryJobQueue):
            try:
                msg = await asyncio.wait_for(self.queue.queue.get(), timeout=block_ms / 1000.0)
                await self.process_job(msg["job_id"], msg["documents"])
                return True
            except TimeoutError:
                return False

        return False

    async def run_loop(self, poll_interval_seconds: float = 0.5) -> None:
        """Run continuous worker event loop until stopped."""
        self._running = True
        logger.info(
            "Ingest worker loop started",
            extra={"consumer": self.consumer_name},
        )
        try:
            while self._running:
                processed = await self.process_next_message(block_ms=1000)
                if not processed:
                    await asyncio.sleep(poll_interval_seconds)
        except asyncio.CancelledError:
            logger.info("Ingest worker loop cancelled")
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop worker execution loop."""
        self._running = False
