"""Unit tests for background job queue and IngestWorker."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from httpx import ASGITransport

from sentinel.llm.fake import FakeProvider
from sentinel.main import build_app
from sentinel.rag.chunker import RecursiveChunker
from sentinel.workers.queue import InMemoryJobQueue
from sentinel.workers.worker import IngestWorker
from tests.conftest import InMemoryResponseCache, InMemoryVectorStore


class MockEmbedder:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        _ = text
        return [0.1] * self.dimension

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]


@pytest.mark.asyncio
async def test_in_memory_job_queue_lifecycle() -> None:
    """Verify job queue enqueueing and status updates."""
    queue = InMemoryJobQueue()
    docs = [
        {"content": "Doc 1 text", "source": "doc1.md"},
        {"content": "Doc 2 text", "source": "doc2.md"},
    ]

    job_id = await queue.enqueue_job(docs)
    assert job_id.startswith("job_")

    status = await queue.get_job_status(job_id)
    assert status is not None
    assert status["status"] == "queued"
    assert status["total_documents"] == 2
    assert status["processed_documents"] == 0
    assert status["progress_percent"] == 0.0

    # Simulate processing
    await queue.update_job_status(job_id, status="processing", processed_documents=1)
    status_mid = await queue.get_job_status(job_id)
    assert status_mid is not None
    assert status_mid["status"] == "processing"
    assert status_mid["processed_documents"] == 1
    assert status_mid["progress_percent"] == 50.0

    # Simulate completion
    await queue.update_job_status(job_id, status="completed", processed_documents=2)
    status_done = await queue.get_job_status(job_id)
    assert status_done is not None
    assert status_done["status"] == "completed"
    assert status_done["progress_percent"] == 100.0
    assert status_done["completed_at"] is not None


@pytest.mark.asyncio
async def test_ingest_worker_processes_job(
    in_memory_vector_store: InMemoryVectorStore,
    in_memory_job_queue: InMemoryJobQueue,
) -> None:
    """Verify IngestWorker chunks, embeds, upserts, and updates job status."""
    embedder = MockEmbedder(384)
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    worker = IngestWorker(
        queue=in_memory_job_queue,
        vector_store=in_memory_vector_store,
        embedder=embedder,  # type: ignore[arg-type]
        chunker=chunker,
    )

    docs = [
        {
            "content": "Kubernetes Architecture: The control plane manages worker nodes.",
            "source": "arch.md",
            "metadata": {"category": "architecture"},
        },
        {
            "content": "Kubelet: An agent that runs on each node in the cluster.",
            "source": "kubelet.md",
            "metadata": {"category": "components"},
        },
    ]

    job_id = await in_memory_job_queue.enqueue_job(docs)
    total_chunks = await worker.process_job(job_id=job_id, documents=docs)

    assert total_chunks >= 2
    assert await in_memory_vector_store.count() == total_chunks

    final_status = await in_memory_job_queue.get_job_status(job_id)
    assert final_status is not None
    assert final_status["status"] == "completed"
    assert final_status["processed_documents"] == 2
    assert final_status["progress_percent"] == 100.0


@pytest.mark.asyncio
async def test_ingest_worker_handles_failure(
    in_memory_job_queue: InMemoryJobQueue,
) -> None:
    """Verify worker catches unhandled errors and transitions job state to failed."""

    class BrokenVectorStore:
        async def upsert(self, chunks: list[Any], embeddings: list[list[float]]) -> None:
            _ = (chunks, embeddings)
            raise ConnectionError("Qdrant unreachable")

    embedder = MockEmbedder(384)
    worker = IngestWorker(
        queue=in_memory_job_queue,
        vector_store=BrokenVectorStore(),  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
    )

    docs = [{"content": "Some text", "source": "err.md"}]
    job_id = await in_memory_job_queue.enqueue_job(docs)

    with pytest.raises(ConnectionError):
        await worker.process_job(job_id=job_id, documents=docs)

    status = await in_memory_job_queue.get_job_status(job_id)
    assert status is not None
    assert status["status"] == "failed"
    assert "Qdrant unreachable" in (status["error"] or "")


@pytest.mark.asyncio
async def test_ingest_api_endpoints(
    in_memory_job_queue: InMemoryJobQueue,
    in_memory_vector_store: InMemoryVectorStore,
    in_memory_cache: InMemoryResponseCache,
    fake_llm_provider: FakeProvider,
) -> None:
    """Verify POST /v1/ingest/jobs and GET /v1/ingest/jobs/{job_id} HTTP endpoints."""
    app = build_app(override_primary_llm=fake_llm_provider)
    app.state.cache = in_memory_cache
    app.state.vector_store = in_memory_vector_store
    app.state.job_queue = in_memory_job_queue

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        # 1. Enqueue job via POST /v1/ingest/jobs
        payload = {
            "documents": [
                {"content": "Content for doc A", "source": "a.md"},
                {"content": "Content for doc B", "source": "b.md"},
            ]
        }
        resp = await client.post("/v1/ingest/jobs", json=payload)
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["total_documents"] == 2
        job_id = data["job_id"]

        # 2. Query status via GET /v1/ingest/jobs/{job_id}
        resp_status = await client.get(f"/v1/ingest/jobs/{job_id}")
        assert resp_status.status_code == 200
        st_data = resp_status.json()
        assert st_data["job_id"] == job_id
        assert st_data["status"] == "queued"
        assert st_data["total_documents"] == 2

        # 3. Query non-existent job returns 404
        resp_404 = await client.get("/v1/ingest/jobs/job_nonexistent123")
        assert resp_404.status_code == 404
