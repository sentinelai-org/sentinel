"""Background job queue backed by Redis Streams and in-memory fallback."""

from __future__ import annotations

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis

from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer

logger = get_logger(__name__)
tracer = get_tracer("sentinel.workers.queue")


class BaseJobQueue(ABC):
    """Abstract interface for ingestion job queues."""

    @abstractmethod
    async def enqueue_job(self, documents: list[dict[str, Any]]) -> str:
        """Enqueue document ingestion job and return unique job ID."""
        ...

    @abstractmethod
    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Retrieve current status of a job."""
        ...

    @abstractmethod
    async def update_job_status(
        self,
        job_id: str,
        status: str,
        processed_documents: int | None = None,
        error: str | None = None,
    ) -> None:
        """Update job execution progress and lifecycle state."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connections or cleanup resources."""
        ...


class RedisJobQueue(BaseJobQueue):
    """Production job queue backed by Redis Streams and Redis Hashes."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        stream_name: str = "sentinel:stream:ingest_jobs",
        consumer_group: str = "sentinel_workers",
    ) -> None:
        self.redis_url = redis_url
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self._redis: aioredis.Redis | None = None

    def _get_client(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def initialize_consumer_group(self) -> None:
        """Create Redis stream consumer group idempotently if not already created."""
        client = self._get_client()
        try:
            await client.xgroup_create(
                name=self.stream_name,
                groupname=self.consumer_group,
                id="0",
                mkstream=True,
            )
            logger.info(
                "Created Redis consumer group",
                extra={"stream": self.stream_name, "group": self.consumer_group},
            )
        except Exception as exc:
            if "BUSYGROUP" in str(exc):
                # Consumer group already exists
                pass
            else:
                logger.warning(
                    "Error creating Redis consumer group",
                    extra={"error": str(exc), "stream": self.stream_name},
                )

    async def enqueue_job(self, documents: list[dict[str, Any]]) -> str:
        """Enqueue documents to Redis Stream and create initial job status record."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC).isoformat()
        total_docs = len(documents)

        client = self._get_client()

        with tracer.start_as_current_span("queue.enqueue_job") as span:
            span.set_attribute("queue.job_id", job_id)
            span.set_attribute("queue.total_documents", total_docs)
            span.set_attribute("queue.stream", self.stream_name)

            # 1. Store initial job status metadata in Hash
            job_key = f"sentinel:job:{job_id}"
            initial_status: Mapping[str | bytes, bytes | float | int | str] = {
                "job_id": job_id,
                "status": "queued",
                "total_documents": str(total_docs),
                "processed_documents": "0",
                "progress_percent": "0.0",
                "created_at": now,
                "completed_at": "",
                "error": "",
            }
            await client.hset(job_key, mapping=initial_status)
            await client.expire(job_key, 86400)  # Keep job status for 24h

            # 2. Add message to Redis Stream
            message_data = {
                "job_id": job_id,
                "documents_json": json.dumps(documents),
                "enqueued_at": now,
            }
            stream_msg_id = await client.xadd(self.stream_name, message_data)
            span.set_attribute("queue.stream_msg_id", stream_msg_id)

            logger.info(
                "Enqueued ingestion job to Redis Stream",
                extra={
                    "job_id": job_id,
                    "total_documents": total_docs,
                    "stream_msg_id": stream_msg_id,
                },
            )

        return job_id

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Fetch job state and progress from Redis Hash."""
        client = self._get_client()
        job_key = f"sentinel:job:{job_id}"
        data = await client.hgetall(job_key)
        if not data:
            return None

        total_docs = int(data.get("total_documents", 0))
        processed_docs = int(data.get("processed_documents", 0))
        progress = float(data.get("progress_percent", 0.0))

        return {
            "job_id": data.get("job_id", job_id),
            "status": data.get("status", "unknown"),
            "total_documents": total_docs,
            "processed_documents": processed_docs,
            "progress_percent": progress,
            "created_at": data.get("created_at") or None,
            "completed_at": data.get("completed_at") or None,
            "error": data.get("error") or None,
        }

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        processed_documents: int | None = None,
        error: str | None = None,
    ) -> None:
        """Update job lifecycle and progress counters."""
        client = self._get_client()
        job_key = f"sentinel:job:{job_id}"

        updates: dict[str | bytes, bytes | float | int | str] = {"status": status}
        if error:
            updates["error"] = error

        if processed_documents is not None:
            updates["processed_documents"] = str(processed_documents)
            # Recompute percentage
            total_raw = await client.hget(job_key, "total_documents")
            total = int(total_raw) if total_raw else 0
            if total > 0:
                pct = round(min(100.0, (processed_documents / total) * 100.0), 1)
                updates["progress_percent"] = str(pct)

        if status in ("completed", "failed"):
            updates["completed_at"] = datetime.now(UTC).isoformat()
            if status == "completed":
                updates["progress_percent"] = "100.0"

        await client.hset(job_key, mapping=updates)

    async def close(self) -> None:
        """Close connection to Redis."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None


class InMemoryJobQueue(BaseJobQueue):
    """In-memory job queue for testing and zero-dependency local environments."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def enqueue_job(self, documents: list[dict[str, Any]]) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC).isoformat()
        total_docs = len(documents)

        status_record = {
            "job_id": job_id,
            "status": "queued",
            "total_documents": total_docs,
            "processed_documents": 0,
            "progress_percent": 0.0,
            "created_at": now,
            "completed_at": None,
            "error": None,
        }
        self.jobs[job_id] = status_record

        await self.queue.put({"job_id": job_id, "documents": documents})
        return job_id

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        return dict(job)

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        processed_documents: int | None = None,
        error: str | None = None,
    ) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return

        job["status"] = status
        if error is not None:
            job["error"] = error
        if processed_documents is not None:
            job["processed_documents"] = processed_documents
            total = job["total_documents"]
            if total > 0:
                job["progress_percent"] = round(
                    min(100.0, (processed_documents / total) * 100.0), 1
                )

        if status in ("completed", "failed"):
            job["completed_at"] = datetime.now(UTC).isoformat()
            if status == "completed":
                job["progress_percent"] = 100.0

    async def get_next_message(self) -> dict[str, Any]:
        """Fetch next job from in-memory queue."""
        return await self.queue.get()

    async def close(self) -> None:
        pass
