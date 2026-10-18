"""Closed-loop user feedback collection, RLHF dataset export, and metrics tracking."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis

from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer
from sentinel.evals.schemas import FeedbackRequest, FeedbackResponse, FeedbackStats

logger = get_logger(__name__)
tracer = get_tracer("sentinel.evals.feedback")


class FeedbackStore:
    """Stores user feedback in Redis with in-memory fallback for testing and offline modes."""

    FEEDBACK_KEY_PREFIX = "sentinel:feedback:"
    TIMELINE_KEY = "sentinel:feedback:timeline"
    REQ_INDEX_PREFIX = "sentinel:request_to_feedback:"
    FEEDBACK_STREAM = "sentinel:stream:feedback"

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None
        self._mem_store: dict[str, dict[str, Any]] = {}

    async def _get_redis(self) -> aioredis.Redis | None:
        if self._redis is None:
            try:
                client = aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=1.5,
                )
                await client.ping()
                self._redis = client
            except Exception as exc:
                logger.warning(
                    "Redis unavailable for feedback store; using in-memory fallback",
                    extra={"error": str(exc)},
                )
                self._redis = None
        return self._redis

    async def record_feedback(self, payload: FeedbackRequest) -> FeedbackResponse:
        """Record user feedback and emit to stream for closed-loop fine-tuning."""
        with tracer.start_as_current_span("feedback.record") as span:
            feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
            now_iso = datetime.now(UTC).isoformat()
            now_ts = time.time()

            record = {
                "feedback_id": feedback_id,
                "request_id": payload.request_id,
                "rating": payload.rating,
                "comment": payload.comment or "",
                "correction": payload.correction or "",
                "metadata": json.dumps(payload.metadata),
                "recorded_at": now_iso,
            }

            span.set_attribute("feedback.id", feedback_id)
            span.set_attribute("feedback.rating", payload.rating)
            span.set_attribute("feedback.request_id", payload.request_id)

            client = await self._get_redis()
            if client is not None:
                try:
                    pipe = client.pipeline()
                    # 1. Store full record hash
                    pipe.hset(f"{self.FEEDBACK_KEY_PREFIX}{feedback_id}", mapping=record)
                    # 2. Add to sorted set timeline for range queries
                    pipe.zadd(self.TIMELINE_KEY, {feedback_id: now_ts})
                    # 3. Request ID index
                    pipe.set(f"{self.REQ_INDEX_PREFIX}{payload.request_id}", feedback_id)
                    # 4. Stream to feedback event stream for fine-tuning pipeline
                    pipe.xadd(self.FEEDBACK_STREAM, record)
                    await pipe.execute()
                except Exception as exc:
                    logger.error("Failed to write feedback to Redis", extra={"error": str(exc)})
                    self._mem_store[feedback_id] = record
            else:
                self._mem_store[feedback_id] = record

            logger.info(
                "User feedback recorded",
                extra={
                    "feedback_id": feedback_id,
                    "request_id": payload.request_id,
                    "rating": payload.rating,
                },
            )

            return FeedbackResponse(
                feedback_id=feedback_id,
                request_id=payload.request_id,
                status="recorded",
                recorded_at=now_iso,
            )

    async def get_stats(self) -> FeedbackStats:
        """Compute aggregated satisfaction rate and feedback counts."""
        with tracer.start_as_current_span("feedback.get_stats"):
            client = await self._get_redis()
            records: list[dict[str, Any]] = []

            if client is not None:
                try:
                    # Fetch recent feedback IDs from timeline
                    fb_ids = await client.zrevrange(self.TIMELINE_KEY, 0, 500)
                    for f_id in fb_ids:
                        data = await client.hgetall(f"{self.FEEDBACK_KEY_PREFIX}{f_id}")
                        if data:
                            records.append(data)
                except Exception as exc:
                    logger.warning("Redis read failed during get_stats", extra={"error": str(exc)})
                    records = list(self._mem_store.values())
            else:
                records = list(self._mem_store.values())

            total = len(records)
            if total == 0:
                return FeedbackStats(
                    total_feedback=0,
                    positive_count=0,
                    negative_count=0,
                    satisfaction_rate=100.0,
                    latest_feedback_at=None,
                )

            positive = sum(1 for r in records if int(r.get("rating", 0)) > 0)
            negative = sum(1 for r in records if int(r.get("rating", 0)) <= 0)
            satisfaction = round((positive / total) * 100.0, 1) if total > 0 else 100.0

            return FeedbackStats(
                total_feedback=total,
                positive_count=positive,
                negative_count=negative,
                satisfaction_rate=satisfaction,
                latest_feedback_at=records[0].get("recorded_at") if records else None,
            )

    async def export_tuning_dataset(self, limit: int = 100) -> list[dict[str, Any]]:
        """Export corrected and high-value feedback formatted as an LLM fine-tuning dataset."""
        with tracer.start_as_current_span("feedback.export_dataset"):
            client = await self._get_redis()
            records: list[dict[str, Any]] = []

            if client is not None:
                try:
                    fb_ids = await client.zrevrange(self.TIMELINE_KEY, 0, limit - 1)
                    for f_id in fb_ids:
                        data = await client.hgetall(f"{self.FEEDBACK_KEY_PREFIX}{f_id}")
                        if data:
                            records.append(data)
                except Exception as exc:
                    logger.warning(
                        "Redis read failed during export_tuning_dataset", extra={"error": str(exc)}
                    )
                    records = list(self._mem_store.values())[:limit]
            else:
                records = list(self._mem_store.values())[:limit]

            dataset: list[dict[str, Any]] = []
            for r in records:
                correction = r.get("correction", "").strip()
                comment = r.get("comment", "").strip()
                rating = int(r.get("rating", 0))
                meta = {}
                try:
                    meta = json.loads(r.get("metadata", "{}"))
                except Exception:
                    pass

                question = meta.get("question", "")

                # Only include entries where ground-truth correction or detailed feedback was given
                if correction and question:
                    sample = {
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": correction},
                        ],
                        "metadata": {
                            "feedback_id": r.get("feedback_id"),
                            "request_id": r.get("request_id"),
                            "rating": rating,
                            "user_comment": comment,
                            "type": "correction_supervision",
                        },
                    }
                    dataset.append(sample)
                elif rating < 0 and question:
                    # Negative example (DPO rejected sample)
                    sample = {
                        "prompt": question,
                        "rejected_answer": meta.get("rejected_answer", ""),
                        "comment": comment,
                        "rating": rating,
                        "type": "dpo_negative",
                    }
                    dataset.append(sample)

            return dataset
