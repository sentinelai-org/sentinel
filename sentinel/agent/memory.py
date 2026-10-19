"""Redis-backed multi-turn conversational session memory."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis

from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer

logger = get_logger(__name__)
tracer = get_tracer("sentinel.memory")


class SessionManager:
    """Manages conversational session threads and sliding-window history using Redis."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 86400,
        max_turns: int = 6,
        enabled: bool = True,
    ) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self.enabled = enabled
        self._redis: aioredis.Redis | None = None
        # In-memory fallback for local dev/testing without active Redis
        self._fallback_store: dict[str, list[dict[str, Any]]] = {}

    def _get_client(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    @staticmethod
    def _key(session_id: str) -> str:
        return f"sentinel:session:{session_id}:messages"

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Append a message turn to the session history."""
        if not self.enabled or not session_id:
            return {}

        message = {
            "role": role,
            "content": content,
            "sources": sources or [],
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Keep fallback store updated
        if session_id not in self._fallback_store:
            self._fallback_store[session_id] = []
        self._fallback_store[session_id].append(message)

        try:
            client = self._get_client()
            key = self._key(session_id)
            payload = json.dumps(message)
            async with client.pipeline(transaction=True) as pipe:
                pipe.rpush(key, payload)
                pipe.expire(key, self.ttl_seconds)
                await pipe.execute()
        except Exception as exc:
            logger.debug(
                "Redis session append failed, used in-memory store",
                extra={"session_id": session_id, "error": str(exc)},
            )

        return message

    async def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all recorded messages for a session."""
        if not self.enabled or not session_id:
            return []

        try:
            client = self._get_client()
            key = self._key(session_id)
            raw_items = await client.lrange(key, 0, -1)
            if raw_items:
                return [json.loads(item) for item in raw_items]
        except Exception as exc:
            logger.debug(
                "Redis session get failed, checking in-memory store",
                extra={"session_id": session_id, "error": str(exc)},
            )

        return list(self._fallback_store.get(session_id, []))

    async def get_recent_turns(
        self, session_id: str, max_turns: int | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve recent conversation turns (pairs of user and assistant messages)."""
        history = await self.get_history(session_id)
        limit = (max_turns or self.max_turns) * 2
        return history[-limit:] if len(history) > limit else history

    async def format_history_for_prompt(
        self, session_id: str, max_turns: int | None = None
    ) -> str:
        """Format recent conversation history as readable dialog text for LLM context."""
        recent_messages = await self.get_recent_turns(session_id, max_turns)
        if not recent_messages:
            return ""

        formatted_lines: list[str] = []
        for msg in recent_messages:
            role_label = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            formatted_lines.append(f"{role_label}: {content}")

        return "\n".join(formatted_lines)

    async def clear_session(self, session_id: str) -> bool:
        """Delete session history completely."""
        self._fallback_store.pop(session_id, None)
        if not self.enabled or not session_id:
            return True

        try:
            client = self._get_client()
            key = self._key(session_id)
            deleted = await client.delete(key)
            return bool(deleted > 0)
        except Exception as exc:
            logger.warning(
                "Redis clear_session failed",
                extra={"session_id": session_id, "error": str(exc)},
            )
            return True

    async def close(self) -> None:
        """Close the underlying Redis connection pool."""
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
