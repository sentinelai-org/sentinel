"""Sentinel Client: Official Python SDK for the Sentinel Enterprise AI & RAG Gateway."""

from __future__ import annotations

from sentinel_client.client import AsyncSentinelClient, SentinelClient
from sentinel_client.exceptions import (
    AuthenticationError,
    PayloadTooLargeError,
    RateLimitError,
    SentinelError,
    ServiceUnavailableError,
)
from sentinel_client.models import (
    ChatMetadata,
    ChatResponse,
    FeedbackResponse,
    IngestDocument,
    IngestJobResponse,
    IngestJobStatus,
    SourceItem,
)

__version__ = "0.1.0"

__all__ = [
    "SentinelClient",
    "AsyncSentinelClient",
    "SentinelError",
    "AuthenticationError",
    "RateLimitError",
    "PayloadTooLargeError",
    "ServiceUnavailableError",
    "ChatResponse",
    "ChatMetadata",
    "SourceItem",
    "IngestDocument",
    "IngestJobResponse",
    "IngestJobStatus",
    "FeedbackResponse",
]
