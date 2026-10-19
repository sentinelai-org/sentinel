"""Data models for Sentinel SDK requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    """A cited documentation chunk returned by Sentinel."""

    source: str = Field(description="Document source name or file path")
    content: str = Field(description="Document text excerpt or snippet")
    score: float = Field(description="Retrieval / re-ranker relevance score")


class ChatMetadata(BaseModel):
    """Operational and accounting metadata for a Sentinel request."""

    latency_ms: float = Field(description="End-to-end execution duration in ms")
    tokens_in: int = Field(description="Input prompt tokens consumed")
    tokens_out: int = Field(description="Output tokens generated")
    provider: str = Field(
        description="Model provider that fulfilled the query (gemini, ollama, cache)"
    )
    model: str = Field(description="Specific model identifier")
    cache_hit: bool = Field(default=False, description="Whether served from cache")
    cache_type: str | None = Field(
        default=None, description="Type of cache (exact, semantic, or none)"
    )
    cost_usd: float | None = Field(default=None, description="Incurred inference cost in USD")
    groundedness_score: float | None = Field(
        default=None, description="Factual token overlap score"
    )
    groundedness_passed: bool = Field(default=True, description="Whether Groundedness Gate passed")
    guardrail_passed: bool = Field(default=True, description="Whether safety guardrails passed")
    request_id: str = Field(description="Unique request tracing correlation ID")
    trace_id: str | None = Field(default=None, description="OpenTelemetry 32-hex trace ID")
    span_id: str | None = Field(default=None, description="OpenTelemetry span ID")


class ChatResponse(BaseModel):
    """Response returned by the /v1/chat endpoint."""

    answer: str = Field(description="The verified generated answer")
    sources: list[SourceItem] = Field(default_factory=list, description="Citations used in answer")
    meta: ChatMetadata = Field(description="Execution and cost metadata")
    session_id: str | None = Field(
        default=None, description="Active conversation thread session ID"
    )


class IngestDocument(BaseModel):
    """A document to be ingested into the knowledge base."""

    content: str = Field(..., min_length=1, description="Raw markdown or text content")
    source: str = Field(default="api_upload", description="Source path or filename")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata tags (e.g. tenant_id)"
    )


class IngestJobResponse(BaseModel):
    """Response returned upon submitting an ingestion job."""

    job_id: str
    status: str
    total_documents: int
    created_at: str


class IngestJobStatus(BaseModel):
    """Current status of an asynchronous ingestion job."""

    job_id: str
    status: str
    total_documents: int
    processed_documents: int
    progress_percent: float
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class FeedbackResponse(BaseModel):
    """Response returned upon submitting feedback."""

    status: str
    request_id: str
    feedback_id: str


class ChatMessage(BaseModel):
    """A message turn within a conversation thread."""

    role: str
    content: str
    timestamp: str | None = None
    sources: list[SourceItem] = Field(default_factory=list)


class SessionHistoryResponse(BaseModel):
    """Complete message history for a conversation thread."""

    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    total_messages: int = 0
