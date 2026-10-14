"""Pydantic v2 schemas for API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload for POST /v1/chat."""

    question: str = Field(
        ...,
        min_length=1,
        description="The question or query to ask Sentinel.",
        examples=["How do Kubernetes Deployments manage rolling updates?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session or conversation identifier for multi-turn tracking.",
    )


class SourceItem(BaseModel):
    """Retrieved document reference included in the answer."""

    source: str = Field(description="Document source filename or path")
    content: str = Field(description="Snippet or excerpt from the document")
    score: float = Field(description="Similarity relevance score")


class ChatMetadata(BaseModel):
    """Operational and performance metadata for the request."""

    latency_ms: float = Field(description="Total end-to-end request latency in milliseconds")
    tokens_in: int = Field(description="Prompt tokens consumed")
    tokens_out: int = Field(description="Completion/candidate tokens produced")
    provider: str = Field(description="LLM provider name (gemini, ollama, cache, fake)")
    model: str = Field(description="Model identifier that answered the query")
    cache_hit: bool = Field(description="Whether the response was served from exact-match cache")
    request_id: str = Field(description="Unique request tracing ID")
    trace_id: str | None = Field(
        default=None, description="OpenTelemetry 32-hex trace ID for distributed tracing"
    )
    span_id: str | None = Field(
        default=None, description="OpenTelemetry 16-hex span ID for distributed tracing"
    )
    groundedness_score: float | None = Field(
        default=None, description="Calculated factual groundedness verification score"
    )
    groundedness_passed: bool = Field(
        default=True, description="Whether the response passed runtime Groundedness Gate"
    )


class ChatResponse(BaseModel):
    """Full response schema for POST /v1/chat."""

    answer: str = Field(description="Generated answer or refusal message")
    sources: list[SourceItem] = Field(
        default_factory=list, description="List of retrieved source citations"
    )
    meta: ChatMetadata = Field(description="Execution metadata")


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = Field(default="ok")


class ReadyResponse(BaseModel):
    """Readiness probe response detailing component health."""

    status: str = Field(description="'ready' if all critical services pass, else 'not_ready'")
    components: dict[str, bool] = Field(
        description="Health status of downstream dependencies (qdrant, redis)"
    )
