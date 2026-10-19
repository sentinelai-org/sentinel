"""Application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General Environment
    environment: str = Field(default="development", description="Runtime environment name")
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")

    # API Server
    host: str = Field(default="0.0.0.0", description="API bind host")
    port: int = Field(default=8000, description="API bind port")

    # Google Gemini Configuration (google-genai SDK)
    gemini_api_key: str = Field(default="", description="Google GenAI API key")
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Google Gemini model identifier (e.g. gemini-2.5-flash, gemini-1.5-flash)",
    )

    # Ollama Fallback Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Base URL for local Ollama service"
    )
    ollama_model: str = Field(
        default="qwen2.5:3b", description="Model identifier for Ollama fallback"
    )
    ollama_timeout_seconds: float = Field(
        default=30.0, description="Timeout in seconds for Ollama HTTP requests"
    )

    # Vector Database & Embeddings
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant service URL")
    qdrant_collection: str = Field(
        default="sentinel_docs", description="Name of the Qdrant collection"
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Local sentence-transformers embedding model name",
    )

    # Redis Cache & Background Queue (Phase 5)
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    cache_ttl_seconds: int = Field(
        default=3600, description="Exact-match response cache TTL in seconds"
    )
    cache_enabled: bool = Field(
        default=True, description="Enable or disable Redis exact-match cache"
    )
    ingest_stream_name: str = Field(
        default="sentinel:stream:ingest_jobs",
        description="Redis stream key for background ingestion jobs",
    )
    ingest_consumer_group: str = Field(
        default="sentinel_workers", description="Redis consumer group for background ingest workers"
    )

    # Conversational Session Memory (Phase 8)
    session_memory_enabled: bool = Field(
        default=True, description="Enable multi-turn conversational session memory in Redis"
    )
    session_history_ttl_seconds: int = Field(
        default=86400, description="TTL in seconds for session conversation history (default 24h)"
    )
    session_max_history_turns: int = Field(
        default=6, description="Maximum previous conversation turns included in LLM context"
    )

    # RAG, Hybrid Search & Re-ranking (RAG Precision)
    retrieval_top_k: int = Field(default=4, description="Number of top chunks to pass to LLM")
    score_threshold: float = Field(
        default=0.45, description="Minimum similarity score threshold for context grading"
    )
    hybrid_search_enabled: bool = Field(
        default=True, description="Enable hybrid dense + BM25 sparse keyword search via RRF"
    )
    hybrid_rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion smoothing constant k")
    retrieval_candidates_k: int = Field(
        default=20, description="Number of candidate chunks retrieved before re-ranking"
    )
    reranker_enabled: bool = Field(
        default=True, description="Enable Cross-Encoder re-ranking on retrieved candidates"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model identifier for re-ranking (e.g. BAAI/bge-reranker-base)",
    )
    bm25_index_path: str = Field(
        default="data/bm25_index.json",
        description="Local path to serialized BM25 sparse index",
    )

    # OpenTelemetry Distributed Tracing (Phase 2)
    otel_enabled: bool = Field(default=True, description="Enable OpenTelemetry distributed tracing")
    otel_service_name: str = Field(
        default="sentinel-gateway", description="Service name reported to tracing collectors"
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4318/v1/traces",
        description="OTLP HTTP collector endpoint for traces (e.g. Jaeger or Phoenix)",
    )

    # Automated Evaluations & Groundedness Gate (Phase 3)
    groundedness_threshold: float = Field(
        default=0.70, description="Minimum groundedness score for runtime gate verification"
    )
    groundedness_gate_enabled: bool = Field(
        default=True, description="Enable runtime Groundedness Gate verification"
    )

    # Safety Guardrails, Semantic Cache & Cost Routing (Phase 4)
    guardrails_enabled: bool = Field(
        default=True, description="Enable input and output safety guardrails"
    )
    semantic_cache_enabled: bool = Field(
        default=True, description="Enable vector-based semantic response caching"
    )
    semantic_cache_threshold: float = Field(
        default=0.90, description="Cosine similarity threshold for semantic cache hits"
    )
    semantic_cache_collection: str = Field(
        default="sentinel_semantic_cache", description="Qdrant collection for semantic cache"
    )
    cost_routing_enabled: bool = Field(
        default=True, description="Enable complexity-based dynamic cost-aware routing"
    )

    # Enterprise Security & Hardening
    api_key_auth_enabled: bool = Field(
        default=False, description="Require Bearer API key or X-API-Key on protected endpoints"
    )
    sentinel_api_keys: list[str] = Field(
        default_factory=lambda: ["sk-sentinel-default-key"],
        description="Authorized API keys for gateway access",
    )
    rate_limit_enabled: bool = Field(
        default=True, description="Enable sliding window rate limiting to prevent DoS attacks"
    )
    rate_limit_requests: int = Field(
        default=120, description="Maximum allowed requests per sliding window"
    )
    rate_limit_window_seconds: int = Field(
        default=60, description="Rate limit window duration in seconds"
    )
    max_request_size_bytes: int = Field(
        default=2 * 1024 * 1024, description="Maximum HTTP request body size in bytes (2MB)"
    )
    security_headers_enabled: bool = Field(
        default=True, description="Enforce OWASP HTTP security headers on all responses"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached singleton application settings."""
    return Settings()
