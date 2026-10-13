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

    # Redis Cache
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    cache_ttl_seconds: int = Field(
        default=3600, description="Exact-match response cache TTL in seconds"
    )
    cache_enabled: bool = Field(
        default=True, description="Enable or disable Redis exact-match cache"
    )

    # RAG & Context Grading
    retrieval_top_k: int = Field(default=4, description="Number of top chunks to retrieve")
    score_threshold: float = Field(
        default=0.45, description="Minimum similarity score threshold for context grading"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached singleton application settings."""
    return Settings()
