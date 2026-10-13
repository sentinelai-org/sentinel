"""Typed domain exceptions for Sentinel."""


class SentinelError(Exception):
    """Base exception for all Sentinel domain errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ProviderError(SentinelError):
    """Raised when an LLM provider encounters an error."""

    def __init__(self, message: str, provider: str, details: dict | None = None) -> None:
        super().__init__(message, details)
        self.provider = provider


class ProviderRateLimitError(ProviderError):
    """Raised when an LLM provider encounters a 429 rate limit or quota exhaustion."""

    def __init__(self, message: str, provider: str, retry_after: float | None = None) -> None:
        details = {"retry_after": retry_after} if retry_after else {}
        super().__init__(message, provider=provider, details=details)
        self.retry_after = retry_after


class VectorStoreError(SentinelError):
    """Raised when an operation on the vector store fails."""


class CacheError(SentinelError):
    """Raised when a cache operation encounters an error."""


class DocumentProcessingError(SentinelError):
    """Raised when document loading or chunking fails."""
