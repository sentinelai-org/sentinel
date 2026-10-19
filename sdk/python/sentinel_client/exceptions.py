"""Exceptions raised by the Sentinel client."""

from __future__ import annotations

from typing import Any


class SentinelError(Exception):
    """Base exception for all Sentinel SDK errors."""

    def __init__(self, message: str, status_code: int | None = None, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details

    def __str__(self) -> str:
        if self.status_code:
            return f"[HTTP {self.status_code}] {self.message}"
        return self.message


class AuthenticationError(SentinelError):
    """Raised when an API key is missing, expired, or invalid (HTTP 401)."""


class RateLimitError(SentinelError):
    """Raised when the client exceeds the rate limit quota (HTTP 429)."""

    def __init__(self, message: str, retry_after: int | None = None, details: Any = None) -> None:
        super().__init__(message, status_code=429, details=details)
        self.retry_after = retry_after


class PayloadTooLargeError(SentinelError):
    """Raised when the request payload exceeds the maximum allowed size (HTTP 413)."""

    def __init__(self, message: str, details: Any = None) -> None:
        super().__init__(message, status_code=413, details=details)


class ServiceUnavailableError(SentinelError):
    """Raised when the Sentinel gateway or downstream dependencies are unhealthy (HTTP 503)."""
