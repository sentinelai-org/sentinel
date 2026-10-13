"""LLM Provider Router with exponential backoff retries and fallback."""

from __future__ import annotations

import asyncio

from sentinel.core.errors import ProviderError
from sentinel.core.logging import get_logger
from sentinel.llm.base import LLMProvider, LLMResult

logger = get_logger(__name__)


class LLMRouter:
    """Routes generation requests to primary provider with retries, then fallback."""

    def __init__(
        self,
        primary_provider: LLMProvider,
        fallback_provider: LLMProvider | None = None,
        max_retries: int = 3,
        initial_backoff_seconds: float = 0.5,
        backoff_multiplier: float = 2.0,
    ) -> None:
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.backoff_multiplier = backoff_multiplier

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        """Attempt primary provider with backoff retries, falling back on persistent error."""
        backoff = self.initial_backoff_seconds
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "Calling primary LLM provider",
                    extra={
                        "provider": self.primary_provider.name,
                        "model": self.primary_provider.model_name,
                        "attempt": attempt,
                    },
                )
                return await self.primary_provider.generate(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Primary provider attempt failed",
                    extra={
                        "provider": self.primary_provider.name,
                        "attempt": attempt,
                        "max_retries": self.max_retries,
                        "error": str(exc),
                    },
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(backoff)
                    backoff *= self.backoff_multiplier

        # Primary failed all retries; attempt fallback provider
        if self.fallback_provider is not None:
            logger.warning(
                "Primary provider exhausted retries, switching to fallback",
                extra={
                    "primary": self.primary_provider.name,
                    "fallback": self.fallback_provider.name,
                    "model": self.fallback_provider.model_name,
                    "primary_last_error": str(last_error),
                },
            )
            try:
                return await self.fallback_provider.generate(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )
            except Exception as fallback_exc:
                logger.error(
                    "Fallback provider also failed",
                    extra={
                        "fallback": self.fallback_provider.name,
                        "error": str(fallback_exc),
                    },
                )
                raise ProviderError(
                    f"Both primary ({self.primary_provider.name}) and fallback "
                    f"({self.fallback_provider.name}) providers failed: {fallback_exc}",
                    provider="router",
                ) from fallback_exc

        raise ProviderError(
            f"Primary provider ({self.primary_provider.name}) failed after {self.max_retries} "
            f"retries, no fallback available: {last_error}",
            provider=self.primary_provider.name,
        ) from last_error
