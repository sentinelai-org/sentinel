"""LLM Provider Router with exponential backoff retries and fallback."""

from __future__ import annotations

import asyncio

from sentinel.core.errors import ProviderError
from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer
from sentinel.llm.base import LLMProvider, LLMResult

logger = get_logger(__name__)
tracer = get_tracer("sentinel.llm")


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
        with tracer.start_as_current_span("llm.route") as route_span:
            route_span.set_attribute("llm.primary_provider", self.primary_provider.name)
            route_span.set_attribute(
                "llm.fallback_provider",
                self.fallback_provider.name if self.fallback_provider else "none",
            )

            backoff = self.initial_backoff_seconds
            last_error: Exception | None = None

            for attempt in range(1, self.max_retries + 1):
                with tracer.start_as_current_span("llm.call") as call_span:
                    call_span.set_attribute("gen_ai.system", self.primary_provider.name)
                    call_span.set_attribute(
                        "gen_ai.request.model", self.primary_provider.model_name
                    )
                    call_span.set_attribute("gen_ai.attempt", attempt)
                    call_span.set_attribute("gen_ai.temperature", temperature)

                    try:
                        logger.info(
                            "Calling primary LLM provider",
                            extra={
                                "provider": self.primary_provider.name,
                                "model": self.primary_provider.model_name,
                                "attempt": attempt,
                            },
                        )
                        result = await self.primary_provider.generate(
                            prompt=prompt,
                            system_instruction=system_instruction,
                            temperature=temperature,
                        )
                        call_span.set_attribute("gen_ai.response.model", result.model)
                        call_span.set_attribute("gen_ai.usage.input_tokens", result.tokens_in)
                        call_span.set_attribute("gen_ai.usage.output_tokens", result.tokens_out)
                        route_span.set_attribute("llm.resolved_provider", result.provider)
                        route_span.set_attribute("llm.resolved_model", result.model)
                        return result
                    except Exception as exc:
                        last_error = exc
                        call_span.record_exception(exc)
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
                route_span.set_attribute("llm.fallback_triggered", True)
                logger.warning(
                    "Primary provider exhausted retries, switching to fallback",
                    extra={
                        "primary": self.primary_provider.name,
                        "fallback": self.fallback_provider.name,
                        "model": self.fallback_provider.model_name,
                        "primary_last_error": str(last_error),
                    },
                )
                with tracer.start_as_current_span("llm.call_fallback") as fallback_span:
                    fallback_span.set_attribute("gen_ai.system", self.fallback_provider.name)
                    fallback_span.set_attribute(
                        "gen_ai.request.model", self.fallback_provider.model_name
                    )
                    fallback_span.set_attribute("gen_ai.fallback_reason", str(last_error))
                    try:
                        result = await self.fallback_provider.generate(
                            prompt=prompt,
                            system_instruction=system_instruction,
                            temperature=temperature,
                        )
                        fallback_span.set_attribute("gen_ai.response.model", result.model)
                        fallback_span.set_attribute("gen_ai.usage.input_tokens", result.tokens_in)
                        fallback_span.set_attribute("gen_ai.usage.output_tokens", result.tokens_out)
                        route_span.set_attribute("llm.resolved_provider", result.provider)
                        route_span.set_attribute("llm.resolved_model", result.model)
                        return result
                    except Exception as fallback_exc:
                        fallback_span.record_exception(fallback_exc)
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
