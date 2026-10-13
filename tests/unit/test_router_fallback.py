"""Unit tests for LLM provider router fallback and retry behavior."""

from __future__ import annotations

import pytest

from sentinel.core.errors import ProviderError, ProviderRateLimitError
from sentinel.llm.base import LLMProvider, LLMResult
from sentinel.llm.fake import FakeProvider
from sentinel.llm.router import LLMRouter


class FailingProvider(LLMProvider):
    """Provider that always fails, tracking call count."""

    def __init__(self, exception: Exception, name: str = "failing", model: str = "fail-v1") -> None:
        self._exception = exception
        self._name = name
        self._model = model
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        self.call_count += 1
        _ = (prompt, system_instruction, temperature)
        raise self._exception


@pytest.mark.asyncio
async def test_router_primary_success() -> None:
    """Primary provider succeeds on first attempt."""
    primary = FakeProvider(default_response="Primary ok", provider_name="gemini")
    fallback = FakeProvider(default_response="Fallback ok", provider_name="ollama")

    router = LLMRouter(primary_provider=primary, fallback_provider=fallback)
    result = await router.generate("test question")

    assert result.text == "Primary ok"
    assert result.provider == "gemini"
    assert primary.call_count == 1
    assert fallback.call_count == 0


@pytest.mark.asyncio
async def test_router_fallback_on_429() -> None:
    """Primary hits 429 rate limit, retries 3 times, then falls back to Ollama."""
    rate_limit_error = ProviderRateLimitError("Resource exhausted (429)", provider="gemini")
    primary = FailingProvider(exception=rate_limit_error, name="gemini", model="gemini-2.5-flash")
    fallback = FakeProvider(
        default_response="Fallback response from Ollama",
        provider_name="ollama",
        model="qwen2.5:3b",
    )

    router = LLMRouter(
        primary_provider=primary,
        fallback_provider=fallback,
        max_retries=3,
        initial_backoff_seconds=0.01,  # fast backoff for tests
        backoff_multiplier=2.0,
    )

    result = await router.generate("What is a pod?")

    assert result.text == "Fallback response from Ollama"
    assert result.provider == "ollama"
    assert result.model == "qwen2.5:3b"
    assert primary.call_count == 3
    assert fallback.call_count == 1


@pytest.mark.asyncio
async def test_router_both_fail_raises_provider_error() -> None:
    """If both primary and fallback fail, ProviderError is raised."""
    primary = FailingProvider(
        exception=ProviderRateLimitError("429 Too Many Requests", provider="gemini"),
        name="gemini",
    )
    fallback = FailingProvider(
        exception=ProviderError("Connection refused", provider="ollama"),
        name="ollama",
    )

    router = LLMRouter(
        primary_provider=primary,
        fallback_provider=fallback,
        max_retries=3,
        initial_backoff_seconds=0.01,
    )

    with pytest.raises(ProviderError) as exc_info:
        await router.generate("hello")

    assert "Both primary (gemini) and fallback (ollama) providers failed" in str(exc_info.value)
    assert primary.call_count == 3
    assert fallback.call_count == 1
