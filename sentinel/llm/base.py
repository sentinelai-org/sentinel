"""Abstract base class and data containers for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResult:
    """Normalized result returned by any LLM provider."""

    text: str
    tokens_in: int
    tokens_out: int
    model: str
    provider: str
    latency_ms: float


class LLMProvider(ABC):
    """Abstract interface that all LLM providers must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider (e.g. 'gemini', 'ollama', 'fake')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model utilized by this provider."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        """Generate a response from the model."""
        ...

    async def generate_stream(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        """Stream tokens from the model. Defaults to single-shot yield."""
        result = await self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )
        yield result.text
