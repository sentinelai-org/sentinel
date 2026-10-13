"""Abstract base class and data containers for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
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
