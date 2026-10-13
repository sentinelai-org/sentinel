"""Deterministic FakeProvider for zero-cost testing and CI environments."""

from __future__ import annotations

import asyncio

from sentinel.llm.base import LLMProvider, LLMResult


class FakeProvider(LLMProvider):
    """Fake LLM provider for unit and integration testing without external API calls."""

    def __init__(
        self,
        default_response: str = "This is a deterministic fake answer from Sentinel.",
        tokens_in: int = 15,
        tokens_out: int = 10,
        latency_ms: float = 12.5,
        model: str = "fake-model-v1",
        provider_name: str = "fake",
    ) -> None:
        self._default_response = default_response
        self._tokens_in = tokens_in
        self._tokens_out = tokens_out
        self._latency_ms = latency_ms
        self._model = model
        self._provider_name = provider_name
        self.call_count = 0
        self.last_prompt: str | None = None

    @property
    def name(self) -> str:
        return self._provider_name

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
        self.last_prompt = prompt
        _ = (system_instruction, temperature)
        # Simulate brief async delay
        await asyncio.sleep(0.001)

        return LLMResult(
            text=self._default_response,
            tokens_in=self._tokens_in,
            tokens_out=self._tokens_out,
            model=self._model,
            provider=self.name,
            latency_ms=self._latency_ms,
        )
