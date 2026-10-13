"""Local Ollama provider implementation."""

from __future__ import annotations

import time

import httpx

from sentinel.core.errors import ProviderError
from sentinel.core.logging import get_logger
from sentinel.llm.base import LLMProvider, LLMResult

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """Local LLM provider backed by Ollama service."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> LLMResult:
        start_time = time.perf_counter()
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system_instruction:
            payload["system"] = system_instruction

        endpoint = f"{self._base_url}/api/generate"

        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            tokens_in = data.get("prompt_eval_count", 0)
            tokens_out = data.get("eval_count", 0)
            text = data.get("response", "")

            return LLMResult(
                text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=self._model,
                provider=self.name,
                latency_ms=round(elapsed_ms, 2),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "Ollama invocation failed",
                extra={"error": str(exc), "model": self._model, "latency_ms": elapsed_ms},
            )
            raise ProviderError(
                f"Ollama generation failed: {exc}",
                provider=self.name,
            ) from exc
