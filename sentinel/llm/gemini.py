"""Google Gemini provider implementation using the official google-genai SDK."""

from __future__ import annotations

import time
from typing import Any

from google import genai
from google.genai import types

from sentinel.core.errors import ProviderError, ProviderRateLimitError
from sentinel.core.logging import get_logger
from sentinel.llm.base import LLMProvider, LLMResult

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """LLM provider backed by Google Gemini via google-genai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        if not api_key:
            raise ProviderError("GEMINI_API_KEY is not set or empty", provider="gemini")
        self._api_key = api_key
        self._model = model
        self._client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        return "gemini"

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
        config: dict[str, Any] = {"temperature": temperature}
        if system_instruction:
            config["system_instruction"] = system_instruction

        generate_config = types.GenerateContentConfig(**config)

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=generate_config,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract token counts safely from usage_metadata
            tokens_in = 0
            tokens_out = 0
            if response.usage_metadata:
                tokens_in = response.usage_metadata.prompt_token_count or 0
                tokens_out = response.usage_metadata.candidates_token_count or 0

            return LLMResult(
                text=response.text or "",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=self._model,
                provider=self.name,
                latency_ms=round(elapsed_ms, 2),
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            error_str = str(exc)
            logger.warning(
                "Gemini invocation failed",
                extra={"error": error_str, "model": self._model, "latency_ms": elapsed_ms},
            )

            # Check for rate limiting / 429 indicators
            if (
                "429" in error_str
                or "RESOURCE_EXHAUSTED" in error_str
                or "quota" in error_str.lower()
            ):
                raise ProviderRateLimitError(
                    f"Gemini quota exhausted / rate limit: {error_str}",
                    provider=self.name,
                ) from exc

            raise ProviderError(
                f"Gemini generation error: {error_str}",
                provider=self.name,
            ) from exc
