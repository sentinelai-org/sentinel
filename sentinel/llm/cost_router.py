"""Dynamic Cost-Aware Router and Token Pricing Accounting."""

from __future__ import annotations

import re
from typing import ClassVar

from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer
from sentinel.llm.base import LLMProvider, LLMResult
from sentinel.llm.router import LLMRouter

logger = get_logger(__name__)
tracer = get_tracer("sentinel.llm.cost_router")

# Pricing table per 1,000,000 tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "qwen2.5:3b": {"input": 0.0, "output": 0.0},
    "fake-model-v1": {"input": 0.0, "output": 0.0},
    "mock-eval-v1": {"input": 0.0, "output": 0.0},
}


def calculate_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Calculate inference cost in USD from token counts."""
    # Find matching pricing rule
    pricing = None
    for model_key, rates in MODEL_PRICING.items():
        if model_key in model.lower():
            pricing = rates
            break

    if pricing is None:
        # Default conservative fallback estimate ($0.10/1M in, $0.40/1M out)
        pricing = {"input": 0.10, "output": 0.40}

    cost_in = (tokens_in / 1_000_000) * pricing["input"]
    cost_out = (tokens_out / 1_000_000) * pricing["output"]
    return round(cost_in + cost_out, 6)


class ComplexityClassifier:
    """Evaluates question complexity to decide whether to route to economy or frontier models."""

    COMPLEX_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(
            r"\b(compare|difference between|analyze|contrast|trade-offs|architecture|troubleshoot|why does|explain in detail|step[- ]by[- ]step)\b",
            re.IGNORECASE,
        ),
    ]

    @classmethod
    def is_complex(cls, question: str) -> bool:
        """Return True if question exhibits reasoning complexity or high token length."""
        # Queries longer than 25 words or matching complex intent patterns
        words = question.split()
        if len(words) > 25:
            return True

        for pattern in cls.COMPLEX_PATTERNS:
            if pattern.search(question):
                return True

        return False


class CostAwareRouter(LLMRouter):
    """Dynamic router balancing cost, complexity, and availability."""

    def __init__(
        self,
        primary_provider: LLMProvider,
        fallback_provider: LLMProvider | None = None,
        economy_provider: LLMProvider | None = None,
        cost_routing_enabled: bool = True,
        max_retries: int = 3,
        initial_backoff_seconds: float = 0.5,
        backoff_multiplier: float = 2.0,
    ) -> None:
        super().__init__(
            primary_provider=primary_provider,
            fallback_provider=fallback_provider,
            max_retries=max_retries,
            initial_backoff_seconds=initial_backoff_seconds,
            backoff_multiplier=backoff_multiplier,
        )
        self.economy_provider = economy_provider
        self.cost_routing_enabled = cost_routing_enabled

    async def route_and_generate(
        self,
        prompt: str,
        question: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> tuple[LLMResult, float, str]:
        """Route to appropriate model tier based on complexity and calculate cost.

        Returns tuple of (LLMResult, cost_usd, complexity_tier).
        """
        complexity = "complex" if ComplexityClassifier.is_complex(question) else "simple"

        with tracer.start_as_current_span("llm.cost_router") as span:
            span.set_attribute("llm.complexity", complexity)
            span.set_attribute("llm.cost_routing_enabled", self.cost_routing_enabled)

            # Route to economy provider if question is simple and economy provider is configured
            if self.cost_routing_enabled and complexity == "simple" and self.economy_provider:
                span.set_attribute("llm.selected_tier", "economy")
                try:
                    logger.info(
                        "Routing simple query to economy provider",
                        extra={"provider": self.economy_provider.name},
                    )
                    result = await self.economy_provider.generate(
                        prompt, system_instruction=system_instruction, temperature=temperature
                    )
                    cost = calculate_cost_usd(result.model, result.tokens_in, result.tokens_out)
                    span.set_attribute("llm.cost_usd", cost)
                    return result, cost, complexity
                except Exception as exc:
                    logger.warning(
                        "Economy provider failed, falling back to primary",
                        extra={"error": str(exc)},
                    )
                    span.record_exception(exc)

            # Otherwise use primary router with fallback
            span.set_attribute("llm.selected_tier", "primary")
            result = await self.generate(
                prompt, system_instruction=system_instruction, temperature=temperature
            )
            cost = calculate_cost_usd(result.model, result.tokens_in, result.tokens_out)
            span.set_attribute("llm.cost_usd", cost)
            return result, cost, complexity
