"""Unit tests for Cost-Aware Router and Token Pricing Accounting."""

import pytest

from sentinel.llm.cost_router import (
    ComplexityClassifier,
    CostAwareRouter,
    calculate_cost_usd,
)
from sentinel.llm.fake import FakeProvider


def test_calculate_cost_usd_accurately_computes_prices() -> None:
    # 1,000,000 in, 1,000,000 out for gemini-2.5-flash: $0.075 + $0.30 = $0.375
    cost_1m = calculate_cost_usd("gemini-2.5-flash", 1_000_000, 1_000_000)
    assert cost_1m == 0.375

    # 1,000 in, 500 out: (1000/1M)*0.075 + (500/1M)*0.30 = 0.000075 + 0.00015 = 0.000225
    cost_small = calculate_cost_usd("gemini-2.5-flash", 1_000, 500)
    assert cost_small == 0.000225

    # Local Ollama is free ($0.0)
    cost_ollama = calculate_cost_usd("qwen2.5:3b", 10_000, 2_000)
    assert cost_ollama == 0.0


def test_complexity_classifier_identifies_simple_vs_complex_queries() -> None:
    simple_q1 = "What is a Pod?"
    assert ComplexityClassifier.is_complex(simple_q1) is False

    simple_q2 = "How to define a ReplicaSet in YAML?"
    assert ComplexityClassifier.is_complex(simple_q2) is False

    complex_q1 = "Compare the difference between ClusterIP and NodePort architecture."
    assert ComplexityClassifier.is_complex(complex_q1) is True

    complex_q2 = "Analyze the trade-offs of rolling update strategies and explain in detail."
    assert ComplexityClassifier.is_complex(complex_q2) is True

    # Long query over 25 words
    long_q = " ".join(["word"] * 26)
    assert ComplexityClassifier.is_complex(long_q) is True


@pytest.mark.asyncio
async def test_cost_aware_router_routes_simple_to_economy() -> None:
    primary = FakeProvider(
        default_response="Frontier model answer", model="gemini-2.5-flash", provider_name="frontier"
    )
    economy = FakeProvider(
        default_response="Economy local answer", model="qwen2.5:3b", provider_name="economy"
    )

    router = CostAwareRouter(
        primary_provider=primary,
        economy_provider=economy,
        cost_routing_enabled=True,
    )

    result, cost, tier = await router.route_and_generate(
        prompt="Explain pod",
        question="What is a Pod?",
    )

    assert tier == "simple"
    assert result.provider == "economy"
    assert result.text == "Economy local answer"
    assert cost == 0.0
    assert economy.call_count == 1
    assert primary.call_count == 0


@pytest.mark.asyncio
async def test_cost_aware_router_routes_complex_to_primary() -> None:
    primary = FakeProvider(
        default_response="Frontier model answer", model="gemini-2.5-flash", provider_name="frontier"
    )
    economy = FakeProvider(
        default_response="Economy local answer", model="qwen2.5:3b", provider_name="economy"
    )

    router = CostAwareRouter(
        primary_provider=primary,
        economy_provider=economy,
        cost_routing_enabled=True,
    )

    result, cost, tier = await router.route_and_generate(
        prompt="Compare architecture",
        question="Analyze the difference between ClusterIP and NodePort and explain in detail.",
    )

    assert tier == "complex"
    assert result.provider == "frontier"
    assert result.text == "Frontier model answer"
    assert cost > 0.0
    assert primary.call_count == 1
    assert economy.call_count == 0
