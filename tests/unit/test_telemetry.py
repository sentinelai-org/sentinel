"""Unit tests for Sentinel OpenTelemetry tracing, span attributes, and log correlation."""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from unittest.mock import AsyncMock

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sentinel.agent.cache import CachedResponse, ResponseCache
from sentinel.core.config import Settings
from sentinel.core.logging import JSONFormatter
from sentinel.core.telemetry import (
    get_current_span_id,
    get_current_trace_id,
    get_tracer,
    setup_in_memory_telemetry,
    setup_telemetry,
    shutdown_telemetry,
)
from sentinel.llm.fake import FakeProvider
from sentinel.llm.router import LLMRouter


@pytest.fixture(autouse=True)
def clean_telemetry() -> Generator[InMemorySpanExporter, None, None]:
    """Ensure in-memory telemetry is isolated for each test."""
    provider, exporter = setup_in_memory_telemetry()
    yield exporter
    shutdown_telemetry()


def test_trace_and_span_id_formatting() -> None:
    """Verify trace_id and span_id are properly formatted hex strings inside a span."""
    assert get_current_trace_id() is None
    assert get_current_span_id() is None

    tracer = get_tracer("test.tracer")
    with tracer.start_as_current_span("test.span"):
        trace_id = get_current_trace_id()
        span_id = get_current_span_id()

        assert trace_id is not None
        assert len(trace_id) == 32
        assert int(trace_id, 16) > 0

        assert span_id is not None
        assert len(span_id) == 16
        assert int(span_id, 16) > 0

    assert get_current_trace_id() is None
    assert get_current_span_id() is None


def test_json_formatter_includes_trace_context() -> None:
    """Verify structured JSON logs contain trace_id and span_id when inside active span."""
    formatter = JSONFormatter()
    logger = logging.getLogger("test.logger")

    # Outside span
    record_outside = logger.makeRecord(
        "test.logger", logging.INFO, "test.py", 1, "Hello outside", (), None
    )
    json_outside = json.loads(formatter.format(record_outside))
    assert json_outside["trace_id"] is None
    assert json_outside["span_id"] is None

    # Inside span
    tracer = get_tracer("test.tracer")
    with tracer.start_as_current_span("active.span"):
        expected_trace_id = get_current_trace_id()
        expected_span_id = get_current_span_id()

        record_inside = logger.makeRecord(
            "test.logger", logging.INFO, "test.py", 2, "Hello inside", (), None
        )
        json_inside = json.loads(formatter.format(record_inside))

        assert json_inside["trace_id"] == expected_trace_id
        assert json_inside["span_id"] == expected_span_id
        assert json_inside["message"] == "Hello inside"


def test_setup_telemetry_disabled() -> None:
    """Verify setup_telemetry functions cleanly when otel_enabled is False."""
    settings = Settings(otel_enabled=False)
    provider = setup_telemetry(settings)
    assert provider is not None
    shutdown_telemetry()


@pytest.mark.asyncio
async def test_router_spans_record_gen_ai_attributes(
    clean_telemetry: InMemorySpanExporter,
) -> None:
    """Verify LLMRouter produces spans adhering to GenAI semantic conventions."""
    exporter = clean_telemetry
    fake_provider = FakeProvider(
        default_response="Test response",
        tokens_in=50,
        tokens_out=25,
        provider_name="test-llm",
        model="test-model-1",
    )
    router = LLMRouter(primary_provider=fake_provider)

    result = await router.generate(prompt="Explain Kubernetes")
    assert result.text == "Test response"

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert "llm.call" in span_names
    assert "llm.route" in span_names

    call_span = next(s for s in spans if s.name == "llm.call")
    assert call_span.attributes is not None
    assert call_span.attributes["gen_ai.system"] == "test-llm"
    assert call_span.attributes["gen_ai.request.model"] == "test-model-1"
    assert call_span.attributes["gen_ai.usage.input_tokens"] == 50
    assert call_span.attributes["gen_ai.usage.output_tokens"] == 25


@pytest.mark.asyncio
async def test_cache_spans_record_hit_and_miss(
    clean_telemetry: InMemorySpanExporter,
) -> None:
    """Verify ResponseCache emits cache.get and cache.set spans with hit/miss attributes."""
    exporter = clean_telemetry
    cache = ResponseCache(enabled=True)

    # Mock internal redis client
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    cache._redis = mock_redis

    # Miss
    cached = await cache.get("non-existent question")
    assert cached is None

    spans = exporter.get_finished_spans()
    get_span = next(s for s in spans if s.name == "cache.get")
    assert get_span.attributes is not None
    assert get_span.attributes["cache.system"] == "redis"
    assert get_span.attributes["cache.hit"] is False

    # Set
    mock_redis.set.return_value = True
    test_response = CachedResponse(
        answer="Cached answer",
        sources=[],
        model="test-model",
        provider="test-provider",
        tokens_in=10,
        tokens_out=20,
    )
    await cache.set("test question", test_response)

    spans = exporter.get_finished_spans()
    set_span = next(s for s in spans if s.name == "cache.set")
    assert set_span.attributes is not None
    assert set_span.attributes["cache.system"] == "redis"
    assert set_span.attributes["cache.ttl_seconds"] == 3600

    # Hit
    mock_redis.get.return_value = json.dumps(
        {
            "answer": "Cached answer",
            "sources": [],
            "model": "test-model",
            "provider": "test-provider",
            "tokens_in": 10,
            "tokens_out": 20,
        }
    )
    hit_result = await cache.get("test question")
    assert hit_result is not None
    assert hit_result.answer == "Cached answer"

    spans = exporter.get_finished_spans()
    hit_spans = [
        s
        for s in spans
        if s.name == "cache.get"
        and s.attributes is not None
        and s.attributes.get("cache.hit") is True
    ]
    assert len(hit_spans) == 1


@pytest.mark.asyncio
async def test_agent_nodes_tracing_spans(
    clean_telemetry: InMemorySpanExporter,
) -> None:
    """Verify AgentNodeHandler generates rag.retrieve, rag.grade_context, and agent.generate spans."""
    from sentinel.agent.nodes import AgentNodeHandler
    from sentinel.rag.chunker import Chunk
    from tests.conftest import InMemoryVectorStore

    exporter = clean_telemetry
    vector_store = InMemoryVectorStore()
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        text="Kubernetes Pods documentation",
        index=0,
        metadata={"source": "d1.md"},
    )
    await vector_store.upsert([chunk], [[0.1] * 384])

    class DummyEmbedder:
        def embed_text(self, text: str) -> list[float]:
            _ = text
            return [0.1] * 384

    fake_llm = FakeProvider(
        default_response="Pods are the smallest unit",
        tokens_in=30,
        tokens_out=15,
        provider_name="test-llm",
        model="test-model",
    )
    router = LLMRouter(primary_provider=fake_llm)

    handler = AgentNodeHandler(
        vector_store=vector_store,
        embedder=DummyEmbedder(),  # type: ignore[arg-type]
        llm_router=router,
        score_threshold=0.5,
        top_k=2,
    )

    # 1. Retrieve
    ret_res = await handler.retrieve({"question": "What is a Pod?"})
    assert len(ret_res["documents"]) == 1

    # 2. Grade context
    grade_res = await handler.grade_context(
        {"question": "What is a Pod?", "documents": ret_res["documents"]}
    )
    assert grade_res["context_relevant"] is True

    # 3. Generate
    gen_res = await handler.generate(
        {
            "question": "What is a Pod?",
            "documents": grade_res["documents"],
            "meta": {},
        }
    )
    assert "smallest unit" in gen_res["answer"]

    # 4. No context answer
    no_ctx_res = await handler.no_context_answer({"question": "Unknown topic"})
    assert "could not find sufficient" in no_ctx_res["answer"]

    # Verify all generated spans
    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    assert "rag.retrieve" in span_names
    assert "rag.grade_context" in span_names
    assert "agent.generate" in span_names
    assert "agent.no_context_answer" in span_names

    retrieve_span = next(s for s in spans if s.name == "rag.retrieve")
    assert retrieve_span.attributes is not None
    assert retrieve_span.attributes["rag.query"] == "What is a Pod?"
    assert retrieve_span.attributes["rag.top_k"] == 2
    assert retrieve_span.attributes["rag.retrieved_chunks_count"] == 1

    grade_span = next(s for s in spans if s.name == "rag.grade_context")
    assert grade_span.attributes is not None
    assert grade_span.attributes["rag.context_relevant"] is True
    assert grade_span.attributes["rag.score_threshold"] == 0.5

    gen_span = next(s for s in spans if s.name == "agent.generate")
    assert gen_span.attributes is not None
    assert gen_span.attributes["gen_ai.system"] == "test-llm"
    assert gen_span.attributes["gen_ai.usage.input_tokens"] == 30
    assert gen_span.attributes["gen_ai.usage.output_tokens"] == 15

    no_ctx_span = next(s for s in spans if s.name == "agent.no_context_answer")
    assert no_ctx_span.attributes is not None
    assert no_ctx_span.attributes["agent.refusal"] is True
