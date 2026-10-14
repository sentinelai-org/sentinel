"""Integration test for POST /v1/chat and health endpoints using FakeProvider."""

from __future__ import annotations

import httpx
import pytest
from httpx import ASGITransport

from sentinel.agent.graph import create_agent_graph
from sentinel.agent.nodes import AgentNodeHandler
from sentinel.llm.fake import FakeProvider
from sentinel.llm.router import LLMRouter
from sentinel.main import build_app
from sentinel.rag.chunker import Chunk
from tests.conftest import InMemoryResponseCache, InMemoryVectorStore


class MockEmbedder:
    """Mock embedder returning fixed vector."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        _ = text
        return [0.1] * self.dimension

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]


@pytest.mark.asyncio
async def test_chat_integration_e2e(
    in_memory_vector_store: InMemoryVectorStore,
    in_memory_cache: InMemoryResponseCache,
    fake_llm_provider: FakeProvider,
) -> None:
    """End-to-end test verifying chat endpoint, context grading, and caching."""
    # Seed mock vector store with document chunk
    chunk = Chunk(
        chunk_id="chk_pod_01",
        doc_id="pods.md",
        text="A Pod is the smallest deployable unit of computing in Kubernetes.",
        index=0,
        metadata={"source": "pods.md", "title": "Kubernetes Pods"},
    )
    await in_memory_vector_store.upsert([chunk], [[0.1] * 384])

    router = LLMRouter(primary_provider=fake_llm_provider)
    embedder = MockEmbedder(384)

    node_handler = AgentNodeHandler(
        vector_store=in_memory_vector_store,
        embedder=embedder,  # type: ignore[arg-type]
        llm_router=router,
        score_threshold=0.5,
        top_k=2,
    )
    compiled_graph = create_agent_graph(node_handler)

    app = build_app(override_primary_llm=fake_llm_provider)
    app.state.cache = in_memory_cache
    app.state.vector_store = in_memory_vector_store
    app.state.embedder = embedder
    app.state.llm_router = router
    app.state.agent_graph = compiled_graph

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        # 1. Health check
        resp_health = await client.get("/healthz")
        assert resp_health.status_code == 200
        assert resp_health.json() == {"status": "ok"}

        # 2. Readiness check
        resp_ready = await client.get("/readyz")
        assert resp_ready.status_code == 200
        assert resp_ready.json()["status"] == "ready"

        # 3. First chat request: Cache Miss -> RAG Agent -> Fake LLM
        req_payload = {
            "question": "What is a Pod in Kubernetes?",
            "session_id": "sess_123",
        }
        resp1 = await client.post("/v1/chat", json=req_payload)
        assert resp1.status_code == 200
        data1 = resp1.json()

        assert "Kubernetes Pods" in data1["answer"]
        assert len(data1["sources"]) > 0
        assert data1["sources"][0]["source"] == "pods.md"
        assert data1["meta"]["cache_hit"] is False
        assert data1["meta"]["provider"] == "fake-gemini"
        assert data1["meta"]["model"] == "fake-gemini-2.5"
        assert data1["meta"]["tokens_in"] == 45
        assert data1["meta"]["tokens_out"] == 18
        assert data1["meta"]["latency_ms"] >= 0.0
        assert data1["meta"]["request_id"].startswith("req_")
        assert "trace_id" in data1["meta"]
        assert "span_id" in data1["meta"]

        # 4. Second identical request: Cache Hit!
        resp2 = await client.post("/v1/chat", json=req_payload)
        assert resp2.status_code == 200
        data2 = resp2.json()

        assert data2["answer"] == data1["answer"]
        assert data2["meta"]["cache_hit"] is True
        assert data2["meta"]["provider"] == "fake-gemini"


@pytest.mark.asyncio
async def test_chat_no_relevant_context(
    in_memory_vector_store: InMemoryVectorStore,
    in_memory_cache: InMemoryResponseCache,
    fake_llm_provider: FakeProvider,
) -> None:
    """When vector store returns no matches or below threshold, routes to refusal."""
    router = LLMRouter(primary_provider=fake_llm_provider)
    embedder = MockEmbedder(384)

    # Empty vector store: 0 docs returned
    node_handler = AgentNodeHandler(
        vector_store=in_memory_vector_store,
        embedder=embedder,  # type: ignore[arg-type]
        llm_router=router,
        score_threshold=0.5,
        top_k=2,
    )
    compiled_graph = create_agent_graph(node_handler)

    app = build_app(override_primary_llm=fake_llm_provider)
    app.state.cache = in_memory_cache
    app.state.vector_store = in_memory_vector_store
    app.state.embedder = embedder
    app.state.llm_router = router
    app.state.agent_graph = compiled_graph

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        resp = await client.post("/v1/chat", json={"question": "What is quantum gravity?"})
        assert resp.status_code == 200
        data = resp.json()

        assert "could not find sufficient relevant documentation" in data["answer"].lower()
        assert data["meta"]["tokens_in"] == 0
        assert data["meta"]["tokens_out"] == 0
        assert data["meta"]["provider"] == "system"
