"""Unit tests for Groundedness Gate and LangGraph interception."""

from __future__ import annotations

import pytest

from sentinel.agent.graph import create_agent_graph
from sentinel.agent.nodes import AgentNodeHandler
from sentinel.agent.state import AgentState
from sentinel.evals.gate import SAFE_UNVERIFIED_REFUSAL, GroundednessGate
from sentinel.llm.fake import FakeProvider
from sentinel.llm.router import LLMRouter
from sentinel.rag.chunker import Chunk
from tests.conftest import InMemoryVectorStore


class DummyEmbedder:
    def embed_text(self, text: str) -> list[float]:
        _ = text
        return [0.1] * 384


def test_gate_decision_logic() -> None:
    """GroundednessGate approves supported content and rejects hallucinations."""
    gate = GroundednessGate(threshold=0.70)
    context = ["Kubernetes Pods run containers and share storage volumes."]

    # Grounded answer
    grounded_ans = "Pods run containers and share storage volumes in Kubernetes."
    res1 = gate.evaluate(grounded_ans, context)
    assert res1.passed is True
    assert res1.score >= 0.70

    # Hallucinated answer
    hallucinated_ans = (
        "Kubernetes Pods utilize artificial superintelligence to bend spacetime dimensions."
    )
    res2 = gate.evaluate(hallucinated_ans, context)
    assert res2.passed is False
    assert res2.score < 0.70


@pytest.mark.asyncio
async def test_graph_intercepts_hallucinations() -> None:
    """When LLM hallucinates claims unsupported by Qdrant docs, gate replaces answer."""
    vector_store = InMemoryVectorStore()
    chunk = Chunk(
        chunk_id="c_doc1",
        doc_id="deployments.md",
        text="Deployments manage declarative updates using RollingUpdate.",
        index=0,
        metadata={"source": "deployments.md"},
    )
    await vector_store.upsert([chunk], [[0.1] * 384])

    # Fake LLM configured to hallucinate completely unsupported facts
    hallucinating_provider = FakeProvider(
        default_response="Quantum warp drives travel through hyperdimensional wormholes.",
        provider_name="hallucinating-llm",
        model="hallucinate-v1",
    )
    router = LLMRouter(primary_provider=hallucinating_provider)

    gate = GroundednessGate(threshold=0.70)
    handler = AgentNodeHandler(
        vector_store=vector_store,
        embedder=DummyEmbedder(),  # type: ignore[arg-type]
        llm_router=router,
        score_threshold=0.5,
        top_k=2,
        groundedness_gate=gate,
    )
    graph = create_agent_graph(handler)

    initial_state: AgentState = {
        "question": "How do Deployments update pods?",
        "session_id": "test_sess",
        "request_id": "req_gate_test",
        "documents": [],
        "context_relevant": False,
        "answer": "",
        "sources": [],
        "meta": {},
        "groundedness_score": None,
        "groundedness_passed": True,
    }

    output = await graph.ainvoke(initial_state)

    # Interception verification
    assert output["answer"] == SAFE_UNVERIFIED_REFUSAL
    assert output["groundedness_passed"] is False
    assert output["groundedness_score"] < 0.70
    assert output["meta"]["groundedness_passed"] is False
