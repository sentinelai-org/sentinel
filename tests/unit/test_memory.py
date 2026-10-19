"""Unit tests for Phase 8 Multi-Turn Conversational Session Memory."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from sentinel.agent.memory import SessionManager
from sentinel.agent.nodes import AgentNodeHandler
from sentinel.agent.state import AgentState
from sentinel.llm.fake import FakeProvider
from sentinel.llm.router import LLMRouter
from sentinel.main import build_app
from tests.conftest import InMemoryResponseCache, InMemoryVectorStore


class MockEmbedder:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        return [0.1] * self.dimension


@pytest.mark.asyncio
async def test_session_manager_append_and_retrieve_history() -> None:
    """Test appending user and assistant messages and retrieving them in order."""
    session_mgr = SessionManager(enabled=True)
    session_id = "test_sess_001"

    # Append first turn
    await session_mgr.append_message(session_id, "user", "What is a Pod?")
    await session_mgr.append_message(
        session_id,
        "assistant",
        "A Pod encapsulates containers.",
        sources=[{"source": "pods.md", "content": "Pods excerpt", "score": 0.9}],
    )

    history = await session_mgr.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is a Pod?"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "A Pod encapsulates containers."
    assert len(history[1]["sources"]) == 1

    # Cleanup
    await session_mgr.clear_session(session_id)
    history_after = await session_mgr.get_history(session_id)
    assert len(history_after) == 0


@pytest.mark.asyncio
async def test_session_manager_sliding_window_turns() -> None:
    """Test that get_recent_turns limits history to max_turns * 2 messages."""
    session_mgr = SessionManager(max_turns=2, enabled=True)
    session_id = "test_sess_sliding"

    for i in range(5):
        await session_mgr.append_message(session_id, "user", f"Question {i}")
        await session_mgr.append_message(session_id, "assistant", f"Answer {i}")

    # Total 10 messages stored
    all_history = await session_mgr.get_history(session_id)
    assert len(all_history) == 10

    # Recent turns should only be the last 2 pairs = 4 messages
    recent = await session_mgr.get_recent_turns(session_id)
    assert len(recent) == 4
    assert recent[0]["content"] == "Question 3"
    assert recent[-1]["content"] == "Answer 4"

    # Format for prompt
    prompt_text = await session_mgr.format_history_for_prompt(session_id)
    assert "User: Question 3" in prompt_text
    assert "Assistant: Answer 4" in prompt_text

    await session_mgr.clear_session(session_id)


@pytest.mark.asyncio
async def test_session_api_endpoints() -> None:
    """Test GET /v1/chat/sessions/{id} and DELETE /v1/chat/sessions/{id} endpoints."""
    app = build_app()
    session_mgr = SessionManager(enabled=True)
    app.state.session_manager = session_mgr
    app.state.cache = InMemoryResponseCache()
    app.state.vector_store = InMemoryVectorStore()

    session_id = "sess_api_test"
    await session_mgr.append_message(session_id, "user", "How does kube-proxy work?")
    await session_mgr.append_message(
        session_id, "assistant", "kube-proxy uses iptables or IPVS modes."
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        # 1. Retrieve session history
        resp = await client.get(f"/v1/chat/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["total_messages"] == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"

        # 2. Clear session history
        del_resp = await client.delete(f"/v1/chat/sessions/{session_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "cleared"

        # 3. Confirm cleared
        resp_after = await client.get(f"/v1/chat/sessions/{session_id}")
        assert resp_after.status_code == 200
        assert resp_after.json()["total_messages"] == 0


@pytest.mark.asyncio
async def test_contextualize_query_node() -> None:
    """Test that contextualize_query rephrases follow-up questions when history exists."""
    fake_llm = FakeProvider(
        default_response="How does Kubernetes Service iptables proxy mode work?",
    )
    router = LLMRouter(primary_provider=fake_llm)
    handler = AgentNodeHandler(
        vector_store=InMemoryVectorStore(),
        embedder=MockEmbedder(384),  # type: ignore[arg-type]
        llm_router=router,
    )

    # 1. Without history -> returns original question unchanged
    state_no_history: AgentState = {
        "question": "What is a Pod?",
        "history_text": "",
    }
    res1 = await handler.contextualize_query(state_no_history)
    assert res1["standalone_query"] == "What is a Pod?"

    # 2. With history -> rephrased via LLM
    state_with_history: AgentState = {
        "question": "Can you give an example of the first mode?",
        "history_text": "User: What are Service proxy modes?\nAssistant: The modes are iptables, IPVS, and userspace.",
    }
    res2 = await handler.contextualize_query(state_with_history)
    assert "iptables" in res2["standalone_query"]
