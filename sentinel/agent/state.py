"""LangGraph explicit state object definition."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Explicit state passed across nodes in the LangGraph workflow."""

    question: str
    session_id: str | None
    request_id: str
    documents: list[dict[str, Any]]
    context_relevant: bool
    answer: str
    sources: list[dict[str, Any]]
    meta: dict[str, Any]
    groundedness_score: float | None
    groundedness_passed: bool
    guardrail_passed: bool
    guardrail_violation: str | None
    cost_usd: float | None
    chat_history: list[dict[str, Any]]
    history_text: str
    standalone_query: str
