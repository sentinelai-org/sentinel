"""LangGraph workflow definition and compilation."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from sentinel.agent.nodes import AgentNodeHandler
from sentinel.agent.state import AgentState


def route_context(state: AgentState) -> Literal["generate", "no_context_answer"]:
    """Conditional edge router based on context relevance."""
    if state.get("context_relevant", False):
        return "generate"
    return "no_context_answer"


def create_agent_graph(handler: AgentNodeHandler) -> Any:
    """Construct and compile the Sentinel RAG LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Register graph nodes
    workflow.add_node("retrieve", handler.retrieve)
    workflow.add_node("grade_context", handler.grade_context)
    workflow.add_node("generate", handler.generate)
    workflow.add_node("validate_groundedness", handler.validate_groundedness)
    workflow.add_node("no_context_answer", handler.no_context_answer)
    workflow.add_node("finalize", handler.finalize)

    # Linear flow: start -> retrieve -> grade_context
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_context")

    # Conditional branching: grade_context -> generate OR no_context_answer
    workflow.add_conditional_edges(
        "grade_context",
        route_context,
        {
            "generate": "generate",
            "no_context_answer": "no_context_answer",
        },
    )

    # Generated answers must pass the Groundedness Gate before finalizing
    workflow.add_edge("generate", "validate_groundedness")
    workflow.add_edge("validate_groundedness", "finalize")
    workflow.add_edge("no_context_answer", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
