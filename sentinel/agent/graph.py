"""LangGraph workflow definition and compilation."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from sentinel.agent.nodes import AgentNodeHandler
from sentinel.agent.state import AgentState


def route_input_guardrail(state: AgentState) -> Literal["contextualize_query", "refuse_unsafe"]:
    """Conditional router based on input safety guardrail check."""
    if state.get("guardrail_passed", True):
        return "contextualize_query"
    return "refuse_unsafe"


def route_context(state: AgentState) -> Literal["generate", "no_context_answer"]:
    """Conditional edge router based on context relevance."""
    if state.get("context_relevant", False):
        return "generate"
    return "no_context_answer"


def create_agent_graph(handler: AgentNodeHandler) -> Any:
    """Construct and compile the Sentinel RAG LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Register graph nodes
    workflow.add_node("guardrail_input", handler.guardrail_input)
    workflow.add_node("refuse_unsafe", handler.refuse_unsafe)
    workflow.add_node("contextualize_query", handler.contextualize_query)
    workflow.add_node("retrieve", handler.retrieve)
    workflow.add_node("grade_context", handler.grade_context)
    workflow.add_node("generate", handler.generate)
    workflow.add_node("validate_groundedness", handler.validate_groundedness)
    workflow.add_node("guardrail_output", handler.guardrail_output)
    workflow.add_node("no_context_answer", handler.no_context_answer)
    workflow.add_node("finalize", handler.finalize)

    # Step 1: Input Guardrail check
    workflow.add_edge(START, "guardrail_input")
    workflow.add_conditional_edges(
        "guardrail_input",
        route_input_guardrail,
        {
            "contextualize_query": "contextualize_query",
            "refuse_unsafe": "refuse_unsafe",
        },
    )
    workflow.add_edge("refuse_unsafe", "finalize")
    workflow.add_edge("contextualize_query", "retrieve")

    # Step 2: Context Retrieval & Grading
    workflow.add_edge("retrieve", "grade_context")
    workflow.add_conditional_edges(
        "grade_context",
        route_context,
        {
            "generate": "generate",
            "no_context_answer": "no_context_answer",
        },
    )

    # Step 3: Generation -> Groundedness Gate -> Output Guardrail -> Finalize
    workflow.add_edge("generate", "validate_groundedness")
    workflow.add_edge("validate_groundedness", "guardrail_output")
    workflow.add_edge("guardrail_output", "finalize")
    workflow.add_edge("no_context_answer", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
