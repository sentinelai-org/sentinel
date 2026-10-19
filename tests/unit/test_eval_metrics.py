"""Unit tests for RAG evaluation metrics."""

from __future__ import annotations

from sentinel.evals.metrics import (
    calculate_answer_relevance,
    calculate_context_recall,
    calculate_groundedness,
)


def test_groundedness_high_when_claims_supported() -> None:
    """Answer completely grounded in retrieved text yields a high score."""
    contexts = [
        "A Pod encapsulates one or more containers that share storage and network.",
        "Pods are the smallest deployable units of computing in Kubernetes.",
    ]
    answer = "A Pod encapsulates one or more containers sharing storage and network in Kubernetes."
    score = calculate_groundedness(answer, contexts)
    assert score >= 0.80


def test_groundedness_zero_when_hallucinated() -> None:
    """Answer with claims entirely absent from context yields a low/zero score."""
    contexts = [
        "A Pod encapsulates one or more containers in Kubernetes.",
    ]
    answer = "Quantum entanglement teleports secret cryptographic keys across extraterrestrial satellites."
    score = calculate_groundedness(answer, contexts)
    assert score == 0.0


def test_groundedness_refusal_is_inherently_grounded() -> None:
    """Standard polite refusal when context is missing is 100% grounded."""
    contexts: list[str] = []
    answer = "I could not find sufficient relevant documentation in the knowledge base to answer your question."
    score = calculate_groundedness(answer, contexts)
    assert score == 1.0


def test_groundedness_not_enough_information_refusal() -> None:
    """Meta-refusals stating lack of information in context are scored as 1.0 grounded."""
    contexts = ["Some basic documentation without details."]
    answer = "Based on the provided context, there is not enough information to explain the specific mechanism."
    score = calculate_groundedness(answer, contexts)
    assert score == 1.0


def test_context_recall_calculation() -> None:
    """Context recall measures ratio of expected source files present in retrieval."""
    expected = ["pods.md", "deployments.md"]

    # Full recall
    retrieved_full = ["pods.md", "deployments.md", "services.md"]
    assert calculate_context_recall(retrieved_full, expected) == 1.0

    # Partial recall
    retrieved_partial = ["pods.md", "services.md"]
    assert calculate_context_recall(retrieved_partial, expected) == 0.5

    # Zero recall
    retrieved_none = ["services.md"]
    assert calculate_context_recall(retrieved_none, expected) == 0.0

    # Empty expected
    assert calculate_context_recall(retrieved_none, []) == 1.0


def test_answer_relevance_calculation() -> None:
    """Measures keyword alignment between question and response."""
    question = "How do Kubernetes Deployments roll out updates?"
    relevant_answer = "Kubernetes Deployments roll out updates using the RollingUpdate strategy."
    irrelevant_answer = "Apples and oranges are nutritious fruits."

    rel_score = calculate_answer_relevance(question, relevant_answer)
    irrel_score = calculate_answer_relevance(question, irrelevant_answer)

    assert rel_score > irrel_score
    assert rel_score >= 0.60
    assert irrel_score == 0.0
