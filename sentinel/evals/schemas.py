"""Data models for automated RAG evaluation and groundedness checking."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    """A single evaluation benchmark item from the golden dataset."""

    id: str = Field(description="Unique benchmark sample identifier")
    question: str = Field(description="User question to evaluate")
    expected_answer: str = Field(description="Reference ideal answer")
    expected_sources: list[str] = Field(
        default_factory=list, description="List of expected document sources"
    )
    ground_truth_claims: list[str] = Field(
        default_factory=list, description="Atomic factual claims required in a correct answer"
    )
    unanswerable: bool = Field(
        default=False,
        description="Whether the question is intentionally out-of-domain to test refusal",
    )


class GateResult(BaseModel):
    """Result of evaluating groundedness on an answer."""

    passed: bool = Field(description="Whether the answer passed the groundedness threshold")
    score: float = Field(description="Groundedness similarity / verification score from 0.0 to 1.0")
    reason: str = Field(description="Diagnostic rationale for gate decision")
    hallucinated_claims: list[str] = Field(
        default_factory=list, description="Specific claims detected as unverified by context"
    )


class EvalResult(BaseModel):
    """Evaluation metrics for a single executed sample."""

    sample_id: str
    question: str
    generated_answer: str
    expected_answer: str
    groundedness_score: float
    context_recall: float
    answer_relevance: float
    gate_passed: bool
    unanswerable: bool
    correctly_refused: bool | None = None
    retrieved_sources: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class EvaluationReport(BaseModel):
    """Aggregated evaluation summary across the entire benchmark suite."""

    total_samples: int
    mean_groundedness: float
    mean_context_recall: float
    mean_answer_relevance: float
    gate_pass_rate: float
    unanswerable_refusal_rate: float
    passed_gate_threshold: bool
    results: list[EvalResult] = Field(default_factory=list)
    timestamp: str
