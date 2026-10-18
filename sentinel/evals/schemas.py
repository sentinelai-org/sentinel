"""Data models for automated RAG evaluation and groundedness checking."""

from typing import Any

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


class FeedbackRequest(BaseModel):
    """User evaluation feedback on a specific gateway interaction."""

    request_id: str = Field(..., min_length=1, description="Target request ID to evaluate")
    rating: int = Field(
        ..., ge=-1, le=5, description="Feedback rating (-1/1 for thumbs down/up, or 1-5 star scale)"
    )
    comment: str | None = Field(default=None, description="Optional qualitative user commentary")
    correction: str | None = Field(
        default=None, description="Optional user-provided correction for RLHF/fine-tuning"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary client metadata tags"
    )


class FeedbackResponse(BaseModel):
    """Confirmation returned when user feedback is recorded."""

    feedback_id: str = Field(description="Unique feedback tracking ID")
    request_id: str = Field(description="Associated request ID")
    status: str = Field(default="recorded", description="Status confirmation")
    recorded_at: str = Field(description="ISO 8601 recording timestamp")


class FeedbackStats(BaseModel):
    """Aggregated statistics of user feedback."""

    total_feedback: int = Field(description="Total feedback entries recorded")
    positive_count: int = Field(description="Number of positive ratings (>= 1)")
    negative_count: int = Field(description="Number of negative ratings (<= 0)")
    satisfaction_rate: float = Field(description="Percentage of positive ratings (0.0 to 100.0)")
    latest_feedback_at: str | None = Field(
        default=None, description="Timestamp of most recent feedback"
    )
