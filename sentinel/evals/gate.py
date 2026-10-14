"""Runtime Groundedness Gate for preventing hallucinations and unverified claims."""

from __future__ import annotations

from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer
from sentinel.evals.metrics import calculate_groundedness
from sentinel.evals.schemas import GateResult

logger = get_logger(__name__)
tracer = get_tracer("sentinel.evals.gate")

SAFE_UNVERIFIED_REFUSAL = (
    "Sentinel Guardrail: The generated answer could not be verified against the "
    "retrieved documentation with sufficient confidence. Please refer to the official documentation."
)


class GroundednessGate:
    """Evaluates whether an answer is sufficiently grounded in retrieved context."""

    def __init__(self, threshold: float = 0.70) -> None:
        self.threshold = threshold

    def evaluate(self, answer: str, context_chunks: list[str]) -> GateResult:
        """Evaluate factual alignment and return a GateResult."""
        with tracer.start_as_current_span("groundedness_gate.evaluate") as span:
            span.set_attribute("gate.threshold", self.threshold)

            score = calculate_groundedness(answer=answer, context_chunks=context_chunks)
            passed = score >= self.threshold

            span.set_attribute("gate.score", score)
            span.set_attribute("gate.passed", passed)

            if passed:
                logger.info(
                    "Groundedness gate passed",
                    extra={"score": score, "threshold": self.threshold},
                )
                return GateResult(
                    passed=True,
                    score=score,
                    reason=f"Answer grounded in retrieved context (score {score:.2f} >= {self.threshold:.2f})",
                )

            logger.warning(
                "Groundedness gate rejected answer: hallucination risk",
                extra={"score": score, "threshold": self.threshold},
            )
            return GateResult(
                passed=False,
                score=score,
                reason=(
                    f"Answer failed groundedness verification: score {score:.2f} "
                    f"is below threshold {self.threshold:.2f}"
                ),
            )
