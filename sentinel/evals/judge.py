"""Evaluation judges supporting heuristic and LLM-as-a-judge approaches."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel.core.logging import get_logger
from sentinel.evals.metrics import (
    calculate_answer_relevance,
    calculate_context_recall,
    calculate_groundedness,
)
from sentinel.evals.schemas import EvalResult, EvalSample
from sentinel.llm.base import LLMProvider

logger = get_logger(__name__)


class BaseJudge(ABC):
    """Abstract base class for RAG evaluation judges."""

    @abstractmethod
    async def evaluate_sample(
        self,
        sample: EvalSample,
        generated_answer: str,
        retrieved_contexts: list[str],
        retrieved_sources: list[str],
        latency_ms: float = 0.0,
    ) -> EvalResult:
        """Evaluate a single question-answering generation."""
        pass


class HeuristicJudge(BaseJudge):
    """Fast, deterministic, zero-cost evaluation judge suitable for offline CI/CD."""

    def __init__(self, groundedness_threshold: float = 0.70) -> None:
        self.groundedness_threshold = groundedness_threshold

    async def evaluate_sample(
        self,
        sample: EvalSample,
        generated_answer: str,
        retrieved_contexts: list[str],
        retrieved_sources: list[str],
        latency_ms: float = 0.0,
    ) -> EvalResult:
        groundedness = calculate_groundedness(generated_answer, retrieved_contexts)
        context_recall = calculate_context_recall(retrieved_sources, sample.expected_sources)
        relevance = calculate_answer_relevance(sample.question, generated_answer)

        # Check refusal for unanswerable queries
        refusal_phrases = [
            "could not find sufficient",
            "not covered",
            "unknown",
            "refusal",
            "not enough information",
            "insufficient information",
            "context does not contain",
            "context does not provide",
        ]
        is_refusal = any(p in generated_answer.lower() for p in refusal_phrases)

        correctly_refused: bool | None = None
        if sample.unanswerable:
            correctly_refused = is_refusal
            if correctly_refused:
                groundedness = 1.0
                relevance = 1.0

        gate_passed = groundedness >= self.groundedness_threshold

        return EvalResult(
            sample_id=sample.id,
            question=sample.question,
            generated_answer=generated_answer,
            expected_answer=sample.expected_answer,
            groundedness_score=groundedness,
            context_recall=context_recall,
            answer_relevance=relevance,
            gate_passed=gate_passed,
            unanswerable=sample.unanswerable,
            correctly_refused=correctly_refused,
            retrieved_sources=retrieved_sources,
            expected_sources=sample.expected_sources,
            latency_ms=round(latency_ms, 2),
        )


class LLMJudge(BaseJudge):
    """LLM-as-a-Judge using a foundation model provider (Gemini or Ollama) for semantic evaluation."""

    def __init__(self, provider: LLMProvider, groundedness_threshold: float = 0.70) -> None:
        self.provider = provider
        self.groundedness_threshold = groundedness_threshold
        self.heuristic_fallback = HeuristicJudge(groundedness_threshold=groundedness_threshold)

    async def evaluate_sample(
        self,
        sample: EvalSample,
        generated_answer: str,
        retrieved_contexts: list[str],
        retrieved_sources: list[str],
        latency_ms: float = 0.0,
    ) -> EvalResult:
        # Compute baseline recall and relevance
        context_recall = calculate_context_recall(retrieved_sources, sample.expected_sources)
        relevance = calculate_answer_relevance(sample.question, generated_answer)

        judge_prompt = f"""You are an expert AI evaluator for technical RAG systems.
Evaluate whether the following ANSWER is strictly grounded in the given CONTEXT without hallucinations.

CONTEXT:
{" ".join(retrieved_contexts)[:2000]}

QUESTION:
{sample.question}

ANSWER:
{generated_answer}

Rate the groundedness of the answer on a scale from 0.0 to 1.0, where 1.0 means every claim is fully supported by the context.
Reply ONLY with a single numeric float between 0.0 and 1.0."""

        try:
            res = await self.provider.generate(
                prompt=judge_prompt,
                temperature=0.0,
            )
            # Parse numeric score safely
            raw_text = res.text.strip()
            score = float(raw_text)
            score = max(0.0, min(1.0, score))
        except Exception as exc:
            logger.warning(
                "LLM Judge call failed, using heuristic judge", extra={"error": str(exc)}
            )
            return await self.heuristic_fallback.evaluate_sample(
                sample=sample,
                generated_answer=generated_answer,
                retrieved_contexts=retrieved_contexts,
                retrieved_sources=retrieved_sources,
                latency_ms=latency_ms,
            )

        gate_passed = score >= self.groundedness_threshold

        return EvalResult(
            sample_id=sample.id,
            question=sample.question,
            generated_answer=generated_answer,
            expected_answer=sample.expected_answer,
            groundedness_score=round(score, 4),
            context_recall=context_recall,
            answer_relevance=relevance,
            gate_passed=gate_passed,
            unanswerable=sample.unanswerable,
            retrieved_sources=retrieved_sources,
            expected_sources=sample.expected_sources,
            latency_ms=round(latency_ms, 2),
        )
