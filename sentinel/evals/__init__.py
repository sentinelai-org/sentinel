"""Automated evaluation harness, metrics, and groundedness gates for Sentinel."""

from sentinel.evals.gate import SAFE_UNVERIFIED_REFUSAL, GroundednessGate
from sentinel.evals.harness import EvaluationHarness
from sentinel.evals.judge import BaseJudge, HeuristicJudge, LLMJudge
from sentinel.evals.metrics import (
    calculate_answer_relevance,
    calculate_context_recall,
    calculate_groundedness,
)
from sentinel.evals.schemas import EvalResult, EvalSample, EvaluationReport, GateResult

__all__ = [
    "GroundednessGate",
    "SAFE_UNVERIFIED_REFUSAL",
    "EvaluationHarness",
    "BaseJudge",
    "HeuristicJudge",
    "LLMJudge",
    "calculate_groundedness",
    "calculate_context_recall",
    "calculate_answer_relevance",
    "EvalSample",
    "EvalResult",
    "EvaluationReport",
    "GateResult",
]
