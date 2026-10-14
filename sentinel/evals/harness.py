"""Automated evaluation harness for benchmarking Sentinel RAG agents."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentinel.core.logging import get_logger
from sentinel.evals.judge import BaseJudge, HeuristicJudge
from sentinel.evals.schemas import EvalResult, EvalSample, EvaluationReport

logger = get_logger(__name__)


class EvaluationHarness:
    """Executes benchmark suites against the Sentinel RAG pipeline and compiles reports."""

    def __init__(self, agent_graph: Any, judge: BaseJudge | None = None) -> None:
        self.agent_graph = agent_graph
        self.judge = judge or HeuristicJudge()

    @staticmethod
    def load_dataset(dataset_path: str | Path) -> list[EvalSample]:
        """Load and parse evaluation benchmark dataset."""
        path = Path(dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

        raw_data = json.loads(path.read_text(encoding="utf-8"))
        return [EvalSample(**item) for item in raw_data]

    async def run(
        self,
        samples: list[EvalSample],
        min_groundedness: float = 0.70,
    ) -> EvaluationReport:
        """Run the full evaluation benchmark and aggregate quality metrics."""
        results: list[EvalResult] = []

        logger.info(
            "Starting automated RAG evaluation benchmark", extra={"samples_count": len(samples)}
        )

        for sample in samples:
            print(f"  -> Evaluating sample {sample.id}: {sample.question[:45]}...", flush=True)
            start_time = time.perf_counter()
            initial_state = {
                "question": sample.question,
                "session_id": "eval_session",
                "request_id": f"eval_{sample.id}",
                "documents": [],
                "context_relevant": False,
                "answer": "",
                "sources": [],
                "meta": {},
                "groundedness_score": None,
                "groundedness_passed": True,
            }

            try:
                output = await self.agent_graph.ainvoke(initial_state)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                generated_answer = output.get("answer", "")
                sources_data = output.get("sources", [])
                retrieved_sources = [s.get("source", "") for s in sources_data]
                documents = output.get("documents", [])
                retrieved_contexts = [d.get("text", "") for d in documents]

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    "Sample execution failed", extra={"sample_id": sample.id, "error": str(exc)}
                )
                generated_answer = f"Error during generation: {exc}"
                retrieved_contexts = []
                retrieved_sources = []

            eval_res = await self.judge.evaluate_sample(
                sample=sample,
                generated_answer=generated_answer,
                retrieved_contexts=retrieved_contexts,
                retrieved_sources=retrieved_sources,
                latency_ms=elapsed_ms,
            )
            results.append(eval_res)

        # Aggregate metrics
        total = len(results)
        if total == 0:
            return EvaluationReport(
                total_samples=0,
                mean_groundedness=0.0,
                mean_context_recall=0.0,
                mean_answer_relevance=0.0,
                gate_pass_rate=0.0,
                unanswerable_refusal_rate=0.0,
                passed_gate_threshold=False,
                results=[],
                timestamp=datetime.now(UTC).isoformat(),
            )

        mean_groundedness = sum(r.groundedness_score for r in results) / total
        mean_context_recall = sum(r.context_recall for r in results) / total
        mean_answer_relevance = sum(r.answer_relevance for r in results) / total
        gate_passed_count = sum(1 for r in results if r.gate_passed)
        gate_pass_rate = gate_passed_count / total

        unanswerable_samples = [r for r in results if r.unanswerable]
        if unanswerable_samples:
            refused_count = sum(1 for r in unanswerable_samples if r.correctly_refused)
            unanswerable_refusal_rate = refused_count / len(unanswerable_samples)
        else:
            unanswerable_refusal_rate = 1.0

        passed_threshold = (
            mean_groundedness >= min_groundedness and unanswerable_refusal_rate >= 0.80
        )

        report = EvaluationReport(
            total_samples=total,
            mean_groundedness=round(mean_groundedness, 4),
            mean_context_recall=round(mean_context_recall, 4),
            mean_answer_relevance=round(mean_answer_relevance, 4),
            gate_pass_rate=round(gate_pass_rate, 4),
            unanswerable_refusal_rate=round(unanswerable_refusal_rate, 4),
            passed_gate_threshold=passed_threshold,
            results=results,
            timestamp=datetime.now(UTC).isoformat(),
        )

        logger.info(
            "Evaluation complete",
            extra={
                "mean_groundedness": report.mean_groundedness,
                "context_recall": report.mean_context_recall,
                "passed_gate": report.passed_gate_threshold,
            },
        )
        return report

    @staticmethod
    def export_markdown_summary(report: EvaluationReport, output_path: str | Path) -> None:
        """Export report summary as formatted Markdown."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        status_emoji = "✅ PASSED" if report.passed_gate_threshold else "❌ FAILED"
        lines = [
            f"# Sentinel Evaluation Report - {status_emoji}",
            f"\n**Timestamp**: `{report.timestamp}`",
            f"**Total Samples**: `{report.total_samples}`\n",
            "## Summary Metrics",
            "| Metric | Score | Target |",
            "| :--- | :--- | :--- |",
            f"| Mean Groundedness | **{report.mean_groundedness * 100:.1f}%** | >= 70.0% |",
            f"| Mean Context Recall | **{report.mean_context_recall * 100:.1f}%** | >= 70.0% |",
            f"| Mean Answer Relevance | **{report.mean_answer_relevance * 100:.1f}%** | >= 60.0% |",
            f"| Gate Pass Rate | **{report.gate_pass_rate * 100:.1f}%** | >= 80.0% |",
            f"| Refusal Rate (Unanswerable) | **{report.unanswerable_refusal_rate * 100:.1f}%** | >= 80.0% |\n",
            "## Sample Breakdown",
            "| ID | Question | Groundedness | Recall | Passed |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for r in report.results:
            passed_icon = "✅" if r.gate_passed else "❌"
            q_short = r.question[:45] + ("..." if len(r.question) > 45 else "")
            lines.append(
                f"| `{r.sample_id}` | {q_short} | {r.groundedness_score * 100:.1f}% | {r.context_recall * 100:.1f}% | {passed_icon} |"
            )

        path.write_text("\n".join(lines), encoding="utf-8")
