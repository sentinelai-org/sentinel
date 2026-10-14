"""Unit tests for EvaluationHarness and dataset benchmark execution."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from sentinel.evals.harness import EvaluationHarness
from sentinel.evals.judge import HeuristicJudge
from sentinel.evals.schemas import EvalSample


def test_load_dataset_parses_golden_dataset() -> None:
    """Ensure golden_dataset.json exists and parses with valid schema."""
    dataset_path = Path("data/evals/golden_dataset.json")
    samples = EvaluationHarness.load_dataset(dataset_path)

    assert len(samples) >= 8
    assert all(isinstance(s, EvalSample) for s in samples)
    assert any(s.unanswerable for s in samples)


@pytest.mark.asyncio
async def test_harness_run_aggregates_metrics() -> None:
    """Verify harness aggregates groundedness, recall, and relevance correctly."""
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "answer": "A Pod encapsulates one or more containers in Kubernetes.",
        "sources": [{"source": "pods.md"}],
        "documents": [{"text": "A Pod encapsulates one or more containers in Kubernetes."}],
    }

    judge = HeuristicJudge(groundedness_threshold=0.70)
    harness = EvaluationHarness(agent_graph=mock_graph, judge=judge)

    samples = [
        EvalSample(
            id="s1",
            question="What is a Pod?",
            expected_answer="A Pod encapsulates containers.",
            expected_sources=["pods.md"],
            unanswerable=False,
        ),
        EvalSample(
            id="s2",
            question="Tell me about aliens?",
            expected_answer="I could not find sufficient relevant documentation.",
            expected_sources=[],
            unanswerable=True,
        ),
    ]

    report = await harness.run(samples, min_groundedness=0.60)

    assert report.total_samples == 2
    assert report.mean_groundedness >= 0.70
    assert report.mean_context_recall >= 0.50
    assert len(report.results) == 2

    # Verify markdown export
    with tempfile.TemporaryDirectory() as tmp_dir:
        md_path = Path(tmp_dir) / "summary.md"
        harness.export_markdown_summary(report, md_path)
        assert md_path.is_file()
        content = md_path.read_text(encoding="utf-8")
        assert "Sentinel Evaluation Report" in content
        assert "Mean Groundedness" in content
