"""Unit tests for user feedback collection, statistics, and fine-tuning export."""

import pytest
from fastapi.testclient import TestClient

from sentinel.api.schemas import FeedbackRequest
from sentinel.evals.feedback import FeedbackStore
from sentinel.main import build_app


@pytest.fixture
def feedback_store() -> FeedbackStore:
    """Provides an isolated feedback store using in-memory fallback."""
    store = FeedbackStore(redis_url="redis://localhost:9999/0")  # Intentional invalid port
    return store


@pytest.mark.asyncio
async def test_record_feedback_success(feedback_store: FeedbackStore) -> None:
    req = FeedbackRequest(
        request_id="req_test_001",
        rating=1,
        comment="Great answer, very detailed.",
        metadata={"client": "web_ui"},
    )
    res = await feedback_store.record_feedback(req)
    assert res.status == "recorded"
    assert res.request_id == "req_test_001"
    assert res.feedback_id.startswith("fb_")


@pytest.mark.asyncio
async def test_feedback_stats_calculation(feedback_store: FeedbackStore) -> None:
    # 2 positive, 1 negative
    await feedback_store.record_feedback(FeedbackRequest(request_id="req_1", rating=1))
    await feedback_store.record_feedback(FeedbackRequest(request_id="req_2", rating=1))
    await feedback_store.record_feedback(FeedbackRequest(request_id="req_3", rating=-1))

    stats = await feedback_store.get_stats()
    assert stats.total_feedback == 3
    assert stats.positive_count == 2
    assert stats.negative_count == 1
    assert stats.satisfaction_rate == 66.7


@pytest.mark.asyncio
async def test_export_tuning_dataset(feedback_store: FeedbackStore) -> None:
    # Add a corrected feedback
    await feedback_store.record_feedback(
        FeedbackRequest(
            request_id="req_cor_1",
            rating=-1,
            comment="Wrong parameter name",
            correction="The parameter name is maxSurge, not maxSurgeCount.",
            metadata={"question": "What is the parameter for rolling updates?"},
        )
    )

    dataset = await feedback_store.export_tuning_dataset()
    assert len(dataset) == 1
    sample = dataset[0]
    assert "messages" in sample
    assert sample["messages"][0]["content"] == "What is the parameter for rolling updates?"
    assert sample["messages"][1]["content"] == "The parameter name is maxSurge, not maxSurgeCount."


def test_feedback_api_endpoints() -> None:
    app = build_app()
    with TestClient(app) as client:
        # Submit feedback
        res_post = client.post(
            "/v1/feedback",
            json={
                "request_id": "req_api_test",
                "rating": 1,
                "comment": "Accurate output",
                "metadata": {"test": True},
            },
        )
        assert res_post.status_code == 201
        data = res_post.json()
        assert data["status"] == "recorded"
        assert data["request_id"] == "req_api_test"

        # Check stats
        res_stats = client.get("/v1/feedback/stats")
        assert res_stats.status_code == 200
        stats = res_stats.json()
        assert "total_feedback" in stats
        assert "satisfaction_rate" in stats

        # Export endpoint
        res_export = client.get("/v1/feedback/export")
        assert res_export.status_code == 200
        assert isinstance(res_export.json(), list)
