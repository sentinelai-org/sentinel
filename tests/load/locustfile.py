"""Production Locust load testing suite for Sentinel LLMOps Gateway."""

import random
from typing import Any

from locust import HttpUser, between, events, task

SAMPLE_QUESTIONS = [
    "What is a Kubernetes Pod?",
    "How do Kubernetes Deployments roll out updates?",
    "What is the difference between ClusterIP and NodePort?",
    "What are the primary control plane components in Kubernetes?",
    "What is the role of kubelet on worker nodes?",
    "How do Services route traffic to Pods in Kubernetes?",
    "Explain the Kubernetes rolling update strategy.",
    "What is Sentinel's security policy for production clusters?",
]


class SentinelLoadUser(HttpUser):
    """Simulates realistic concurrent user traffic against Sentinel Gateway."""

    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        """Initialize user session context."""
        self.recorded_request_ids: list[str] = []

    @task(5)
    def query_chat_cached_and_uncached(self) -> None:
        """Test synchronous chat endpoint with repeated and unique questions."""
        question = random.choice(SAMPLE_QUESTIONS)
        with self.client.post(
            "/v1/chat",
            json={"question": question, "session_id": "locust_load_test"},
            catch_response=True,
            name="/v1/chat",
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    meta = data.get("meta", {})
                    req_id = meta.get("request_id")
                    if req_id:
                        self.recorded_request_ids.append(req_id)
                        # Keep list bounded
                        if len(self.recorded_request_ids) > 100:
                            self.recorded_request_ids.pop(0)

                    # Verify structure
                    if not data.get("answer"):
                        response.failure("Missing answer field in response payload")
                    else:
                        response.success()
                except Exception as exc:
                    response.failure(f"JSON decode failed: {exc}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(3)
    def query_chat_stream(self) -> None:
        """Test Server-Sent Events (SSE) streaming throughput under load."""
        question = random.choice(SAMPLE_QUESTIONS)
        with self.client.post(
            "/v1/chat/stream",
            json={"question": question, "session_id": "locust_stream_test"},
            stream=True,
            catch_response=True,
            name="/v1/chat/stream",
        ) as response:
            if response.status_code == 200:
                token_count = 0
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                        if "event: token" in decoded:
                            token_count += 1
                if token_count > 0:
                    response.success()
                else:
                    response.failure("Stream yielded 0 tokens")
            else:
                response.failure(f"Stream failed with HTTP {response.status_code}")

    @task(2)
    def submit_user_feedback(self) -> None:
        """Simulate user thumbs up / down feedback submission."""
        if not self.recorded_request_ids:
            return

        req_id = random.choice(self.recorded_request_ids)
        rating = random.choice([1, 1, 1, -1])  # 75% positive bias
        comment = "Accurate response" if rating > 0 else "Needs more detail"

        with self.client.post(
            "/v1/feedback",
            json={
                "request_id": req_id,
                "rating": rating,
                "comment": comment,
                "metadata": {"load_test": True},
            },
            catch_response=True,
            name="/v1/feedback",
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"Feedback submission failed: {response.status_code}")

    @task(1)
    def health_probes(self) -> None:
        """Simulate orchestrator liveness and readiness probing."""
        self.client.get("/healthz", name="/healthz")
        self.client.get("/readyz", name="/readyz")


@events.test_stop.add_listener
def on_test_stop(environment: Any, **_kwargs: Any) -> None:
    """Log performance benchmark summary on completion."""
    stats = environment.runner.stats.total
    print("\n==================================================")
    print("  Sentinel Load Test Summary")
    print("==================================================")
    print(f"Total Requests:      {stats.num_requests}")
    print(f"Failed Requests:     {stats.num_failures}")
    print(f"Mean Latency:        {stats.avg_response_time:.2f} ms")
    print(f"P95 Latency:         {stats.get_current_response_time_percentile(0.95):.2f} ms")
    print(f"P99 Latency:         {stats.get_current_response_time_percentile(0.99):.2f} ms")
    print("==================================================\n")
