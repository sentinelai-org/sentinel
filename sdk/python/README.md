# Sentinel Python SDK

The official Python client library for the **Sentinel Enterprise AI & RAG Gateway**.

[![PyPI version](https://img.shields.io/pypi/v/sentinel-client.svg)](https://pypi.org/project/sentinel-client/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/your-org/sentinel/blob/main/LICENSE)

---

## Installation

```bash
pip install sentinel-client
```

Or with `poetry`:
```bash
poetry add sentinel-client
```

---

## Quickstart

### Synchronous Usage

```python
from sentinel_client import SentinelClient

client = SentinelClient(
    base_url="https://api.sentinel.yourdomain.com",
    api_key="sk-live-enterprise-key",
)

response = client.chat(
    question="How do Kubernetes Deployments handle rolling updates?",
    session_id="tenant_acme_user_101",
)

print("Answer:", response.answer)
print("Grounded:", response.meta.groundedness_passed)
print("Cost:", f"${response.meta.cost_usd:.6f}")

for source in response.sources:
    print(f"- {source.source} (score: {source.score:.2f})")
```

### Asynchronous Usage & Real-time Streaming

```python
import asyncio
from sentinel_client import AsyncSentinelClient


async def main():
    async with AsyncSentinelClient(api_key="sk-live-enterprise-key") as client:
        # Stream answer tokens live
        async for chunk in client.chat_stream("Explain Kubernetes Pods."):
            if chunk.get("token"):
                print(chunk["token"], end="", flush=True)
            elif chunk.get("type") == "meta":
                print(f"\nLatency: {chunk['meta']['latency_ms']}ms")


asyncio.run(main())
```

### Background Ingestion

```python
from sentinel_client import SentinelClient

client = SentinelClient(api_key="sk-live-enterprise-key")

job = client.ingest(
    [
        {
            "content": "# Runbook\n\nIf redis latency > 50ms, failover.",
            "source": "runbooks/redis.md",
            "metadata": {"team": "sre"},
        }
    ]
)

print("Queued Job ID:", job.job_id)
```

---

## License

Apache-2.0
