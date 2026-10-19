# Sentinel Enterprise SaaS & API Integration Guide

This guide details how **Sentinel** can be deployed and consumed as an **Enterprise AI Gateway / RAG-as-a-Service (SaaS)**. External applications—such as internal employee portals, customer-facing web apps, mobile clients, Slack/Teams bots, or CRM systems—interact with Sentinel purely over standard REST and Server-Sent Events (SSE) APIs.

---

## 1. High-Level Architecture: SaaS / Gateway Pattern

When used as a SaaS or centralized internal platform, Sentinel sits between your applications and underlying LLMs/datastores:

```mermaid
flowchart LR
    subgraph Clients["Enterprise Applications"]
        Web[Web Portal / React]
        Mobile[Mobile iOS / Android]
        Bot[Slack / Teams Bot]
        Internal[ERP / CRM Backend]
    end

    subgraph SentinelSaaS["Sentinel Gateway (SaaS)"]
        Ingress["FastAPI Gateway (/v1/chat, /v1/chat/stream)"]
        Guard["Input / Output Guardrails (PII & Injection)"]
        Cache["2-Tier Cache (Redis Exact + Qdrant Semantic)"]
        Hybrid["Hybrid Search (BM25 + Dense RRF)"]
        Rerank["Cross-Encoder Re-Ranker"]
        Router["Cost-Aware Resilient Router (Gemini -> Ollama)"]
        Gate["Groundedness Gate (Hallucination Prevention)"]
    end

    subgraph Datastores["Managed Infrastructure"]
        Qdrant[(Qdrant Vector DB)]
        Redis[(Redis Cache & Stream Queue)]
        Workers["Ingest Workers"]
        OLLM["Ollama / Gemini APIs"]
    end

    Clients -->|Bearer API Key| Ingress
    Ingress --> Guard
    Guard --> Cache
    Cache --> Hybrid
    Hybrid --> Rerank
    Rerank --> Router
    Router --> OLLM
    OLLM --> Gate
    Gate --> Ingress
    Ingress -->|SSE Tokens or JSON| Clients

    Ingress -.->|Async Ingest /v1/ingest/jobs| Workers
    Workers -.-> Qdrant
```

### Why Enterprise Applications Use Sentinel as an API
1. **Zero LLM Lock-in**: Client apps call one unified endpoint. If Gemini is down or throttled (`429`), Sentinel fails over to local Ollama (`qwen2.5`) with zero client-side code changes.
2. **Built-in Hallucination Prevention**: Answers failing the Groundedness Gate ($\ge 70\%$ factual support) are intercepted before reaching users.
3. **Sub-5ms Exact & Semantic Caching**: Identical or semantically similar queries are returned immediately without incurring LLM cost or latency.
4. **Complete Cost & Token Attribution**: Every response reports exact `cost_usd`, input/output tokens, and latency for enterprise chargeback accounting.

---

## 2. API Reference & Client Implementation

### A. Non-Streaming Query (`POST /v1/chat`)

Use this for backend integrations, asynchronous workers, Slack bots, or batch jobs.

#### HTTP Request
```http
POST /v1/chat HTTP/1.1
Host: api.sentinel.company.com
Content-Type: application/json
Authorization: Bearer sk_live_enterprise_token_xyz

{
  "question": "What is the default ServiceType in Kubernetes?",
  "session_id": "tenant_acme_user_42"
}
```

#### HTTP Response (200 OK)
```json
{
  "answer": "The default ServiceType in Kubernetes is ClusterIP, which exposes the Service on a cluster-internal IP accessible only from within the cluster.",
  "sources": [
    {
      "source": "services.md",
      "score": 0.88,
      "snippet": "ClusterIP: Exposes the Service on an internal IP in the cluster. This is the default ServiceType."
    }
  ],
  "session_id": "tenant_acme_user_42",
  "request_id": "req_8f192bce-110a",
  "meta": {
    "provider": "google-gemini",
    "model": "gemini-2.5-flash",
    "tokens_in": 320,
    "tokens_out": 38,
    "cost_usd": 0.000032,
    "llm_latency_ms": 284.1,
    "cached": false,
    "groundedness_score": 1.0,
    "groundedness_passed": true,
    "guardrail_passed": true,
    "complexity": "simple"
  }
}
```

---

### B. Real-Time Token Streaming (`POST /v1/chat/stream`)

Use this for ChatGPT-like interactive web portals, chat widgets, and mobile apps.

#### TypeScript / React Hook Example
```typescript
import { useState } from 'react';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{ source: string; score: number }>;
}

export function useSentinelChat(apiBaseUrl: string, apiKey: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const sendMessage = async (question: string, sessionId: string) => {
    setIsStreaming(true);
    setMessages((prev) => [...prev, { role: 'user', content: question }]);

    let assistantText = '';
    setMessages((prev) => [...prev, { role: 'assistant', content: '' }]);

    const response = await fetch(`${apiBaseUrl}/v1/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ question, session_id: sessionId }),
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder('utf-8');

    if (!reader) return;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const payload = line.replace('data: ', '').trim();
          if (payload === '[DONE]') {
            setIsStreaming(false);
            break;
          }

          try {
            const event = JSON.parse(payload);
            if (event.token) {
              assistantText += event.token;
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1].content = assistantText;
                return updated;
              });
            }
          } catch {
            // Ignore non-JSON control heartbeats
          }
        }
      }
    }
    setIsStreaming(false);
  };

  return { messages, sendMessage, isStreaming };
}
```

---

### C. Python Backend Integration SDK

Enterprise Python microservices can interact with Sentinel via standard `httpx` or `requests`:

```python
"""Enterprise client wrapper for Sentinel RAG Gateway."""

from typing import Any
import httpx


class SentinelClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def ask(self, question: str, session_id: str = "default_session") -> dict[str, Any]:
        """Submit a RAG query to Sentinel and receive verified domain answer."""
        url = f"{self.base_url}/v1/chat"
        payload = {"question": question, "session_id": session_id}

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()

    def submit_feedback(self, request_id: str, rating: int, comment: str | None = None) -> bool:
        """Submit user satisfaction rating for fine-tuning dataset curation."""
        url = f"{self.base_url}/v1/feedback"
        payload = {"request_id": request_id, "rating": rating, "comment": comment}
        with httpx.Client(timeout=10.0) as client:
            res = client.post(url, json=payload, headers=self.headers)
            return res.status_code == 200


# Example Usage:
client = SentinelClient(base_url="https://sentinel.internal.corp", api_key="sk_live_12345")
result = client.ask("How do Deployments handle rolling updates?")
print("Answer:", result["answer"])
print("Cost:", f"${result['meta']['cost_usd']:.6f}")
print("Grounded:", result["meta"]["groundedness_passed"])
```

---

## 3. Asynchronous Enterprise Ingestion (`POST /v1/ingest/jobs`)

Enterprise systems frequently sync knowledge from sources like Confluence, Notion, ServiceNow, or GitHub markdown repos. Rather than re-indexing on client request threads, client apps submit document batches to Sentinel's Redis Stream queue:

```bash
curl -X POST https://sentinel.internal.corp/v1/ingest/jobs \
  -H "Authorization: Bearer sk_live_enterprise_token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "# Production Incident Runbook\n\nIf database latency exceeds 500ms, failover to replica cluster using command: failover-db --cluster=prod-east",
        "source": "runbooks/database_failover.md",
        "metadata": {"department": "SRE", "classification": "internal"}
      }
    ]
  }'
```

#### Response (Job Queued)
```json
{
  "job_id": "c92e70a1-432a-4db3-9821-6893e4b01e12",
  "status": "queued",
  "document_count": 1,
  "created_at": "2026-10-19T02:40:00Z"
}
```

The background worker automatically chunks the content, generates dense vectors, updates Qdrant, and indexes keywords into the BM25 inverted index.

---

## 4. Multi-Tenant SaaS Architecture

To operate Sentinel as a commercial multi-tenant SaaS or multi-department enterprise platform:

### 1. Tenant Data Isolation
- **Collection-Per-Tenant**: Configure collection naming dynamically as `sentinel_docs_{tenant_id}`.
- **Payload Filter Isolation (Recommended for scale)**: Store all documents in one collection with metadata tag `"tenant_id": "org_xyz"`. When querying, inject Qdrant payload filters:
  ```python
  Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))])
  ```

### 2. Tenant API Key Authentication & Rate Limiting
- Use a lightweight FastAPI dependency or API Gateway (Kong / Envoy / Cloudflare):
  - Read `Authorization: Bearer <api_key>`
  - Look up tenant plan, active status, and rate limit bucket in Redis (`rate:{tenant_id}:{minute}`)
  - Inject `tenant_id` into `request.state.tenant_id`.

### 3. Usage & Billing Metering
Sentinel already computes and tracks:
- `tokens_in` and `tokens_out`
- `cost_usd` per query
- Exact model executed (`gemini-2.5-flash`, `qwen2.5:3b`, or cache hit)

Enterprises can stream these metrics to Stripe, Datadog, or an internal billing table via OpenTelemetry trace events or by logging the `meta` dictionary.

---

## 5. Security & Compliance Checklist for Enterprise Deployment

| Capability | Sentinel Implementation | Enterprise Benefit |
| :--- | :--- | :--- |
| **Data Privacy (PII)** | Regex & Presidio pattern masking (email, SSN, API keys) | Private data is sanitized before reaching cloud LLMs |
| **Prompt Injection** | Heuristic & pattern detection blocking jailbreak attempts | Protects internal systems from adversarial prompt hacking |
| **Audit Trails** | Distributed tracing with OpenTelemetry & OTLP export to Jaeger | Every retrieval, prompt, cost, and token is permanently auditable |
| **Data Sovereignty** | Self-hosted fallback (Ollama on private Kubernetes cluster) | Critical data never leaves the corporate boundary during cloud outages |
| **Model Verification** | Runtime Groundedness Gate ($\ge 70\%$ threshold) | Eliminates hallucinated facts from reaching end users |
