# ADR 0005: Server-Sent Events (SSE) Streaming and Redis Streams Background Worker

## Status
Accepted

## Date
2026-10-14

## Context
Production LLMOps gateways encounter two critical latency and throughput bottlenecks:
1. **Perceived Latency in Interactive Conversational RAG**: Full-response generation via LLMs often requires 1.5 to 5.0 seconds. Blocking HTTP client requests until the entire answer is formed creates high Time-To-First-Token (TTFT) and an unresponsive user experience.
2. **Blocking Heavy Document Ingestion on HTTP Endpoints**: Chunking large technical documentation, calculating high-dimensional embeddings, and upserting vectors into Qdrant in synchronous API request handlers introduces HTTP timeout risks (504 Gateway Timeout), hogs API server worker threads, and prevents scalable batch indexing.

## Decision
We implement a real-time Server-Sent Events (SSE) streaming gateway and an asynchronous background ingestion job queue powered by Redis Streams:

1. **Real-Time Token Streaming via Server-Sent Events (`POST /v1/chat/stream`)**:
   - Built on native HTTP/1.1 and HTTP/2 `text/event-stream` using FastAPI's `StreamingResponse`.
   - Emits structured event envelopes:
     - `event: sources`: Dispatched immediately upon vector retrieval to provide instant source citations before token generation begins.
     - `event: token`: Emits token chunks (`{"token": "..."}`) incrementally with $< 200\text{ ms}$ TTFT.
     - `event: done`: Emits final operational metadata (`ChatMetadata`) including latency, token counts, Groundedness Gate scores, and cache status.
     - `event: error`: Delivers structured error envelopes in case of upstream provider exceptions.
   - Preserves multi-tier caching: Cache hits (L1 Redis or L2 Semantic) stream directly without invoking LLM providers.
   - Enforces perimeter safety guardrails before streaming and verifies runtime factual groundedness upon stream completion.

2. **Asynchronous Background Ingestion Queue (`sentinel.workers`)**:
   - **`RedisJobQueue` (`sentinel.workers.queue`)**:
     - Submissions via `POST /v1/ingest/jobs` are enqueued non-blockingly into Redis Stream `sentinel:stream:ingest_jobs` with HTTP 202 Accepted.
     - Tracks granular lifecycle state (`queued`, `processing`, `completed`, `failed`) and real-time progress percentages (`0.0%` to `100.0%`) inside Redis Hashes (`sentinel:job:{job_id}`).
     - Inspectable via `GET /v1/ingest/jobs/{job_id}`.
     - Includes `InMemoryJobQueue` for zero-dependency offline automated testing.
   - **`IngestWorker` (`sentinel.workers.worker`)**:
     - Redis Consumer Group worker (`sentinel_workers`) consuming ingestion payloads asynchronously.
     - Automatically chunks documents via `RecursiveChunker`, generates vector embeddings via `LocalEmbedder`, and batches upserts into `QdrantVectorStore`.
     - Uses Redis stream message acknowledgment (`XACK`) to guarantee resilient at-least-once processing.

3. **Modern Interactive Web UI (`sentinel/api/ui.html`)**:
   - Enhanced UI supporting live SSE token streaming via `ReadableStream` with typing animation, live citation rendering, and latency/model tracking.

## Consequences

### Positive
- **Sub-200ms TTFT**: Users observe instantaneous interactive feedback while answers generate.
- **Zero Ingestion Timeouts**: Massive document sets are ingested asynchronously in the background without affecting API latency or throughput.
- **At-Least-Once Reliability**: Redis Streams Consumer Groups guarantee no document batches are lost if a worker restarts.
- **Full Operational Traceability**: Every streaming token and background job is instrumented with OpenTelemetry distributed tracing.

### Neutral / Trade-offs
- Client connection must support persistent HTTP connections for streaming (disabled buffering via `X-Accel-Buffering: no`).
