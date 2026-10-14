# ADR 0002: OpenTelemetry Distributed Tracing & GenAI Semantic Conventions

## Status
Accepted

## Date
2026-10-14

## Context
As Sentinel orchestrates multi-step RAG workflows across Redis exact-match caching, dense vector search in Qdrant, context grading, and resilient LLM routing with retries/fallbacks, understanding system behavior from flat request logs alone becomes insufficient:
1. **Latency Attribution**: When a user request experiences high latency, developers must be able to visually pinpoint whether time was spent in embedding vector computation, Qdrant cosine similarity search, or foundation model inference.
2. **Failover Observability**: Upstream rate-limiting (HTTP 429) triggers backoff retries and fallback to Ollama. Without tracing, verifying the timing, attempt count, and fallback initiation requires assembling fragmented log records.
3. **GenAI Token and Cost Accounting**: Tracking input and completion tokens across diverse providers requires standardized telemetry schema matching industry conventions.
4. **Log-to-Trace Correlation**: Every structured log message should be queryable in tracing platforms using a common identifier.

## Decision
We integrate OpenTelemetry distributed tracing and GenAI semantic conventions across Sentinel:

1. **Vendor-Neutral OpenTelemetry Core**:
   - `TracerProvider` configured with standard resource attributes (`service.name`, `service.version`, `deployment.environment`).
   - Standard OTLP HTTP trace exporter (`opentelemetry-exporter-otlp-proto-http`) sending spans to any compliant backend (Jaeger, Arize Phoenix, Langfuse, or OpenTelemetry Collector).
   - In-memory trace exporter (`InMemorySpanExporter`) provided for fast, hermetic, zero-dependency unit and integration testing.

2. **Hierarchical Span Architecture**:
   - **Gateway / Root Span**: Automatically managed via `FastAPIInstrumentor` for all inbound HTTP requests.
   - **Cache Span (`cache.get`, `cache.set`)**: Captures `cache.system=redis`, `cache.key`, and boolean `cache.hit`.
   - **RAG Spans (`rag.retrieve`, `rag.grade_context`)**: Captures retrieval parameters (`rag.top_k`, `rag.retrieved_chunks_count`, `rag.top_score`) and grading decisions (`rag.score_threshold`, `rag.context_relevant`).
   - **LLM Routing Spans (`llm.route`, `llm.call`, `llm.call_fallback`)**: Follows OpenTelemetry GenAI semantic conventions:
     - `gen_ai.system` (e.g. `gemini`, `ollama`, `fake`)
     - `gen_ai.request.model` & `gen_ai.response.model`
     - `gen_ai.usage.input_tokens` & `gen_ai.usage.output_tokens`
     - `gen_ai.attempt` and backoff retry errors.

3. **Log & Response Correlation**:
   - `JSONFormatter` in `sentinel.core.logging` injects active `trace_id` and `span_id` alongside `request_id` into every JSON log record.
   - Response payload `ChatMetadata` exposes `trace_id` and `span_id` to clients.

4. **Zero-Cost Local Visualization**:
   - Pre-configured `jaeger` service in `docker-compose.yml` (`jaegertracing/all-in-one`) providing immediate web UI waterfall analysis at `http://localhost:16686`.

## Consequences

### Positive
- **Instant Root Cause Analysis**: Developers can inspect trace waterfalls and instantly determine which component introduced latency or failed.
- **Unified Observability**: Direct correlation between log lines, trace spans, and client response metadata.
- **Vendor Agnostic**: Seamless export to any OTLP-compatible collector or monitoring platform without changing application code.
- **Zero Overhead in CI**: In-memory tracing fixtures ensure tests run entirely offline with zero network overhead.

### Negative / Trade-offs
- **Batch Export Delay**: Traces sent via `BatchSpanProcessor` are flushed asynchronously, causing a minor lag (a few hundred milliseconds) before appearing in the Jaeger UI.
- **Memory Footprint**: Keeping span buffers in memory adds a minor memory overhead, bounded by the batch processor queue size.
