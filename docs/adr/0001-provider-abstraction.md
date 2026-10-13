# ADR 0001: Provider Abstraction and Multi-Tier Fallback

## Status
Accepted

## Date
2026-10-14

## Context
Production LLMOps platforms must reliably serve user traffic despite three major operational challenges:
1. **Model Churn and API Instability**: Cloud AI providers continuously update API schemas, deprecate model names, and introduce breaking SDK changes. Direct vendor coupling makes future migrations costly and brittle.
2. **Strict Quota and Rate Limits (HTTP 429)**: Free-tier and developer accounts (e.g., Google AI Studio Gemini API) enforce low Requests Per Minute (RPM) and Requests Per Day (RPD) caps. A surge in requests or temporary upstream exhaustion immediately crashes unshielded applications.
3. **Offline and Cost Constraints ($0 Budget)**: Local developer workstations and offline CI/CD runners must function without incurring cloud bills or requiring external API credentials.

## Decision
We decouple Sentinel's business logic from any specific LLM provider through a two-fold architectural strategy:

1. **Uniform `LLMProvider` Abstract Interface**:
   - Standardizes inputs (`prompt`, `system_instruction`, `temperature`) and returns a normalized `LLMResult` containing text, tokens in/out, latency in milliseconds, model name, and provider origin.
   - Concretely implemented by:
     - `GeminiProvider`: Backed by the official `google-genai` SDK.
     - `OllamaProvider`: Backed by local inference (e.g. `qwen2.5:3b`) via HTTP.
     - `FakeProvider`: Deterministic in-memory provider for zero-cost, zero-latency unit and integration tests in CI.

2. **Resilient `LLMRouter` with Exponential Backoff and Fallback**:
   - Routes requests to the primary provider (`GeminiProvider`).
   - On transient network failure or HTTP 429 (`ResourceExhausted`), executes exponential backoff retries (3 attempts).
   - If retries are exhausted, seamlessly falls back to the local `OllamaProvider`.
   - Records provider metadata (`provider`, `model`, `latency_ms`) directly onto the request audit trail.

## Consequences

### Positive
- **Zero Downtime on Quota Spikes**: User requests survive upstream rate limits without failing or losing state.
- **Cost Isolation**: CI tests execute 100% locally with `FakeProvider`, ensuring zero cloud quota burn during continuous integration.
- **Provider Interchangeability**: Swapping Gemini for Anthropic, OpenAI, or specialized open-source models in future phases requires writing a single subclass without altering LangGraph agent nodes or API endpoints.

### Negative / Trade-offs
- **Latency Variation**: In the event of a primary provider failure, exponential backoff retries add latency before switching to the fallback model.
- **Capability Drift**: Fallback models (such as local quantized `qwen2.5:3b`) may exhibit different reasoning styles or formatting subtleties compared to larger frontier models.
