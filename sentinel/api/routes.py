"""API endpoints for Sentinel service."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from sentinel.agent.cache import CachedResponse, ResponseCache
from sentinel.agent.memory import SessionManager
from sentinel.agent.nodes import SYSTEM_PROMPT
from sentinel.agent.semantic_cache import SemanticCache
from sentinel.agent.state import AgentState
from sentinel.api.schemas import (
    ChatMessage,
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStats,
    HealthResponse,
    IngestJobRequest,
    IngestJobResponse,
    IngestJobStatus,
    ReadyResponse,
    SessionHistoryResponse,
    SourceItem,
)
from sentinel.api.security import verify_api_key
from sentinel.core.config import get_settings
from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_current_span_id, get_current_trace_id
from sentinel.core.tenancy import (
    Enterprise,
    EnterpriseAuthorizeRequest,
    EnterpriseCreateRequest,
    EnterpriseManager,
)
from sentinel.evals.feedback import FeedbackStore
from sentinel.evals.gate import GroundednessGate
from sentinel.guardrails.input import InputGuardrail
from sentinel.guardrails.output import OutputGuardrail
from sentinel.llm.cost_router import CostAwareRouter
from sentinel.rag.vector_store import VectorStore
from sentinel.workers.queue import BaseJobQueue

logger = get_logger(__name__)


router = APIRouter()


def get_cache(request: Request) -> ResponseCache:
    """Dependency provider for response cache."""
    return request.app.state.cache


def get_vector_store(request: Request) -> VectorStore:
    """Dependency provider for vector store."""
    return request.app.state.vector_store


def get_agent_graph(request: Request) -> Any:
    """Dependency provider for compiled LangGraph agent."""
    return request.app.state.agent_graph


def get_semantic_cache(request: Request) -> SemanticCache | None:
    """Dependency provider for vector-based semantic response cache."""
    return getattr(request.app.state, "semantic_cache", None)


def get_job_queue(request: Request) -> BaseJobQueue:
    """Dependency provider for asynchronous ingestion job queue."""
    return request.app.state.job_queue


def get_llm_router(request: Request) -> Any:
    """Dependency provider for LLM router."""
    return request.app.state.llm_router


def get_embedder(request: Request) -> Any:
    """Dependency provider for text embedder."""
    return request.app.state.embedder


def get_feedback_store(request: Request) -> FeedbackStore:
    """Dependency provider for user feedback store."""
    return request.app.state.feedback_store


def get_session_manager(request: Request) -> SessionManager:
    """Dependency provider for conversational session memory manager."""
    session_mgr = getattr(request.app.state, "session_manager", None)
    if session_mgr is None:
        session_mgr = SessionManager(enabled=True)
    return session_mgr


def get_enterprise_manager(request: Request) -> EnterpriseManager:
    """Dependency provider for enterprise commercialization and tenancy manager."""
    em = getattr(request.app.state, "enterprise_manager", None)
    if em is None:
        em = EnterpriseManager()
        request.app.state.enterprise_manager = em
    return em


@router.get("/ui", response_class=HTMLResponse, tags=["UI"])
@router.get("/admin", response_class=HTMLResponse, tags=["UI"])
async def get_ui() -> str:
    """Serve the web UI and Enterprise Admin Console."""
    ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
    with open(ui_path, encoding="utf-8") as f:
        return f.read()


@router.get("/", tags=["Info"])
async def root() -> dict[str, Any]:
    """Root metadata endpoint with links to documentation and health probes."""
    return {
        "service": "Sentinel LLMOps Gateway",
        "version": "0.1.0",
        "status": "running",
        "docs_url": "/docs",
        "health_url": "/healthz",
        "ready_url": "/readyz",
        "chat_url": "/v1/chat",
        "tracing_ui": "http://localhost:16686",
    }


@router.get("/health", include_in_schema=False)
@router.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def healthz() -> HealthResponse:
    """Liveness probe returning 200 OK if service process is running."""
    return HealthResponse(status="ok")


@router.get("/ready", include_in_schema=False)
@router.get("/readyz", response_model=ReadyResponse, tags=["Health"])
async def readyz(
    cache: ResponseCache = Depends(get_cache),
    vector_store: VectorStore = Depends(get_vector_store),
) -> ReadyResponse:
    """Readiness probe checking connectivity to Qdrant and Redis."""
    qdrant_healthy = await vector_store.health_check()
    redis_healthy = await cache.health_check()

    is_ready = qdrant_healthy and redis_healthy
    components = {
        "qdrant": qdrant_healthy,
        "redis": redis_healthy,
    }

    if not is_ready:
        logger.warning("Readiness probe failed", extra={"components": components})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "components": components},
        )

    return ReadyResponse(status="ready", components=components)


@router.post("/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    payload: ChatRequest,
    request: Request,
    cache: ResponseCache = Depends(get_cache),
    semantic_cache: SemanticCache | None = Depends(get_semantic_cache),
    session_manager: SessionManager = Depends(get_session_manager),
    agent_graph: Any = Depends(get_agent_graph),
    _api_key: str = Depends(verify_api_key),
) -> ChatResponse:
    """Main chat endpoint fronting the LangGraph RAG agent with multi-tier caching and safety guardrails."""
    req_id = getattr(request.state, "request_id", "req_unknown")
    start_time = getattr(request.state, "start_time", time.perf_counter())

    session_id = payload.session_id or str(uuid.uuid4())
    chat_history = await session_manager.get_recent_turns(session_id)
    history_text = await session_manager.format_history_for_prompt(session_id)

    # Step 1: Check L1 Redis exact-match cache
    cached = await cache.get(payload.question)
    if cached is not None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Exact cache hit for query",
            extra={"question": payload.question, "latency_ms": round(elapsed_ms, 2)},
        )
        await session_manager.append_message(session_id, "user", payload.question)
        await session_manager.append_message(
            session_id, "assistant", cached.answer, sources=cached.sources
        )
        return ChatResponse(
            answer=cached.answer,
            sources=[SourceItem(**s) for s in cached.sources],
            session_id=session_id,
            meta=ChatMetadata(
                latency_ms=round(elapsed_ms, 2),
                tokens_in=cached.tokens_in,
                tokens_out=cached.tokens_out,
                provider=cached.provider,
                model=cached.model,
                cache_hit=True,
                cache_type="exact",
                cost_usd=0.0,
                guardrail_passed=True,
                request_id=req_id,
                trace_id=get_current_trace_id(),
                span_id=get_current_span_id(),
            ),
        )

    # Step 2: Check L2 Qdrant semantic vector cache (for standalone queries without prior context)
    if not chat_history and semantic_cache is not None:
        sem_cached, sim_score = await semantic_cache.get(payload.question)
        if sem_cached is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Semantic cache hit for query",
                extra={
                    "question": payload.question,
                    "similarity": round(sim_score, 4),
                    "latency_ms": round(elapsed_ms, 2),
                },
            )
            await session_manager.append_message(session_id, "user", payload.question)
            await session_manager.append_message(
                session_id, "assistant", sem_cached.answer, sources=sem_cached.sources
            )
            return ChatResponse(
                answer=sem_cached.answer,
                sources=[SourceItem(**s) for s in sem_cached.sources],
                session_id=session_id,
                meta=ChatMetadata(
                    latency_ms=round(elapsed_ms, 2),
                    tokens_in=sem_cached.tokens_in,
                    tokens_out=sem_cached.tokens_out,
                    provider=sem_cached.provider,
                    model=sem_cached.model,
                    cache_hit=True,
                    cache_type="semantic",
                    cost_usd=0.0,
                    guardrail_passed=True,
                    request_id=req_id,
                    trace_id=get_current_trace_id(),
                    span_id=get_current_span_id(),
                ),
            )

    # Step 3: Invoke LangGraph agent on double cache miss or multi-turn query
    logger.info("Cache miss, executing LangGraph agent workflow", extra={"session_id": session_id})
    initial_state: AgentState = {
        "question": payload.question,
        "session_id": session_id,
        "request_id": req_id,
        "chat_history": chat_history,
        "history_text": history_text,
        "documents": [],
        "context_relevant": False,
        "answer": "",
        "sources": [],
        "meta": {},
        "groundedness_score": None,
        "groundedness_passed": True,
        "guardrail_passed": True,
        "guardrail_violation": None,
        "cost_usd": None,
    }

    graph_output = await agent_graph.ainvoke(initial_state)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    meta_info = graph_output.get("meta", {})
    sources_data = graph_output.get("sources", [])
    answer = graph_output.get("answer", "")
    cost_usd = meta_info.get("cost_usd", graph_output.get("cost_usd", 0.0))
    guardrail_passed = meta_info.get("guardrail_passed", graph_output.get("guardrail_passed", True))

    meta = ChatMetadata(
        latency_ms=round(elapsed_ms, 2),
        tokens_in=meta_info.get("tokens_in", 0),
        tokens_out=meta_info.get("tokens_out", 0),
        provider=meta_info.get("provider", "unknown"),
        model=meta_info.get("model", "unknown"),
        cache_hit=False,
        cache_type="none",
        cost_usd=cost_usd,
        guardrail_passed=guardrail_passed,
        request_id=req_id,
        trace_id=get_current_trace_id(),
        span_id=get_current_span_id(),
        groundedness_score=meta_info.get("groundedness_score"),
        groundedness_passed=meta_info.get("groundedness_passed", True),
    )

    # Step 4: Populate both L1 and L2 caches if context was grounded, safe, and answered
    if graph_output.get("context_relevant", False) and answer and guardrail_passed:
        cached_entry = CachedResponse(
            answer=answer,
            sources=sources_data,
            model=meta.model,
            provider=meta.provider,
            tokens_in=meta.tokens_in,
            tokens_out=meta.tokens_out,
        )
        await cache.set(payload.question, cached_entry)
        if semantic_cache is not None:
            await semantic_cache.set(payload.question, cached_entry)

    # Step 5: Save message turn to conversational session memory
    await session_manager.append_message(session_id, "user", payload.question)
    await session_manager.append_message(
        session_id,
        "assistant",
        answer,
        sources=sources_data,
    )

    return ChatResponse(
        answer=answer,
        sources=[SourceItem(**s) for s in sources_data],
        session_id=session_id,
        meta=meta,
    )


@router.post(
    "/v1/ingest/jobs",
    response_model=IngestJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Ingest"],
)
async def submit_ingest_job(
    payload: IngestJobRequest,
    job_queue: BaseJobQueue = Depends(get_job_queue),
    _api_key: str = Depends(verify_api_key),
) -> IngestJobResponse:
    """Submit document batch for background asynchronous ingestion via Redis Streams."""
    docs_payload = [doc.model_dump() for doc in payload.documents]
    job_id = await job_queue.enqueue_job(docs_payload)
    return IngestJobResponse(
        job_id=job_id,
        status="queued",
        total_documents=len(payload.documents),
        created_at=datetime.now(UTC).isoformat(),
    )


@router.get(
    "/v1/ingest/jobs/{job_id}",
    response_model=IngestJobStatus,
    tags=["Ingest"],
)
async def get_ingest_job_status(
    job_id: str,
    job_queue: BaseJobQueue = Depends(get_job_queue),
) -> IngestJobStatus:
    """Retrieve status, progress percentage, and lifecycle state of an ingestion job."""
    status_record = await job_queue.get_job_status(job_id)
    if status_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job '{job_id}' not found",
        )
    return IngestJobStatus(**status_record)


@router.post("/v1/chat/stream", tags=["Chat"])
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    cache: ResponseCache = Depends(get_cache),
    semantic_cache: SemanticCache | None = Depends(get_semantic_cache),
    session_manager: SessionManager = Depends(get_session_manager),
    vector_store: VectorStore = Depends(get_vector_store),
    embedder: Any = Depends(get_embedder),
    llm_router: Any = Depends(get_llm_router),
    _api_key: str = Depends(verify_api_key),
) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming chat endpoint with real-time token delivery."""
    settings = get_settings()
    req_id = getattr(request.state, "request_id", "req_unknown")
    start_time = getattr(request.state, "start_time", time.perf_counter())

    session_id = payload.session_id or str(uuid.uuid4())
    chat_history = await session_manager.get_recent_turns(session_id)
    history_text = await session_manager.format_history_for_prompt(session_id)

    async def sse_event_stream() -> AsyncIterator[str]:
        # Step 1: L1 Exact-match cache hit (only if standalone query)
        if not chat_history:
            cached = await cache.get(payload.question)
            if cached is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                await session_manager.append_message(session_id, "user", payload.question)
                await session_manager.append_message(
                    session_id, "assistant", cached.answer, sources=cached.sources
                )
                yield f"event: sources\ndata: {json.dumps(cached.sources)}\n\n"
                yield f"event: token\ndata: {json.dumps({'token': cached.answer})}\n\n"
                meta = {
                    "latency_ms": round(elapsed_ms, 2),
                    "tokens_in": cached.tokens_in,
                    "tokens_out": cached.tokens_out,
                    "provider": cached.provider,
                    "model": cached.model,
                    "cache_hit": True,
                    "cache_type": "exact",
                    "cost_usd": 0.0,
                    "guardrail_passed": True,
                    "groundedness_passed": True,
                    "groundedness_score": 1.0,
                    "request_id": req_id,
                    "session_id": session_id,
                    "trace_id": get_current_trace_id(),
                    "span_id": get_current_span_id(),
                }
                yield f"event: done\ndata: {json.dumps({'meta': meta})}\n\n"
                return

            # Step 2: L2 Semantic cache hit (only if standalone query)
            if semantic_cache is not None:
                sem_cached, sim_score = await semantic_cache.get(payload.question)
                if sem_cached is not None:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    await session_manager.append_message(session_id, "user", payload.question)
                    await session_manager.append_message(
                        session_id, "assistant", sem_cached.answer, sources=sem_cached.sources
                    )
                    yield f"event: sources\ndata: {json.dumps(sem_cached.sources)}\n\n"
                    yield f"event: token\ndata: {json.dumps({'token': sem_cached.answer})}\n\n"
                    meta = {
                        "latency_ms": round(elapsed_ms, 2),
                        "tokens_in": sem_cached.tokens_in,
                        "tokens_out": sem_cached.tokens_out,
                        "provider": sem_cached.provider,
                        "model": sem_cached.model,
                        "cache_hit": True,
                        "cache_type": "semantic",
                        "cost_usd": 0.0,
                        "guardrail_passed": True,
                        "groundedness_passed": True,
                        "groundedness_score": 1.0,
                        "request_id": req_id,
                        "session_id": session_id,
                        "trace_id": get_current_trace_id(),
                        "span_id": get_current_span_id(),
                    }
                    yield f"event: done\ndata: {json.dumps({'meta': meta})}\n\n"
                    return

        # Step 3: Input Guardrails
        input_guardrail = InputGuardrail(enabled=settings.guardrails_enabled)
        eval_result = input_guardrail.validate(payload.question)
        if not eval_result.allowed:
            refusal_text = f"Sentinel Guardrail: Request blocked for safety ({eval_result.reason})."
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            yield "event: sources\ndata: []\n\n"
            yield f"event: token\ndata: {json.dumps({'token': refusal_text})}\n\n"
            meta = {
                "latency_ms": round(elapsed_ms, 2),
                "tokens_in": 0,
                "tokens_out": 0,
                "provider": "guardrail",
                "model": "safety-filter",
                "cache_hit": False,
                "cache_type": "none",
                "cost_usd": 0.0,
                "guardrail_passed": False,
                "groundedness_passed": True,
                "groundedness_score": 1.0,
                "request_id": req_id,
                "trace_id": get_current_trace_id(),
                "span_id": get_current_span_id(),
            }
            yield f"event: done\ndata: {json.dumps({'meta': meta})}\n\n"
            return

        sanitized_question = eval_result.sanitized_text

        # Step 4: Vector Retrieval & Grading
        query_vec = embedder.embed_text(sanitized_question)
        scored_docs = await vector_store.search(
            query_vector=query_vec,
            top_k=settings.retrieval_top_k,
            score_threshold=0.0,
        )
        relevant_docs = [doc for doc in scored_docs if doc.score >= settings.score_threshold]

        if not relevant_docs:
            refusal = (
                "I could not find sufficient relevant documentation in the knowledge base "
                "to answer your question. Please ensure the topic is covered in the documentation."
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            yield "event: sources\ndata: []\n\n"
            yield f"event: token\ndata: {json.dumps({'token': refusal})}\n\n"
            meta = {
                "latency_ms": round(elapsed_ms, 2),
                "tokens_in": 0,
                "tokens_out": 0,
                "provider": "system",
                "model": "none",
                "cache_hit": False,
                "cache_type": "none",
                "cost_usd": 0.0,
                "guardrail_passed": True,
                "groundedness_passed": True,
                "groundedness_score": 1.0,
                "request_id": req_id,
                "trace_id": get_current_trace_id(),
                "span_id": get_current_span_id(),
            }
            yield f"event: done\ndata: {json.dumps({'meta': meta})}\n\n"
            return

        # Format sources
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.text[:250] + ("..." if len(doc.text) > 250 else ""),
                "score": round(doc.score, 4),
            }
            for doc in relevant_docs
        ]
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        context_blocks = "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')} (Score: {doc.score:.2f})]\n{doc.text}"
            for doc in relevant_docs
        )
        history_section = (
            f"Conversation History:\n{history_text}\n\n" if history_text else ""
        )
        prompt = (
            f"Context:\n{context_blocks}\n\n"
            f"{history_section}"
            f"Question: {sanitized_question}\n\n"
            f"Answer based on the above context and conversation history:"
        )

        # Step 5: Stream token generator
        stream_tokens: list[str] = []
        if isinstance(llm_router, CostAwareRouter):
            token_iter = llm_router.route_and_stream(
                prompt=prompt,
                question=sanitized_question,
                system_instruction=SYSTEM_PROMPT,
            )
        else:
            token_iter = llm_router.generate_stream(
                prompt=prompt,
                system_instruction=SYSTEM_PROMPT,
            )

        try:
            async for token in token_iter:
                stream_tokens.append(token)
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
        except Exception as exc:
            logger.error("Streaming error during token generation: %s", exc)
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            return

        full_answer = "".join(stream_tokens)

        # Step 6: Verify Groundedness & Output Guardrail
        gate = GroundednessGate(threshold=settings.groundedness_threshold)
        gate_res = gate.evaluate(
            answer=full_answer,
            context_chunks=[doc.text for doc in relevant_docs],
        )

        out_guard = OutputGuardrail(enabled=settings.guardrails_enabled)
        out_res = out_guard.validate(full_answer)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        primary = getattr(llm_router, "primary_provider", None)
        provider_name = getattr(primary, "name", "llm")
        model_name = getattr(primary, "model_name", "model")
        tokens_in = len(prompt.split())
        tokens_out = len(stream_tokens)

        meta = {
            "latency_ms": round(elapsed_ms, 2),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "provider": provider_name,
            "model": model_name,
            "cache_hit": False,
            "cache_type": "none",
            "cost_usd": 0.0,
            "guardrail_passed": out_res.allowed,
            "groundedness_passed": gate_res.passed,
            "groundedness_score": gate_res.score,
            "request_id": req_id,
            "session_id": session_id,
            "trace_id": get_current_trace_id(),
            "span_id": get_current_span_id(),
        }

        # Cache if valid and standalone
        if not chat_history and gate_res.passed and out_res.allowed:
            cached_entry = CachedResponse(
                answer=full_answer,
                sources=sources,
                model=model_name,
                provider=provider_name,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
            await cache.set(payload.question, cached_entry)
            if semantic_cache is not None:
                await semantic_cache.set(payload.question, cached_entry)

        # Save turn to conversational session memory
        await session_manager.append_message(session_id, "user", payload.question)
        await session_manager.append_message(
            session_id, "assistant", full_answer, sources=sources
        )

        yield f"event: done\ndata: {json.dumps({'meta': meta})}\n\n"

    return StreamingResponse(
        sse_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Phase 7: Closed-Loop Feedback & RLHF Dataset Collection
# ============================================================================


@router.post(
    "/v1/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Feedback"],
)
async def submit_feedback(
    payload: FeedbackRequest,
    feedback_store: FeedbackStore = Depends(get_feedback_store),
    _api_key: str = Depends(verify_api_key),
) -> FeedbackResponse:
    """Submit user evaluation feedback, ratings, and corrections for a query."""
    return await feedback_store.record_feedback(payload)


@router.get(
    "/v1/feedback/stats",
    response_model=FeedbackStats,
    tags=["Feedback"],
)
async def get_feedback_stats(
    feedback_store: FeedbackStore = Depends(get_feedback_store),
) -> FeedbackStats:
    """Retrieve aggregate user satisfaction metrics and rating distributions."""
    return await feedback_store.get_stats()


@router.get(
    "/v1/feedback/export",
    tags=["Feedback"],
)
async def export_tuning_dataset(
    limit: int = 100,
    feedback_store: FeedbackStore = Depends(get_feedback_store),
) -> list[dict[str, Any]]:
    """Export corrected feedback pairs formatted for fine-tuning or DPO alignment."""
    return await feedback_store.export_tuning_dataset(limit=limit)


# ============================================================================
# Phase 8: Multi-Turn Conversational Memory & Session Management
# ============================================================================


@router.get(
    "/v1/chat/sessions/{session_id}",
    response_model=SessionHistoryResponse,
    tags=["Chat"],
)
async def get_session_history(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
    _api_key: str = Depends(verify_api_key),
) -> SessionHistoryResponse:
    """Retrieve chronological message turns for a conversation thread."""
    raw_messages = await session_manager.get_history(session_id)
    messages = [
        ChatMessage(
            role=m.get("role", "user"),
            content=m.get("content", ""),
            timestamp=m.get("timestamp"),
            sources=[SourceItem(**s) for s in m.get("sources", [])],
        )
        for m in raw_messages
    ]
    return SessionHistoryResponse(
        session_id=session_id,
        messages=messages,
        total_messages=len(messages),
    )


@router.delete(
    "/v1/chat/sessions/{session_id}",
    tags=["Chat"],
)
async def clear_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Clear conversation history for a given session ID."""
    deleted = await session_manager.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id, "success": deleted}


# ---------------------------------------------------------------------------
# Enterprise & Commercialization Admin Endpoints
# ---------------------------------------------------------------------------


class AdminTestQueryRequest(BaseModel):
    """Payload for admin running a test query as a specific enterprise in test or live mode."""

    enterprise_id: str
    environment: str = "test"  # "test" or "production"
    question: str


@router.get(
    "/v1/admin/enterprises",
    response_model=list[Enterprise],
    tags=["Enterprise Admin"],
)
async def list_enterprises(
    enterprise_manager: EnterpriseManager = Depends(get_enterprise_manager),
) -> list[Enterprise]:
    """List all registered enterprise clients with their environment credentials and authorization status."""
    return await enterprise_manager.list_enterprises()


@router.post(
    "/v1/admin/enterprises",
    response_model=Enterprise,
    status_code=status.HTTP_201_CREATED,
    tags=["Enterprise Admin"],
)
async def create_enterprise(
    body: EnterpriseCreateRequest,
    enterprise_manager: EnterpriseManager = Depends(get_enterprise_manager),
) -> Enterprise:
    """Register and onboard a new enterprise with sandbox (test) and production keys."""
    return await enterprise_manager.create_enterprise(body)


@router.post(
    "/v1/admin/enterprises/{enterprise_id}/authorize",
    response_model=Enterprise,
    tags=["Enterprise Admin"],
)
async def authorize_enterprise(
    enterprise_id: str,
    body: EnterpriseAuthorizeRequest,
    enterprise_manager: EnterpriseManager = Depends(get_enterprise_manager),
) -> Enterprise:
    """Admin grants production authorization to an enterprise."""
    ent = await enterprise_manager.authorize_enterprise(
        enterprise_id=enterprise_id,
        admin_user=body.admin_user,
        rate_limit=body.rate_limit_per_minute,
        notes=body.notes,
    )
    if not ent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enterprise '{enterprise_id}' not found.",
        )
    return ent


@router.post(
    "/v1/admin/enterprises/{enterprise_id}/revoke",
    response_model=Enterprise,
    tags=["Enterprise Admin"],
)
async def revoke_enterprise(
    enterprise_id: str,
    admin_user: str = "admin@sentinel.internal",
    reason: str | None = None,
    enterprise_manager: EnterpriseManager = Depends(get_enterprise_manager),
) -> Enterprise:
    """Admin revokes or suspends production access for an enterprise."""
    ent = await enterprise_manager.revoke_enterprise(
        enterprise_id=enterprise_id,
        admin_user=admin_user,
        reason=reason,
    )
    if not ent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enterprise '{enterprise_id}' not found.",
        )
    return ent


@router.post(
    "/v1/admin/enterprises/{enterprise_id}/keys/rotate",
    tags=["Enterprise Admin"],
)
async def rotate_enterprise_key(
    enterprise_id: str,
    environment: str = "test",
    enterprise_manager: EnterpriseManager = Depends(get_enterprise_manager),
) -> dict[str, Any]:
    """Rotate the API key for an enterprise in either test or production environment."""
    if environment not in ("test", "production"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Environment must be either 'test' or 'production'.",
        )
    ent, new_key = await enterprise_manager.rotate_key(enterprise_id, environment)
    if not ent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enterprise '{enterprise_id}' not found.",
        )
    return {
        "enterprise_id": enterprise_id,
        "environment": environment,
        "new_key": new_key,
        "enterprise": ent,
    }


@router.post(
    "/v1/admin/test-query",
    tags=["Enterprise Admin"],
)
async def admin_test_query(
    body: AdminTestQueryRequest,
    request: Request,
    enterprise_manager: EnterpriseManager = Depends(get_enterprise_manager),
) -> dict[str, Any]:
    """Execute an interactive test query as an enterprise in either test or production environment."""
    ent = await enterprise_manager.get_enterprise(body.enterprise_id)
    if not ent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enterprise '{body.enterprise_id}' not found.",
        )

    api_key = ent.test_api_key if body.environment == "test" else ent.live_api_key
    is_valid, _, _, is_authorized, error_msg = enterprise_manager.validate_key(api_key)

    if not is_authorized:
        return {
            "authorized": False,
            "environment": body.environment,
            "status_code": 403,
            "error": "Enterprise Authorization Denied",
            "message": error_msg,
            "enterprise": {
                "id": ent.id,
                "name": ent.name,
                "status": ent.status,
                "production_allowed": ent.production_allowed,
            },
        }

    # Execute query through agent graph if loaded
    start_time = time.perf_counter()
    enterprise_manager.record_usage(ent.id, body.environment)

    graph = getattr(request.app.state, "agent_graph", None)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    if graph:
        initial_state = AgentState(
            question=body.question,
            raw_question=body.question,
            request_id=f"adm_test_{uuid.uuid4().hex[:8]}",
            session_id=f"sess_adm_{ent.id}",
        )
        result_state = await graph.ainvoke(initial_state)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "authorized": True,
            "environment": body.environment,
            "status_code": 200,
            "enterprise": {
                "id": ent.id,
                "name": ent.name,
                "status": ent.status,
                "production_allowed": ent.production_allowed,
            },
            "answer": result_state.get("answer", ""),
            "sources": [
                {
                    "source": s.get("source", ""),
                    "content": s.get("content", ""),
                    "score": round(float(s.get("score", 0.0)), 3),
                }
                for s in result_state.get("sources", [])
            ],
            "meta": {
                "latency_ms": round(elapsed_ms, 2),
                "groundedness_score": result_state.get("groundedness_score"),
                "groundedness_passed": result_state.get("groundedness_passed", True),
                "model": result_state.get("model", "gemini-3.8-flash"),
                "tokens_in": result_state.get("tokens_in", 0),
                "tokens_out": result_state.get("tokens_out", 0),
                "api_key_used": f"{api_key[:12]}...{api_key[-4:]}",
            },
        }

    return {
        "authorized": True,
        "environment": body.environment,
        "status_code": 200,
        "enterprise": {
            "id": ent.id,
            "name": ent.name,
            "status": ent.status,
            "production_allowed": ent.production_allowed,
        },
        "answer": f"[Sentinel Diagnostic Response for {ent.name}]: Query executed successfully in {body.environment} mode.",
        "sources": [{"source": "deployments.md", "content": "Deployments documentation", "score": 0.95}],
        "meta": {
            "latency_ms": round(elapsed_ms, 2),
            "groundedness_score": 0.95,
            "groundedness_passed": True,
            "model": "diagnostic-engine",
            "tokens_in": 15,
            "tokens_out": 25,
            "api_key_used": f"{api_key[:12]}...{api_key[-4:]}",
        },
    }

