"""API endpoints for Sentinel service."""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse

from sentinel.agent.cache import CachedResponse, ResponseCache
from sentinel.agent.state import AgentState
from sentinel.api.schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ReadyResponse,
    SourceItem,
)
from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_current_span_id, get_current_trace_id
from sentinel.rag.vector_store import VectorStore

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


@router.get("/ui", response_class=HTMLResponse, tags=["UI"])
async def get_ui() -> str:
    """Serve the simple web UI."""
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


@router.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def healthz() -> HealthResponse:
    """Liveness probe returning 200 OK if service process is running."""
    return HealthResponse(status="ok")


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
    agent_graph: Any = Depends(get_agent_graph),
) -> ChatResponse:
    """Main chat endpoint fronting the LangGraph RAG agent."""
    req_id = getattr(request.state, "request_id", "req_unknown")
    start_time = getattr(request.state, "start_time", time.perf_counter())

    # Step 1: Check Redis exact-match cache
    cached = await cache.get(payload.question)
    if cached is not None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Cache hit for query",
            extra={"question": payload.question, "latency_ms": round(elapsed_ms, 2)},
        )
        return ChatResponse(
            answer=cached.answer,
            sources=[SourceItem(**s) for s in cached.sources],
            meta=ChatMetadata(
                latency_ms=round(elapsed_ms, 2),
                tokens_in=cached.tokens_in,
                tokens_out=cached.tokens_out,
                provider=cached.provider,
                model=cached.model,
                cache_hit=True,
                request_id=req_id,
                trace_id=get_current_trace_id(),
                span_id=get_current_span_id(),
            ),
        )

    # Step 2: Invoke LangGraph agent on cache miss
    logger.info("Cache miss, executing LangGraph agent workflow")
    initial_state: AgentState = {
        "question": payload.question,
        "session_id": payload.session_id,
        "request_id": req_id,
        "documents": [],
        "context_relevant": False,
        "answer": "",
        "sources": [],
        "meta": {},
        "groundedness_score": None,
        "groundedness_passed": True,
    }

    graph_output = await agent_graph.ainvoke(initial_state)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    meta_info = graph_output.get("meta", {})
    sources_data = graph_output.get("sources", [])
    answer = graph_output.get("answer", "")

    meta = ChatMetadata(
        latency_ms=round(elapsed_ms, 2),
        tokens_in=meta_info.get("tokens_in", 0),
        tokens_out=meta_info.get("tokens_out", 0),
        provider=meta_info.get("provider", "unknown"),
        model=meta_info.get("model", "unknown"),
        cache_hit=False,
        request_id=req_id,
        trace_id=get_current_trace_id(),
        span_id=get_current_span_id(),
        groundedness_score=meta_info.get("groundedness_score"),
        groundedness_passed=meta_info.get("groundedness_passed", True),
    )

    # Step 3: Populate cache if context was grounded and an answer was produced
    if graph_output.get("context_relevant", False) and answer:
        cached_entry = CachedResponse(
            answer=answer,
            sources=sources_data,
            model=meta.model,
            provider=meta.provider,
            tokens_in=meta.tokens_in,
            tokens_out=meta.tokens_out,
        )
        await cache.set(payload.question, cached_entry)

    return ChatResponse(
        answer=answer,
        sources=[SourceItem(**s) for s in sources_data],
        meta=meta,
    )
