"""Main FastAPI application entrypoint and lifespan management."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from sentinel.agent.cache import ResponseCache
from sentinel.agent.graph import create_agent_graph
from sentinel.agent.memory import SessionManager
from sentinel.agent.nodes import AgentNodeHandler
from sentinel.agent.semantic_cache import SemanticCache
from sentinel.api.middleware import RequestContextMiddleware
from sentinel.api.routes import router
from sentinel.api.security import (
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
    SlidingWindowRateLimiter,
)
from sentinel.core.config import Settings, get_settings
from sentinel.core.errors import SentinelError
from sentinel.core.logging import get_logger, setup_logging
from sentinel.core.telemetry import setup_telemetry, shutdown_telemetry
from sentinel.core.tenancy import EnterpriseManager
from sentinel.evals.feedback import FeedbackStore
from sentinel.evals.gate import GroundednessGate
from sentinel.guardrails.input import InputGuardrail
from sentinel.guardrails.output import OutputGuardrail
from sentinel.llm.base import LLMProvider
from sentinel.llm.cost_router import CostAwareRouter
from sentinel.llm.fake import FakeProvider
from sentinel.llm.gemini import GeminiProvider
from sentinel.llm.ollama import OllamaProvider
from sentinel.rag.bm25 import BM25Index
from sentinel.rag.embedder import LocalEmbedder
from sentinel.rag.qdrant import QdrantVectorStore
from sentinel.rag.reranker import CrossEncoderReranker
from sentinel.workers.queue import RedisJobQueue
from sentinel.workers.worker import IngestWorker

logger = get_logger(__name__)


def build_app(
    settings: Settings | None = None,
    override_primary_llm: LLMProvider | None = None,
    override_fallback_llm: LLMProvider | None = None,
) -> FastAPI:
    """Application factory supporting testing overrides."""
    current_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # 1. Structured Logging and OpenTelemetry setup
        setup_logging(current_settings.log_level)
        setup_telemetry(current_settings)
        logger.info(
            "Starting Sentinel LLMOps Gateway",
            extra={
                "environment": current_settings.environment,
                "model": current_settings.gemini_model,
                "otel_enabled": current_settings.otel_enabled,
            },
        )

        # 2. Redis Cache
        cache = ResponseCache(
            redis_url=current_settings.redis_url,
            ttl_seconds=current_settings.cache_ttl_seconds,
            enabled=current_settings.cache_enabled,
        )
        app.state.cache = cache

        # 3. Vector DB & Embeddings
        vector_store = QdrantVectorStore(
            url=current_settings.qdrant_url,
            collection_name=current_settings.qdrant_collection,
            vector_size=384,
        )
        app.state.vector_store = vector_store
        try:
            await vector_store.initialize()
        except Exception as exc:
            logger.warning(
                "Could not pre-initialize Qdrant collection at startup", extra={"error": str(exc)}
            )

        embedder = LocalEmbedder(model_name=current_settings.embedding_model)
        app.state.embedder = embedder

        # 4. LLM Providers & Cost Router (Phase 4)
        if override_primary_llm is not None:
            primary_llm = override_primary_llm
        elif (
            current_settings.gemini_api_key
            and current_settings.gemini_api_key != "your_gemini_api_key_here"
        ):
            primary_llm = GeminiProvider(
                api_key=current_settings.gemini_api_key,
                model=current_settings.gemini_model,
            )
        else:
            logger.warning("GEMINI_API_KEY not set or placeholder; using FakeProvider as primary")
            primary_llm = FakeProvider(
                default_response=(
                    "Based on Kubernetes documentation, Deployments manage rolling updates "
                    "declaratively using the RollingUpdate strategy by controlling maxSurge and "
                    "maxUnavailable to guarantee zero downtime."
                ),
                provider_name="fake-gemini",
                model=current_settings.gemini_model,
            )

        if override_fallback_llm is not None:
            fallback_llm = override_fallback_llm
        else:
            fallback_llm = OllamaProvider(
                base_url=current_settings.ollama_base_url,
                model=current_settings.ollama_model,
                timeout_seconds=current_settings.ollama_timeout_seconds,
            )

        cost_router = CostAwareRouter(
            primary_provider=primary_llm,
            fallback_provider=fallback_llm,
            economy_provider=fallback_llm,
            cost_routing_enabled=current_settings.cost_routing_enabled,
        )
        app.state.llm_router = cost_router

        # 5. Semantic Cache (Phase 4)
        semantic_vector_store = QdrantVectorStore(
            url=current_settings.qdrant_url,
            collection_name=current_settings.semantic_cache_collection,
            vector_size=384,
        )
        semantic_cache = SemanticCache(
            vector_store=semantic_vector_store,
            embedder=embedder,
            similarity_threshold=current_settings.semantic_cache_threshold,
            enabled=current_settings.semantic_cache_enabled,
        )
        app.state.semantic_cache = semantic_cache
        try:
            await semantic_cache.initialize()
        except Exception as exc:
            logger.warning(
                "Could not pre-initialize Semantic Cache collection", extra={"error": str(exc)}
            )

        # 6. Safety Guardrails & Groundedness Gate (Phases 3 & 4)
        gate = GroundednessGate(threshold=current_settings.groundedness_threshold)
        input_guardrail = InputGuardrail(enabled=current_settings.guardrails_enabled)
        output_guardrail = OutputGuardrail(enabled=current_settings.guardrails_enabled)

        # 7. Hybrid Search (BM25) & Cross-Encoder Re-ranker (RAG Precision)
        bm25_index = BM25Index.load_from_file(current_settings.bm25_index_path)
        app.state.bm25_index = bm25_index

        reranker = CrossEncoderReranker(model_name=current_settings.reranker_model)
        app.state.reranker = reranker

        node_handler = AgentNodeHandler(
            vector_store=vector_store,
            embedder=embedder,
            llm_router=cost_router,
            score_threshold=current_settings.score_threshold,
            top_k=current_settings.retrieval_top_k,
            groundedness_gate=gate,
            input_guardrail=input_guardrail,
            output_guardrail=output_guardrail,
            bm25_index=bm25_index,
            reranker=reranker,
            hybrid_search_enabled=current_settings.hybrid_search_enabled,
            reranker_enabled=current_settings.reranker_enabled,
            candidates_k=current_settings.retrieval_candidates_k,
            rrf_k=current_settings.hybrid_rrf_k,
        )
        app.state.agent_graph = create_agent_graph(node_handler)

        # 8. Background Ingestion Worker & Queue (Phase 5)
        job_queue = RedisJobQueue(
            redis_url=current_settings.redis_url,
            stream_name=current_settings.ingest_stream_name,
            consumer_group=current_settings.ingest_consumer_group,
        )
        app.state.job_queue = job_queue
        try:
            await job_queue.initialize_consumer_group()
        except Exception as exc:
            logger.warning(
                "Could not initialize Redis consumer group at startup", extra={"error": str(exc)}
            )

        ingest_worker = IngestWorker(
            queue=job_queue,
            vector_store=vector_store,
            embedder=embedder,
            bm25_index=bm25_index,
            bm25_index_path=current_settings.bm25_index_path,
        )
        app.state.ingest_worker = ingest_worker
        worker_task = asyncio.create_task(ingest_worker.run_loop())

        # 9. User Feedback & RLHF Store (Phase 7)
        app.state.feedback_store = FeedbackStore(redis_url=current_settings.redis_url)

        # 10. Multi-Turn Conversational Memory (Phase 8)
        session_manager = SessionManager(
            redis_url=current_settings.redis_url,
            ttl_seconds=current_settings.session_history_ttl_seconds,
            max_turns=current_settings.session_max_history_turns,
            enabled=current_settings.session_memory_enabled,
        )
        app.state.session_manager = session_manager

        # 11. Enterprise Commercialization & Tenancy Manager
        enterprise_manager = EnterpriseManager(redis_url=current_settings.redis_url)
        app.state.enterprise_manager = enterprise_manager

        yield

        # Graceful Shutdown
        logger.info("Shutting down Sentinel services")
        ingest_worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        await session_manager.close()
        await job_queue.close()
        await cache.close()
        await vector_store.close()
        await semantic_vector_store.close()
        shutdown_telemetry()

    app = FastAPI(
        title="Sentinel LLMOps Gateway",
        description="Production gateway for resilient, observable RAG agents",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = current_settings

    # Middleware stack
    if current_settings.security_headers_enabled:
        app.add_middleware(SecurityHeadersMiddleware)

    if current_settings.rate_limit_enabled:
        rate_limiter = SlidingWindowRateLimiter(
            max_requests=current_settings.rate_limit_requests,
            window_seconds=current_settings.rate_limit_window_seconds,
        )
        app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_size_bytes=current_settings.max_request_size_bytes,
    )
    app.add_middleware(RequestContextMiddleware)

    # Exception Handlers
    @app.exception_handler(SentinelError)
    async def handle_sentinel_error(request: Request, exc: SentinelError) -> JSONResponse:
        req_id = getattr(request.state, "request_id", None)
        logger.error(
            "Domain exception occurred", extra={"error": exc.message, "details": exc.details}
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": exc.message,
                "type": exc.__class__.__name__,
                "details": exc.details,
                "request_id": req_id,
            },
        )

    # Routes
    app.include_router(router)

    # OpenTelemetry FastAPI Auto-Instrumentation
    if current_settings.otel_enabled:
        FastAPIInstrumentor.instrument_app(app)

    return app


# Default ASGI application instance
app = build_app()
