"""Main FastAPI application entrypoint and lifespan management."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sentinel.agent.cache import ResponseCache
from sentinel.agent.graph import create_agent_graph
from sentinel.agent.nodes import AgentNodeHandler
from sentinel.api.middleware import RequestContextMiddleware
from sentinel.api.routes import router
from sentinel.core.config import Settings, get_settings
from sentinel.core.errors import SentinelError
from sentinel.core.logging import get_logger, setup_logging
from sentinel.llm.base import LLMProvider
from sentinel.llm.fake import FakeProvider
from sentinel.llm.gemini import GeminiProvider
from sentinel.llm.ollama import OllamaProvider
from sentinel.llm.router import LLMRouter
from sentinel.rag.embedder import LocalEmbedder
from sentinel.rag.qdrant import QdrantVectorStore

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
        # 1. Structured Logging setup
        setup_logging(current_settings.log_level)
        logger.info(
            "Starting Sentinel LLMOps Gateway",
            extra={
                "environment": current_settings.environment,
                "model": current_settings.gemini_model,
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

        # 4. LLM Providers & Router
        if override_primary_llm is not None:
            primary_llm = override_primary_llm
        elif current_settings.gemini_api_key:
            primary_llm = GeminiProvider(
                api_key=current_settings.gemini_api_key,
                model=current_settings.gemini_model,
            )
        else:
            logger.warning("GEMINI_API_KEY not set; using FakeProvider as primary")
            primary_llm = FakeProvider(
                default_response="Simulated Sentinel response (GEMINI_API_KEY was not configured).",
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

        router_instance = LLMRouter(
            primary_provider=primary_llm,
            fallback_provider=fallback_llm,
        )
        app.state.llm_router = router_instance

        # 5. LangGraph Agent compilation
        node_handler = AgentNodeHandler(
            vector_store=vector_store,
            embedder=embedder,
            llm_router=router_instance,
            score_threshold=current_settings.score_threshold,
            top_k=current_settings.retrieval_top_k,
        )
        app.state.agent_graph = create_agent_graph(node_handler)

        yield

        # Graceful Shutdown
        logger.info("Shutting down Sentinel services")
        await cache.close()
        await vector_store.close()

    app = FastAPI(
        title="Sentinel LLMOps Gateway",
        description="Production gateway for resilient, observable RAG agents",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware
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

    return app


# Default ASGI application instance
app = build_app()
