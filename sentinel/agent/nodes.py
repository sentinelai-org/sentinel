"""LangGraph nodes for the Sentinel RAG workflow."""

from __future__ import annotations

from typing import Any

from sentinel.agent.state import AgentState
from sentinel.core.logging import get_logger
from sentinel.llm.router import LLMRouter
from sentinel.rag.embedder import LocalEmbedder
from sentinel.rag.vector_store import VectorStore

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are Sentinel, an expert AI platform assistant for technical documentation.
Answer the user's question accurately using ONLY the provided context snippets.
If the context does not contain enough information to answer the question, state that clearly.
Do not fabricate information. Keep your answers concise, well-structured, and helpful."""


class AgentNodeHandler:
    """Encapsulates node execution logic with injected dependencies."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: LocalEmbedder,
        llm_router: LLMRouter,
        score_threshold: float = 0.45,
        top_k: int = 4,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_router = llm_router
        self.score_threshold = score_threshold
        self.top_k = top_k

    async def retrieve(self, state: AgentState) -> dict[str, Any]:
        """Retrieve relevant context chunks from vector store."""
        question = state["question"]
        logger.info("Retrieving documents for question", extra={"question": question})

        query_vec = self.embedder.embed_text(question)
        scored_docs = await self.vector_store.search(
            query_vector=query_vec,
            top_k=self.top_k,
            score_threshold=0.0,
        )

        documents: list[dict[str, Any]] = [
            {
                "text": doc.text,
                "score": doc.score,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.chunk_id,
            }
            for doc in scored_docs
        ]

        logger.info(
            "Retrieved documents",
            extra={
                "count": len(documents),
                "top_score": documents[0]["score"] if documents else 0.0,
            },
        )
        return {"documents": documents}

    async def grade_context(self, state: AgentState) -> dict[str, Any]:
        """Grade whether retrieved documents are relevant enough to answer."""
        documents = state.get("documents", [])

        # Filter documents that meet similarity score threshold
        relevant_docs = [doc for doc in documents if doc["score"] >= self.score_threshold]

        is_relevant = len(relevant_docs) > 0
        logger.info(
            "Graded context relevance",
            extra={
                "total_docs": len(documents),
                "relevant_docs": len(relevant_docs),
                "threshold": self.score_threshold,
                "is_relevant": is_relevant,
            },
        )

        return {
            "context_relevant": is_relevant,
            "documents": relevant_docs if is_relevant else [],
        }

    async def generate(self, state: AgentState) -> dict[str, Any]:
        """Generate answer using LLM router and verified context."""
        question = state["question"]
        documents = state.get("documents", [])

        context_blocks = "\n\n---\n\n".join(
            f"[Source: {doc['source']} (Score: {doc['score']:.2f})]\n{doc['text']}"
            for doc in documents
        )
        prompt = (
            f"Context:\n{context_blocks}\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on the above context:"
        )

        result = await self.llm_router.generate(
            prompt=prompt,
            system_instruction=SYSTEM_PROMPT,
        )

        meta = state.get("meta", {})
        meta.update(
            {
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "provider": result.provider,
                "model": result.model,
                "llm_latency_ms": result.latency_ms,
            }
        )

        return {
            "answer": result.text,
            "meta": meta,
        }

    async def no_context_answer(self, state: AgentState) -> dict[str, Any]:
        """Deterministic response when context is irrelevant to prevent hallucination."""
        logger.info("Routing to no_context_answer: no relevant context found")
        meta = state.get("meta", {})
        meta.update(
            {
                "tokens_in": 0,
                "tokens_out": 0,
                "provider": "system",
                "model": "none",
                "llm_latency_ms": 0.0,
            }
        )
        return {
            "answer": (
                "I could not find sufficient relevant documentation in the knowledge base "
                "to answer your question. Please ensure the topic is covered in the documentation."
            ),
            "meta": meta,
        }

    async def finalize(self, state: AgentState) -> dict[str, Any]:
        """Build sources and final output payload."""
        documents = state.get("documents", [])
        sources = [
            {
                "source": doc["source"],
                "content": doc["text"][:250] + ("..." if len(doc["text"]) > 250 else ""),
                "score": round(doc["score"], 4),
            }
            for doc in documents
        ]
        return {"sources": sources}
