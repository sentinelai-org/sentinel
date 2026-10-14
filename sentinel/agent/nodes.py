"""LangGraph nodes for the Sentinel RAG workflow."""

from __future__ import annotations

from typing import Any

from sentinel.agent.state import AgentState
from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer
from sentinel.evals.gate import SAFE_UNVERIFIED_REFUSAL, GroundednessGate
from sentinel.llm.router import LLMRouter
from sentinel.rag.embedder import LocalEmbedder
from sentinel.rag.vector_store import VectorStore

logger = get_logger(__name__)
tracer = get_tracer("sentinel.agent")

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
        groundedness_gate: GroundednessGate | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_router = llm_router
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.groundedness_gate = groundedness_gate or GroundednessGate(threshold=0.70)

    async def retrieve(self, state: AgentState) -> dict[str, Any]:
        """Retrieve relevant context chunks from vector store."""
        question = state["question"]
        logger.info("Retrieving documents for question", extra={"question": question})

        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("rag.query", question)
            span.set_attribute("rag.top_k", self.top_k)

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

            span.set_attribute("rag.retrieved_chunks_count", len(documents))
            top_score = documents[0]["score"] if documents else 0.0
            span.set_attribute("rag.top_score", top_score)

            logger.info(
                "Retrieved documents",
                extra={
                    "count": len(documents),
                    "top_score": top_score,
                },
            )
            return {"documents": documents}

    async def grade_context(self, state: AgentState) -> dict[str, Any]:
        """Grade whether retrieved documents are relevant enough to answer."""
        documents = state.get("documents", [])

        with tracer.start_as_current_span("rag.grade_context") as span:
            span.set_attribute("rag.score_threshold", self.score_threshold)
            span.set_attribute("rag.total_docs", len(documents))

            # Filter documents that meet similarity score threshold
            relevant_docs = [doc for doc in documents if doc["score"] >= self.score_threshold]

            is_relevant = len(relevant_docs) > 0
            span.set_attribute("rag.relevant_docs_count", len(relevant_docs))
            span.set_attribute("rag.context_relevant", is_relevant)

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

        with tracer.start_as_current_span("agent.generate") as span:
            span.set_attribute("agent.context_chunks_count", len(documents))

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

            span.set_attribute("gen_ai.system", result.provider)
            span.set_attribute("gen_ai.response.model", result.model)
            span.set_attribute("gen_ai.usage.input_tokens", result.tokens_in)
            span.set_attribute("gen_ai.usage.output_tokens", result.tokens_out)

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

    async def validate_groundedness(self, state: AgentState) -> dict[str, Any]:
        """Validate generated answer against retrieved context using GroundednessGate."""
        answer = state.get("answer", "")
        documents = state.get("documents", [])
        context_texts = [str(d.get("text", "")) for d in documents]

        with tracer.start_as_current_span("agent.validate_groundedness") as span:
            gate_result = self.groundedness_gate.evaluate(
                answer=answer,
                context_chunks=context_texts,
            )

            span.set_attribute("groundedness.score", gate_result.score)
            span.set_attribute("groundedness.passed", gate_result.passed)

            final_answer = answer if gate_result.passed else SAFE_UNVERIFIED_REFUSAL

            meta = state.get("meta", {})
            meta.update(
                {
                    "groundedness_score": gate_result.score,
                    "groundedness_passed": gate_result.passed,
                }
            )

            return {
                "answer": final_answer,
                "groundedness_score": gate_result.score,
                "groundedness_passed": gate_result.passed,
                "meta": meta,
            }

    async def no_context_answer(self, state: AgentState) -> dict[str, Any]:
        """Deterministic response when context is irrelevant to prevent hallucination."""
        with tracer.start_as_current_span("agent.no_context_answer") as span:
            span.set_attribute("agent.refusal", True)
            logger.info("Routing to no_context_answer: no relevant context found")
            meta = state.get("meta", {})
            meta.update(
                {
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "provider": "system",
                    "model": "none",
                    "llm_latency_ms": 0.0,
                    "groundedness_score": 1.0,
                    "groundedness_passed": True,
                }
            )
            return {
                "answer": (
                    "I could not find sufficient relevant documentation in the knowledge base "
                    "to answer your question. Please ensure the topic is covered in the documentation."
                ),
                "groundedness_score": 1.0,
                "groundedness_passed": True,
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
