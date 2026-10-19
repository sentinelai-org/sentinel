"""LangGraph nodes for the Sentinel RAG workflow."""

from __future__ import annotations

from typing import Any

from sentinel.agent.state import AgentState
from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer
from sentinel.evals.gate import SAFE_UNVERIFIED_REFUSAL, GroundednessGate
from sentinel.guardrails.input import InputGuardrail
from sentinel.guardrails.output import OutputGuardrail
from sentinel.llm.cost_router import CostAwareRouter, calculate_cost_usd
from sentinel.llm.router import LLMRouter
from sentinel.rag.bm25 import BM25Index, reciprocal_rank_fusion
from sentinel.rag.embedder import LocalEmbedder
from sentinel.rag.reranker import BaseReranker
from sentinel.rag.vector_store import ScoredDocument, VectorStore

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
        input_guardrail: InputGuardrail | None = None,
        output_guardrail: OutputGuardrail | None = None,
        bm25_index: BM25Index | None = None,
        reranker: BaseReranker | None = None,
        hybrid_search_enabled: bool = True,
        reranker_enabled: bool = True,
        candidates_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_router = llm_router
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.groundedness_gate = groundedness_gate or GroundednessGate(threshold=0.70)
        self.input_guardrail = input_guardrail or InputGuardrail(enabled=True)
        self.output_guardrail = output_guardrail or OutputGuardrail(enabled=True)
        self.bm25_index = bm25_index
        self.reranker = reranker
        self.hybrid_search_enabled = hybrid_search_enabled
        self.reranker_enabled = reranker_enabled
        self.candidates_k = candidates_k
        self.rrf_k = rrf_k

    async def guardrail_input(self, state: AgentState) -> dict[str, Any]:
        """Evaluate incoming question for safety violations and redact PII."""
        question = state["question"]
        eval_result = self.input_guardrail.validate(question)
        if not eval_result.allowed:
            return {
                "guardrail_passed": False,
                "guardrail_violation": (
                    eval_result.violation_type.value if eval_result.violation_type else "unsafe"
                ),
                "answer": f"Sentinel Guardrail: Request blocked for safety ({eval_result.reason}).",
            }
        return {
            "question": eval_result.sanitized_text,
            "guardrail_passed": True,
            "guardrail_violation": None,
        }

    async def refuse_unsafe(self, state: AgentState) -> dict[str, Any]:
        """Terminal refusal when input fails safety checks."""
        meta = state.get("meta", {})
        meta.update(
            {
                "tokens_in": 0,
                "tokens_out": 0,
                "provider": "guardrail",
                "model": "safety-filter",
                "llm_latency_ms": 0.0,
                "guardrail_passed": False,
                "cost_usd": 0.0,
                "groundedness_score": 1.0,
                "groundedness_passed": True,
            }
        )
        return {
            "answer": state.get(
                "answer",
                "Sentinel Guardrail: Request blocked for safety reasons.",
            ),
            "guardrail_passed": False,
            "cost_usd": 0.0,
            "meta": meta,
        }

    async def contextualize_query(self, state: AgentState) -> dict[str, Any]:
        """Contextualize follow-up questions with conversation history into a standalone search query."""
        question = state["question"]
        history_text = state.get("history_text", "")

        if not history_text:
            return {"standalone_query": question}

        with tracer.start_as_current_span("agent.contextualize_query") as span:
            span.set_attribute("agent.original_question", question)
            try:
                prompt = (
                    f"Given the conversation history and a follow-up question, rephrase the question to be a standalone, "
                    f"self-contained search query. Do NOT answer the question, only output the rephrased query.\n\n"
                    f"Conversation History:\n{history_text}\n\n"
                    f"Follow-up Question: {question}\n"
                    f"Standalone Query:"
                )
                res = await self.llm_router.generate(
                    prompt=prompt,
                    system_instruction="You are a query reformulator. Rewrite the follow-up question to be a standalone search query based on the conversation history. Output ONLY the single rewritten query line, nothing else.",
                )
                standalone = res.text.strip().strip('"').strip("'")
                if not standalone or len(standalone) < 3:
                    standalone = question

                span.set_attribute("agent.standalone_query", standalone)
                logger.info(
                    "Contextualized follow-up query",
                    extra={"original": question, "standalone": standalone},
                )
                return {"standalone_query": standalone}
            except Exception as exc:
                logger.warning(
                    "Query contextualization failed, falling back to original question",
                    extra={"error": str(exc)},
                )
                return {"standalone_query": question}

    async def retrieve(self, state: AgentState) -> dict[str, Any]:
        """Retrieve relevant context chunks using Hybrid Search (BM25 + Dense RRF) and Cross-Encoder Re-ranking."""
        question = state["question"]
        retrieval_query = state.get("standalone_query") or question
        logger.info(
            "Retrieving documents for question",
            extra={"question": question, "retrieval_query": retrieval_query},
        )

        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("rag.query", question)
            span.set_attribute("rag.retrieval_query", retrieval_query)
            span.set_attribute("rag.top_k", self.top_k)
            span.set_attribute(
                "rag.hybrid_enabled", self.hybrid_search_enabled and bool(self.bm25_index)
            )
            span.set_attribute(
                "rag.reranker_enabled", self.reranker_enabled and bool(self.reranker)
            )

            query_vec = self.embedder.embed_text(retrieval_query)
            retrieval_k = (
                self.candidates_k
                if (self.hybrid_search_enabled or self.reranker_enabled)
                else self.top_k
            )

            # 1. Candidate Retrieval (Dense vs Hybrid BM25)
            dense_docs = await self.vector_store.search(
                query_vector=query_vec,
                top_k=retrieval_k,
                score_threshold=0.0,
            )

            candidate_docs: list[ScoredDocument]
            if (
                self.hybrid_search_enabled
                and self.bm25_index is not None
                and len(self.bm25_index) > 0
            ):
                sparse_docs = self.bm25_index.search(
                    query=retrieval_query,
                    top_k=retrieval_k,
                )
                candidate_docs = reciprocal_rank_fusion(
                    dense_results=dense_docs,
                    sparse_results=sparse_docs,
                    k=self.rrf_k,
                    top_k=retrieval_k,
                )
                span.set_attribute("rag.dense_candidates_count", len(dense_docs))
                span.set_attribute("rag.sparse_candidates_count", len(sparse_docs))
                span.set_attribute("rag.fused_candidates_count", len(candidate_docs))
            else:
                candidate_docs = dense_docs

            # 2. Cross-Encoder Re-ranking (Top-20 candidates -> Top-N final)
            final_docs: list[ScoredDocument]
            if self.reranker_enabled and self.reranker is not None and candidate_docs:
                final_docs = self.reranker.rerank(
                    query=retrieval_query,
                    documents=candidate_docs,
                    top_n=self.top_k,
                )
                span.set_attribute("rag.reranked_count", len(final_docs))
            else:
                final_docs = candidate_docs[: self.top_k]

            documents: list[dict[str, Any]] = [
                {
                    "text": doc.text,
                    "score": doc.score,
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_id": doc.chunk_id,
                }
                for doc in final_docs
            ]

            span.set_attribute("rag.retrieved_chunks_count", len(documents))
            top_score = documents[0]["score"] if documents else 0.0
            span.set_attribute("rag.top_score", top_score)

            logger.info(
                "Retrieved and processed documents",
                extra={
                    "count": len(documents),
                    "top_score": top_score,
                    "hybrid_active": self.hybrid_search_enabled and bool(self.bm25_index),
                    "rerank_active": self.reranker_enabled and bool(self.reranker),
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
        history_text = state.get("history_text", "")

        with tracer.start_as_current_span("agent.generate") as span:
            span.set_attribute("agent.context_chunks_count", len(documents))

            context_blocks = "\n\n---\n\n".join(
                f"[Source: {doc['source']} (Score: {doc['score']:.2f})]\n{doc['text']}"
                for doc in documents
            )
            history_section = f"Conversation History:\n{history_text}\n\n" if history_text else ""
            prompt = (
                f"Context:\n{context_blocks}\n\n"
                f"{history_section}"
                f"Question: {question}\n\n"
                f"Answer based on the above context and conversation history:"
            )

            cost_usd = 0.0
            complexity = "simple"
            if isinstance(self.llm_router, CostAwareRouter):
                result, cost_usd, complexity = await self.llm_router.route_and_generate(
                    prompt=prompt,
                    question=question,
                    system_instruction=SYSTEM_PROMPT,
                )
                span.set_attribute("agent.complexity", complexity)
            else:
                result = await self.llm_router.generate(
                    prompt=prompt,
                    system_instruction=SYSTEM_PROMPT,
                )
                cost_usd = calculate_cost_usd(result.model, result.tokens_in, result.tokens_out)

            span.set_attribute("gen_ai.system", result.provider)
            span.set_attribute("gen_ai.response.model", result.model)
            span.set_attribute("gen_ai.usage.input_tokens", result.tokens_in)
            span.set_attribute("gen_ai.usage.output_tokens", result.tokens_out)
            span.set_attribute("gen_ai.usage.cost_usd", cost_usd)

            meta = state.get("meta", {})
            meta.update(
                {
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "provider": result.provider,
                    "model": result.model,
                    "llm_latency_ms": result.latency_ms,
                    "cost_usd": cost_usd,
                    "complexity": complexity,
                }
            )

            return {
                "answer": result.text,
                "cost_usd": cost_usd,
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

    async def guardrail_output(self, state: AgentState) -> dict[str, Any]:
        """Validate generated answer for system prompt leaks or unredacted secrets."""
        answer = state.get("answer", "")
        eval_result = self.output_guardrail.validate(answer)

        meta = state.get("meta", {})
        meta.update({"guardrail_passed": eval_result.allowed})

        return {
            "answer": eval_result.sanitized_text,
            "guardrail_passed": eval_result.allowed,
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
