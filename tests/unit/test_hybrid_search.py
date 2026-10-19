"""Unit tests for Hybrid Search (BM25 + Dense RRF) and Cross-Encoder Re-ranking."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from sentinel.agent.nodes import AgentNodeHandler
from sentinel.llm.router import LLMRouter
from sentinel.rag.bm25 import BM25Index, reciprocal_rank_fusion, tokenize
from sentinel.rag.chunker import Chunk
from sentinel.rag.reranker import CrossEncoderReranker
from sentinel.rag.vector_store import ScoredDocument


class TestBM25TokenizerAndIndex:
    """Test suite for technical tokenization and BM25Okapi scoring."""

    def test_tokenize_kubernetes_technical_tokens(self) -> None:
        text = "Kubelet failed with ErrImagePull on flag --kubelet-extra-args and CrashLoopBackOff"
        tokens = tokenize(text)
        assert "errimagepull" in tokens
        assert "kubelet-extra-args" in tokens
        assert "kubelet" in tokens
        assert "crashloopbackoff" in tokens

    def test_bm25_exact_keyword_ranking(self) -> None:
        index = BM25Index()
        chunk1 = Chunk(
            chunk_id="chunk_general_pods",
            doc_id="pods.md",
            text="Pods represent groups of containers running in Kubernetes with shared storage and network.",
            index=0,
            metadata={"source": "pods.md"},
        )
        chunk2 = Chunk(
            chunk_id="chunk_error_codes",
            doc_id="troubleshooting.md",
            text="When an image cannot be pulled from registry, the container enters ErrImagePull state.",
            index=0,
            metadata={"source": "troubleshooting.md"},
        )
        chunk3 = Chunk(
            chunk_id="chunk_kubelet_flags",
            doc_id="kubelet.md",
            text="Configure node flags using --kubelet-extra-args to customize worker node behaviour.",
            index=0,
            metadata={"source": "kubelet.md"},
        )
        index.add_documents([chunk1, chunk2, chunk3])

        # Query for exact technical error code
        results = index.search(query="Why am I getting ErrImagePull in my pod?", top_k=2)
        assert len(results) > 0
        assert results[0].chunk_id == "chunk_error_codes"
        assert results[0].score > 0.0

        # Query for exact flag
        results_flag = index.search(query="how to pass --kubelet-extra-args to node", top_k=2)
        assert len(results_flag) > 0
        assert results_flag[0].chunk_id == "chunk_kubelet_flags"

    def test_bm25_save_and_load(self) -> None:
        index = BM25Index()
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_id="deploy.md",
                text="Deployments manage declarative rolling updates.",
                index=0,
                metadata={"source": "deploy.md"},
            ),
            Chunk(
                chunk_id="c2",
                doc_id="svc.md",
                text="Services provide stable ClusterIP networking.",
                index=0,
                metadata={"source": "svc.md"},
            ),
        ]
        index.add_documents(chunks)

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "bm25_test.json"
            index.save_to_file(file_path)
            assert file_path.exists()

            loaded_index = BM25Index.load_from_file(file_path)
            assert len(loaded_index) == 2
            results = loaded_index.search("ClusterIP networking", top_k=1)
            assert len(results) == 1
            assert results[0].chunk_id == "c2"


class TestReciprocalRankFusion:
    """Test suite for RRF merging of dense and sparse result sets."""

    def test_rrf_boosts_documents_ranked_in_both(self) -> None:
        dense_results = [
            ScoredDocument(
                chunk_id="doc_a", text="Document A text", score=0.88, metadata={"source": "a"}
            ),
            ScoredDocument(
                chunk_id="doc_b", text="Document B text", score=0.85, metadata={"source": "b"}
            ),
            ScoredDocument(
                chunk_id="doc_c", text="Document C text", score=0.80, metadata={"source": "c"}
            ),
        ]
        sparse_results = [
            ScoredDocument(
                chunk_id="doc_b", text="Document B text", score=4.5, metadata={"source": "b"}
            ),
            ScoredDocument(
                chunk_id="doc_d", text="Document D text", score=3.8, metadata={"source": "d"}
            ),
            ScoredDocument(
                chunk_id="doc_a", text="Document A text", score=2.1, metadata={"source": "a"}
            ),
        ]

        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_k=10)

        # Doc B is ranked #2 in dense (1/62) and #1 in sparse (1/61) -> ~0.0325
        # Doc A is ranked #1 in dense (1/61) and #3 in sparse (1/63) -> ~0.0322
        # Doc B should be top or near top
        doc_ids = [d.chunk_id for d in fused]
        assert "doc_b" in doc_ids[:2]
        assert "doc_a" in doc_ids[:2]
        assert len(fused) == 4  # A, B, C, D deduplicated
        assert all(d.score > 0 for d in fused)

        # Check metadata preservation
        doc_b_entry = next(d for d in fused if d.chunk_id == "doc_b")
        assert doc_b_entry.metadata["dense_rank"] == 2
        assert doc_b_entry.metadata["sparse_rank"] == 1


class TestCrossEncoderReranker:
    """Test suite for Cross-Encoder candidate re-ranking."""

    def test_heuristic_reranker_selects_exact_matches(self) -> None:
        reranker = CrossEncoderReranker(force_mock=True)

        candidates = [
            ScoredDocument(
                chunk_id="cand_1",
                text="General Kubernetes containers and networking configuration.",
                score=0.90,
                metadata={},
            ),
            ScoredDocument(
                chunk_id="cand_2",
                text="Fixing the ErrImagePull error by checking image pull secrets and docker tags.",
                score=0.75,
                metadata={},
            ),
            ScoredDocument(
                chunk_id="cand_3",
                text="Setting worker flags with --kubelet-extra-args in kubeadm.",
                score=0.70,
                metadata={},
            ),
        ]

        reranked = reranker.rerank(
            query="How to resolve ErrImagePull?", documents=candidates, top_n=2
        )
        assert len(reranked) == 2
        # The document mentioning ErrImagePull explicitly should be re-ranked to the top
        assert reranked[0].chunk_id == "cand_2"
        assert reranked[0].metadata["pre_rerank_score"] == 0.75
        assert "rerank_score" in reranked[0].metadata


@pytest.mark.asyncio
class TestAgentNodeHandlerHybridIntegration:
    """Test AgentNodeHandler end-to-end retrieve step with Hybrid Search and Re-ranking."""

    async def test_retrieve_pipeline_hybrid_and_rerank(self) -> None:
        # Mock Vector Store (Dense)
        mock_vector_store = MagicMock()
        mock_vector_store.search = AsyncMock(
            return_value=[
                ScoredDocument(
                    chunk_id="dense_doc_1",
                    text="Kubernetes Pods group containers together.",
                    score=0.80,
                    metadata={"source": "pods.md"},
                )
            ]
        )

        # Mock Embedder
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384

        # BM25 Index with exact keyword match
        bm25_index = BM25Index()
        bm25_index.add_documents(
            [
                Chunk(
                    chunk_id="sparse_doc_flags",
                    doc_id="kubelet.md",
                    text="Pass custom kubelet flags via --kubelet-extra-args in kind or kubeadm.",
                    index=0,
                    metadata={"source": "kubelet.md"},
                )
            ]
        )

        reranker = CrossEncoderReranker(force_mock=True)

        handler = AgentNodeHandler(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            llm_router=MagicMock(spec=LLMRouter),
            bm25_index=bm25_index,
            reranker=reranker,
            hybrid_search_enabled=True,
            reranker_enabled=True,
            top_k=2,
            candidates_k=10,
        )

        state = {"question": "What is --kubelet-extra-args?"}
        res = await handler.retrieve(state)  # type: ignore[arg-type]

        documents = res["documents"]
        assert len(documents) > 0
        # The sparse keyword matched document should be retrieved and re-ranked into final documents
        chunk_ids = [d["chunk_id"] for d in documents]
        assert "sparse_doc_flags" in chunk_ids
