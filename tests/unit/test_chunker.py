"""Unit tests for the document chunker."""

import pytest

from sentinel.rag.chunker import RecursiveChunker


def test_chunker_empty_text() -> None:
    """Empty or whitespace text returns empty list of chunks."""
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    assert chunker.chunk_document("doc1", "") == []
    assert chunker.chunk_document("doc1", "   \n\n  ") == []


def test_chunker_invalid_overlap() -> None:
    """Overlap >= size raises ValueError."""
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)


def test_chunker_splits_and_preserves_metadata() -> None:
    """Documents larger than chunk_size are split and retain metadata."""
    sample_text = (
        "# Kubernetes Pods\n\n"
        "Pods are the smallest deployable units of computing that you can create "
        "and manage in Kubernetes.\n\n"
        "A Pod is a group of one or more containers, with shared storage and network "
        "resources, and a specification for how to run the containers.\n\n"
        "## Pod Lifecycle\n\n"
        "Like individual application containers, pods are considered to be relatively "
        "ephemeral (rather than durable) entities. Pods are created, assigned a unique ID, "
        "and scheduled to nodes where they remain until termination."
    )

    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.chunk_document("k8s_pods", sample_text, metadata={"source": "pods.md"})

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.doc_id == "k8s_pods"
        assert chunk.index == i
        assert chunk.metadata["source"] == "pods.md"
        assert len(chunk.chunk_id) == 64  # SHA-256


def test_chunker_deterministic_ids() -> None:
    """Same doc_id and content produces identical chunk IDs."""
    text = "Kubernetes is an open-source system for automating deployment."
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)

    chunks1 = chunker.chunk_document("doc1", text)
    chunks2 = chunker.chunk_document("doc1", text)

    assert len(chunks1) == len(chunks2)
    assert chunks1[0].chunk_id == chunks2[0].chunk_id
