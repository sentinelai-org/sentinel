"""Evaluation metrics for RAG groundedness, context recall, and answer relevance."""

from __future__ import annotations

import re


def _tokenize(text: str) -> set[str]:
    """Extract normalized lowercase alphanumeric tokens, filtering common stopwords."""
    stopwords = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "of",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "should",
        "it",
        "this",
        "that",
        "these",
        "those",
        "which",
        "what",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "as",
        "if",
        "through",
        "across",
        "into",
        "over",
        "between",
        "each",
        "other",
        "such",
        "also",
        "their",
        "they",
        "them",
        "then",
        "there",
        "about",
        "all",
        "both",
        "more",
        "most",
        "so",
        "some",
        "than",
        "too",
        "very",
        "while",
        "its",
        "via",
        "using",
    }
    words = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
    return {w for w in words if w not in stopwords}


def calculate_context_recall(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Measure the ratio of expected sources successfully retrieved."""
    if not expected_sources:
        return 1.0

    retrieved_set = {s.strip().lower() for s in retrieved_sources}
    expected_set = {s.strip().lower() for s in expected_sources}

    matched = len(retrieved_set.intersection(expected_set))
    return round(matched / len(expected_set), 4)


def calculate_groundedness(answer: str, context_chunks: list[str]) -> float:
    """Calculate the groundedness score of an answer against retrieved context snippets.

    Returns a score between 0.0 and 1.0, where 1.0 indicates full factual alignment.
    Refusals (when context is missing) are inherently grounded (score 1.0).
    """
    refusal_keywords = [
        "could not find sufficient",
        "not covered in the documentation",
        "cannot find enough information",
        "does not contain enough information",
        "not enough information",
        "there is not enough information",
        "insufficient information",
        "not mentioned in the context",
        "context does not contain",
        "context does not provide",
        "not covered in the provided",
        "no information provided",
    ]
    if any(kw in answer.lower() for kw in refusal_keywords):
        return 1.0

    if not context_chunks:
        # Generated an answer without any context -> hallucination
        return 0.0

    # Combine all context text
    full_context = " ".join(context_chunks).lower()
    context_tokens = _tokenize(full_context)
    if not context_tokens:
        return 0.0

    # Split answer into distinct sentences
    sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 8]
    if not sentences:
        return 1.0

    supported_count = 0
    for sentence in sentences:
        sentence_tokens = _tokenize(sentence)
        if not sentence_tokens:
            supported_count += 1
            continue

        overlap = len(sentence_tokens.intersection(context_tokens))
        ratio = overlap / len(sentence_tokens)
        # Sentence is considered supported if >= 60% of informative tokens appear in context
        if ratio >= 0.60:
            supported_count += 1

    return round(supported_count / len(sentences), 4)


def calculate_answer_relevance(question: str, answer: str) -> float:
    """Measure keyword relevance of answer to user question."""
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 1.0

    a_tokens = _tokenize(answer)
    if not a_tokens:
        return 0.0

    overlap = len(q_tokens.intersection(a_tokens))
    coverage = overlap / len(q_tokens)
    return round(min(1.0, coverage * 1.5), 4)
