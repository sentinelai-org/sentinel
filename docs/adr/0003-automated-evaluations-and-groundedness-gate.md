# ADR 0003: Automated Evaluations and Runtime Groundedness Gate

## Status
Accepted

## Date
2026-10-14

## Context
Non-deterministic text generation by Large Language Models (LLMs) poses critical challenges in mission-critical technical platforms:
1. **Hallucination Risk**: An agent answering documentation questions may generate plausibly sounding but entirely inaccurate facts (e.g. incorrect Kubernetes flags, nonexistent configuration keys).
2. **Regression Detection**: Modifying chunking strategies, embeddings, retrieval top-k, system instructions, or foundation model versions can inadvertently degrade response quality across edge cases.
3. **CI/CD Quality Enforcement**: Traditional binary assertions (`assert answer == expected`) are useless for generative outputs. A continuous evaluation harness with standardized semantic quality metrics is necessary to prevent merging regressions.

## Decision
We implement a two-tier evaluation and safety framework for Sentinel:

1. **Golden Dataset Benchmark**:
   - Curated evaluation set (`data/evals/golden_dataset.json`) containing technical question-answer pairs, ground truth claims, expected source document citations, and explicitly unanswerable/out-of-domain queries.

2. **Standardized RAG Triad Metrics (`sentinel.evals.metrics`)**:
   - **Groundedness / Faithfulness**: Verifies whether factual claims in the generated response are directly supported by retrieved context snippets.
   - **Context Recall**: Measures whether expected source documents were successfully retrieved from Qdrant.
   - **Answer Relevance**: Measures keyword and semantic alignment between the user's question and the generated answer.

3. **Pluggable Evaluation Judges (`sentinel.evals.judge`)**:
   - **HeuristicJudge**: A deterministic, 100% offline, zero-cost semantic evaluator computing claim-context token overlap and sentence support. Used in CI/CD and unit tests without external cloud dependencies.
   - **LLMJudge**: An LLM-as-a-Judge mechanism employing foundation models (Gemini / Ollama) for nuanced reasoning and semantic score verification.

4. **Runtime Groundedness Gate (`sentinel.evals.gate`)**:
   - Integrated directly into the LangGraph state machine as a post-generation verification node (`validate_groundedness`).
   - If the generated answer scores below `groundedness_threshold` (default 0.70), the response is safely intercepted and replaced with a polite refusal (`SAFE_UNVERIFIED_REFUSAL`), recording `groundedness_passed=False` on the audit passport.

5. **Automated CI/CD Quality Gate (`scripts/evaluate.py`)**:
   - Evaluates the agent on the golden dataset during continuous integration.
   - Fails the build with exit code 1 if mean groundedness drops below target threshold or unanswerable queries fail to be refused.

## Consequences

### Positive
- **Guaranteed Hallucination Shield**: Unverified claims are intercepted before reaching clients.
- **Zero-Cost Continuous Integration**: Heuristic evaluation and FakeProvider allow CI pipelines to evaluate 100% locally with zero cloud token consumption.
- **Auditable Quality Passports**: Clients receive `groundedness_score` and `groundedness_passed` flags directly in response metadata.

### Negative / Trade-offs
- **Latency Overhead**: Adding runtime groundedness verification adds a slight computational step prior to returning the response.
- **Refusal Conservatism**: Strict thresholds (e.g., > 0.85) might occasionally refuse correct answers that use synonyms not present in the source chunks.
