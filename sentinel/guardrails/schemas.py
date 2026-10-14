"""Data schemas and types for Safety Guardrails."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ViolationType(StrEnum):
    """Categorized safety guardrail violation types."""

    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    PII_LEAKAGE = "pii_leakage"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    UNSAFE_CONTENT = "unsafe_content"


class GuardrailResult(BaseModel):
    """Result returned by input or output guardrail evaluation."""

    allowed: bool = Field(description="Whether the content is safe to proceed")
    sanitized_text: str = Field(description="Content after PII redaction or sanitization")
    violation_type: ViolationType | None = Field(
        default=None, description="Category of violation if rejected"
    )
    reason: str | None = Field(default=None, description="Human-readable explanation of rejection")
    risk_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Estimated safety risk score"
    )
