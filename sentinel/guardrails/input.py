"""Input Safety Guardrail for prompt injection detection and PII masking."""

from __future__ import annotations

import re
from typing import ClassVar

from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer
from sentinel.guardrails.schemas import GuardrailResult, ViolationType

logger = get_logger(__name__)


class InputGuardrail:
    """Validates user prompts for prompt injections, jailbreaks, and redacts PII."""

    # High-risk prompt injection and jailbreak regex patterns
    INJECTION_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (
            re.compile(
                r"(ignore|disregard|forget|bypass)\s+(all\s+)?(previous|prior|above|system)\s+(instructions|directives|prompts|rules)",
                re.IGNORECASE,
            ),
            "Instruction override attempt detected",
        ),
        (
            re.compile(
                r"((you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(in\s+)?(DAN|jailbroken|unrestricted|an\s+evil|developer\s+mode)|\b(developer|dan|jailbreak)\s+mode\b)",
                re.IGNORECASE,
            ),
            "Persona jailbreak attempt detected",
        ),
        (
            re.compile(
                r"(print|reveal|display|output|show|repeat)\s+(your\s+)?(system\s+prompt|initial\s+instructions|hidden\s+rules)",
                re.IGNORECASE,
            ),
            "System prompt extraction attempt detected",
        ),
        (
            re.compile(
                r"(<\s*/?\s*(system|context|instruction|prompt)\s*>|\[\s*system\s*\])",
                re.IGNORECASE,
            ),
            "System delimiter injection detected",
        ),
    ]

    # PII patterns for redaction
    EMAIL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )
    API_KEY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(sk-[A-Za-z0-9]{20,}|AIzaSy[A-Za-z0-9_-]{33}|AKIA[0-9A-Z]{16})\b"
    )
    CREDIT_CARD_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
    IP_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )

    def __init__(self, enabled: bool = True, redact_pii: bool = True) -> None:
        self.enabled = enabled
        self.redact_pii = redact_pii
        self._tracer = get_tracer("sentinel.guardrails")

    def sanitize_pii(self, text: str) -> tuple[str, bool]:
        """Redact sensitive PII elements from input text."""
        sanitized = text
        redacted = False

        if self.EMAIL_PATTERN.search(sanitized):
            sanitized = self.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)
            redacted = True

        if self.API_KEY_PATTERN.search(sanitized):
            sanitized = self.API_KEY_PATTERN.sub("[REDACTED_API_KEY]", sanitized)
            redacted = True

        if self.CREDIT_CARD_PATTERN.search(sanitized):
            sanitized = self.CREDIT_CARD_PATTERN.sub("[REDACTED_CREDIT_CARD]", sanitized)
            redacted = True

        return sanitized, redacted

    def validate(self, text: str) -> GuardrailResult:
        """Validate input text against injection heuristics and apply PII sanitization."""
        if not self.enabled:
            return GuardrailResult(
                allowed=True,
                sanitized_text=text,
                risk_score=0.0,
            )

        with self._tracer.start_as_current_span("guardrail.input.validate") as span:
            span.set_attribute("guardrail.input.length", len(text))

            # 1. Check for prompt injection / jailbreak
            for pattern, reason in self.INJECTION_PATTERNS:
                if pattern.search(text):
                    logger.warning(
                        "Prompt injection rejected by input guardrail",
                        extra={"reason": reason, "snippet": text[:60]},
                    )
                    span.set_attribute("guardrail.allowed", False)
                    span.set_attribute(
                        "guardrail.violation_type", ViolationType.PROMPT_INJECTION.value
                    )
                    span.set_attribute("guardrail.risk_score", 1.0)
                    return GuardrailResult(
                        allowed=False,
                        sanitized_text=text,
                        violation_type=ViolationType.PROMPT_INJECTION,
                        reason=reason,
                        risk_score=1.0,
                    )

            # 2. Check and redact PII
            sanitized_text, pii_detected = (
                self.sanitize_pii(text) if self.redact_pii else (text, False)
            )

            span.set_attribute("guardrail.allowed", True)
            span.set_attribute("guardrail.pii_detected", pii_detected)
            span.set_attribute("guardrail.risk_score", 0.0)

            return GuardrailResult(
                allowed=True,
                sanitized_text=sanitized_text,
                violation_type=ViolationType.PII_LEAKAGE if pii_detected else None,
                reason="PII detected and redacted" if pii_detected else None,
                risk_score=0.1 if pii_detected else 0.0,
            )
