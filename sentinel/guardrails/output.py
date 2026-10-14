"""Output Safety Guardrail for system prompt leak prevention and secret redaction."""

from __future__ import annotations

import re
from typing import ClassVar

from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer
from sentinel.guardrails.schemas import GuardrailResult, ViolationType

logger = get_logger(__name__)


class OutputGuardrail:
    """Validates LLM-generated output for system prompt leaks and sensitive secret disclosure."""

    # Patterns indicating inadvertent disclosure of system prompts
    SYSTEM_LEAK_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (
            re.compile(
                r"(my\s+instructions\s+(are|state)|as\s+instructed\s+by\s+my\s+system\s+prompt|here\s+is\s+my\s+system\s+prompt)",
                re.IGNORECASE,
            ),
            "System instruction leakage detected in answer",
        ),
    ]

    API_KEY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(sk-[A-Za-z0-9]{20,}|AIzaSy[A-Za-z0-9_-]{33}|AKIA[0-9A-Z]{16})\b"
    )

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._tracer = get_tracer("sentinel.guardrails")

    def validate(self, text: str) -> GuardrailResult:
        """Validate generated text against leakage patterns and redact sensitive secrets."""
        if not self.enabled:
            return GuardrailResult(
                allowed=True,
                sanitized_text=text,
                risk_score=0.0,
            )

        with self._tracer.start_as_current_span("guardrail.output.validate") as span:
            span.set_attribute("guardrail.output.length", len(text))

            # 1. Check for system prompt leaks
            for pattern, reason in self.SYSTEM_LEAK_PATTERNS:
                if pattern.search(text):
                    logger.warning(
                        "System prompt leak detected in LLM answer",
                        extra={"reason": reason},
                    )
                    span.set_attribute("guardrail.allowed", False)
                    span.set_attribute(
                        "guardrail.violation_type", ViolationType.SYSTEM_PROMPT_LEAK.value
                    )
                    span.set_attribute("guardrail.risk_score", 0.9)
                    return GuardrailResult(
                        allowed=False,
                        sanitized_text="The generated answer contained internal instructions and was withheld for safety.",
                        violation_type=ViolationType.SYSTEM_PROMPT_LEAK,
                        reason=reason,
                        risk_score=0.9,
                    )

            # 2. Redact any accidentally exposed API keys or secrets
            sanitized = text
            secrets_redacted = False
            if self.API_KEY_PATTERN.search(sanitized):
                sanitized = self.API_KEY_PATTERN.sub("[REDACTED_API_KEY]", sanitized)
                secrets_redacted = True

            span.set_attribute("guardrail.allowed", True)
            span.set_attribute("guardrail.secrets_redacted", secrets_redacted)
            span.set_attribute("guardrail.risk_score", 0.0)

            return GuardrailResult(
                allowed=True,
                sanitized_text=sanitized,
                violation_type=ViolationType.PII_LEAKAGE if secrets_redacted else None,
                reason="Sensitive token redacted from output" if secrets_redacted else None,
                risk_score=0.1 if secrets_redacted else 0.0,
            )
