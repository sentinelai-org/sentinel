"""Output Safety Guardrail for system prompt leak prevention, secret redaction, and XSS sanitization."""

from __future__ import annotations

import re
from typing import ClassVar

from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer
from sentinel.guardrails.schemas import GuardrailResult, ViolationType

logger = get_logger(__name__)


class OutputGuardrail:
    """Validates LLM-generated output for system prompt leaks, sensitive secret disclosure, and XSS exploits."""

    # Patterns indicating inadvertent disclosure of system prompts
    SYSTEM_LEAK_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (
            re.compile(
                r"(my\s+instructions\s+(are|state)|as\s+instructed\s+by\s+my\s+system\s+prompt|here\s+is\s+my\s+system\s+prompt|i\s+am\s+programmed\s+to\s+follow\s+these\s+instructions)",
                re.IGNORECASE,
            ),
            "System instruction leakage detected in answer",
        ),
    ]

    # Secret & Token patterns
    API_KEY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(sk-[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|AIzaSy[A-Za-z0-9_-]{33}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36})\b"
    )
    JWT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\beyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\b"
    )
    DB_URI_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:postgres|postgresql|mysql|mongodb|redis):\/\/[^\s@]+:[^\s@]+@[^\s\/]+",
        re.IGNORECASE,
    )
    PRIVATE_KEY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC )?PRIVATE KEY-----"
    )

    # Malicious HTML/XSS injection patterns (prevents DOM XSS when markdown is rendered by clients)
    XSS_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (
            re.compile(r"<\s*script\b[^>]*>[\s\S]*?<\s*/\s*script\s*>", re.IGNORECASE),
            "Malicious script tag stripped",
        ),
        (
            re.compile(r"javascript:\s*[^\s\"']+", re.IGNORECASE),
            "Dangerous javascript: URI stripped",
        ),
        (
            re.compile(r"\bon[a-z]+\s*=\s*['\"][^'\"]*['\"]", re.IGNORECASE),
            "Inline event handler stripped",
        ),
    ]

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._tracer = get_tracer("sentinel.guardrails")

    def sanitize_xss(self, text: str) -> str:
        """Neutralize malicious HTML/JavaScript exploit vectors."""
        sanitized = text
        for pattern, _ in self.XSS_PATTERNS:
            sanitized = pattern.sub("", sanitized)
        return sanitized

    def sanitize_secrets(self, text: str) -> tuple[str, bool]:
        """Redact discovered secrets, connection strings, and private keys."""
        sanitized = text
        redacted = False

        if self.PRIVATE_KEY_PATTERN.search(sanitized):
            sanitized = self.PRIVATE_KEY_PATTERN.sub("[REDACTED_PRIVATE_KEY]", sanitized)
            redacted = True

        if self.DB_URI_PATTERN.search(sanitized):
            sanitized = self.DB_URI_PATTERN.sub("[REDACTED_DATABASE_URI]", sanitized)
            redacted = True

        if self.API_KEY_PATTERN.search(sanitized):
            sanitized = self.API_KEY_PATTERN.sub("[REDACTED_API_KEY]", sanitized)
            redacted = True

        if self.JWT_PATTERN.search(sanitized):
            sanitized = self.JWT_PATTERN.sub("[REDACTED_JWT_TOKEN]", sanitized)
            redacted = True

        return sanitized, redacted

    def validate(self, text: str) -> GuardrailResult:
        """Validate generated text against leakage patterns, redact secrets, and sanitize XSS."""
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

            # 2. Redact any accidentally exposed API keys, tokens, or credentials
            sanitized, secrets_redacted = self.sanitize_secrets(text)

            # 3. Strip XSS payloads
            sanitized = self.sanitize_xss(sanitized)

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
