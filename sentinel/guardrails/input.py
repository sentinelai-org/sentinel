"""Input Safety Guardrail for advanced prompt injection detection and PII masking."""

from __future__ import annotations

import re
import unicodedata
from typing import ClassVar

from sentinel.core.logging import get_logger
from sentinel.core.telemetry import get_tracer
from sentinel.guardrails.schemas import GuardrailResult, ViolationType

logger = get_logger(__name__)

# Zero-width spaces, directional overrides, and invisible Unicode codepoints used to evade filters
ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF\u00AD\u202A-\u202E\u2060-\u206F]")


class InputGuardrail:
    """Validates user prompts for prompt injections, jailbreaks, and redacts PII."""

    # High-risk prompt injection and jailbreak regex patterns
    INJECTION_PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (
            re.compile(
                r"(ignore|disregard|forget|bypass|override)\s+(all\s+)?(previous|prior|above|system|safety)\s+(instructions|directives|prompts|rules|guidelines)",
                re.IGNORECASE,
            ),
            "Instruction override attempt detected",
        ),
        (
            re.compile(
                r"((you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(in\s+)?(DAN|jailbroken|unrestricted|an\s+evil|developer\s+mode|chaosGPT)|\b(developer|dan|jailbreak)\s+mode\b)",
                re.IGNORECASE,
            ),
            "Persona jailbreak attempt detected",
        ),
        (
            re.compile(
                r"(print|reveal|display|output|show|repeat|dump)\s+(your\s+)?(system\s+prompt|initial\s+instructions|hidden\s+rules|core\s+directives)",
                re.IGNORECASE,
            ),
            "System prompt extraction attempt detected",
        ),
        (
            re.compile(
                r"(<\s*/?\s*(system|context|instruction|prompt|meta)\s*>|\[\s*system\s*\]|===\s*HUMAN\s*===)",
                re.IGNORECASE,
            ),
            "System delimiter injection detected",
        ),
        (
            re.compile(
                r"(base64\s*:\s*[A-Za-z0-9+/=]{20,}|(?:eval|exec)\s*\(\s*(?:base64|decode|atob))",
                re.IGNORECASE,
            ),
            "Obfuscated execution payload detected",
        ),
        (
            re.compile(
                r"(from\s+now\s+on\s+you\s+have\s+no\s+(rules|restrictions|filters)|disable\s+(all\s+)?(safety|filters|guardrails))",
                re.IGNORECASE,
            ),
            "Safety bypass attempt detected",
        ),
    ]

    # Extended PII & Secret Patterns
    EMAIL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )
    API_KEY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(sk-[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|AIzaSy[A-Za-z0-9_-]{33}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|bearer\s+[A-Za-z0-9_\-\.]{25,})\b",
        re.IGNORECASE,
    )
    CREDIT_CARD_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b(?:\d{4}[-\s]?){2}\d{5}\b"
    )
    SSN_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    PHONE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    )
    IPV4_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )
    PRIVATE_KEY_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC )?PRIVATE KEY-----"
    )
    PASSWORD_ASSIGNMENT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:password|passwd|secret|auth_token)\s*[:=]\s*['\"]?([^\s'\"]{6,})['\"]?",
        re.IGNORECASE,
    )
    JWT_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\beyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\b"
    )

    def __init__(self, enabled: bool = True, redact_pii: bool = True) -> None:
        self.enabled = enabled
        self.redact_pii = redact_pii
        self._tracer = get_tracer("sentinel.guardrails")

    @staticmethod
    def normalize_text(text: str) -> str:
        """Strip invisible zero-width unicode characters and normalize NFKC text."""
        stripped = ZERO_WIDTH_PATTERN.sub("", text)
        return unicodedata.normalize("NFKC", stripped)

    def sanitize_pii(self, text: str) -> tuple[str, bool]:
        """Redact sensitive PII, credentials, keys, and tokens from input text."""
        sanitized = text
        redacted = False

        if self.PRIVATE_KEY_PATTERN.search(sanitized):
            sanitized = self.PRIVATE_KEY_PATTERN.sub("[REDACTED_PRIVATE_KEY]", sanitized)
            redacted = True

        if self.JWT_PATTERN.search(sanitized):
            sanitized = self.JWT_PATTERN.sub("[REDACTED_JWT_TOKEN]", sanitized)
            redacted = True

        if self.EMAIL_PATTERN.search(sanitized):
            sanitized = self.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)
            redacted = True

        if self.API_KEY_PATTERN.search(sanitized):
            sanitized = self.API_KEY_PATTERN.sub("[REDACTED_API_KEY]", sanitized)
            redacted = True

        if self.CREDIT_CARD_PATTERN.search(sanitized):
            sanitized = self.CREDIT_CARD_PATTERN.sub("[REDACTED_CREDIT_CARD]", sanitized)
            redacted = True

        if self.SSN_PATTERN.search(sanitized):
            sanitized = self.SSN_PATTERN.sub("[REDACTED_SSN]", sanitized)
            redacted = True

        if self.PHONE_PATTERN.search(sanitized):
            sanitized = self.PHONE_PATTERN.sub("[REDACTED_PHONE]", sanitized)
            redacted = True

        if self.PASSWORD_ASSIGNMENT_PATTERN.search(sanitized):
            sanitized = self.PASSWORD_ASSIGNMENT_PATTERN.sub(
                lambda m: m.group(0).replace(m.group(1), "[REDACTED_SECRET]"), sanitized
            )
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

            # 0. Canonical normalization
            normalized = self.normalize_text(text)

            # 1. Check for prompt injection / jailbreak against normalized text
            for pattern, reason in self.INJECTION_PATTERNS:
                if pattern.search(normalized):
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

            # 2. Check and redact PII / sensitive data
            sanitized_text, pii_detected = (
                self.sanitize_pii(normalized) if self.redact_pii else (normalized, False)
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
