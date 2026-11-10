"""Input guardrails leveraging LangChain's built-in RedactionRule plus RAG prompt-injection defense."""

from __future__ import annotations

import re

from langchain.agents.middleware import RedactionRule

from agentic_rag_notebook.domain.models import GuardrailStepResult


class InputLengthGuardrail:
    """Rejects empty queries or excessively long payloads."""

    name = "input_length_guard"

    def __init__(self, *, max_chars: int = 4000) -> None:
        self._max_chars = max_chars

    def check(self, text: str) -> tuple[str, GuardrailStepResult]:
        stripped = text.strip()
        if not stripped:
            return stripped, GuardrailStepResult(
                guard_name=self.name,
                stage="input",
                passed=False,
                action="block",
                reason="Please enter a non-empty question.",
            )
        if len(stripped) > self._max_chars:
            return stripped[: self._max_chars], GuardrailStepResult(
                guard_name=self.name,
                stage="input",
                passed=False,
                action="block",
                reason=f"Input exceeds maximum allowed length of {self._max_chars} characters.",
            )
        return stripped, GuardrailStepResult(
            guard_name=self.name,
            stage="input",
            passed=True,
            action="allow",
            reason="Input length within bounds.",
        )


class PromptInjectionGuardrail:
    """Detects common prompt-injection and system-prompt exfiltration attempts."""

    name = "prompt_injection_guard"

    _PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(
            r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(reveal|print|show|dump|leak)\s+(your\s+)?(system\s+prompt|hidden\s+instructions)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(jailbreak|bypass\s+guardrails?|disable\s+safety|override\s+system)\b",
            re.IGNORECASE,
        ),
        re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE),
    )

    def check(self, text: str) -> tuple[str, GuardrailStepResult]:
        for pattern in self._PATTERNS:
            if pattern.search(text):
                return text, GuardrailStepResult(
                    guard_name=self.name,
                    stage="input",
                    passed=False,
                    action="block",
                    reason=(
                        "Security Guardrail: Potential prompt injection or system-override "
                        "instruction detected and blocked."
                    ),
                )
        return text, GuardrailStepResult(
            guard_name=self.name,
            stage="input",
            passed=True,
            action="allow",
            reason="No prompt-injection patterns detected.",
        )


class PIIRedactionGuardrail:
    """Redacts PII and API keys using LangChain's built-in `RedactionRule` utility."""

    name = "langchain_pii_redaction_guard"

    _RULES: tuple[RedactionRule, ...] = (
        RedactionRule("email", strategy="redact"),
        RedactionRule("credit_card", strategy="redact"),
        RedactionRule("ip", strategy="redact"),
        RedactionRule("ssn", strategy="redact", detector=r"\b\d{3}-\d{2}-\d{4}\b"),
        RedactionRule(
            "phone",
            strategy="redact",
            detector=r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b",
        ),
        RedactionRule(
            "api_key",
            strategy="redact",
            detector=r"\b(?:sk-[A-Za-z0-9_-]{12,}|gsk_[A-Za-z0-9_-]{12,})\b",
        ),
    )

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._resolved_rules = [rule.resolve() for rule in self._RULES]

    def check(self, text: str) -> tuple[str, GuardrailStepResult]:
        if not self._enabled:
            return text, GuardrailStepResult(
                guard_name=self.name,
                stage="input",
                passed=True,
                action="allow",
                reason="PII redaction disabled by configuration.",
            )

        redacted_text = text
        found_types: list[str] = []
        for resolved in self._resolved_rules:
            redacted_text, matches = resolved.apply(redacted_text)
            if matches:
                found_types.append(resolved.pii_type)

        if found_types:
            return redacted_text, GuardrailStepResult(
                guard_name=self.name,
                stage="input",
                passed=True,
                action="redact",
                reason=f"Redacted PII/secrets via LangChain RedactionRule ({', '.join(found_types)}).",
            )

        return text, GuardrailStepResult(
            guard_name=self.name,
            stage="input",
            passed=True,
            action="allow",
            reason="No PII detected.",
        )
