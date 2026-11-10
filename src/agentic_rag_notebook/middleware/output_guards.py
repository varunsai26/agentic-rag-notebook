"""Output guardrails using LangChain's RedactionRule plus RAG groundedness/citation validation."""

from __future__ import annotations

import re

from langchain.agents.middleware import RedactionRule

from agentic_rag_notebook.domain.models import GuardrailStepResult, RetrievedChunk


class SecretLeakageOutputGuardrail:
    """Scrubs leaked API keys, bearer tokens, or private keys using LangChain RedactionRule."""

    name = "langchain_secret_output_guard"

    _RULES: tuple[RedactionRule, ...] = (
        RedactionRule(
            "secret",
            strategy="redact",
            detector=r"\b(?:sk-[A-Za-z0-9_-]{12,}|gsk_[A-Za-z0-9_-]{12,})\b",
        ),
        RedactionRule(
            "bearer_token",
            strategy="redact",
            detector=r"Bearer\s+[A-Za-z0-9._-]{16,}",
        ),
        RedactionRule(
            "private_key",
            strategy="redact",
            detector=r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        ),
    )

    def __init__(self) -> None:
        self._resolved_rules = [rule.resolve() for rule in self._RULES]

    def check(
        self,
        answer: str,
        *,
        retrieved_chunks: list[RetrievedChunk],
        has_active_sources: bool,
    ) -> tuple[str, GuardrailStepResult]:
        del retrieved_chunks, has_active_sources
        sanitized = answer
        found: list[str] = []
        for resolved in self._resolved_rules:
            sanitized, matches = resolved.apply(sanitized)
            if matches:
                found.append(resolved.pii_type)

        if found:
            return sanitized, GuardrailStepResult(
                guard_name=self.name,
                stage="output",
                passed=True,
                action="redact",
                reason=f"Redacted secret pattern(s) via LangChain RedactionRule ({', '.join(found)}).",
            )
        return answer, GuardrailStepResult(
            guard_name=self.name,
            stage="output",
            passed=True,
            action="allow",
            reason="No leaked secrets detected.",
        )


class UnsafeContentOutputGuardrail:
    """Blocks overtly harmful outputs before displaying to the user."""

    name = "unsafe_content_output_guard"

    _UNSAFE_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(
            r"\b(?:how to build a bomb|synthesize\s+(?:ricin|sarin)|steal\s+credit\s+card)\b",
            re.IGNORECASE,
        ),
    )

    def check(
        self,
        answer: str,
        *,
        retrieved_chunks: list[RetrievedChunk],
        has_active_sources: bool,
    ) -> tuple[str, GuardrailStepResult]:
        del retrieved_chunks, has_active_sources
        for pattern in self._UNSAFE_PATTERNS:
            if pattern.search(answer):
                return (
                    "I cannot provide that response because it triggered the safety output guardrail.",
                    GuardrailStepResult(
                        guard_name=self.name,
                        stage="output",
                        passed=False,
                        action="block",
                        reason="Output matched unsafe content guardrail policy.",
                    ),
                )
        return answer, GuardrailStepResult(
            guard_name=self.name,
            stage="output",
            passed=True,
            action="allow",
            reason="Output passed safety check.",
        )


class GroundednessCitationGuardrail:
    """Validates RAG grounding & appends transparent source citations or fallback notice."""

    name = "groundedness_citation_guard"

    _GREETING_RE = re.compile(
        r"^(?:hi|hello|hey|thanks|thank you|good\s+(?:morning|afternoon|evening))\b",
        re.IGNORECASE,
    )

    def __init__(self, *, strict_mode: bool = False) -> None:
        self._strict_mode = strict_mode

    def check(
        self,
        answer: str,
        *,
        retrieved_chunks: list[RetrievedChunk],
        has_active_sources: bool,
    ) -> tuple[str, GuardrailStepResult]:
        clean_answer = answer.strip()
        if not clean_answer:
            return (
                "I could not generate a grounded answer. Please try rephrasing your question "
                "or verify your active session sources.",
                GuardrailStepResult(
                    guard_name=self.name,
                    stage="output",
                    passed=False,
                    action="warn",
                    reason="Empty response replaced by safe fallback.",
                ),
            )

        if not has_active_sources:
            return clean_answer, GuardrailStepResult(
                guard_name=self.name,
                stage="output",
                passed=True,
                action="allow",
                reason="No active sources selected in this session; conversational reply allowed.",
            )

        if not retrieved_chunks:
            if self._GREETING_RE.match(clean_answer) and len(clean_answer) < 280:
                return clean_answer, GuardrailStepResult(
                    guard_name=self.name,
                    stage="output",
                    passed=True,
                    action="allow",
                    reason="Conversational greeting/status reply without retrieval.",
                )
            if self._strict_mode:
                return (
                    "I could not find supporting evidence in your active session sources for this "
                    "question. Please add relevant PDF, CSV, or Wikipedia sources or refine your query.",
                    GuardrailStepResult(
                        guard_name=self.name,
                        stage="output",
                        passed=False,
                        action="block",
                        reason="Strict grounding enabled and no document chunks were retrieved.",
                    ),
                )
            notice = (
                "\n\n> **Grounding Notice:** No matching excerpts were retrieved from the active "
                "session sources for this reply; verify claims against your sources."
            )
            if "Grounding Notice:" not in clean_answer:
                clean_answer = f"{clean_answer}{notice}"
            return clean_answer, GuardrailStepResult(
                guard_name=self.name,
                stage="output",
                passed=True,
                action="warn",
                reason="Active sources exist, but zero chunks were retrieved for this response.",
            )

        source_identifiers = list(
            dict.fromkeys(
                ident
                for chunk in retrieved_chunks
                for ident in (chunk.source, chunk.title)
                if ident
            )
        )
        has_source_mention = any(ident in clean_answer for ident in source_identifiers)
        if not has_source_mention:
            formatted_refs: list[str] = []
            for c in retrieved_chunks[:3]:
                if c.page is not None:
                    formatted_refs.append(f"`{c.source}` (p.{c.page})")
                elif c.row is not None:
                    formatted_refs.append(f"`{c.source}` (row {c.row + 1})")
                elif c.url:
                    formatted_refs.append(f"[{c.title or c.source}]({c.url})")
                else:
                    formatted_refs.append(f"`{c.source}`")
            refs = ", ".join(formatted_refs)
            clean_answer = f"{clean_answer}\n\n**Sources:** {refs}"
            return clean_answer, GuardrailStepResult(
                guard_name=self.name,
                stage="output",
                passed=True,
                action="warn",
                reason="Model omitted explicit source names; guardrail appended verified citations.",
            )

        return clean_answer, GuardrailStepResult(
            guard_name=self.name,
            stage="output",
            passed=True,
            action="allow",
            reason=f"Response grounded across {len(retrieved_chunks)} retrieved chunk(s).",
        )
