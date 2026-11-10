"""Composable input/output guardrail middleware pipeline (KISS + OCP)."""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_rag_notebook.domain.interfaces import InputGuardrail, OutputGuardrail
from agentic_rag_notebook.domain.models import (
    GuardrailReport,
    GuardrailStepResult,
    RetrievedChunk,
)


@dataclass
class InputCheckResult:
    """Outcome of running all configured input guardrails."""

    sanitized_text: str
    blocked: bool = False
    block_message: str = ""
    steps: list[GuardrailStepResult] = field(default_factory=list)


@dataclass
class OutputCheckResult:
    """Outcome of running all configured output guardrails."""

    final_answer: str
    blocked: bool = False
    steps: list[GuardrailStepResult] = field(default_factory=list)


class MiddlewarePipeline:
    """Runs ordered input and output guardrails and aggregates an audit report."""

    def __init__(
        self,
        *,
        input_guardrails: list[InputGuardrail] | None = None,
        output_guardrails: list[OutputGuardrail] | None = None,
        enable_input: bool = True,
        enable_output: bool = True,
    ) -> None:
        self._input_guardrails = input_guardrails or []
        self._output_guardrails = output_guardrails or []
        self._enable_input = enable_input
        self._enable_output = enable_output

    def run_input(self, text: str) -> InputCheckResult:
        """Execute all input guardrails sequentially, short-circuiting on block."""
        current = text
        steps: list[GuardrailStepResult] = []
        if not self._enable_input:
            return InputCheckResult(sanitized_text=current, steps=steps)

        for guard in self._input_guardrails:
            current, step = guard.check(current)
            steps.append(step)
            if not step.passed and step.action == "block":
                return InputCheckResult(
                    sanitized_text=current,
                    blocked=True,
                    block_message=step.reason or "Request blocked by input guardrail policy.",
                    steps=steps,
                )
        return InputCheckResult(sanitized_text=current, steps=steps)

    def run_output(
        self,
        answer: str,
        *,
        retrieved_chunks: list[RetrievedChunk],
        has_active_sources: bool,
    ) -> OutputCheckResult:
        """Execute all output guardrails sequentially on the generated answer."""
        current = answer
        steps: list[GuardrailStepResult] = []
        if not self._enable_output:
            return OutputCheckResult(final_answer=current, steps=steps)

        blocked = False
        for guard in self._output_guardrails:
            current, step = guard.check(
                current,
                retrieved_chunks=retrieved_chunks,
                has_active_sources=has_active_sources,
            )
            steps.append(step)
            if not step.passed and step.action == "block":
                blocked = True
                break
        return OutputCheckResult(final_answer=current, blocked=blocked, steps=steps)

    @staticmethod
    def build_report(
        input_steps: list[GuardrailStepResult],
        output_steps: list[GuardrailStepResult],
    ) -> GuardrailReport:
        """Merge input and output steps into a single GuardrailReport."""
        all_steps = [*input_steps, *output_steps]
        blocked = any(step.action == "block" and not step.passed for step in all_steps)
        redacted = any(step.action == "redact" for step in all_steps)
        warnings = [
            f"[{step.guard_name}] {step.reason}"
            for step in all_steps
            if step.action == "warn" and step.reason
        ]
        return GuardrailReport(
            blocked=blocked,
            redacted=redacted,
            warnings=warnings,
            steps=all_steps,
        )
