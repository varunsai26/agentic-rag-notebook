"""Mandatory input/output guardrails and LangChain middleware adapters."""

from agentic_rag_notebook.config.settings import Settings
from agentic_rag_notebook.middleware.base import (
    InputCheckResult,
    MiddlewarePipeline,
    OutputCheckResult,
)
from agentic_rag_notebook.middleware.input_guards import (
    InputLengthGuardrail,
    PIIRedactionGuardrail,
    PromptInjectionGuardrail,
)
from agentic_rag_notebook.middleware.langchain_adapter import (
    GuardrailRunContext,
    LangChainGuardrailMiddleware,
    build_langchain_middlewares,
    build_tool_call_logger_middleware,
)
from agentic_rag_notebook.middleware.output_guards import (
    GroundednessCitationGuardrail,
    SecretLeakageOutputGuardrail,
    UnsafeContentOutputGuardrail,
)


def create_default_middleware_pipeline(settings: Settings) -> MiddlewarePipeline:
    """Build the standard mandatory guardrail pipeline from application Settings."""
    return MiddlewarePipeline(
        input_guardrails=[
            InputLengthGuardrail(max_chars=settings.max_input_chars),
            PromptInjectionGuardrail(),
            PIIRedactionGuardrail(enabled=settings.pii_redaction_enabled),
        ],
        output_guardrails=[
            SecretLeakageOutputGuardrail(),
            UnsafeContentOutputGuardrail(),
            GroundednessCitationGuardrail(strict_mode=settings.strict_grounding_mode),
        ],
        enable_input=settings.enable_input_guardrails,
        enable_output=settings.enable_output_guardrails,
    )


__all__ = [
    "GroundednessCitationGuardrail",
    "GuardrailRunContext",
    "InputCheckResult",
    "InputLengthGuardrail",
    "LangChainGuardrailMiddleware",
    "MiddlewarePipeline",
    "OutputCheckResult",
    "PIIRedactionGuardrail",
    "PromptInjectionGuardrail",
    "SecretLeakageOutputGuardrail",
    "UnsafeContentOutputGuardrail",
    "build_langchain_middlewares",
    "build_tool_call_logger_middleware",
    "create_default_middleware_pipeline",
]
