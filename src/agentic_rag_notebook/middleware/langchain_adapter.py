"""LangChain built-in middleware stack + thin domain RAG guardrail hook."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolCallRequest,
    ToolRetryMiddleware,
    hook_config,
    wrap_tool_call,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from agentic_rag_notebook.domain.models import GuardrailStepResult, RetrievedChunk
from agentic_rag_notebook.middleware.base import MiddlewarePipeline
from agentic_rag_notebook.prompts.templates import extract_thinking_and_answer

logger = logging.getLogger(__name__)


@dataclass
class GuardrailRunContext:
    """Request-scoped mutable trace passed into middleware and tools for a single turn."""

    has_active_sources: bool = False
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    input_steps: list[GuardrailStepResult] = field(default_factory=list)
    output_steps: list[GuardrailStepResult] = field(default_factory=list)
    blocked: bool = False
    block_message: str = ""
    sanitized_query: str = ""
    reasoning: str = ""
    final_answer: str = ""


class LangChainGuardrailMiddleware(AgentMiddleware[Any, Any]):
    """Thin AgentMiddleware hook for RAG-specific input injection & output groundedness checks."""

    def __init__(self, pipeline: MiddlewarePipeline, context: GuardrailRunContext) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._context = context

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Run input guardrails and short-circuit via `jump_to='end'` if blocked."""
        del runtime
        messages = list(state.get("messages", []))
        last_user_idx: int | None = None
        last_user_msg: HumanMessage | None = None
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if isinstance(msg, HumanMessage):
                last_user_idx = idx
                last_user_msg = msg
                break

        if last_user_idx is None or last_user_msg is None:
            return None

        raw_text = (
            last_user_msg.content
            if isinstance(last_user_msg.content, str)
            else str(last_user_msg.content)
        )
        result = self._pipeline.run_input(raw_text)
        self._context.input_steps.extend(result.steps)
        self._context.sanitized_query = result.sanitized_text

        if result.blocked:
            self._context.blocked = True
            self._context.block_message = result.block_message
            self._context.final_answer = result.block_message
            return {
                "messages": [AIMessage(content=result.block_message)],
                "jump_to": "end",
            }

        if result.sanitized_text != raw_text:
            updated_msg = last_user_msg.model_copy(update={"content": result.sanitized_text})
            messages[last_user_idx] = updated_msg
            return {"messages": messages}

        return None

    def after_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Run output guardrails (secret redaction, safety, groundedness/citations) on final reply."""
        del runtime
        if self._context.blocked:
            return None

        messages = list(state.get("messages", []))
        last_ai_idx: int | None = None
        last_ai_msg: AIMessage | None = None
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if isinstance(msg, AIMessage):
                last_ai_idx = idx
                last_ai_msg = msg
                break

        if last_ai_idx is None or last_ai_msg is None:
            return None

        raw_content = (
            last_ai_msg.content
            if isinstance(last_ai_msg.content, str)
            else str(last_ai_msg.content)
        )
        reasoning, answer_body = extract_thinking_and_answer(raw_content)
        self._context.reasoning = reasoning

        output_res = self._pipeline.run_output(
            answer_body,
            retrieved_chunks=self._context.retrieved_chunks,
            has_active_sources=self._context.has_active_sources,
        )
        self._context.output_steps.extend(output_res.steps)
        self._context.final_answer = output_res.final_answer
        if output_res.blocked:
            self._context.blocked = True

        if output_res.final_answer != answer_body:
            updated_msg = last_ai_msg.model_copy(update={"content": output_res.final_answer})
            messages[last_ai_idx] = updated_msg
            return {"messages": messages}

        return None


def build_tool_call_logger_middleware(
    context: GuardrailRunContext,
    *,
    session_id: str = "",
) -> AgentMiddleware[Any, Any]:
    """Create `@wrap_tool_call` middleware logging and timing tool executions (e.g. `search_knowledge_base`)."""

    @wrap_tool_call(name="RagToolLoggerMiddleware")
    def log_tool_call(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_name = str(request.tool_call.get("name", "unknown"))
        args = dict(request.tool_call.get("args", {}))
        if tool_name == "search_knowledge_base":
            call_label = f"search_knowledge_base({args.get('query', '')!r})"
        elif not args:
            call_label = f"{tool_name}()"
        else:
            call_label = f"{tool_name}({args!r})"

        sid_label = session_id or "default"
        started = time.perf_counter()
        logger.info(
            "Tool call started: tool=%s session_id=%s args=%s",
            tool_name,
            sid_label,
            args,
        )
        try:
            result = handler(request)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.exception(
                "Tool call failed: tool=%s session_id=%s elapsed_ms=%.1f error=%s",
                tool_name,
                sid_label,
                elapsed_ms,
                exc,
            )
            context.tool_calls.append(f"{call_label} [error: {type(exc).__name__}]")
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        context.tool_calls.append(f"{call_label} ({elapsed_ms:.1f}ms)")
        logger.info(
            "Tool call completed: tool=%s session_id=%s elapsed_ms=%.1f",
            tool_name,
            sid_label,
            elapsed_ms,
        )
        return result

    return log_tool_call


def build_langchain_middlewares(
    pipeline: MiddlewarePipeline,
    context: GuardrailRunContext,
    *,
    session_id: str = "",
    chat_model: BaseChatModel | None = None,
    pii_redaction_enabled: bool = True,
    max_history_messages: int = 24,
) -> list[AgentMiddleware[Any, Any]]:
    """Assemble built-in LangChain middlewares alongside our domain RAG guardrail hook."""
    middlewares: list[AgentMiddleware[Any, Any]] = [
        LangChainGuardrailMiddleware(pipeline, context),
        build_tool_call_logger_middleware(context, session_id=session_id),
        ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=6, exit_behavior="continue"),
        ToolRetryMiddleware(max_retries=2, initial_delay=0.2),
    ]
    if pii_redaction_enabled:
        middlewares.extend(
            [
                PIIMiddleware(
                    "email",
                    strategy="redact",
                    apply_to_input=True,
                    apply_to_output=True,
                ),
                PIIMiddleware(
                    "credit_card",
                    strategy="redact",
                    apply_to_input=True,
                    apply_to_output=True,
                ),
                PIIMiddleware(
                    "ip",
                    strategy="redact",
                    apply_to_input=True,
                    apply_to_output=False,
                ),
                PIIMiddleware(
                    "api_key",
                    strategy="redact",
                    detector=r"\b(?:sk-[A-Za-z0-9_-]{12,}|gsk_[A-Za-z0-9_-]{12,})\b",
                    apply_to_input=True,
                    apply_to_output=True,
                ),
            ]
        )
    if chat_model is not None:
        middlewares.append(
            SummarizationMiddleware(
                model=chat_model,
                trigger=("messages", max_history_messages),
                keep=("messages", max(4, max_history_messages // 2)),
            )
        )
    return middlewares
