"""Phase 1 Primary: LangChain v1 orchestration with PromptTemplates, SQLite history, and Middleware."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agentic_rag_notebook.config.settings import Settings
from agentic_rag_notebook.domain.interfaces import VectorRepository
from agentic_rag_notebook.domain.models import AgentResponse
from agentic_rag_notebook.infrastructure.llm_factory import LLMFactory, RuntimeLLMOverrides
from agentic_rag_notebook.infrastructure.sqlite_session_store import SQLiteSessionStore
from agentic_rag_notebook.middleware import (
    GuardrailRunContext,
    MiddlewarePipeline,
    build_langchain_middlewares,
    create_default_middleware_pipeline,
)
from agentic_rag_notebook.orchestration.tools import (
    build_rag_tools,
    chunks_to_citations,
    coerce_message_text,
    format_chunks_for_prompt,
)
from agentic_rag_notebook.prompts.templates import (
    build_langchain_rag_prompt,
    build_langgraph_answer_prompt,
    extract_thinking_and_answer,
)

ModelResolver = Callable[[], tuple[BaseChatModel, str]]

logger = logging.getLogger(__name__)


class LangChainRagOrchestrator:
    """Phase 1 Agentic RAG orchestrator powered by LangChain create_agent + AgentMiddleware."""

    engine_name = "langchain"

    def __init__(
        self,
        *,
        settings: Settings,
        vector_repo: VectorRepository,
        session_store: SQLiteSessionStore,
        llm_factory: LLMFactory | None = None,
        middleware_pipeline: MiddlewarePipeline | None = None,
        model_resolver: ModelResolver | None = None,
        llm_overrides: RuntimeLLMOverrides | None = None,
    ) -> None:
        self._settings = settings
        self._vector_repo = vector_repo
        self._session_store = session_store
        self._llm_factory = llm_factory or LLMFactory(settings)
        self._pipeline = middleware_pipeline or create_default_middleware_pipeline(settings)
        self._model_resolver = model_resolver
        self._llm_overrides = llm_overrides

    def _resolve_model(self) -> tuple[BaseChatModel, str]:
        if self._model_resolver is not None:
            return self._model_resolver()
        return self._llm_factory.create(self._llm_overrides)

    def _load_recent_history(self, session_id: str) -> list[BaseMessage]:
        history = self._session_store.get_langchain_history(session_id)
        max_msgs = self._settings.max_history_turns * 2
        messages = history.messages
        if len(messages) > max_msgs:
            return messages[-max_msgs:]
        return messages

    def _fallback_direct_rag(
        self,
        *,
        session_id: str,
        chat_model: BaseChatModel,
        history_messages: list[BaseMessage],
        active_sources: list[str] | None,
        context: GuardrailRunContext,
    ) -> str:
        """Fallback path if a local OpenAI server lacks tool-calling flags."""
        input_res = self._pipeline.run_input(context.sanitized_query or "")
        if not context.input_steps:
            context.input_steps.extend(input_res.steps)
        if input_res.blocked:
            context.blocked = True
            context.block_message = input_res.block_message
            return input_res.block_message

        chunks = (
            self._vector_repo.similarity_search(
                input_res.sanitized_text,
                k=self._settings.rag_top_k,
                session_id=session_id,
                active_sources=active_sources,
            )
            if active_sources
            else []
        )
        context.retrieved_chunks.extend(chunks)
        context.tool_calls.append("fallback_similarity_search()")

        prompt = build_langgraph_answer_prompt()
        sources_label = ", ".join(active_sources) if active_sources else "None selected"
        chain = prompt | chat_model
        raw_msg = chain.invoke(
            {
                "active_sources_label": sources_label,
                "history": history_messages,
                "question": input_res.sanitized_text,
                "context": format_chunks_for_prompt(chunks),
            }
        )
        raw_text = coerce_message_text(getattr(raw_msg, "content", raw_msg))
        reasoning, answer_body = extract_thinking_and_answer(raw_text)
        context.reasoning = reasoning
        output_res = self._pipeline.run_output(
            answer_body,
            retrieved_chunks=context.retrieved_chunks,
            has_active_sources=context.has_active_sources,
        )
        context.output_steps.extend(output_res.steps)
        if output_res.blocked:
            context.blocked = True
        return output_res.final_answer

    def ask(
        self,
        *,
        session_id: str,
        query: str,
        active_sources: list[str] | None = None,
    ) -> AgentResponse:
        """Execute LangChain v1 agentic RAG with SQLite context and mandatory middleware."""
        chat_model, model_label = self._resolve_model()
        history_messages = self._load_recent_history(session_id)

        context = GuardrailRunContext(
            has_active_sources=bool(active_sources),
            sanitized_query=query,
        )
        tools = build_rag_tools(
            vector_repo=self._vector_repo,
            session_id=session_id,
            active_sources=active_sources,
            context=context,
            top_k=self._settings.rag_top_k,
        )
        middlewares = build_langchain_middlewares(
            self._pipeline,
            context,
            session_id=session_id,
            chat_model=chat_model,
            pii_redaction_enabled=self._settings.pii_redaction_enabled,
            max_history_messages=self._settings.max_history_turns * 2,
        )
        system_prompt = build_langchain_rag_prompt(active_sources)

        agent = create_agent(
            model=chat_model,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middlewares,
        )

        final_answer: str
        agent_input: Any = {"messages": [*history_messages, HumanMessage(content=query)]}
        try:
            state_result: dict[str, Any] = agent.invoke(agent_input)
            if context.final_answer:
                final_answer = context.final_answer
            else:
                result_messages = state_result.get("messages", [])
                last_msg = result_messages[-1] if result_messages else AIMessage(content="")
                raw_text = coerce_message_text(getattr(last_msg, "content", ""))
                reasoning, clean_answer = extract_thinking_and_answer(raw_text)
                if reasoning and not context.reasoning:
                    context.reasoning = reasoning
                final_answer = clean_answer
        except Exception as exc:
            # Graceful fallback only if remote/local server rejects tool choice parameters
            err_msg = str(exc).lower()
            if "tool" in err_msg and ("choice" in err_msg or "support" in err_msg):
                logger.warning(
                    "Falling back to direct RAG (no tool-calling support) session_id=%s model=%s",
                    session_id,
                    model_label,
                )
                final_answer = self._fallback_direct_rag(
                    session_id=session_id,
                    chat_model=chat_model,
                    history_messages=history_messages,
                    active_sources=active_sources,
                    context=context,
                )
            else:
                raise

        if not context.blocked:
            lc_history = self._session_store.get_langchain_history(session_id)
            lc_history.add_messages(
                [
                    HumanMessage(content=context.sanitized_query or query),
                    AIMessage(content=final_answer),
                ]
            )

        report = MiddlewarePipeline.build_report(context.input_steps, context.output_steps)
        citations = chunks_to_citations(context.retrieved_chunks)
        return AgentResponse(
            answer=final_answer,
            reasoning=context.reasoning,
            citations=citations,
            guardrail_report=report,
            tool_calls=context.tool_calls,
            engine=self.engine_name,
            model_label=model_label,
            blocked=context.blocked,
        )
