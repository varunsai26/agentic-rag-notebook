"""Phase 2 Learning Path: Explicit LangGraph StateGraph Agentic RAG with SqliteSaver."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agentic_rag_notebook.config.settings import Settings
from agentic_rag_notebook.domain.interfaces import VectorRepository
from agentic_rag_notebook.domain.models import AgentResponse, GuardrailStepResult, RetrievedChunk
from agentic_rag_notebook.infrastructure.llm_factory import LLMFactory, RuntimeLLMOverrides
from agentic_rag_notebook.infrastructure.sqlite_session_store import SQLiteSessionStore
from agentic_rag_notebook.middleware import MiddlewarePipeline, create_default_middleware_pipeline
from agentic_rag_notebook.orchestration.tools import (
    chunks_to_citations,
    coerce_message_text,
    format_chunks_for_prompt,
)
from agentic_rag_notebook.prompts.templates import (
    QUERY_REWRITE_PROMPT,
    RELEVANCE_GRADER_PROMPT,
    build_langgraph_answer_prompt,
    extract_thinking_and_answer,
)

ModelResolver = Callable[[], tuple[BaseChatModel, str]]

logger = logging.getLogger(__name__)

_CONVERSATIONAL_PREFIXES = (
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "who are you",
    "what can you do",
)


class AgenticRagState(TypedDict, total=False):
    """Explicit graph state for Phase 2 LangGraph corrective/adaptive RAG."""

    session_id: str
    question: str
    rewritten_query: str
    active_sources: list[str]
    history_messages: list[BaseMessage]
    retrieved_chunks: list[RetrievedChunk]
    needs_retrieval: bool
    docs_relevant: bool
    retries: int
    raw_answer: str
    final_answer: str
    reasoning: str
    blocked: bool
    tool_calls: list[str]
    input_steps: list[GuardrailStepResult]
    output_steps: list[GuardrailStepResult]


class LangGraphRagOrchestrator:
    """Phase 2 Agentic RAG orchestrator using explicit LangGraph StateGraph + SqliteSaver."""

    engine_name = "langgraph"
    _MAX_RETRIES = 2

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

    def _build_graph(self, chat_model: BaseChatModel) -> StateGraph[AgenticRagState]:
        graph = StateGraph(AgenticRagState)

        def input_guardrail_node(state: AgenticRagState) -> dict[str, Any]:
            result = self._pipeline.run_input(state["question"])
            if result.blocked:
                return {
                    "question": result.sanitized_text,
                    "blocked": True,
                    "final_answer": result.block_message,
                    "input_steps": result.steps,
                }
            return {
                "question": result.sanitized_text,
                "rewritten_query": result.sanitized_text,
                "blocked": False,
                "input_steps": result.steps,
            }

        def route_query_node(state: AgenticRagState) -> dict[str, Any]:
            active_sources = state.get("active_sources", [])
            q_lower = state["question"].strip().lower()
            is_smalltalk = len(q_lower) < 40 and any(
                q_lower == prefix or q_lower.startswith(f"{prefix} ")
                for prefix in _CONVERSATIONAL_PREFIXES
            )
            needs_retrieval = bool(active_sources) and not is_smalltalk
            return {"needs_retrieval": needs_retrieval}

        def retrieve_node(state: AgenticRagState) -> dict[str, Any]:
            query = state.get("rewritten_query") or state["question"]
            active_sources = state.get("active_sources", [])
            chunks = self._vector_repo.similarity_search(
                query,
                k=self._settings.rag_top_k,
                session_id=state["session_id"],
                active_sources=active_sources,
            )
            tool_calls = [*state.get("tool_calls", []), f"retrieve_chunks({query!r})"]
            return {"retrieved_chunks": chunks, "tool_calls": tool_calls}

        def grade_documents_node(state: AgenticRagState) -> dict[str, Any]:
            chunks = state.get("retrieved_chunks", [])
            if not chunks:
                return {"docs_relevant": False}
            chain = RELEVANCE_GRADER_PROMPT | chat_model
            verdict_msg = chain.invoke(
                {
                    "question": state["question"],
                    "context": format_chunks_for_prompt(chunks[:3]),
                }
            )
            verdict = (
                coerce_message_text(getattr(verdict_msg, "content", verdict_msg)).strip().upper()
            )
            # Default to relevant if a model emits conversational explanation rather than NO
            is_relevant = not verdict.startswith("NO")
            return {"docs_relevant": is_relevant}

        def rewrite_query_node(state: AgenticRagState) -> dict[str, Any]:
            history_msgs = state.get("history_messages", [])
            history_summary = "\n".join(
                f"{msg.type}: {coerce_message_text(msg.content)[:120]}" for msg in history_msgs[-4:]
            )
            chain = QUERY_REWRITE_PROMPT | chat_model
            rewritten_msg = chain.invoke(
                {
                    "history_summary": history_summary or "None",
                    "question": state["question"],
                }
            )
            _, rewritten = extract_thinking_and_answer(
                coerce_message_text(getattr(rewritten_msg, "content", rewritten_msg))
            )
            return {
                "rewritten_query": rewritten.strip() or state["question"],
                "retries": state.get("retries", 0) + 1,
                "tool_calls": [*state.get("tool_calls", []), "rewrite_query()"],
            }

        def generate_node(state: AgenticRagState) -> dict[str, Any]:
            prompt = build_langgraph_answer_prompt()
            active_sources = state.get("active_sources", [])
            sources_label = ", ".join(active_sources) if active_sources else "None selected"
            chunks = state.get("retrieved_chunks", [])
            chain = prompt | chat_model
            ai_msg = chain.invoke(
                {
                    "active_sources_label": sources_label,
                    "history": state.get("history_messages", []),
                    "question": state["question"],
                    "context": format_chunks_for_prompt(chunks),
                }
            )
            raw_text = coerce_message_text(getattr(ai_msg, "content", ai_msg))
            reasoning, clean_answer = extract_thinking_and_answer(raw_text)
            return {"raw_answer": clean_answer, "reasoning": reasoning}

        def output_guardrail_node(state: AgenticRagState) -> dict[str, Any]:
            chunks = state.get("retrieved_chunks", [])
            has_active_sources = bool(state.get("active_sources", []))
            output_res = self._pipeline.run_output(
                state.get("raw_answer", ""),
                retrieved_chunks=chunks,
                has_active_sources=has_active_sources,
            )
            return {
                "final_answer": output_res.final_answer,
                "blocked": output_res.blocked,
                "output_steps": output_res.steps,
            }

        graph.add_node("input_guardrail", input_guardrail_node)
        graph.add_node("route_query", route_query_node)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("grade_documents", grade_documents_node)
        graph.add_node("rewrite_query", rewrite_query_node)
        graph.add_node("generate", generate_node)
        graph.add_node("output_guardrail", output_guardrail_node)

        graph.add_edge(START, "input_guardrail")
        graph.add_conditional_edges(
            "input_guardrail",
            lambda s: END if s.get("blocked") else "route_query",
        )
        graph.add_conditional_edges(
            "route_query",
            lambda s: "retrieve" if s.get("needs_retrieval") else "generate",
        )
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents",
            lambda s: (
                "generate"
                if s.get("docs_relevant") or s.get("retries", 0) >= self._MAX_RETRIES
                else "rewrite_query"
            ),
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("generate", "output_guardrail")
        graph.add_edge("output_guardrail", END)
        return graph

    def ask(
        self,
        *,
        session_id: str,
        query: str,
        active_sources: list[str] | None = None,
    ) -> AgentResponse:
        """Execute Phase 2 StateGraph Agentic RAG backed by SqliteSaver."""
        chat_model, model_label = self._resolve_model()
        history_messages = self._load_recent_history(session_id)
        graph = self._build_graph(chat_model)

        initial_state: AgenticRagState = {
            "session_id": session_id,
            "question": query,
            "rewritten_query": query,
            "active_sources": list(active_sources) if active_sources else [],
            "history_messages": history_messages,
            "retrieved_chunks": [],
            "retries": 0,
            "tool_calls": [],
            "input_steps": [],
            "output_steps": [],
        }
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}

        with SqliteSaver.from_conn_string(str(self._session_store.db_path)) as checkpointer:
            compiled = graph.compile(checkpointer=checkpointer)
            final_state: dict[str, Any] = compiled.invoke(initial_state, config=config)

        blocked = bool(final_state.get("blocked", False))
        final_answer = str(final_state.get("final_answer", ""))
        reasoning = str(final_state.get("reasoning", ""))
        retrieved_chunks = list(final_state.get("retrieved_chunks", []))
        input_steps = list(final_state.get("input_steps", []))
        output_steps = list(final_state.get("output_steps", []))
        tool_calls = list(final_state.get("tool_calls", []))

        if not blocked:
            lc_history = self._session_store.get_langchain_history(session_id)
            lc_history.add_messages(
                [
                    HumanMessage(content=str(final_state.get("question", query))),
                    AIMessage(content=final_answer),
                ]
            )

        report = MiddlewarePipeline.build_report(input_steps, output_steps)
        citations = chunks_to_citations(retrieved_chunks)
        logger.info(
            "LangGraph run completed session_id=%s model=%s blocked=%s retries=%d citations=%d",
            session_id,
            model_label,
            blocked,
            int(final_state.get("retries", 0)),
            len(citations),
        )
        return AgentResponse(
            answer=final_answer,
            reasoning=reasoning,
            citations=citations,
            guardrail_report=report,
            tool_calls=tool_calls,
            engine=self.engine_name,
            model_label=model_label,
            blocked=blocked,
        )
