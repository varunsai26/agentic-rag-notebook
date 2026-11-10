"""Versioned modular ChatPromptTemplate definitions and Qwen reasoning parser."""

from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

NOTEBOOK_SYSTEM_PROMPT = """You are Agentic RAG Notebook, a meticulous, source-grounded AI research assistant inspired by Gemini NotebookLM.

Core rules:
1. Active knowledge sources (PDFs, CSVs, and Wikipedia articles) in this session: {active_sources_label}.
2. For any question about the active sources, ALWAYS call `search_knowledge_base` before answering. You may call it multiple times with refined queries if initial chunks are incomplete.
3. Ground every factual claim strictly in retrieved excerpts. Cite sources inline as `[source, p.X]` for PDFs, `[source, row Y]` for CSV rows, or `[title]` for Wikipedia articles.
4. If the retrieved sources do not contain enough evidence to answer, state clearly what is missing rather than inventing facts.
5. For simple conversational greetings or meta-questions about active sources, you may respond concisely or call `list_active_sources`.
6. Keep answers well-structured with concise headings, bullet points, or tables where helpful."""

RELEVANCE_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval relevance grader. Given a user question and retrieved source excerpts "
            "(PDF, CSV, or Wikipedia), reply ONLY with 'YES' if the excerpts contain useful evidence to answer "
            "the question, or 'NO' otherwise.",
        ),
        (
            "human",
            "Question: {question}\n\nRetrieved excerpts:\n{context}\n\nRelevant (YES or NO)?",
        ),
    ]
)

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite user questions into concise, keyword-rich semantic search queries optimized for vector retrieval "
            "over uploaded PDFs, CSV datasets, and Wikipedia articles. Return ONLY the rewritten search query text, with no commentary.",
        ),
        (
            "human",
            "Conversation context:\n{history_summary}\n\nOriginal question: {question}",
        ),
    ]
)

LANGGRAPH_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", NOTEBOOK_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            "User question: {question}\n\nRetrieved source context:\n{context}\n\n"
            "Write a grounded answer with inline citations (`[source, p.X]`, `[source, row Y]`, or `[title]`).",
        ),
    ]
)


def build_langchain_rag_prompt(active_sources: list[str] | None) -> str:
    """Render the system prompt for Phase 1 LangChain agent orchestration."""
    sources_label = ", ".join(active_sources) if active_sources else "None uploaded or enabled yet"
    return NOTEBOOK_SYSTEM_PROMPT.format(active_sources_label=sources_label)


def build_langgraph_answer_prompt() -> ChatPromptTemplate:
    """Return the ChatPromptTemplate used by the Phase 2 LangGraph generator node."""
    return LANGGRAPH_ANSWER_PROMPT


def extract_thinking_and_answer(raw_text: str) -> tuple[str, str]:
    """Separate Qwen/reasoning `<think>...</think>` blocks from the user-facing answer."""
    if not raw_text:
        return "", ""

    thoughts = [match.strip() for match in THINK_BLOCK_PATTERN.findall(raw_text) if match.strip()]
    cleaned_answer = THINK_BLOCK_PATTERN.sub("", raw_text).strip()
    return "\n\n".join(thoughts), cleaned_answer
