"""LangChain RAG tools and shared formatting helpers."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, tool

from agentic_rag_notebook.domain.interfaces import VectorRepository
from agentic_rag_notebook.domain.models import Citation, RetrievedChunk
from agentic_rag_notebook.middleware.langchain_adapter import GuardrailRunContext


def coerce_message_text(content: Any) -> str:
    """Safely normalize string or multimodal block lists into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks with explicit source type, page/row, URL, and heading labels."""
    if not chunks:
        return "No matching excerpts were found in the active session sources."
    formatted: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta_parts: list[str] = [chunk.source_type.upper()]
        if chunk.page is not None:
            meta_parts.append(f"p.{chunk.page}")
        if chunk.row is not None:
            meta_parts.append(f"row {chunk.row + 1}")
        if chunk.heading:
            meta_parts.append(f"section: {chunk.heading}")
        if chunk.url:
            meta_parts.append(f"url: {chunk.url}")
        meta_str = " | ".join(meta_parts)
        formatted.append(f"[{idx}] Source: {chunk.source} ({meta_str})\n{chunk.content}")
    return "\n\n---\n\n".join(formatted)


def chunks_to_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Deduplicate retrieved chunks into numbered user-facing Citations."""
    citations: list[Citation] = []
    seen: set[tuple[str, int | None, int | None, str]] = set()
    for chunk in chunks:
        fingerprint = (chunk.source, chunk.page, chunk.row, chunk.content[:80])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        excerpt = " ".join(chunk.content.split())
        if len(excerpt) > 280:
            excerpt = f"{excerpt[:277]}..."
        citations.append(
            Citation(
                index=len(citations) + 1,
                source=chunk.source,
                source_type=chunk.source_type,
                url=chunk.url,
                title=chunk.title,
                page=chunk.page,
                row=chunk.row,
                heading=chunk.heading,
                excerpt=excerpt,
            )
        )
    return citations


def build_rag_tools(
    *,
    vector_repo: VectorRepository,
    session_id: str,
    active_sources: list[str] | None,
    context: GuardrailRunContext,
    top_k: int = 4,
) -> list[BaseTool]:
    """Build session-isolated RAG tools bound to active sources and trace context."""
    effective_sources = list(active_sources) if active_sources is not None else []

    @tool("search_knowledge_base")
    def search_knowledge_base(query: str) -> str:
        """Search the active uploaded PDFs, CSVs, and Wikipedia sources in this session."""
        if active_sources is not None and not effective_sources:
            return "No active knowledge sources are selected in this session."
        chunks = vector_repo.similarity_search(
            query,
            k=top_k,
            session_id=session_id,
            active_sources=active_sources,
        )
        context.retrieved_chunks.extend(chunks)
        return format_chunks_for_prompt(chunks)

    @tool("list_active_sources")
    def list_active_sources() -> str:
        """List the source identifiers currently active in this notebook session."""
        if not effective_sources:
            return "No knowledge sources are currently active in this session."
        return "Active sources in this session:\n" + "\n".join(
            f"- {name}" for name in effective_sources
        )

    return [search_knowledge_base, list_active_sources]
