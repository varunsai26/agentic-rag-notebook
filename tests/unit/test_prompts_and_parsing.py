"""Unit tests for PromptTemplates, Qwen <think> parsing, and Docling metadata extraction."""

from agentic_rag_notebook.infrastructure.docling_parser import _extract_page_and_heading
from agentic_rag_notebook.prompts.templates import (
    build_langchain_rag_prompt,
    build_langgraph_answer_prompt,
    extract_thinking_and_answer,
)


def test_extract_thinking_and_answer_for_qwen_blocks() -> None:
    raw = "<think>\nFirst check source A.\nSecond check table B.\n</think>\nFinal grounded answer."
    reasoning, answer = extract_thinking_and_answer(raw)
    assert "First check source A." in reasoning
    assert answer == "Final grounded answer."


def test_prompt_templates_render_active_sources() -> None:
    system_text = build_langchain_rag_prompt(["paper.pdf", "notes.pdf"])
    assert "paper.pdf, notes.pdf" in system_text

    prompt = build_langgraph_answer_prompt()
    formatted = prompt.format_messages(
        active_sources_label="paper.pdf",
        history=[],
        question="What is X?",
        context="[1] X is Y.",
    )
    assert len(formatted) == 2
    assert "paper.pdf" in str(formatted[0].content)


def test_docling_metadata_extraction() -> None:
    dl_meta = {
        "headings": ["Chapter 1", "Overview"],
        "doc_items": [
            {"prov": [{"page_no": 4}]},
        ],
    }
    page, heading = _extract_page_and_heading(dl_meta)
    assert page == 4
    assert heading == "Chapter 1 > Overview"
