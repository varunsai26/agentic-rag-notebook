"""Modular Streamlit UI components styled after Gemini / NotebookLM."""

from __future__ import annotations

import html

import streamlit as st

from agentic_rag_notebook.domain.models import (
    ChatTurn,
    Citation,
    ConversationSession,
    DocumentRecord,
    GuardrailReport,
)

NOTEBOOK_CSS = """
<style>
.notebook-header {
    padding: 0.75rem 1rem;
    border-radius: 14px;
    border: 1px solid rgba(128, 128, 128, 0.22);
    background: linear-gradient(135deg, rgba(66, 133, 244, 0.08), rgba(155, 114, 203, 0.08));
    margin-bottom: 1rem;
}
.notebook-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.45rem;
}
.nb-pill {
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    border: 1px solid rgba(128, 128, 128, 0.28);
    background-color: rgba(128, 128, 128, 0.08);
}
.citation-card {
    padding: 0.55rem 0.75rem;
    border-radius: 10px;
    border-left: 3px solid #4285F4;
    background-color: rgba(66, 133, 244, 0.06);
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
}
</style>
"""

SUGGESTED_PROMPTS: tuple[str, ...] = (
    "Summarize the core ideas and key takeaways across my active sources.",
    "Extract key metrics, tables, or structured facts with source citations.",
    "Cross-reference concepts across my PDFs, CSV datasets, and Wikipedia articles.",
)


def inject_notebook_styles() -> None:
    """Inject subtle Gemini/NotebookLM custom CSS."""
    st.markdown(NOTEBOOK_CSS, unsafe_allow_html=True)


def render_notebook_header(
    *,
    session: ConversationSession,
    active_sources: list[str],
    engine: str,
    model_label: str,
) -> None:
    """Render the top Notebook header bar with URL session ID and active runtime pills."""
    safe_title = html.escape(session.title)
    short_sid = html.escape(session.session_id[:8])
    safe_engine = html.escape(engine)
    safe_model = html.escape(model_label)
    sources_count = len(active_sources)
    st.markdown(
        f"""
        <div class="notebook-header">
            <div style="font-size: 1.15rem; font-weight: 600;">📓 {safe_title}</div>
            <div class="notebook-badges">
                <span class="nb-pill">🔗 session: {short_sid}…</span>
                <span class="nb-pill">📚 active sources: {sources_count}</span>
                <span class="nb-pill">⚙️ engine: {safe_engine}</span>
                <span class="nb-pill">🤖 model: {safe_model}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_citations(citations: list[Citation]) -> None:
    """Render expandable citation cards with source type, page/row, URL, heading, and excerpt."""
    if not citations:
        return
    with st.expander(f"📚 Sources & Citations ({len(citations)})", expanded=False):
        for cit in citations:
            icon = {"pdf": "📄", "csv": "📊", "wikipedia": "🌐"}.get(cit.source_type, "📄")
            display_name = cit.title if cit.source_type == "wikipedia" and cit.title else cit.source
            if cit.page is not None:
                loc_txt = f"PDF · Page {cit.page}"
            elif cit.row is not None:
                loc_txt = f"CSV · Row {cit.row + 1}"
            elif cit.source_type == "wikipedia":
                loc_txt = "Wikipedia"
            else:
                loc_txt = cit.source_type.upper()

            heading_txt = (
                f" · {html.escape(cit.heading)}"
                if cit.heading and cit.heading != display_name
                else ""
            )
            url_html = ""
            if cit.url and cit.url.startswith(("http://", "https://")):
                safe_url = html.escape(cit.url, quote=True)
                url_html = (
                    f' · <a href="{safe_url}" target="_blank" '
                    f'rel="noopener noreferrer">{safe_url}</a>'
                )

            safe_src = html.escape(display_name)
            safe_excerpt = html.escape(cit.excerpt)
            st.markdown(
                f"""
                <div class="citation-card">
                    <strong>[{cit.index}] {icon} {safe_src}</strong> ({loc_txt}{heading_txt}){url_html}<br/>
                    <span style="opacity: 0.85;">{safe_excerpt}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_guardrail_audit(
    report: GuardrailReport | None,
    *,
    tool_calls: list[str] | None = None,
    engine: str = "",
    model_label: str = "",
) -> None:
    """Render middleware guardrail audit steps and agent tool trace."""
    if report is None and not tool_calls:
        return

    status_icon = "🛡️"
    status_label = "Guardrails Passed"
    if report is not None:
        if report.blocked:
            status_icon = "⛔"
            status_label = "Blocked by Guardrail"
        elif report.redacted:
            status_icon = "🧹"
            status_label = "Passed (PII/Secret Redacted)"
        elif report.warnings:
            status_icon = "⚠️"
            status_label = f"Passed with {len(report.warnings)} Notice(s)"

    with st.expander(f"{status_icon} Middleware & Trace · {status_label}", expanded=False):
        if engine or model_label:
            st.caption(f"Engine: `{engine}` | Model: `{model_label}`")
        if tool_calls:
            st.markdown("**Agent Tool Calls:**")
            for call in tool_calls:
                st.code(call, language="text")
        if report is not None and report.steps:
            st.markdown("**Guardrail Pipeline Steps:**")
            for step in report.steps:
                badge = "✅" if step.passed else "❌"
                st.markdown(
                    f"- {badge} `{step.stage.upper()}` **{step.guard_name}** "
                    f"(`{step.action}`): {step.reason}"
                )


def render_chat_turn(turn: ChatTurn) -> None:
    """Render a persisted user or assistant turn cleanly."""
    with st.chat_message(turn.role):
        if turn.role == "assistant" and turn.reasoning:
            with st.expander("💭 Model Reasoning (<think>)", expanded=False):
                st.markdown(turn.reasoning)
        st.markdown(turn.content)
        if turn.role == "assistant":
            render_citations(turn.citations)
            render_guardrail_audit(
                turn.guardrail_report,
                engine=turn.engine,
                model_label=turn.model_label,
            )


def render_sidebar_sessions(
    *,
    sessions: list[ConversationSession],
    current_session_id: str,
) -> tuple[bool, str | None, str | None]:
    """Render ChatGPT-style conversation list in sidebar.

    Returns `(create_new, select_session_id, delete_session_id)`.
    """
    st.subheader("💬 Notebook Chats")
    create_new = st.button("+ New Notebook Chat", use_container_width=True, type="primary")
    selected_id: str | None = None
    delete_id: str | None = None

    for sess in sessions[:20]:
        col_btn, col_del = st.columns([5, 1])
        is_current = sess.session_id == current_session_id
        prefix = "▸ " if is_current else ""
        label = f"{prefix}{sess.title}"
        with col_btn:
            if st.button(
                label,
                key=f"sess_select_{sess.session_id}",
                use_container_width=True,
                type="secondary" if not is_current else "primary",
            ):
                selected_id = sess.session_id
        with col_del:
            if st.button("🗑", key=f"sess_del_{sess.session_id}", help="Delete conversation"):
                delete_id = sess.session_id

    return create_new, selected_id, delete_id


def render_sidebar_sources(
    documents: list[DocumentRecord],
    *,
    session_id: str = "default",
) -> tuple[list[tuple[str, bool]], str | None]:
    """Render session-isolated NotebookLM source toggles (PDF, CSV, Wikipedia) & delete controls.

    Returns `(toggled_states, delete_filename)`.
    """
    st.subheader("📚 Session Sources")
    toggled: list[tuple[str, bool]] = []
    delete_filename: str | None = None

    if not documents:
        st.caption(
            "No sources indexed in this session yet. Upload PDFs/CSVs or load Wikipedia below."
        )
        return toggled, delete_filename

    for doc in documents:
        col_chk, col_del = st.columns([5, 1])
        with col_chk:
            if doc.source_type == "csv":
                label = f"📊 {doc.filename} ({doc.chunk_count} rows)"
            elif doc.source_type == "wikipedia":
                wiki_title = doc.title or doc.filename
                label = f"🌐 {wiki_title} ({doc.chunk_count} chunks)"
            else:
                ocr_badge = " · OCR" if doc.ocr_enabled else ""
                label = f"📄 {doc.filename} ({doc.chunk_count} chunks{ocr_badge})"

            checked = st.checkbox(
                label,
                value=doc.active,
                key=f"doc_active_{session_id}_{doc.filename}",
            )
            if checked != doc.active:
                toggled.append((doc.filename, checked))
            if doc.source_type == "wikipedia" and doc.url:
                st.caption(f"🔗 [{doc.url}]({doc.url})")
        with col_del:
            if st.button(
                "✕",
                key=f"doc_del_{session_id}_{doc.filename}",
                help=f"Remove {doc.filename}",
            ):
                delete_filename = doc.filename

    return toggled, delete_filename
