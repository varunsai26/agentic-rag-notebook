"""Streamlit entrypoint for Agentic RAG Notebook (Gemini NotebookLM + ChatGPT URL Sessions)."""

from __future__ import annotations

import logging

import streamlit as st

from agentic_rag_notebook.config.logging_config import configure_logging
from agentic_rag_notebook.config.settings import (
    LLMProviderType,
    OrchestrationEngineType,
    Settings,
    get_settings,
)
from agentic_rag_notebook.domain.interfaces import RagOrchestrator
from agentic_rag_notebook.infrastructure.chroma_store import ChromaVectorRepository
from agentic_rag_notebook.infrastructure.csv_loader import LangChainCSVLoader
from agentic_rag_notebook.infrastructure.docling_parser import DoclingDocumentParser
from agentic_rag_notebook.infrastructure.embedding_factory import EmbeddingFactory
from agentic_rag_notebook.infrastructure.llm_factory import LLMFactory, RuntimeLLMOverrides
from agentic_rag_notebook.infrastructure.sqlite_session_store import SQLiteSessionStore
from agentic_rag_notebook.infrastructure.wikipedia_loader import LangChainWikipediaLoader
from agentic_rag_notebook.orchestration.langchain_agent import LangChainRagOrchestrator
from agentic_rag_notebook.orchestration.langgraph_agent import LangGraphRagOrchestrator
from agentic_rag_notebook.services.notebook_service import NotebookService
from agentic_rag_notebook.ui.components import (
    SUGGESTED_PROMPTS,
    inject_notebook_styles,
    render_chat_turn,
    render_citations,
    render_guardrail_audit,
    render_notebook_header,
    render_sidebar_sessions,
    render_sidebar_sources,
)

logger = logging.getLogger("agentic_rag_notebook.ui.app")


@st.cache_resource(show_spinner="Loading local embedding model and ChromaDB...")
def _get_cached_core_resources(
    embedding_provider: str,
    embedding_model: str,
    chroma_dir: str,
    collection_name: str,
    sqlite_path: str,
) -> tuple[ChromaVectorRepository, SQLiteSessionStore, DoclingDocumentParser]:
    """Cache heavyweight embedding model, Chroma store, and Docling parser across reruns."""
    del embedding_provider, embedding_model, chroma_dir, sqlite_path
    settings = get_settings()
    embeddings = EmbeddingFactory(settings).create()
    vector_repo = ChromaVectorRepository(
        persist_directory=settings.chroma_persist_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    session_store = SQLiteSessionStore(settings.sqlite_db_path)
    tokenizer_name = (
        settings.huggingface_embedding_model
        if settings.embedding_provider == "huggingface"
        else None
    )
    parser = DoclingDocumentParser(
        tokenizer_model_name=tokenizer_name,
        max_chunk_tokens=settings.docling_max_chunk_tokens,
    )
    return vector_repo, session_store, parser


def _build_service(
    settings: Settings,
    overrides: RuntimeLLMOverrides | None = None,
) -> NotebookService:
    vector_repo, session_store, parser = _get_cached_core_resources(
        settings.embedding_provider,
        settings.active_embedding_model_name(),
        str(settings.chroma_persist_dir),
        settings.embedding_collection_name(),
        str(settings.sqlite_db_path),
    )
    llm_factory = LLMFactory(settings)
    orchestrators: dict[str, RagOrchestrator] = {
        "langchain": LangChainRagOrchestrator(
            settings=settings,
            vector_repo=vector_repo,
            session_store=session_store,
            llm_factory=llm_factory,
            llm_overrides=overrides,
        ),
        "langgraph": LangGraphRagOrchestrator(
            settings=settings,
            vector_repo=vector_repo,
            session_store=session_store,
            llm_factory=llm_factory,
            llm_overrides=overrides,
        ),
    }
    return NotebookService(
        settings=settings,
        parser=parser,
        csv_loader=LangChainCSVLoader(),
        wiki_loader=LangChainWikipediaLoader(),
        vector_repo=vector_repo,
        session_repo=session_store,
        orchestrators=orchestrators,
    )


def _sync_url_session(service: NotebookService) -> str:
    """Resolve `?session_id=<uuid>` from URL or generate a new ChatGPT-style UUID."""
    raw_sid = st.query_params.get("session_id")
    session_id = str(raw_sid).strip() if raw_sid else ""
    session = service.ensure_session(session_id or None)
    if st.query_params.get("session_id") != session.session_id:
        st.query_params["session_id"] = session.session_id
    return session.session_id


def main() -> None:
    """Run the Streamlit Notebook UI."""
    st.set_page_config(
        page_title="Agentic RAG Notebook",
        page_icon="📓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_notebook_styles()

    settings = get_settings()
    configure_logging(settings)

    # Initialize runtime override state from .env defaults
    if "ui_llm_provider" not in st.session_state:
        st.session_state.ui_llm_provider = settings.llm_provider
    if "ui_engine" not in st.session_state:
        st.session_state.ui_engine = settings.orchestration_engine
    if "ui_local_base_url" not in st.session_state:
        st.session_state.ui_local_base_url = settings.openai_local_base_url
    if "ui_local_api_key" not in st.session_state:
        st.session_state.ui_local_api_key = settings.openai_local_api_key
    if "ui_local_model" not in st.session_state:
        st.session_state.ui_local_model = settings.openai_local_model
    if "ui_groq_model" not in st.session_state:
        st.session_state.ui_groq_model = settings.groq_model
    if "ui_groq_api_key" not in st.session_state:
        st.session_state.ui_groq_api_key = settings.groq_api_key
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    selected_provider: LLMProviderType = st.session_state.ui_llm_provider

    llm_overrides = RuntimeLLMOverrides(
        provider=selected_provider,
        model=(
            st.session_state.ui_groq_model
            if selected_provider == "groq"
            else st.session_state.ui_local_model
        ),
        base_url=(None if selected_provider == "groq" else st.session_state.ui_local_base_url),
        api_key=(
            st.session_state.ui_groq_api_key
            if selected_provider == "groq"
            else st.session_state.ui_local_api_key
        ),
    )

    service = _build_service(settings, llm_overrides)
    current_session_id = _sync_url_session(service)
    current_session = service.ensure_session(current_session_id)

    # --- SIDEBAR: Sessions, Multi-Format Sources, and Environment Model Config ---
    with st.sidebar:
        st.title("📓 Agentic RAG Notebook")
        st.caption("Docling · CSVLoader · WikipediaLoader · Groq & Local Qwen")

        sessions = service.list_sessions()
        create_new, switch_sid, delete_sid = render_sidebar_sessions(
            sessions=sessions,
            current_session_id=current_session_id,
        )
        if create_new:
            new_sess = service.create_new_session()
            st.query_params["session_id"] = new_sess.session_id
            st.rerun()
        if switch_sid and switch_sid != current_session_id:
            st.query_params["session_id"] = switch_sid
            st.rerun()
        if delete_sid:
            service.delete_session(delete_sid)
            if delete_sid == current_session_id:
                remaining = service.list_sessions()
                next_sess = remaining[0] if remaining else service.create_new_session()
                st.query_params["session_id"] = next_sess.session_id
            st.rerun()

        st.divider()

        # Sources panel (strictly scoped to current_session_id)
        docs = service.list_documents(current_session_id)
        toggled_docs, delete_doc_name = render_sidebar_sources(
            docs,
            session_id=current_session_id,
        )
        for fname, is_active in toggled_docs:
            service.set_document_active(fname, is_active, session_id=current_session_id)
            st.rerun()
        if delete_doc_name:
            service.delete_document(delete_doc_name, session_id=current_session_id)
            st.rerun()

        uploaded_files = st.file_uploader(
            "Upload PDF or CSV documents",
            type=["pdf", "csv"],
            accept_multiple_files=True,
            help=(
                "PDFs use Docling HybridChunker; CSVs use LangChain CSVLoader with "
                "session_id, filename, row, and scalar metadata."
            ),
        )
        use_ocr = st.checkbox(
            "Enable Docling OCR (for scanned PDFs)",
            value=settings.docling_default_ocr,
        )
        with st.expander("CSV Loader Metadata Options", expanded=False):
            csv_source_col = st.text_input(
                "CSV source_column (optional)",
                value="",
                key="ui_csv_source_col",
                help="Column whose value labels each CSV row in vector metadata.",
            )
            csv_meta_cols_raw = st.text_input(
                "CSV metadata_columns (comma-separated)",
                value="",
                key="ui_csv_meta_cols",
                help="Additional CSV columns stored as scalar metadata (`csv_<col>`) in Chroma.",
            )

        if uploaded_files and st.button("⚡ Index Uploaded Files", use_container_width=True):
            meta_cols = tuple(c.strip() for c in csv_meta_cols_raw.split(",") if c.strip())
            indexed_any = False
            for uploaded in uploaded_files:
                try:
                    if uploaded.name.lower().endswith(".csv"):
                        with st.spinner(f"Loading `{uploaded.name}` via CSVLoader..."):
                            record = service.ingest_csv_bytes(
                                filename=uploaded.name,
                                content=uploaded.getvalue(),
                                session_id=current_session_id,
                                source_column=csv_source_col or None,
                                metadata_columns=meta_cols,
                            )
                        st.success(f"Indexed CSV `{record.filename}` ({record.chunk_count} rows)")
                    else:
                        with st.spinner(f"Parsing `{uploaded.name}` with Docling..."):
                            record = service.ingest_pdf_bytes(
                                filename=uploaded.name,
                                content=uploaded.getvalue(),
                                session_id=current_session_id,
                                use_ocr=use_ocr,
                            )
                        st.success(f"Indexed PDF `{record.filename}` ({record.chunk_count} chunks)")
                    indexed_any = True
                except Exception as exc:
                    logger.exception(
                        "Failed to index uploaded file filename=%s session_id=%s",
                        uploaded.name,
                        current_session_id,
                    )
                    st.error(f"Failed to index `{uploaded.name}`: {exc}")
            if indexed_any:
                st.rerun()

        st.markdown("**🌐 Wikipedia Loader**")
        wiki_query = st.text_input(
            "Wikipedia topic or article URL",
            placeholder="e.g. Retrieval-augmented generation or https://en.wikipedia.org/wiki/...",
            key="ui_wiki_query",
        )
        wiki_max_docs = int(
            st.number_input(
                "Max Wikipedia articles",
                min_value=1,
                max_value=5,
                value=1,
                step=1,
                key="ui_wiki_max_docs",
            )
        )
        if st.button("🌐 Load Wikipedia", use_container_width=True):
            if not wiki_query.strip():
                st.warning("Enter a Wikipedia topic or article URL first.")
            else:
                try:
                    with st.spinner(f"Fetching Wikipedia `{wiki_query.strip()}`..."):
                        wiki_records = service.ingest_wikipedia(
                            query_or_url=wiki_query,
                            session_id=current_session_id,
                            load_max_docs=wiki_max_docs,
                        )
                    if not wiki_records:
                        st.warning("No Wikipedia articles matched that query or URL.")
                    else:
                        total_chunks = sum(r.chunk_count for r in wiki_records)
                        st.success(
                            f"Indexed {len(wiki_records)} Wikipedia article(s) "
                            f"({total_chunks} chunks)"
                        )
                        st.rerun()
                except Exception as exc:
                    logger.exception(
                        "Wikipedia load failed query=%r session_id=%s",
                        wiki_query,
                        current_session_id,
                    )
                    st.error(f"Wikipedia load failed: {exc}")

        st.divider()

        with st.expander("⚙️ Client & Engine Settings (.env)", expanded=False):
            provider_options: list[LLMProviderType] = ["groq", "openai_local"]
            st.selectbox(
                "LLM Client (`LLM_PROVIDER`)",
                options=provider_options,
                key="ui_llm_provider",
                format_func=lambda p: (
                    "Groq Cloud (ChatGroq)" if p == "groq" else "Local OpenAI / Qwen (base_url)"
                ),
            )
            engine_options: list[OrchestrationEngineType] = ["langchain", "langgraph"]
            st.selectbox(
                "Orchestration Engine",
                options=engine_options,
                key="ui_engine",
                format_func=lambda e: (
                    "Phase 1: LangChain (create_agent + Middleware)"
                    if e == "langchain"
                    else "Phase 2: LangGraph (StateGraph + SqliteSaver)"
                ),
            )

            if st.session_state.ui_llm_provider == "groq":
                st.text_input("Groq Model", key="ui_groq_model")
                st.text_input("Groq API Key", key="ui_groq_api_key", type="password")
            else:
                st.text_input("Local OpenAI Base URL", key="ui_local_base_url")
                st.text_input("Local OpenAI API Key", key="ui_local_api_key", type="password")
                st.text_input("Local Qwen Model", key="ui_local_model")

            st.caption(
                f"Embeddings: `{settings.embedding_provider}` / "
                f"`{settings.active_embedding_model_name()}`"
            )

    # --- MAIN NOTEBOOK WORKSPACE ---
    active_sources = service.active_source_names(current_session_id)
    active_model_display = (
        f"groq/{st.session_state.ui_groq_model}"
        if st.session_state.ui_llm_provider == "groq"
        else f"openai_local/{st.session_state.ui_local_model}"
    )

    render_notebook_header(
        session=current_session,
        active_sources=active_sources,
        engine=st.session_state.ui_engine,
        model_label=active_model_display,
    )

    # Non-blocking credential check notice if running with default placeholder .env
    if st.session_state.ui_llm_provider == "groq" and (
        not st.session_state.ui_groq_api_key
        or st.session_state.ui_groq_api_key.startswith("gsk_your_")
    ):
        st.warning(
            "Configure `GROQ_API_KEY` in `.env` (or in the sidebar settings) — "
            "or switch to **Local OpenAI / Qwen** (`LLM_PROVIDER=openai_local`)."
        )

    turns = service.get_session_turns(current_session_id)

    if not turns:
        st.info(
            "Upload PDFs (parsed via **Docling**), upload CSV datasets (**CSVLoader**), or load "
            "Wikipedia articles (**WikipediaLoader**) in the left sidebar. Every source is tagged "
            "and strictly isolated to this URL `?session_id=` inside ChromaDB and SQLite."
        )
        cols = st.columns(len(SUGGESTED_PROMPTS))
        for idx, suggestion in enumerate(SUGGESTED_PROMPTS):
            with cols[idx]:
                if st.button(suggestion, key=f"suggest_{idx}", use_container_width=True):
                    st.session_state.pending_prompt = suggestion
                    st.rerun()
    else:
        for turn in turns:
            render_chat_turn(turn)

    user_input = st.chat_input("Ask a question across your PDF, CSV, and Wikipedia sources...")
    prompt_to_run = user_input or st.session_state.pending_prompt
    if st.session_state.pending_prompt is not None and not user_input:
        st.session_state.pending_prompt = None

    if prompt_to_run:
        with st.chat_message("user"):
            st.markdown(prompt_to_run)

        with (
            st.chat_message("assistant"),
            st.spinner("Running guardrails, session-scoped retrieval, and agent reasoning..."),
        ):
            try:
                response = service.ask(
                    session_id=current_session_id,
                    query=prompt_to_run,
                    engine=st.session_state.ui_engine,
                )
                if response.reasoning:
                    with st.expander("💭 Model Reasoning (<think>)", expanded=False):
                        st.markdown(response.reasoning)
                st.markdown(response.answer)
                render_citations(response.citations)
                render_guardrail_audit(
                    response.guardrail_report,
                    tool_calls=response.tool_calls,
                    engine=response.engine,
                    model_label=response.model_label,
                )
                st.rerun()
            except Exception as exc:
                logger.exception(
                    "Error executing agentic RAG pipeline session_id=%s engine=%s",
                    current_session_id,
                    st.session_state.ui_engine,
                )
                st.error(f"Error executing agentic RAG pipeline: {exc}")


if __name__ == "__main__":
    main()
