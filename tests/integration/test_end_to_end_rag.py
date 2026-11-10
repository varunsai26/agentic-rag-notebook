"""End-to-end integration test covering NotebookService, Chroma, SQLite URL sessions, and Guardrails."""

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from agentic_rag_notebook.config.settings import Settings
from agentic_rag_notebook.domain.models import ParsedChunk
from agentic_rag_notebook.infrastructure.chroma_store import ChromaVectorRepository
from agentic_rag_notebook.infrastructure.sqlite_session_store import SQLiteSessionStore
from agentic_rag_notebook.orchestration.langchain_agent import LangChainRagOrchestrator
from agentic_rag_notebook.orchestration.langgraph_agent import LangGraphRagOrchestrator
from agentic_rag_notebook.services.notebook_service import NotebookService
from tests.conftest import DeterministicFakeEmbeddings, FakeToolCallingChatModel


class _StubDoclingParser:
    """Stub parser simulating Docling HybridChunker output for deterministic fast CI."""

    def parse(
        self,
        file_path: Path,
        *,
        session_id: str = "default",
        use_ocr: bool = False,
    ) -> list[ParsedChunk]:
        return [
            ParsedChunk(
                text=f"Parsed with ocr={use_ocr}: Qwen runs on local vLLM with API key.",
                source=file_path.name,
                source_type="pdf",
                session_id=session_id,
                filename=file_path.name,
                title=file_path.stem,
                page=1,
                heading="Deployment",
                chunk_index=0,
                metadata={"ocr_enabled": use_ocr},
            )
        ]


@pytest.mark.integration
def test_notebook_service_end_to_end_url_session_flow(tmp_settings: Settings) -> None:
    vector_repo = ChromaVectorRepository(
        persist_directory=tmp_settings.chroma_persist_dir,
        collection_name="integration_test_col",
        embedding_function=DeterministicFakeEmbeddings(),
    )
    session_store = SQLiteSessionStore(tmp_settings.sqlite_db_path)

    fake_llm = FakeToolCallingChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {"query": "Qwen local vLLM"},
                        "id": "call_int_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="<think>Found Deployment section.</think>"
                "Qwen runs on a local vLLM endpoint as documented in [guide.pdf, p.1]."
            ),
        ]
    )

    service = NotebookService(
        settings=tmp_settings,
        parser=_StubDoclingParser(),
        vector_repo=vector_repo,
        session_repo=session_store,
        orchestrators={
            "langchain": LangChainRagOrchestrator(
                settings=tmp_settings,
                vector_repo=vector_repo,
                session_store=session_store,
                model_resolver=lambda: (fake_llm, "openai_local/Qwen2.5-7B-Instruct"),
            ),
            "langgraph": LangGraphRagOrchestrator(
                settings=tmp_settings,
                vector_repo=vector_repo,
                session_store=session_store,
                model_resolver=lambda: (fake_llm, "openai_local/Qwen2.5-7B-Instruct"),
            ),
        },
    )

    # 1. Generate two distinct ChatGPT-style UUID sessions & simulate URL reload
    created = service.ensure_session(None)
    reloaded = service.ensure_session(created.session_id)
    assert reloaded.session_id == created.session_id
    session_b = service.create_new_session()

    # 2. Ingest PDF, real CSVLoader CSV, and mocked WikipediaLoader article into session A
    pdf_record = service.ingest_pdf_bytes(
        filename="guide.pdf",
        content=b"%PDF-1.4 dummy test bytes",
        session_id=created.session_id,
        use_ocr=False,
    )
    assert pdf_record.chunk_count == 1

    csv_record = service.ingest_csv_bytes(
        filename="models.csv",
        content=b"model,endpoint\nQwen2.5-7B,http://localhost:8000/v1\n",
        session_id=created.session_id,
        source_column="model",
        metadata_columns=("endpoint",),
    )
    assert csv_record.source_type == "csv"
    assert csv_record.chunk_count == 1

    wiki_doc = Document(
        page_content="Qwen is a family of large language models developed by Alibaba Cloud.",
        metadata={
            "title": "Qwen",
            "page_url": "https://en.wikipedia.org/wiki/Qwen",
            "categories": ["Large language models"],
            "revision_id": 4242,
        },
    )
    with patch("agentic_rag_notebook.infrastructure.wikipedia_loader.WikipediaLoader") as mock_wiki:
        mock_wiki.return_value.load.return_value = [wiki_doc]
        wiki_records = service.ingest_wikipedia(
            query_or_url="https://en.wikipedia.org/wiki/Qwen",
            session_id=created.session_id,
            load_max_docs=1,
        )

    assert len(wiki_records) == 1
    assert wiki_records[0].url == "https://en.wikipedia.org/wiki/Qwen"
    assert set(service.active_source_names(created.session_id)) == {
        "guide.pdf",
        "models.csv",
        "wiki:Qwen",
    }

    # Session B must see ZERO documents and ZERO vectors from Session A
    assert service.list_documents(session_b.session_id) == []
    assert vector_repo.similarity_search("Qwen", k=5, session_id=session_b.session_id) == []

    # 3. Ask question (Phase 1 LangChain + middleware) in Session A
    reply = service.ask(
        session_id=created.session_id,
        query="How does Qwen run locally? Contact ops@example.com",
        engine="langchain",
    )
    assert reply.blocked is False
    assert reply.guardrail_report.redacted is True
    assert any(
        c.session_id == created.session_id
        for c in vector_repo.similarity_search("Qwen", k=5, session_id=created.session_id)
    )
    assert any(c.source == "guide.pdf" for c in reply.citations)

    # 4. Reload session turns from SQLite as happens on browser refresh
    turns = service.get_session_turns(created.session_id)
    assert len(turns) == 2

    # 5. Switch to Phase 2 LangGraph on same URL session
    reply_lg = service.ask(
        session_id=created.session_id,
        query="Confirm local Qwen deployment details",
        engine="langgraph",
    )
    assert reply_lg.engine == "langgraph"
    assert len(service.get_session_turns(created.session_id)) == 4

    # 6. Delete single CSV source in Session A, then delete entire Session A cleanly
    service.delete_document("models.csv", session_id=created.session_id)
    assert "models.csv" not in service.active_source_names(created.session_id)

    service.delete_session(created.session_id)
    assert service.list_documents(created.session_id) == []
    assert vector_repo.similarity_search("Qwen", k=5, session_id=created.session_id) == []
    assert not (tmp_settings.upload_dir / created.session_id).exists()
