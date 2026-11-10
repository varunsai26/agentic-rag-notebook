"""Unit tests for SQLiteSessionStore, URL session persistence, and LangChain history."""

import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agentic_rag_notebook.domain.models import (
    ChatTurn,
    Citation,
    DocumentRecord,
    GuardrailReport,
    GuardrailStepResult,
)
from agentic_rag_notebook.infrastructure.sqlite_session_store import SQLiteSessionStore


def test_url_session_persistence_and_title_derivation(session_store: SQLiteSessionStore) -> None:
    session = session_store.get_or_create_session("chat-uuid-101")
    assert session.session_id == "chat-uuid-101"
    assert session.title == "New Notebook Chat"

    session_store.add_turn(
        ChatTurn(
            session_id=session.session_id,
            role="user",
            content="Explain how Docling HybridChunker handles tables in PDFs",
        )
    )
    session_store.add_turn(
        ChatTurn(
            session_id=session.session_id,
            role="assistant",
            content="It preserves headers across split tables.",
            reasoning="Checked page 4 table section.",
            citations=[
                Citation(
                    index=1,
                    source="docling.pdf",
                    source_type="pdf",
                    page=4,
                    heading="Tables",
                    excerpt="HybridChunker repeats table headers.",
                )
            ],
            guardrail_report=GuardrailReport(
                steps=[
                    GuardrailStepResult(
                        guard_name="groundedness_citation_guard",
                        stage="output",
                        passed=True,
                        action="allow",
                        reason="Grounded",
                    )
                ]
            ),
            engine="langchain",
            model_label="groq/llama-3.3-70b-versatile",
        )
    )

    reloaded = session_store.get_session("chat-uuid-101")
    assert reloaded is not None
    assert reloaded.title.startswith("Explain how Docling")

    turns = session_store.get_turns("chat-uuid-101")
    assert len(turns) == 2
    assert turns[1].reasoning == "Checked page 4 table section."
    assert turns[1].citations[0].source == "docling.pdf"
    assert turns[1].guardrail_report is not None


def test_langchain_history_and_session_deletion(session_store: SQLiteSessionStore) -> None:
    sid = "chat-uuid-202"
    session_store.get_or_create_session(sid)
    lc_history = session_store.get_langchain_history(sid)
    lc_history.add_messages([HumanMessage(content="Hi"), AIMessage(content="Hello!")])

    reloaded_history = session_store.get_langchain_history(sid)
    assert len(reloaded_history.messages) == 2
    assert reloaded_history.messages[0].content == "Hi"

    session_store.register_document(
        DocumentRecord(
            session_id=sid,
            filename="wiki:Python",
            source_type="wikipedia",
            url="https://en.wikipedia.org/wiki/Python",
            title="Python",
            file_hash="hash-wiki",
            chunk_count=3,
        )
    )
    session_store.register_document(
        DocumentRecord(
            session_id="other-session",
            filename="wiki:Python",
            source_type="wikipedia",
            url="https://en.wikipedia.org/wiki/Python",
            title="Python",
            file_hash="hash-wiki",
            chunk_count=3,
        )
    )

    session_store.delete_session(sid)
    assert session_store.get_session(sid) is None
    assert session_store.get_langchain_history(sid).messages == []
    assert session_store.list_documents(sid) == []
    assert len(session_store.list_documents("other-session")) == 1


def test_document_registry_crud_is_session_scoped(session_store: SQLiteSessionStore) -> None:
    rec_a = DocumentRecord(
        session_id="sess-a",
        filename="paper.pdf",
        source_type="pdf",
        title="paper",
        file_hash="abc123",
        chunk_count=12,
        ocr_enabled=True,
        active=True,
    )
    rec_b = DocumentRecord(
        session_id="sess-b",
        filename="paper.pdf",
        source_type="csv",
        title="paper",
        file_hash="def456",
        chunk_count=5,
        ocr_enabled=False,
        active=True,
    )
    session_store.register_document(rec_a)
    session_store.register_document(rec_b)

    docs_a = session_store.list_documents("sess-a")
    docs_b = session_store.list_documents("sess-b")
    assert len(docs_a) == 1
    assert len(docs_b) == 1
    assert docs_a[0].ocr_enabled is True
    assert docs_b[0].source_type == "csv"

    session_store.set_document_active("paper.pdf", False, session_id="sess-a")
    assert session_store.list_documents("sess-a")[0].active is False
    assert session_store.list_documents("sess-b")[0].active is True

    session_store.delete_document_record("paper.pdf", session_id="sess-a")
    assert session_store.list_documents("sess-a") == []
    assert len(session_store.list_documents("sess-b")) == 1


def test_legacy_documents_table_migrates_to_per_session_schema(tmp_path: Path) -> None:
    legacy_db = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy_db)
    conn.executescript(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            filename TEXT UNIQUE NOT NULL,
            file_hash TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            ocr_enabled INTEGER NOT NULL DEFAULT 0,
            uploaded_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        );
        INSERT INTO documents (doc_id, filename, file_hash, chunk_count, ocr_enabled, uploaded_at, active)
        VALUES ('d1', 'old.pdf', 'h1', 4, 0, '2026-01-01T00:00:00Z', 1);
        """
    )
    conn.commit()
    conn.close()

    store = SQLiteSessionStore(legacy_db)
    migrated = store.list_documents("default")
    assert len(migrated) == 1
    assert migrated[0].filename == "old.pdf"
    assert migrated[0].source_type == "pdf"
    assert migrated[0].session_id == "default"
