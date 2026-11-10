"""SQLite persistence for ChatGPT-style URL sessions, chat turns, per-session sources, and LangChain history."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)

from agentic_rag_notebook.domain.models import (
    ChatTurn,
    Citation,
    ConversationSession,
    DocumentRecord,
    GuardrailReport,
    SourceType,
    utc_now_iso,
)


def _derive_session_title(first_message: str, max_len: int = 52) -> str:
    """Create a concise ChatGPT-like session title from the first prompt."""
    collapsed = " ".join(first_message.strip().split())
    if not collapsed:
        return "New Notebook Chat"
    if len(collapsed) <= max_len:
        return collapsed
    return f"{collapsed[: max_len - 3].rstrip()}..."


def _coerce_source_type(raw: str) -> SourceType:
    if raw in {"pdf", "csv", "wikipedia"}:
        return raw  # type: ignore[return-value]
    return "pdf"


class SQLiteSessionStore:
    """Thread-safe SQLite repository implementing SessionRepository with per-session sources."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def db_path(self) -> Path:
        """Return resolved path to the SQLite database file."""
        return self._db_path

    @property
    def connection_string(self) -> str:
        """Return SQLAlchemy-style connection URL for reference."""
        return f"sqlite:///{self._db_path.resolve()}"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    active_sources_json TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS chat_turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    reasoning TEXT NOT NULL DEFAULT '',
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    guardrail_report_json TEXT,
                    engine TEXT NOT NULL DEFAULT '',
                    model_label TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chat_turns_session_created
                    ON chat_turns(session_id, created_at);

                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL DEFAULT 'default',
                    filename TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'pdf',
                    url TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    file_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    ocr_enabled INTEGER NOT NULL DEFAULT 0,
                    uploaded_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    UNIQUE(session_id, filename)
                );

                CREATE TABLE IF NOT EXISTS langchain_message_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_lc_msg_session
                    ON langchain_message_store(session_id, id);
                """
            )
            self._migrate_documents_if_needed(conn)

    def _migrate_documents_if_needed(self, conn: sqlite3.Connection) -> None:
        """Upgrade legacy single-tenant `documents` table to per-session schema if detected."""
        cols = {str(row["name"]) for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "session_id" in cols and "source_type" in cols and "url" in cols and "title" in cols:
            return

        conn.executescript(
            """
            ALTER TABLE documents RENAME TO documents_legacy;
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL DEFAULT 'default',
                filename TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'pdf',
                url TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                file_hash TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                ocr_enabled INTEGER NOT NULL DEFAULT 0,
                uploaded_at TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                UNIQUE(session_id, filename)
            );
            INSERT INTO documents (
                doc_id, session_id, filename, source_type, url, title,
                file_hash, chunk_count, ocr_enabled, uploaded_at, active
            )
            SELECT
                doc_id, 'default', filename, 'pdf', '', filename,
                file_hash, chunk_count, ocr_enabled, uploaded_at, active
            FROM documents_legacy;
            DROP TABLE documents_legacy;
            """
        )

    def create_session(
        self,
        *,
        session_id: str | None = None,
        title: str = "New Notebook Chat",
        active_sources: list[str] | None = None,
    ) -> ConversationSession:
        sid = session_id or str(uuid4())
        now = utc_now_iso()
        sources = (
            active_sources
            if active_sources is not None
            else [doc.filename for doc in self.list_documents(sid) if doc.active]
        )
        session = ConversationSession(
            session_id=sid,
            title=title,
            created_at=now,
            updated_at=now,
            active_sources=sources,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, title, created_at, updated_at, active_sources_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.title,
                    session.created_at,
                    session.updated_at,
                    json.dumps(session.active_sources),
                ),
            )
        return session

    def get_session(self, session_id: str) -> ConversationSession | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return ConversationSession(
            session_id=str(row["session_id"]),
            title=str(row["title"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            active_sources=list(json.loads(str(row["active_sources_json"]))),
        )

    def get_or_create_session(self, session_id: str) -> ConversationSession:
        existing = self.get_session(session_id)
        if existing is not None:
            return existing
        return self.create_session(session_id=session_id)

    def list_sessions(self) -> list[ConversationSession]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
        return [
            ConversationSession(
                session_id=str(row["session_id"]),
                title=str(row["title"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                active_sources=list(json.loads(str(row["active_sources_json"]))),
            )
            for row in rows
        ]

    def update_session_title(self, session_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title, utc_now_iso(), session_id),
            )

    def update_active_sources(self, session_id: str, active_sources: list[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET active_sources_json = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(active_sources), utc_now_iso(), session_id),
            )

    def delete_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))
            has_lc_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='langchain_message_store'"
            ).fetchone()
            if has_lc_table is not None:
                conn.execute(
                    "DELETE FROM langchain_message_store WHERE session_id = ?",
                    (session_id,),
                )

    def add_turn(self, turn: ChatTurn) -> None:
        session = self.get_or_create_session(turn.session_id)
        citations_json = json.dumps([c.model_dump() for c in turn.citations])
        report_json = (
            turn.guardrail_report.model_dump_json() if turn.guardrail_report is not None else None
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_turns (
                    turn_id, session_id, role, content, reasoning,
                    citations_json, guardrail_report_json, engine, model_label, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.turn_id,
                    turn.session_id,
                    turn.role,
                    turn.content,
                    turn.reasoning,
                    citations_json,
                    report_json,
                    turn.engine,
                    turn.model_label,
                    turn.created_at,
                ),
            )
            new_title = session.title
            if turn.role == "user" and session.title == "New Notebook Chat":
                new_title = _derive_session_title(turn.content)
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (new_title, utc_now_iso(), turn.session_id),
            )

    def get_turns(self, session_id: str) -> list[ChatTurn]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_turns WHERE session_id = ? ORDER BY created_at ASC, rowid ASC",
                (session_id,),
            ).fetchall()

        turns: list[ChatTurn] = []
        for row in rows:
            raw_citations = json.loads(str(row["citations_json"]))
            citations = [Citation.model_validate(item) for item in raw_citations]
            raw_report = row["guardrail_report_json"]
            report = GuardrailReport.model_validate_json(str(raw_report)) if raw_report else None
            turns.append(
                ChatTurn(
                    turn_id=str(row["turn_id"]),
                    session_id=str(row["session_id"]),
                    role="user" if row["role"] == "user" else "assistant",
                    content=str(row["content"]),
                    reasoning=str(row["reasoning"]),
                    citations=citations,
                    guardrail_report=report,
                    engine=str(row["engine"]),
                    model_label=str(row["model_label"]),
                    created_at=str(row["created_at"]),
                )
            )
        return turns

    def register_document(self, record: DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, session_id, filename, source_type, url, title,
                    file_hash, chunk_count, ocr_enabled, uploaded_at, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, filename) DO UPDATE SET
                    source_type = excluded.source_type,
                    url = excluded.url,
                    title = excluded.title,
                    file_hash = excluded.file_hash,
                    chunk_count = excluded.chunk_count,
                    ocr_enabled = excluded.ocr_enabled,
                    uploaded_at = excluded.uploaded_at,
                    active = excluded.active
                """,
                (
                    record.doc_id,
                    record.session_id,
                    record.filename,
                    record.source_type,
                    record.url,
                    record.title,
                    record.file_hash,
                    record.chunk_count,
                    int(record.ocr_enabled),
                    record.uploaded_at,
                    int(record.active),
                ),
            )

    def list_documents(self, session_id: str | None = None) -> list[DocumentRecord]:
        with self._connect() as conn:
            if session_id is not None:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE session_id = ? ORDER BY uploaded_at DESC",
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        return [
            DocumentRecord(
                doc_id=str(row["doc_id"]),
                session_id=str(row["session_id"]),
                filename=str(row["filename"]),
                source_type=_coerce_source_type(str(row["source_type"])),
                url=str(row["url"]),
                title=str(row["title"]),
                file_hash=str(row["file_hash"]),
                chunk_count=int(row["chunk_count"]),
                ocr_enabled=bool(row["ocr_enabled"]),
                uploaded_at=str(row["uploaded_at"]),
                active=bool(row["active"]),
            )
            for row in rows
        ]

    def set_document_active(
        self,
        filename: str,
        active: bool,
        *,
        session_id: str = "default",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE documents SET active = ? WHERE session_id = ? AND filename = ?",
                (int(active), session_id, filename),
            )

    def delete_document_record(
        self,
        filename: str,
        *,
        session_id: str = "default",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM documents WHERE session_id = ? AND filename = ?",
                (session_id, filename),
            )

    def get_langchain_history(self, session_id: str) -> SQLiteLangChainChatHistory:
        """Return a LangChain BaseChatMessageHistory bound to this SQLite store."""
        return SQLiteLangChainChatHistory(session_id=session_id, db_path=self._db_path)


class SQLiteLangChainChatHistory(BaseChatMessageHistory):
    """Pure sqlite3 implementation of LangChain's BaseChatMessageHistory."""

    def __init__(self, *, session_id: str, db_path: Path) -> None:
        self._session_id = session_id
        self._db_path = db_path

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        try:
            rows = conn.execute(
                "SELECT message_json FROM langchain_message_store WHERE session_id = ? ORDER BY id ASC",
                (self._session_id,),
            ).fetchall()
        finally:
            conn.close()
        raw_items = [json.loads(str(row[0])) for row in rows]
        return list(messages_from_dict(raw_items))

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        now = utc_now_iso()
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        try:
            with conn:
                conn.executemany(
                    """
                    INSERT INTO langchain_message_store (session_id, message_json, created_at)
                    VALUES (?, ?, ?)
                    """,
                    [(self._session_id, json.dumps(message_to_dict(msg)), now) for msg in messages],
                )
        finally:
            conn.close()

    def clear(self) -> None:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        try:
            with conn:
                conn.execute(
                    "DELETE FROM langchain_message_store WHERE session_id = ?",
                    (self._session_id,),
                )
        finally:
            conn.close()
