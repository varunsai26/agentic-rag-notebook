"""SOLID Protocol abstractions for dependency inversion across layers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from agentic_rag_notebook.domain.models import (
    AgentResponse,
    ChatTurn,
    ConversationSession,
    DocumentRecord,
    GuardrailStepResult,
    ParsedChunk,
    RetrievedChunk,
)


class DocumentParser(Protocol):
    """Parses PDF files into structured layout-aware chunks."""

    def parse(
        self,
        file_path: Path,
        *,
        session_id: str = "default",
        use_ocr: bool = False,
    ) -> list[ParsedChunk]:
        """Extract normalized chunks from a document file."""
        ...


class CSVDocumentLoader(Protocol):
    """Loads structured rows from CSV files into normalized chunks."""

    def parse(
        self,
        file_path: Path,
        *,
        session_id: str = "default",
        source_column: str | None = None,
        metadata_columns: tuple[str, ...] = (),
    ) -> list[ParsedChunk]:
        """Extract row chunks and metadata from a CSV file."""
        ...


class WikipediaTopicLoader(Protocol):
    """Loads encyclopedic articles from Wikipedia into normalized chunks."""

    def load(
        self,
        query_or_url: str,
        *,
        session_id: str = "default",
        load_max_docs: int = 2,
    ) -> list[ParsedChunk]:
        """Fetch and chunk Wikipedia articles for a query or URL."""
        ...


class VectorRepository(Protocol):
    """Stores and retrieves embedded chunks with strict session_id & source filtering."""

    def add_chunks(self, chunks: list[ParsedChunk]) -> int:
        """Index parsed chunks and return number of chunks stored."""
        ...

    def similarity_search(
        self,
        query: str,
        *,
        k: int = 4,
        session_id: str | None = None,
        active_sources: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Search indexed chunks filtered strictly by session_id and active sources."""
        ...

    def delete_source(
        self,
        source_name: str,
        *,
        session_id: str | None = None,
    ) -> None:
        """Delete all vectors belonging to the given source (optionally scoped to session_id)."""
        ...

    def delete_session_vectors(self, session_id: str) -> None:
        """Delete all vectors belonging to an entire conversation session."""
        ...


class SessionRepository(Protocol):
    """Persists ChatGPT-style sessions, turns, and per-session document records."""

    def create_session(
        self,
        *,
        session_id: str | None = None,
        title: str = "New Notebook Chat",
        active_sources: list[str] | None = None,
    ) -> ConversationSession:
        """Create and persist a new conversation session."""
        ...

    def get_session(self, session_id: str) -> ConversationSession | None:
        """Return a session by ID or None if not found."""
        ...

    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """Load existing session or create a new one with the provided UUID."""
        ...

    def list_sessions(self) -> list[ConversationSession]:
        """Return sessions ordered by most recently updated."""
        ...

    def update_session_title(self, session_id: str, title: str) -> None:
        """Update the display title of a session."""
        ...

    def update_active_sources(self, session_id: str, active_sources: list[str]) -> None:
        """Persist active source names selected for this session."""
        ...

    def delete_session(self, session_id: str) -> None:
        """Delete a session, its chat turns, and its registered source metadata."""
        ...

    def add_turn(self, turn: ChatTurn) -> None:
        """Append a chat turn to the persistent session history."""
        ...

    def get_turns(self, session_id: str) -> list[ChatTurn]:
        """Retrieve ordered chat turns for a session."""
        ...

    def register_document(self, record: DocumentRecord) -> None:
        """Insert or update metadata for an indexed source within a session."""
        ...

    def list_documents(self, session_id: str | None = None) -> list[DocumentRecord]:
        """List indexed documents (filtered by session_id when supplied)."""
        ...

    def set_document_active(
        self,
        filename: str,
        active: bool,
        *,
        session_id: str = "default",
    ) -> None:
        """Toggle whether a source is active within a session."""
        ...

    def delete_document_record(
        self,
        filename: str,
        *,
        session_id: str = "default",
    ) -> None:
        """Remove a document metadata record for a session."""
        ...


class InputGuardrail(Protocol):
    """Single input guardrail check executed before the LLM/agent runs."""

    name: str

    def check(self, text: str) -> tuple[str, GuardrailStepResult]:
        """Validate/transform input text and return updated text + audit step."""
        ...


class OutputGuardrail(Protocol):
    """Single output guardrail check executed before returning an answer to the user."""

    name: str

    def check(
        self,
        answer: str,
        *,
        retrieved_chunks: list[RetrievedChunk],
        has_active_sources: bool,
    ) -> tuple[str, GuardrailStepResult]:
        """Validate/transform output text and return updated text + audit step."""
        ...


class RagOrchestrator(Protocol):
    """Agentic RAG orchestration strategy (LangChain or LangGraph)."""

    engine_name: str

    def ask(
        self,
        *,
        session_id: str,
        query: str,
        active_sources: list[str] | None = None,
    ) -> AgentResponse:
        """Run end-to-end guarded RAG Q&A for a user query within a session."""
        ...
