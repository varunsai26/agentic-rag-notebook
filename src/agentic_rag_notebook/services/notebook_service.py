"""Application facade coordinating PDF, CSV, and Wikipedia ingestion, URL sessions, and Q&A."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

from agentic_rag_notebook.config.settings import OrchestrationEngineType, Settings
from agentic_rag_notebook.domain.interfaces import (
    CSVDocumentLoader,
    DocumentParser,
    RagOrchestrator,
    SessionRepository,
    VectorRepository,
    WikipediaTopicLoader,
)
from agentic_rag_notebook.domain.models import (
    AgentResponse,
    ChatTurn,
    ConversationSession,
    DocumentRecord,
    ParsedChunk,
)
from agentic_rag_notebook.infrastructure.chroma_store import ChromaVectorRepository
from agentic_rag_notebook.infrastructure.csv_loader import LangChainCSVLoader
from agentic_rag_notebook.infrastructure.docling_parser import DoclingDocumentParser
from agentic_rag_notebook.infrastructure.embedding_factory import EmbeddingFactory
from agentic_rag_notebook.infrastructure.llm_factory import LLMFactory, RuntimeLLMOverrides
from agentic_rag_notebook.infrastructure.sqlite_session_store import SQLiteSessionStore
from agentic_rag_notebook.infrastructure.wikipedia_loader import LangChainWikipediaLoader
from agentic_rag_notebook.orchestration.langchain_agent import LangChainRagOrchestrator
from agentic_rag_notebook.orchestration.langgraph_agent import LangGraphRagOrchestrator

logger = logging.getLogger(__name__)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class NotebookService:
    """High-level application service following Facade and Dependency Inversion patterns."""

    def __init__(
        self,
        *,
        settings: Settings,
        parser: DocumentParser,
        vector_repo: VectorRepository,
        session_repo: SessionRepository,
        orchestrators: dict[str, RagOrchestrator],
        csv_loader: CSVDocumentLoader | None = None,
        wiki_loader: WikipediaTopicLoader | None = None,
    ) -> None:
        self._settings = settings
        self._parser = parser
        self._csv_loader = csv_loader or LangChainCSVLoader()
        self._wiki_loader = wiki_loader or LangChainWikipediaLoader()
        self._vector_repo = vector_repo
        self._session_repo = session_repo
        self._orchestrators = orchestrators

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        llm_overrides: RuntimeLLMOverrides | None = None,
    ) -> NotebookService:
        """Assemble default production dependencies from Settings."""
        settings.ensure_directories()
        embeddings = EmbeddingFactory(settings).create()
        vector_repo = ChromaVectorRepository(
            persist_directory=settings.chroma_persist_dir,
            collection_name=settings.embedding_collection_name(),
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
        llm_factory = LLMFactory(settings)
        orchestrators: dict[str, RagOrchestrator] = {
            "langchain": LangChainRagOrchestrator(
                settings=settings,
                vector_repo=vector_repo,
                session_store=session_store,
                llm_factory=llm_factory,
                llm_overrides=llm_overrides,
            ),
            "langgraph": LangGraphRagOrchestrator(
                settings=settings,
                vector_repo=vector_repo,
                session_store=session_store,
                llm_factory=llm_factory,
                llm_overrides=llm_overrides,
            ),
        }
        return cls(
            settings=settings,
            parser=parser,
            vector_repo=vector_repo,
            session_repo=session_store,
            orchestrators=orchestrators,
        )

    # --- URL Session Lifecycle ---

    def ensure_session(self, session_id: str | None) -> ConversationSession:
        """Return existing session for URL `?session_id=` or create a fresh UUID session."""
        if session_id and session_id.strip():
            return self._session_repo.get_or_create_session(session_id.strip())
        return self._session_repo.create_session()

    def create_new_session(self) -> ConversationSession:
        """Create a brand new ChatGPT-style conversation session."""
        return self._session_repo.create_session()

    def list_sessions(self) -> list[ConversationSession]:
        """List all persisted conversation sessions ordered by recency."""
        return self._session_repo.list_sessions()

    def get_session_turns(self, session_id: str) -> list[ChatTurn]:
        """Return persisted chat history turns for a conversation."""
        return self._session_repo.get_turns(session_id)

    def delete_session(self, session_id: str) -> None:
        """Delete a conversation session, its SQLite history/docs, Chroma vectors, and uploads."""
        self._vector_repo.delete_session_vectors(session_id)
        self._session_repo.delete_session(session_id)
        session_upload_dir = self._settings.upload_dir / session_id
        if session_upload_dir.exists():
            shutil.rmtree(session_upload_dir, ignore_errors=True)

    # --- Source Ingestion & Session-Scoped Knowledge Management ---

    def _session_upload_path(self, session_id: str, filename: str) -> Path:
        safe_name = Path(filename).name
        target_dir = self._settings.upload_dir / session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / safe_name

    def ingest_pdf_bytes(
        self,
        *,
        filename: str,
        content: bytes,
        session_id: str = "default",
        use_ocr: bool = False,
    ) -> DocumentRecord:
        """Persist uploaded PDF bytes under `session_id`, parse via Docling, and index in Chroma."""
        self._session_repo.get_or_create_session(session_id)
        target_path = self._session_upload_path(session_id, filename)
        target_path.write_bytes(content)

        chunks = self._parser.parse(target_path, session_id=session_id, use_ocr=use_ocr)
        indexed_count = self._vector_repo.add_chunks(chunks)

        record = DocumentRecord(
            session_id=session_id,
            filename=target_path.name,
            source_type="pdf",
            title=target_path.stem,
            file_hash=_sha256_bytes(content),
            chunk_count=indexed_count,
            ocr_enabled=use_ocr,
            active=True,
        )
        self._session_repo.register_document(record)
        logger.info(
            "Indexed PDF filename=%s session_id=%s chunks=%d ocr=%s",
            record.filename,
            session_id,
            indexed_count,
            use_ocr,
        )
        return record

    def ingest_csv_bytes(
        self,
        *,
        filename: str,
        content: bytes,
        session_id: str = "default",
        source_column: str | None = None,
        metadata_columns: tuple[str, ...] = (),
    ) -> DocumentRecord:
        """Persist uploaded CSV bytes under `session_id`, parse via CSVLoader, and index."""
        self._session_repo.get_or_create_session(session_id)
        target_path = self._session_upload_path(session_id, filename)
        target_path.write_bytes(content)

        chunks = self._csv_loader.parse(
            target_path,
            session_id=session_id,
            source_column=source_column,
            metadata_columns=metadata_columns,
        )
        indexed_count = self._vector_repo.add_chunks(chunks)

        record = DocumentRecord(
            session_id=session_id,
            filename=target_path.name,
            source_type="csv",
            title=target_path.stem,
            file_hash=_sha256_bytes(content),
            chunk_count=indexed_count,
            ocr_enabled=False,
            active=True,
        )
        self._session_repo.register_document(record)
        logger.info(
            "Indexed CSV filename=%s session_id=%s rows=%d",
            record.filename,
            session_id,
            indexed_count,
        )
        return record

    def ingest_wikipedia(
        self,
        *,
        query_or_url: str,
        session_id: str = "default",
        load_max_docs: int = 2,
    ) -> list[DocumentRecord]:
        """Fetch Wikipedia articles via WikipediaLoader, index under `session_id`, and register."""
        self._session_repo.get_or_create_session(session_id)
        chunks = self._wiki_loader.load(
            query_or_url,
            session_id=session_id,
            load_max_docs=load_max_docs,
        )
        if not chunks:
            return []

        grouped: dict[str, list[ParsedChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.source, []).append(chunk)

        records: list[DocumentRecord] = []
        for source_key, source_chunks in grouped.items():
            indexed_count = self._vector_repo.add_chunks(source_chunks)
            first = source_chunks[0]
            record = DocumentRecord(
                session_id=session_id,
                filename=source_key,
                source_type="wikipedia",
                url=first.url,
                title=first.title or source_key,
                file_hash=_sha256_bytes(f"{source_key}:{first.url}".encode()),
                chunk_count=indexed_count,
                ocr_enabled=False,
                active=True,
            )
            self._session_repo.register_document(record)
            records.append(record)
        logger.info(
            "Indexed Wikipedia query=%r session_id=%s articles=%d total_chunks=%d",
            query_or_url.strip(),
            session_id,
            len(records),
            sum(r.chunk_count for r in records),
        )
        return records

    def list_documents(self, session_id: str | None = None) -> list[DocumentRecord]:
        """Return indexed source records scoped to `session_id`."""
        return self._session_repo.list_documents(session_id=session_id)

    def set_document_active(
        self,
        filename: str,
        active: bool,
        *,
        session_id: str = "default",
    ) -> None:
        """Enable or disable a source inside a specific session."""
        self._session_repo.set_document_active(filename, active, session_id=session_id)

    def delete_document(
        self,
        filename: str,
        *,
        session_id: str = "default",
    ) -> None:
        """Delete a source from ChromaDB, SQLite registry, and local session upload dir."""
        self._vector_repo.delete_source(filename, session_id=session_id)
        self._session_repo.delete_document_record(filename, session_id=session_id)
        uploaded_file = self._settings.upload_dir / session_id / Path(filename).name
        if uploaded_file.exists():
            uploaded_file.unlink()

    def active_source_names(self, session_id: str | None = None) -> list[str]:
        """Return active source identifiers for the specified session."""
        return [
            doc.filename
            for doc in self._session_repo.list_documents(session_id=session_id)
            if doc.active
        ]

    # --- Q&A Execution ---

    def ask(
        self,
        *,
        session_id: str,
        query: str,
        engine: OrchestrationEngineType | None = None,
    ) -> AgentResponse:
        """Execute session-isolated guarded RAG Q&A and persist both turns to SQLite."""
        selected_engine = engine or self._settings.orchestration_engine
        orchestrator = self._orchestrators.get(selected_engine)
        if orchestrator is None:
            raise ValueError(f"Unsupported orchestration engine: {selected_engine}")

        self._session_repo.get_or_create_session(session_id)
        active_sources = self.active_source_names(session_id)
        self._session_repo.update_active_sources(session_id, active_sources)

        logger.info(
            "Executing RAG ask session_id=%s engine=%s active_sources=%d",
            session_id,
            selected_engine,
            len(active_sources),
        )
        response = orchestrator.ask(
            session_id=session_id,
            query=query,
            active_sources=active_sources,
        )
        logger.info(
            "Completed RAG ask session_id=%s engine=%s model=%s blocked=%s citations=%d",
            session_id,
            response.engine,
            response.model_label,
            response.blocked,
            len(response.citations),
        )

        self._session_repo.add_turn(
            ChatTurn(
                session_id=session_id,
                role="user",
                content=query,
            )
        )
        self._session_repo.add_turn(
            ChatTurn(
                session_id=session_id,
                role="assistant",
                content=response.answer,
                reasoning=response.reasoning,
                citations=response.citations,
                guardrail_report=response.guardrail_report,
                engine=response.engine,
                model_label=response.model_label,
            )
        )
        return response
