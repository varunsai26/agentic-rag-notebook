"""Persistent ChromaDB vector store with rich metadata & strict session_id isolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agentic_rag_notebook.domain.models import ParsedChunk, RetrievedChunk, SourceType


def _coerce_source_type(raw: Any) -> SourceType:
    if raw == "csv":
        return "csv"
    if raw == "wikipedia":
        return "wikipedia"
    return "pdf"


class ChromaVectorRepository:
    """ChromaDB-backed VectorRepository enforcing session_id and active-source filtering."""

    def __init__(
        self,
        *,
        persist_directory: Path,
        collection_name: str,
        embedding_function: Embeddings,
    ) -> None:
        self._persist_directory = persist_directory
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=str(persist_directory),
        )

    @property
    def collection_name(self) -> str:
        """Return active Chroma collection name."""
        return self._collection_name

    def add_chunks(self, chunks: list[ParsedChunk]) -> int:
        """Store parsed chunks with rich metadata in ChromaDB and return inserted count."""
        if not chunks:
            return 0

        documents: list[Document] = []
        ids: list[str] = []
        for chunk in chunks:
            metadata: dict[str, Any] = {
                "session_id": chunk.session_id,
                "source": chunk.source,
                "source_type": chunk.source_type,
                "filename": chunk.filename,
                "url": chunk.url,
                "title": chunk.title,
                "page": chunk.page if chunk.page is not None else -1,
                "row": chunk.row if chunk.row is not None else -1,
                "heading": chunk.heading or "",
                "chunk_index": chunk.chunk_index,
                "ingested_at": chunk.ingested_at,
                **chunk.metadata,
            }
            documents.append(Document(page_content=chunk.text, metadata=metadata))
            ids.append(f"{chunk.session_id}::{chunk.source}::chunk_{chunk.chunk_index}")

        # Replace previous vectors for the same (session_id, source) pair cleanly
        unique_pairs = {(chunk.session_id, chunk.source) for chunk in chunks}
        for sid, source_name in unique_pairs:
            self.delete_source(source_name, session_id=sid)

        self._store.add_documents(documents=documents, ids=ids)
        return len(documents)

    def similarity_search(
        self,
        query: str,
        *,
        k: int = 4,
        session_id: str | None = None,
        active_sources: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Query ChromaDB filtered strictly by `session_id` and `active_sources`."""
        if active_sources is not None and len(active_sources) == 0:
            return []

        clauses: list[dict[str, Any]] = []
        if session_id is not None:
            clauses.append({"session_id": session_id})
        if active_sources is not None:
            if len(active_sources) == 1:
                clauses.append({"source": active_sources[0]})
            else:
                clauses.append({"source": {"$in": active_sources}})

        where_filter: dict[str, Any] | None = None
        if len(clauses) == 1:
            where_filter = clauses[0]
        elif len(clauses) > 1:
            where_filter = {"$and": clauses}

        results = self._store.similarity_search_with_score(
            query=query,
            k=k,
            filter=where_filter,
        )

        chunks: list[RetrievedChunk] = []
        for doc, distance in results:
            raw_page = doc.metadata.get("page")
            page = int(raw_page) if isinstance(raw_page, int) and raw_page > 0 else None
            raw_row = doc.metadata.get("row")
            row = int(raw_row) if isinstance(raw_row, int) and raw_row >= 0 else None
            heading_raw = str(doc.metadata.get("heading", "")).strip()
            chunks.append(
                RetrievedChunk(
                    content=doc.page_content,
                    source=str(doc.metadata.get("source", "unknown")),
                    source_type=_coerce_source_type(doc.metadata.get("source_type")),
                    session_id=str(doc.metadata.get("session_id", "")),
                    filename=str(doc.metadata.get("filename", "")),
                    url=str(doc.metadata.get("url", "")),
                    title=str(doc.metadata.get("title", "")),
                    page=page,
                    row=row,
                    heading=heading_raw or None,
                    score=float(distance),
                )
            )
        return chunks

    def delete_source(
        self,
        source_name: str,
        *,
        session_id: str | None = None,
    ) -> None:
        """Delete vectors for `source_name`, scoped to `session_id` when provided."""
        if session_id is not None:
            self._store.delete(
                where={"$and": [{"session_id": session_id}, {"source": source_name}]}
            )
            return
        self._store.delete(where={"source": source_name})

    def delete_session_vectors(self, session_id: str) -> None:
        """Delete every vector belonging to `session_id`."""
        self._store.delete(where={"session_id": session_id})

    def get_session_metadata(self, session_id: str) -> list[dict[str, Any]]:
        """Return stored Chroma metadata dictionaries scoped to `session_id`."""
        payload = self._store.get(where={"session_id": session_id}, include=["metadatas"])
        raw_metas = payload.get("metadatas") or []
        return [dict(meta) for meta in raw_metas if isinstance(meta, dict)]
