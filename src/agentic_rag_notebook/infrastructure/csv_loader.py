"""CSV document loader built on LangChain's CSVLoader with session-scoped rich metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import CSVLoader

from agentic_rag_notebook.domain.models import ParsedChunk, ScalarMetaValue

logger = logging.getLogger(__name__)


def _to_scalar(value: Any) -> ScalarMetaValue:
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


class LangChainCSVLoader:
    """Implements CSVDocumentLoader using LangChain's built-in CSVLoader."""

    def parse(
        self,
        file_path: Path,
        *,
        session_id: str = "default",
        source_column: str | None = None,
        metadata_columns: tuple[str, ...] = (),
    ) -> list[ParsedChunk]:
        """Parse CSV rows into normalized ParsedChunks with `row`, `filename`, and `session_id`."""
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        clean_source_col = (
            source_column.strip() if source_column and source_column.strip() else None
        )
        clean_meta_cols = tuple(c.strip() for c in metadata_columns if c.strip())

        loader = CSVLoader(
            file_path=str(file_path),
            source_column=clean_source_col,
            metadata_columns=clean_meta_cols,
            autodetect_encoding=True,
        )
        raw_docs = loader.load()

        parsed_chunks: list[ParsedChunk] = []
        for idx, doc in enumerate(raw_docs):
            content = doc.page_content.strip()
            if not content:
                continue

            raw_row = doc.metadata.get("row")
            row_idx = int(raw_row) if isinstance(raw_row, int) else idx
            raw_source_meta = str(doc.metadata.get("source", "")).strip()

            extra_meta: dict[str, ScalarMetaValue] = {}
            for col_name in clean_meta_cols:
                if col_name in doc.metadata and doc.metadata[col_name] is not None:
                    extra_meta[f"csv_{col_name}"] = _to_scalar(doc.metadata[col_name])
            if clean_source_col and raw_source_meta and raw_source_meta != str(file_path):
                extra_meta["csv_source_value"] = raw_source_meta

            heading = (
                f"Row {row_idx + 1}"
                if not clean_source_col or raw_source_meta == str(file_path)
                else f"Row {row_idx + 1} · {raw_source_meta}"
            )
            parsed_chunks.append(
                ParsedChunk(
                    text=content,
                    source=file_path.name,
                    source_type="csv",
                    session_id=session_id,
                    filename=file_path.name,
                    title=file_path.stem,
                    row=row_idx,
                    heading=heading,
                    chunk_index=idx,
                    metadata=extra_meta,
                )
            )
        logger.info(
            "Parsed CSV filename=%s session_id=%s rows=%d",
            file_path.name,
            session_id,
            len(parsed_chunks),
        )
        return parsed_chunks
