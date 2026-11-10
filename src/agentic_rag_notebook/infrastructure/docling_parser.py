"""Docling PDF document parser using DoclingLoader and layout-aware HybridChunker."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from agentic_rag_notebook.domain.models import ParsedChunk

logger = logging.getLogger(__name__)


def _extract_page_and_heading(dl_meta: Any) -> tuple[int | None, str | None]:
    """Extract first page number and section heading breadcrumb from Docling chunk metadata."""
    if not isinstance(dl_meta, dict):
        return None, None

    page_no: int | None = None
    doc_items = dl_meta.get("doc_items")
    if isinstance(doc_items, list):
        for item in doc_items:
            if not isinstance(item, dict):
                continue
            prov_list = item.get("prov")
            if isinstance(prov_list, list):
                for prov in prov_list:
                    if isinstance(prov, dict) and isinstance(prov.get("page_no"), int):
                        page_no = int(prov["page_no"])
                        break
            if page_no is not None:
                break

    heading: str | None = None
    headings = dl_meta.get("headings")
    if isinstance(headings, list) and headings:
        clean_headings = [str(h).strip() for h in headings if str(h).strip()]
        if clean_headings:
            heading = " > ".join(clean_headings)

    return page_no, heading


class DoclingDocumentParser:
    """Implements DocumentParser using DoclingLoader + HybridChunker."""

    def __init__(
        self,
        *,
        tokenizer_model_name: str | None = None,
        max_chunk_tokens: int = 512,
    ) -> None:
        self._tokenizer_model_name = tokenizer_model_name
        self._max_chunk_tokens = max_chunk_tokens

    def _build_chunker(self) -> HybridChunker:
        if self._tokenizer_model_name:
            try:
                tokenizer = HuggingFaceTokenizer.from_pretrained(
                    model_name=self._tokenizer_model_name,
                    max_tokens=self._max_chunk_tokens,
                )
                return HybridChunker(tokenizer=tokenizer)
            except Exception:
                return HybridChunker()
        return HybridChunker()

    def parse(
        self,
        file_path: Path,
        *,
        session_id: str = "default",
        use_ocr: bool = False,
    ) -> list[ParsedChunk]:
        """Parse and chunk a PDF document while preserving headings, tables, and page numbers."""
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        pipeline_options = PdfPipelineOptions(do_ocr=use_ocr)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        loader = DoclingLoader(
            file_path=str(file_path),
            converter=converter,
            export_type=ExportType.DOC_CHUNKS,
            chunker=self._build_chunker(),
        )
        raw_docs = loader.load()

        parsed_chunks: list[ParsedChunk] = []
        for idx, doc in enumerate(raw_docs):
            content = doc.page_content.strip()
            if not content:
                continue
            page_no, heading = _extract_page_and_heading(doc.metadata.get("dl_meta"))
            parsed_chunks.append(
                ParsedChunk(
                    text=content,
                    source=file_path.name,
                    source_type="pdf",
                    session_id=session_id,
                    filename=file_path.name,
                    title=file_path.stem,
                    page=page_no,
                    heading=heading,
                    chunk_index=idx,
                    metadata={"ocr_enabled": use_ocr},
                )
            )
        logger.info(
            "Parsed PDF filename=%s session_id=%s chunks=%d ocr=%s",
            file_path.name,
            session_id,
            len(parsed_chunks),
            use_ocr,
        )
        return parsed_chunks
