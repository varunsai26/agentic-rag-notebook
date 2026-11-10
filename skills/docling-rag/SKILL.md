---
name: docling-rag
description: Docling HybridChunker, CSVLoader, WikipediaLoader, flat Chroma metadata, and strict session_id isolation rules.
---

# Multi-Source RAG + Chroma Session Isolation Skill

## Rules
1. Parse PDFs with `DoclingLoader(export_type=ExportType.DOC_CHUNKS, chunker=HybridChunker(...))` so headings, tables, and page metadata remain intact, exposing `use_ocr: bool` via `PdfPipelineOptions(do_ocr=use_ocr)`.
2. Parse CSVs using `langchain_community.document_loaders.CSVLoader` with optional `source_column`, `metadata_columns`, and `autodetect_encoding=True`, preserving `row`, `filename`, `csv_source_value`, and scalar `csv_<col>` metadata.
3. Fetch Wikipedia topics or full `https://<lang>.wikipedia.org/wiki/...` URLs using `WikipediaLoader(load_all_available_meta=True)` + `RecursiveCharacterTextSplitter`, flattening list metadata (`categories`, `sections`) to Chroma-safe scalars and preserving `url`, `title`, and `wiki_revision_id`.
4. Namespace Chroma collections by embedding provider/model signature (`settings.embedding_collection_name()`) to prevent vector dimension mismatch errors when switching embeddings in `.env`.
5. Strictly scope vector IDs, Chroma `similarity_search` `$and` filters, SQLite `documents` (`UNIQUE(session_id, filename)`), and delete calls by `session_id` alongside `active_sources`, so sessions never leak vectors across chats.


