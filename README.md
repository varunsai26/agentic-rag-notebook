# Agentic RAG Notebook

A modern Python **Agentic RAG Notebook** template built with **`uv`**, **LangChain**, **Docling**, **ChromaDB**, and **Streamlit** (styled after Gemini / NotebookLM), supporting both **Groq** and **Local OpenAI-compatible (Qwen)** endpoints, SQLite-backed ChatGPT-style **URL session persistence**, and **mandatory middleware guardrails**.

## Key Features

- **Modern `uv` `src/` Layout**: `pyproject.toml` (PEP 621), `uv.lock`, `ruff`, strict `mypy`, `pytest`, `Makefile`, and GitHub Actions CI.
- **KISS + SOLID + LLD Architecture**: Clean domain `Protocol` interfaces, factories, repositories, and composable services ([ARCHITECTURE_LLD.md](docs/ARCHITECTURE_LLD.md)).
- **Environment-Driven LLM & Embedding Clients**:
  - **Groq** (`ChatGroq`) via `GROQ_API_KEY` & `GROQ_MODEL`.
  - **Local OpenAI-Compatible Qwen** (`ChatOpenAI(base_url=..., api_key=...)`) for vLLM, Ollama, LM Studio, or custom gateways.
  - **Configurable Embeddings**: Defaults to zero-key local **HuggingFace** (`BAAI/bge-small-en-v1.5`), with optional **OpenAI** or **OpenAI-compatible custom URL+key** embeddings (note: Groq does not expose an `/embeddings` API).
- **Multi-Source Ingestion with Rich Vector Metadata**:
  - **PDF (Docling)**: `DoclingLoader` + `HybridChunker` preserving headings, tables, and page numbers (`ocr_enabled`, `page`, `heading`, `filename`).
  - **CSV (`CSVLoader`)**: LangChain `CSVLoader` with optional `source_column` & `metadata_columns`, recording `row`, `filename`, `csv_source_value`, and `csv_<col>` scalar metadata.
  - **Wikipedia (`WikipediaLoader`)**: LangChain `WikipediaLoader` accepting topics or full `https://<lang>.wikipedia.org/wiki/...` URLs + `RecursiveCharacterTextSplitter`, recording `url`, `title`, `wiki_summary`, `wiki_categories`, `wiki_sections`, and `wiki_revision_id`.
- **Strict Per-Session Vector & Source Isolation (`session_id`)**: Every chunk in ChromaDB and source record in SQLite carries `session_id`, `source`, `source_type`, `filename`, `url`, `title`, `chunk_index`, and `ingested_at`. Retrieval and deletion strictly filter by `{"session_id": session_id}`, preventing cross-session data leakage.
- **Persistent ChromaDB Vector Store**: Local disk storage namespaced by embedding signature so switching embedding models never causes dimension mismatch errors.
- **Phase 1 LangChain -> Phase 2 LangGraph Learning Path**:
  - **Phase 1 (Default)**: LangChain v1 orchestration (`create_agent`, modular `ChatPromptTemplate` templates, `@tool` retrieval, SQLite `BaseChatMessageHistory`, and built-in LangChain middlewares: `PIIMiddleware`, `SummarizationMiddleware`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, `ToolRetryMiddleware`, plus RAG guardrail hooks).
  - **Phase 2 (`ORCHESTRATION_ENGINE=langgraph`)**: Explicit LangGraph `StateGraph` (`route -> retrieve -> grade -> rewrite -> generate`) with `SqliteSaver` checkpointer ([LANGCHAIN_TO_LANGGRAPH.md](docs/LANGCHAIN_TO_LANGGRAPH.md)).
- **Mandatory Input & Output Guardrails via Middleware**:
  - **Input**: Empty/length validation, prompt-injection detection, and PII redaction (email, phone, SSN, credit card, API keys).
  - **Output**: Secret leakage redaction, unsafe-content filter, and groundedness/citation verification against retrieved PDF chunks.
- **ChatGPT-Style URL Conversation Persistence**: Every chat generates a UUID `session_id` synced to the browser URL (`?session_id=<uuid>`) and persisted in SQLite (`./data/sessions.db`). Refreshing or sharing the URL restores history and conversational context.

## Quick Start

```bash
# 1. Install dependencies & create .venv
uv sync

# 2. Configure environment variables
cp .env.example .env
# Edit .env: set LLM_PROVIDER=groq (with GROQ_API_KEY) or LLM_PROVIDER=openai_local (Qwen URL + key)

# 3. Launch the Streamlit Notebook UI
uv run agentic-rag-ui
# Or directly:
uv run streamlit run src/agentic_rag_notebook/ui/app.py
```

## Developer Commands

```bash
make sync       # Sync virtual environment with uv
make run        # Start Streamlit UI
make format     # Format code with ruff
make lint       # Lint code with ruff
make typecheck  # Run strict mypy checks
make test       # Run pytest suite
make check      # Run format check, lint, mypy, and pytest
```
