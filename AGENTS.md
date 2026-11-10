# AGENTS.md — Project Engineering & Architecture Rules

This project (`agentic-rag-notebook`) follows **KISS**, **SOLID**, and **Low-Level Design (LLD)** principles inside a modern Python `uv` `src/` layout.

## Tooling & Framework Rules
- Always use `uv run ...` instead of raw `python` or `pip`.
- Verify changes with `make check` (`ruff format --check`, `ruff check`, `mypy src tests`, `pytest`).
- All runtime configuration lives in `.env` via `pydantic-settings` (`config/settings.py`). Never hardcode secrets or URLs.
- **Prefer built-in LangChain utilities first** (`create_agent`, `PIIMiddleware`, `SummarizationMiddleware`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, `ToolRetryMiddleware`, `RedactionRule`, `BaseChatMessageHistory`) rather than reimplementing custom equivalents. Only write custom `AgentMiddleware` hooks for domain-specific RAG checks (e.g., prompt-injection defense and citation groundedness).

## Layer Dependency Rule (SOLID / DIP)
`ui` -> `services` -> `orchestration` -> `domain` <- `infrastructure`
- `domain/` must remain pure (no `streamlit`, `chromadb`, `docling`, or `sqlite3` imports).
- `infrastructure/` implements `domain/interfaces.py` Protocols (`DocumentParser`, `VectorRepository`, `SessionRepository`).
- `middleware/` combines built-in LangChain middlewares with thin RAG input/output guardrails.
- `orchestration/` defaults to **Phase 1 LangChain** (`langchain_agent.py`) and includes **Phase 2 LangGraph** (`langgraph_agent.py`) as a clean progression path.

## Project Skills
Read the relevant file in `skills/` before extending a subsystem:
- `skills/python-uv-lld/SKILL.md`
- `skills/docling-rag/SKILL.md`
- `skills/langchain-middleware/SKILL.md`
