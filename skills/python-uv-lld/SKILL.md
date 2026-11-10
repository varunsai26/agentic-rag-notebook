---
name: python-uv-lld
description: Modern Python uv src/ layout, KISS, SOLID, and Low-Level Design (LLD) standards for agentic-rag-notebook.
---

# Python `uv` + KISS + SOLID + LLD Skill

## Principles
1. **KISS**: Keep each class and function focused, small, and readable. Avoid premature metaprogramming or over-engineered plugin loaders.
2. **SRP (Single Responsibility)**: Separate config (`config/`), pure domain contracts (`domain/`), concrete I/O adapters (`infrastructure/`), guardrails (`middleware/`), agent orchestration (`orchestration/`), app use-cases (`services/`), and Streamlit rendering (`ui/`).
3. **OCP (Open/Closed)**: Add a new guardrail by implementing `InputGuardrail` or `OutputGuardrail` and appending it to the middleware pipeline without editing orchestrator logic.
4. **LSP & ISP**: Keep `Protocol` interfaces minimal and cohesive (`VectorRepository`, `SessionRepository`, `DocumentParser`, `RagOrchestrator`).
5. **DIP**: High-level `services/` and `orchestration/` depend on `domain/interfaces.py` Protocols, injected from factories at startup.

## Commands
- `uv add <pkg>` / `uv add --dev <pkg>`
- `uv run ruff format .`
- `uv run ruff check --fix .`
- `uv run mypy src tests`
- `uv run pytest`

