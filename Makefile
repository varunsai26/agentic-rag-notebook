.PHONY: sync run check-env lint format typecheck test check clean

sync:
	uv sync

run:
	uv run streamlit run src/agentic_rag_notebook/ui/app.py

check-env:
	uv run agentic-rag-check-env

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src tests

test:
	uv run pytest

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src tests
	uv run pytest

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage

