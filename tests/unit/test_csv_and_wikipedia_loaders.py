"""Unit tests for LangChain CSVLoader and WikipediaLoader adapters and metadata normalization."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import wikipedia.wikipedia as wiki_mod
from langchain_core.documents import Document

from agentic_rag_notebook.infrastructure.csv_loader import LangChainCSVLoader
from agentic_rag_notebook.infrastructure.wikipedia_loader import (
    WIKIPEDIA_USER_AGENT,
    LangChainWikipediaLoader,
    _configure_wikipedia_client,
    normalize_wikipedia_query,
)


def test_csv_loader_extracts_rows_and_scalar_metadata(tmp_path: Path) -> None:
    csv_file = tmp_path / "models_benchmark.csv"
    csv_file.write_text(
        "model_id,provider,context_window,notes\n"
        "qwen2.5-7b,openai_local,32768,Fast local reasoning model\n"
        "llama-3.3-70b,groq,128000,High throughput cloud model\n",
        encoding="utf-8",
    )

    loader = LangChainCSVLoader()
    chunks = loader.parse(
        csv_file,
        session_id="sess-csv-1",
        source_column="model_id",
        metadata_columns=("provider", "context_window"),
    )

    assert len(chunks) == 2
    first, second = chunks[0], chunks[1]
    assert first.source == "models_benchmark.csv"
    assert first.filename == "models_benchmark.csv"
    assert first.source_type == "csv"
    assert first.session_id == "sess-csv-1"
    assert first.title == "models_benchmark"
    assert first.row == 0
    assert first.heading == "Row 1 · qwen2.5-7b"
    assert first.metadata["csv_provider"] == "openai_local"
    assert first.metadata["csv_context_window"] == "32768"
    assert first.metadata["csv_source_value"] == "qwen2.5-7b"
    assert "Fast local reasoning model" in first.text

    assert second.row == 1
    assert second.heading == "Row 2 · llama-3.3-70b"
    assert second.metadata["csv_provider"] == "groq"


def test_normalize_wikipedia_query_accepts_topics_and_urls() -> None:
    assert normalize_wikipedia_query("Retrieval-augmented generation") == (
        "Retrieval-augmented generation",
        "en",
    )
    assert normalize_wikipedia_query("https://en.wikipedia.org/wiki/Model_Context_Protocol") == (
        "Model Context Protocol",
        "en",
    )
    assert normalize_wikipedia_query(
        "https://www.wikipedia.org/wiki/Python_(programming_language)"
    ) == (
        "Python (programming language)",
        "en",
    )
    assert normalize_wikipedia_query("https://fr.wikipedia.org/wiki/Intelligence_artificielle") == (
        "Intelligence artificielle",
        "fr",
    )

    with pytest.raises(ValueError, match="Wikipedia topic or URL"):
        normalize_wikipedia_query("   ")


def test_wikipedia_loader_flattens_metadata_and_splits_chunks() -> None:
    long_body = ("Retrieval-augmented generation grounds LLM answers. " * 35).strip()
    fake_docs = [
        Document(
            page_content=long_body,
            metadata={
                "title": "Retrieval-augmented generation",
                "summary": "RAG combines retrieval with LLM generation.",
                "source": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
                "page_url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
                "categories": ["Artificial intelligence", "Natural language processing"],
                "sections": ["Process", "Evaluation", "Limitations"],
                "revision_id": 987654,
            },
        )
    ]

    with patch(
        "agentic_rag_notebook.infrastructure.wikipedia_loader.WikipediaLoader"
    ) as mock_wiki_cls:
        mock_wiki_cls.return_value.load.return_value = fake_docs

        wiki_loader = LangChainWikipediaLoader(chunk_size=400, chunk_overlap=50)
        chunks = wiki_loader.load(
            "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
            session_id="sess-wiki-1",
            load_max_docs=2,
        )

    mock_wiki_cls.assert_called_once_with(
        query="Retrieval-augmented generation",
        lang="en",
        load_max_docs=2,
        load_all_available_meta=False,
        doc_content_chars_max=8000,
    )
    assert len(chunks) >= 2
    for idx, chunk in enumerate(chunks):
        assert chunk.source == "wiki:Retrieval-augmented_generation"
        assert chunk.source_type == "wikipedia"
        assert chunk.session_id == "sess-wiki-1"
        assert chunk.url == "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
        assert chunk.title == "Retrieval-augmented generation"
        assert chunk.chunk_index == idx
        assert chunk.metadata["wiki_revision_id"] == 987654
        assert isinstance(chunk.metadata["wiki_categories"], str)
        assert isinstance(chunk.metadata["wiki_sections"], str)
        assert isinstance(chunk.metadata["wiki_summary"], str)


def test_wikipedia_client_uses_https_user_agent_and_handles_json_error() -> None:
    _configure_wikipedia_client("en")
    assert wiki_mod.API_URL == "https://en.wikipedia.org/w/api.php"
    assert wiki_mod.USER_AGENT == WIKIPEDIA_USER_AGENT

    with patch(
        "agentic_rag_notebook.infrastructure.wikipedia_loader.WikipediaLoader"
    ) as mock_wiki_cls:
        mock_wiki_cls.return_value.load.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        wiki_loader = LangChainWikipediaLoader()
        with pytest.raises(RuntimeError, match="non-JSON response"):
            wiki_loader.load("Python", session_id="sess-wiki-err")


def test_resilient_wiki_request_retries_empty_or_rate_limited_responses() -> None:
    _configure_wikipedia_client("en")

    rate_limited = MagicMock(status_code=429, headers={"Retry-After": "0.1"}, text="")
    empty_ok = MagicMock(status_code=200, headers={}, text="   ")
    valid_ok = MagicMock(
        status_code=200,
        headers={},
        text='{"query": {"search": [{"title": "Python"}]}}',
    )
    valid_ok.json.return_value = {"query": {"search": [{"title": "Python"}]}}

    with (
        patch("requests.Session.get", side_effect=[rate_limited, empty_ok, valid_ok]) as mock_get,
        patch("time.sleep"),
    ):
        result = wiki_mod._wiki_request({"list": "search", "srsearch": "Python"})

    assert mock_get.call_count == 3
    assert result == {"query": {"search": [{"title": "Python"}]}}


def test_resilient_wiki_request_raises_clear_error_when_empty_persists() -> None:
    _configure_wikipedia_client("en")

    empty_resp = MagicMock(status_code=200, headers={}, text="")
    with (
        patch("requests.Session.get", return_value=empty_resp),
        patch("time.sleep"),
        pytest.raises(RuntimeError, match="empty response"),
    ):
        wiki_mod._wiki_request({"list": "search", "srsearch": "Python"})
