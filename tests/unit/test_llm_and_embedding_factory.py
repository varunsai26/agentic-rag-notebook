"""Unit tests for LLMFactory (Groq + Local OpenAI Qwen) and EmbeddingFactory."""

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from agentic_rag_notebook.config.settings import Settings
from agentic_rag_notebook.infrastructure.embedding_factory import EmbeddingFactory
from agentic_rag_notebook.infrastructure.llm_factory import LLMFactory, RuntimeLLMOverrides


def test_llm_factory_creates_groq_client(tmp_settings: Settings) -> None:
    model, label = LLMFactory(tmp_settings).create()
    assert isinstance(model, ChatGroq)
    assert label == f"groq/{tmp_settings.groq_model}"


def test_llm_factory_creates_local_openai_qwen_client(tmp_settings: Settings) -> None:
    overrides = RuntimeLLMOverrides(
        provider="openai_local",
        base_url="http://127.0.0.1:11434/v1",
        api_key="qwen-secret-key",
        model="qwen2.5:14b",
    )
    model, label = LLMFactory(tmp_settings).create(overrides)
    assert isinstance(model, ChatOpenAI)
    assert label == "openai_local/qwen2.5:14b"
    assert model.openai_api_base == "http://127.0.0.1:11434/v1"
    assert isinstance(model.openai_api_key, SecretStr)
    assert model.openai_api_key.get_secret_value() == "qwen-secret-key"


def test_llm_factory_rejects_placeholder_groq_key(tmp_path: Path) -> None:
    bad_settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path,
        llm_provider="groq",
        groq_api_key="gsk_your_groq_api_key_here",
    )
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        LLMFactory(bad_settings).create()


def test_embedding_factory_supports_local_openai_and_huggingface(tmp_path: Path) -> None:
    local_emb_settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path,
        embedding_provider="openai_local",
        openai_local_embedding_base_url="http://localhost:8001/v1",
        openai_local_embedding_api_key="emb-key",
        openai_local_embedding_model="Qwen/Qwen3-Embedding-0.6B",
    )
    emb = EmbeddingFactory(local_emb_settings).create()
    assert isinstance(emb, OpenAIEmbeddings)
    assert emb.openai_api_base == "http://localhost:8001/v1"

    hf_settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path,
        embedding_provider="huggingface",
    )
    with patch(
        "agentic_rag_notebook.infrastructure.embedding_factory.HuggingFaceEmbeddings"
    ) as mock_hf:
        EmbeddingFactory(hf_settings).create()
        mock_hf.assert_called_once()
