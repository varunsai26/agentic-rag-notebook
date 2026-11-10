"""Factory for creating HuggingFace (local default) or OpenAI-compatible embeddings."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from agentic_rag_notebook.config.settings import Settings


class EmbeddingFactory:
    """Builds LangChain Embeddings instances from Settings."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def create(self) -> Embeddings:
        """Instantiate the configured embeddings model."""
        provider = self._settings.embedding_provider

        if provider == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=self._settings.huggingface_embedding_model,
                model_kwargs={"device": self._settings.huggingface_embedding_device},
                encode_kwargs={"normalize_embeddings": True},
            )

        if provider == "openai":
            if not self._settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai.")
            return OpenAIEmbeddings(
                api_key=SecretStr(self._settings.openai_api_key),
                model=self._settings.openai_embedding_model,
            )

        if not self._settings.openai_local_embedding_base_url:
            raise ValueError(
                "OPENAI_LOCAL_EMBEDDING_BASE_URL is required when EMBEDDING_PROVIDER=openai_local."
            )
        effective_key = self._settings.openai_local_embedding_api_key or "local-no-key-required"
        return OpenAIEmbeddings(
            base_url=self._settings.openai_local_embedding_base_url,
            api_key=SecretStr(effective_key),
            model=self._settings.openai_local_embedding_model,
        )
