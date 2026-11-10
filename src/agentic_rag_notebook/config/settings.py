"""Environment-driven application configuration using pydantic-settings."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderType = Literal["groq", "openai_local"]
EmbeddingProviderType = Literal["huggingface", "openai", "openai_local"]
OrchestrationEngineType = Literal["langchain", "langgraph"]
LogLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Typed application settings loaded from `.env` and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Active Engine & Provider Selection ---
    llm_provider: LLMProviderType = "groq"
    orchestration_engine: OrchestrationEngineType = "langchain"

    # --- Groq Chat Model Settings ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Local OpenAI-Compatible Qwen Settings ---
    openai_local_base_url: str = "http://localhost:8000/v1"
    openai_local_api_key: str = ""
    openai_local_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # --- Common LLM Generation Settings ---
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=64, le=32768)

    # --- Embedding Settings ---
    embedding_provider: EmbeddingProviderType = "huggingface"
    huggingface_embedding_model: str = "BAAI/bge-small-en-v1.5"
    huggingface_embedding_device: str = "cpu"

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    openai_local_embedding_base_url: str = "http://localhost:8001/v1"
    openai_local_embedding_api_key: str = ""
    openai_local_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # --- Persistence Paths ---
    data_dir: Path = Path("./data")
    chroma_persist_dir: Path = Path("./data/chroma_db")
    sqlite_db_path: Path = Path("./data/sessions.db")
    upload_dir: Path = Path("./data/uploads")

    # --- Logging Configuration ---
    log_dir: Path = Path("./logs")
    log_file_name: str = "app.log"
    log_level: LogLevelType = "INFO"
    log_backup_count: int = Field(default=30, ge=0, le=365)

    # --- Docling & RAG Settings ---
    docling_default_ocr: bool = False
    docling_max_chunk_tokens: int = Field(default=512, ge=128, le=4096)
    rag_top_k: int = Field(default=4, ge=1, le=20)

    # --- Mandatory Guardrail Middleware Settings ---
    enable_input_guardrails: bool = True
    enable_output_guardrails: bool = True
    pii_redaction_enabled: bool = True
    max_input_chars: int = Field(default=4000, ge=100, le=50000)
    max_history_turns: int = Field(default=12, ge=2, le=50)
    strict_grounding_mode: bool = False

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value

    def ensure_directories(self) -> None:
        """Ensure required local data and log directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def active_llm_model_name(self) -> str:
        """Return the configured model identifier for the active LLM provider."""
        if self.llm_provider == "groq":
            return self.groq_model
        return self.openai_local_model

    def active_embedding_model_name(self) -> str:
        """Return the configured model identifier for the active embedding provider."""
        if self.embedding_provider == "huggingface":
            return self.huggingface_embedding_model
        if self.embedding_provider == "openai":
            return self.openai_embedding_model
        return self.openai_local_embedding_model

    def embedding_collection_name(self) -> str:
        """Build a deterministic Chroma collection name scoped to the embedding signature.

        Prevents dimension mismatch errors when switching embedding models in `.env`.
        """
        signature = f"{self.embedding_provider}:{self.active_embedding_model_name()}"
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", self.embedding_provider).strip("_").lower()
        return f"notebook_{slug}_{digest}"

    def sqlite_connection_string(self) -> str:
        """Return SQLAlchemy connection string for SQLite session persistence."""
        return f"sqlite:///{self.sqlite_db_path.resolve()}"

    def validate_active_configuration(self) -> list[str]:
        """Return human-readable warnings for missing/placeholder credentials."""
        warnings: list[str] = []
        if self.llm_provider == "groq":
            if not self.groq_api_key or self.groq_api_key.startswith("gsk_your_"):
                warnings.append(
                    "GROQ_API_KEY is missing or still set to the placeholder in `.env`."
                )
        elif self.llm_provider == "openai_local":
            if not self.openai_local_base_url:
                warnings.append("OPENAI_LOCAL_BASE_URL is required for local Qwen client.")
            if (
                not self.openai_local_api_key
                or self.openai_local_api_key == "your_local_server_api_key"
            ):
                warnings.append(
                    "OPENAI_LOCAL_API_KEY is missing or placeholder in `.env` "
                    "(set a dummy key like `local-key` if your local server doesn't check it)."
                )

        if self.embedding_provider == "openai" and not self.openai_api_key:
            warnings.append("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai.")
        if self.embedding_provider == "openai_local" and not self.openai_local_embedding_base_url:
            warnings.append(
                "OPENAI_LOCAL_EMBEDDING_BASE_URL is required when EMBEDDING_PROVIDER=openai_local."
            )
        return warnings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
