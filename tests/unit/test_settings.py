"""Unit tests for Settings and environment-driven client configuration."""

from pathlib import Path

from agentic_rag_notebook.config.doctor import inspect_environment
from agentic_rag_notebook.config.settings import Settings


def test_settings_defaults_and_collection_isolation(tmp_path: Path) -> None:
    s1 = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path,
        embedding_provider="huggingface",
        huggingface_embedding_model="BAAI/bge-small-en-v1.5",
    )
    s2 = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path,
        embedding_provider="openai",
        openai_embedding_model="text-embedding-3-small",
    )
    assert s1.llm_provider == "groq"
    assert s1.orchestration_engine == "langchain"
    assert s1.embedding_collection_name() != s2.embedding_collection_name()
    assert s1.embedding_collection_name().startswith("notebook_huggingface_")


def test_settings_logging_defaults_and_ensure_directories(tmp_path: Path) -> None:
    cfg = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        chroma_persist_dir=tmp_path / "data" / "chroma",
        sqlite_db_path=tmp_path / "data" / "sessions.db",
        upload_dir=tmp_path / "data" / "uploads",
        log_dir=tmp_path / "logs",
        log_file_name="app.log",
        log_level="debug",  # type: ignore[arg-type]
        log_backup_count=14,
    )
    assert cfg.log_file_name == "app.log"
    assert cfg.log_level == "DEBUG"
    assert cfg.log_backup_count == 14
    cfg.ensure_directories()
    assert (tmp_path / "logs").is_dir()


def test_settings_validation_warnings_for_missing_keys(tmp_path: Path) -> None:
    groq_missing = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path,
        llm_provider="groq",
        groq_api_key="gsk_your_groq_api_key_here",
    )
    warnings = groq_missing.validate_active_configuration()
    assert len(warnings) == 1
    assert "GROQ_API_KEY" in warnings[0]

    local_ok = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path,
        llm_provider="openai_local",
        openai_local_base_url="http://localhost:8000/v1",
        openai_local_api_key="my-local-secret",
        openai_local_model="Qwen/Qwen2.5-7B-Instruct",
    )
    assert local_ok.validate_active_configuration() == []
    assert local_ok.active_llm_model_name() == "Qwen/Qwen2.5-7B-Instruct"

    healthy_report = inspect_environment(local_ok, env_path=tmp_path / ".env", ping=False)
    assert healthy_report.ok is True
    assert healthy_report.errors == []

    bad_report = inspect_environment(groq_missing, env_path=tmp_path / ".env", ping=False)
    assert bad_report.ok is False
    assert bad_report.warnings or bad_report.errors
