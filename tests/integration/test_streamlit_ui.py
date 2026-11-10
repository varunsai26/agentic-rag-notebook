"""Smoke test for the Gemini NotebookLM-style Streamlit UI using AppTest."""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from agentic_rag_notebook.config.logging_config import (
    _CONFIG_SIG_ATTR,
    _MANAGED_HANDLER_ATTR,
    LOGGER_NAME,
)
from agentic_rag_notebook.config.settings import Settings, get_settings
from agentic_rag_notebook.infrastructure.sqlite_session_store import SQLiteSessionStore
from tests.conftest import DeterministicFakeEmbeddings


@pytest.mark.integration
def test_streamlit_notebook_ui_renders_and_creates_url_session(
    tmp_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app_file = (
        Path(__file__).resolve().parents[2] / "src" / "agentic_rag_notebook" / "ui" / "app.py"
    )

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_valid_key_1234567890")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("DATA_DIR", str(tmp_settings.data_dir))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_settings.chroma_persist_dir))
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_settings.sqlite_db_path))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_settings.upload_dir))
    monkeypatch.setenv("LOG_DIR", str(tmp_settings.log_dir))

    get_settings.cache_clear()
    st.cache_resource.clear()

    try:
        with patch(
            "agentic_rag_notebook.infrastructure.embedding_factory.EmbeddingFactory.create",
            return_value=DeterministicFakeEmbeddings(),
        ):
            at = AppTest.from_file(str(app_file), default_timeout=20)
            at.run()

            assert not at.exception
            assert any("Agentic RAG Notebook" in t.value for t in at.sidebar.title)

            store = SQLiteSessionStore(tmp_settings.sqlite_db_path)
            sessions = store.list_sessions()
            assert len(sessions) == 1
            first_sid = sessions[0].session_id

            # Clicking "+ New Notebook Chat" generates a second distinct UUID session
            new_chat_btn = next(b for b in at.sidebar.button if "New Notebook Chat" in b.label)
            new_chat_btn.click().run()

            assert not at.exception
            updated_sessions = store.list_sessions()
            assert len(updated_sessions) == 2
            assert {s.session_id for s in updated_sessions} != {first_sid}

            # Sidebar exposes both Wikipedia loader button and CSV/PDF uploader controls
            wiki_btn = next(b for b in at.sidebar.button if "Load Wikipedia" in b.label)
            wiki_btn.click().run()
            assert not at.exception
            assert any("Wikipedia" in w.value for w in at.sidebar.warning)

            # Active app.log file is created inside LOG_DIR
            expected_log = tmp_settings.log_dir / tmp_settings.log_file_name
            assert expected_log.is_file()
    finally:
        pkg_logger = logging.getLogger(LOGGER_NAME)
        for handler in list(pkg_logger.handlers):
            if getattr(handler, _MANAGED_HANDLER_ATTR, False):
                pkg_logger.removeHandler(handler)
                handler.close()
        if hasattr(pkg_logger, _CONFIG_SIG_ATTR):
            delattr(pkg_logger, _CONFIG_SIG_ATTR)
        st.cache_resource.clear()
        get_settings.cache_clear()
