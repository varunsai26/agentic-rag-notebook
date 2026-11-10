"""Shared pytest fixtures, deterministic fake embeddings, and fake tool-calling chat model."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field

from agentic_rag_notebook.config.settings import Settings
from agentic_rag_notebook.domain.models import ParsedChunk
from agentic_rag_notebook.infrastructure.chroma_store import ChromaVectorRepository
from agentic_rag_notebook.infrastructure.sqlite_session_store import SQLiteSessionStore


class DeterministicFakeEmbeddings(Embeddings):
    """Offline deterministic 16-dim unit-normalized embeddings for Chroma tests."""

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [float(b) / 255.0 for b in digest[:16]]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeToolCallingChatModel(BaseChatModel):
    """Deterministic fake chat model supporting bind_tools, Qwen <think> blocks, and CRAG grading."""

    responses: list[AIMessage] = Field(default_factory=list)
    call_count: int = 0
    grade_sequence: list[str] = Field(default_factory=lambda: ["YES"])
    grade_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-chat"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> FakeToolCallingChatModel:
        del tools, tool_choice, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        last_text = str(messages[-1].content) if messages else ""

        # Handle LangGraph relevance grader prompt
        if "Relevant (YES or NO)?" in last_text:
            verdict = self.grade_sequence[min(self.grade_index, len(self.grade_sequence) - 1)]
            self.grade_index += 1
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=verdict))])

        # Handle LangGraph query rewrite prompt
        if any("semantic search queries" in str(m.content) for m in messages):
            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content="docling hybrid chunker architecture"))
                ]
            )

        if self.responses and self.call_count < len(self.responses):
            msg = self.responses[self.call_count]
            self.call_count += 1
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Default behavior: if last message is a ToolMessage, synthesize grounded answer
        if messages and isinstance(messages[-1], ToolMessage):
            self.call_count += 1
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="<think>Reviewed the retrieved chunk carefully.</think>"
                            "Docling preserves tables and headings in [architecture.pdf, p.2]."
                        )
                    )
                ]
            )

        self.call_count += 1
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="<think>Checking notebook context.</think>"
                        "Based on [architecture.pdf, p.2], Docling uses HybridChunker."
                    )
                )
            ]
        )


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Create isolated Settings pointing all storage to pytest tmp_path."""
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider="groq",
        groq_api_key="gsk_test_valid_key_1234567890",
        openai_local_base_url="http://localhost:8000/v1",
        openai_local_api_key="local-test-key",
        openai_local_model="Qwen/Qwen2.5-7B-Instruct",
        data_dir=tmp_path / "data",
        chroma_persist_dir=tmp_path / "data" / "chroma",
        sqlite_db_path=tmp_path / "data" / "sessions.db",
        upload_dir=tmp_path / "data" / "uploads",
        log_dir=tmp_path / "logs",
    )
    settings.ensure_directories()
    return settings


@pytest.fixture
def sample_chunks() -> list[ParsedChunk]:
    """Return realistic Docling parsed chunks scoped to `sess-test-1`."""
    return [
        ParsedChunk(
            text="Docling parses PDF layouts, headings, and tables cleanly using HybridChunker.",
            source="architecture.pdf",
            source_type="pdf",
            session_id="sess-test-1",
            filename="architecture.pdf",
            title="architecture",
            page=2,
            heading="Document Parsing > Docling",
            chunk_index=0,
            metadata={"ocr_enabled": False},
        ),
        ParsedChunk(
            text="Groq provides ultra-fast LPU inference while local Qwen runs via OpenAI base_url.",
            source="models.pdf",
            source_type="pdf",
            session_id="sess-test-1",
            filename="models.pdf",
            title="models",
            page=5,
            heading="Model Providers",
            chunk_index=0,
            metadata={"ocr_enabled": False},
        ),
    ]


@pytest.fixture
def seeded_chroma(
    tmp_settings: Settings, sample_chunks: list[ParsedChunk]
) -> ChromaVectorRepository:
    """Return a ChromaVectorRepository pre-populated with deterministic fake embeddings."""
    repo = ChromaVectorRepository(
        persist_directory=tmp_settings.chroma_persist_dir,
        collection_name="test_collection",
        embedding_function=DeterministicFakeEmbeddings(),
    )
    repo.add_chunks(sample_chunks)
    return repo


@pytest.fixture
def session_store(tmp_settings: Settings) -> SQLiteSessionStore:
    """Return an isolated SQLiteSessionStore."""
    return SQLiteSessionStore(tmp_settings.sqlite_db_path)
