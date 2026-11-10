"""Pure domain data models (zero dependency on UI, vector DB, or LLM SDKs)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

SourceType = Literal["pdf", "csv", "wikipedia"]
ScalarMetaValue = str | int | float | bool


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(tz=UTC).isoformat()


class ParsedChunk(BaseModel):
    """Normalized document chunk extracted by any source loader (Docling, CSV, Wikipedia)."""

    text: str
    source: str
    source_type: SourceType = "pdf"
    session_id: str = "default"
    filename: str = ""
    url: str = ""
    title: str = ""
    page: int | None = None
    row: int | None = None
    heading: str | None = None
    chunk_index: int = 0
    ingested_at: str = Field(default_factory=utc_now_iso)
    metadata: dict[str, ScalarMetaValue] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    """Metadata record for an indexed notebook source scoped to a session."""

    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = "default"
    filename: str
    source_type: SourceType = "pdf"
    url: str = ""
    title: str = ""
    file_hash: str
    chunk_count: int
    ocr_enabled: bool = False
    uploaded_at: str = Field(default_factory=utc_now_iso)
    active: bool = True


class RetrievedChunk(BaseModel):
    """Chunk returned from vector similarity search with rich source metadata."""

    content: str
    source: str
    source_type: SourceType = "pdf"
    session_id: str = ""
    filename: str = ""
    url: str = ""
    title: str = ""
    page: int | None = None
    row: int | None = None
    heading: str | None = None
    score: float | None = None


class Citation(BaseModel):
    """User-facing citation derived from retrieved chunks."""

    index: int
    source: str
    source_type: SourceType = "pdf"
    url: str = ""
    title: str = ""
    page: int | None = None
    row: int | None = None
    heading: str | None = None
    excerpt: str


class GuardrailStepResult(BaseModel):
    """Outcome of a single input or output guardrail middleware check."""

    guard_name: str
    stage: Literal["input", "output"]
    passed: bool
    action: Literal["allow", "redact", "warn", "block"]
    reason: str = ""


class GuardrailReport(BaseModel):
    """Aggregated audit report across input and output guardrail middleware."""

    blocked: bool = False
    redacted: bool = False
    warnings: list[str] = Field(default_factory=list)
    steps: list[GuardrailStepResult] = Field(default_factory=list)


class ChatTurn(BaseModel):
    """Persisted turn inside a conversation session."""

    turn_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    reasoning: str = ""
    citations: list[Citation] = Field(default_factory=list)
    guardrail_report: GuardrailReport | None = None
    engine: str = ""
    model_label: str = ""
    created_at: str = Field(default_factory=utc_now_iso)


class ConversationSession(BaseModel):
    """Metadata for a ChatGPT-style conversation bound to a URL session_id."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = "New Notebook Chat"
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    active_sources: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Complete structured answer returned by any RagOrchestrator."""

    answer: str
    reasoning: str = ""
    citations: list[Citation] = Field(default_factory=list)
    guardrail_report: GuardrailReport = Field(default_factory=GuardrailReport)
    tool_calls: list[str] = Field(default_factory=list)
    engine: str = "langchain"
    model_label: str = ""
    blocked: bool = False
