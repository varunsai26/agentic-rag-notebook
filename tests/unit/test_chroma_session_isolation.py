"""Unit tests verifying ChromaDB rich scalar metadata and strict session_id isolation."""

from agentic_rag_notebook.config.settings import Settings
from agentic_rag_notebook.domain.models import ParsedChunk
from agentic_rag_notebook.infrastructure.chroma_store import ChromaVectorRepository
from tests.conftest import DeterministicFakeEmbeddings


def test_chroma_stores_rich_metadata_and_enforces_strict_session_isolation(
    tmp_settings: Settings,
) -> None:
    repo = ChromaVectorRepository(
        persist_directory=tmp_settings.chroma_persist_dir,
        collection_name="session_isolation_test",
        embedding_function=DeterministicFakeEmbeddings(),
    )

    assert repo.similarity_search("empty query", k=4, session_id="session-A") == []

    chunks_session_a = [
        ParsedChunk(
            text="Session A quarterly metrics show 42 percent revenue growth.",
            source="metrics.csv",
            source_type="csv",
            session_id="session-A",
            filename="metrics.csv",
            title="metrics",
            row=0,
            heading="Row 1 · Q1",
            chunk_index=0,
            metadata={"csv_department": "cloud", "csv_source_value": "Q1"},
        ),
        ParsedChunk(
            text="Wikipedia Python language overview from session A.",
            source="wiki:Python",
            source_type="wikipedia",
            session_id="session-A",
            filename="",
            url="https://en.wikipedia.org/wiki/Python_(programming_language)",
            title="Python",
            heading="Python",
            chunk_index=0,
            metadata={"wiki_revision_id": 1001, "wiki_categories": "Programming languages"},
        ),
    ]
    chunks_session_b = [
        ParsedChunk(
            text="Session B secret budget metrics must never appear in Session A.",
            source="metrics.csv",
            source_type="csv",
            session_id="session-B",
            filename="metrics.csv",
            title="metrics",
            row=0,
            heading="Row 1 · Secret",
            chunk_index=0,
            metadata={"csv_department": "finance", "csv_source_value": "Secret"},
        ),
        ParsedChunk(
            text="Session B PDF architectural notes on local Qwen deployment.",
            source="notes.pdf",
            source_type="pdf",
            session_id="session-B",
            filename="notes.pdf",
            title="notes",
            page=3,
            heading="Deployment",
            chunk_index=0,
            metadata={"ocr_enabled": True},
        ),
    ]

    assert repo.add_chunks(chunks_session_a) == 2
    assert repo.add_chunks(chunks_session_b) == 2

    # Querying Session A must NEVER match Session B even for the exact same filename `metrics.csv`
    hits_a = repo.similarity_search(
        "quarterly secret budget metrics",
        k=10,
        session_id="session-A",
    )
    assert len(hits_a) == 2
    assert all(hit.session_id == "session-A" for hit in hits_a)
    assert not any("Session B" in hit.content for hit in hits_a)

    # Filter inside Session A by active_sources
    hits_wiki_only = repo.similarity_search(
        "metrics Python",
        k=10,
        session_id="session-A",
        active_sources=["wiki:Python"],
    )
    assert len(hits_wiki_only) == 1
    assert hits_wiki_only[0].source_type == "wikipedia"
    assert hits_wiki_only[0].url == "https://en.wikipedia.org/wiki/Python_(programming_language)"
    assert hits_wiki_only[0].title == "Python"

    # Inspect raw ChromaDB metadata schema directly
    metadatas = repo.get_session_metadata("session-A")
    assert len(metadatas) == 2
    csv_meta = next(m for m in metadatas if m["source_type"] == "csv")
    assert csv_meta["session_id"] == "session-A"
    assert csv_meta["filename"] == "metrics.csv"
    assert csv_meta["row"] == 0
    assert csv_meta["csv_department"] == "cloud"
    assert isinstance(csv_meta["ingested_at"], str)

    # Deleting source in Session A must NOT delete the identically named CSV in Session B
    repo.delete_source("metrics.csv", session_id="session-A")
    remaining_a = repo.similarity_search("metrics", k=10, session_id="session-A")
    remaining_b = repo.similarity_search("metrics", k=10, session_id="session-B")
    assert all(hit.source != "metrics.csv" for hit in remaining_a)
    assert any(hit.source == "metrics.csv" for hit in remaining_b)

    # Deleting Session B vectors leaves Session A intact
    repo.delete_session_vectors("session-B")
    assert repo.similarity_search("metrics Qwen", k=10, session_id="session-B") == []
    assert len(repo.similarity_search("Python", k=10, session_id="session-A")) == 1
