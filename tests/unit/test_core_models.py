"""Contract tests for hybrid RAG domain models."""

from typing import Literal

import pytest
from pydantic import ValidationError

from src.core import Chunk, ChunkMetadata, Document, SearchResult
from src.core.models import Chunk as ModelsChunk
from src.core.models import ChunkMetadata as ModelsChunkMetadata
from src.core.models import Document as ModelsDocument
from src.core.models import SearchResult as ModelsSearchResult

RetrievalStage = Literal["dense", "sparse", "rrf", "reranked"]


def _metadata(*, chunk_index: int = 0, char_count: int = 4, token_count: int = 1) -> ChunkMetadata:
    return ChunkMetadata(
        document_id="doc-1",
        chunk_index=chunk_index,
        headers=["Title", "Section"],
        char_count=char_count,
        token_count=token_count,
    )


def _chunk(*, embedding: list[float] | None = None) -> Chunk:
    return Chunk(id="chunk-1", content="text", metadata=_metadata(), embedding=embedding)


def test_public_exports_match_domain_models() -> None:
    assert Document is ModelsDocument
    assert ChunkMetadata is ModelsChunkMetadata
    assert Chunk is ModelsChunk
    assert SearchResult is ModelsSearchResult


def test_document_defaults_metadata_per_instance() -> None:
    first = Document(id="doc-1", content="raw")
    second = Document(id="doc-2", content="other")

    assert first.metadata == {}
    assert second.metadata == {}
    assert first.metadata is not second.metadata


def test_document_keeps_source_metadata() -> None:
    document = Document(
        id="doc-1",
        content="raw text",
        metadata={"source_path": "/docs/a.md", "file_type": "markdown"},
    )

    assert document.metadata["source_path"] == "/docs/a.md"
    assert document.metadata["file_type"] == "markdown"


def test_chunk_metadata_accepts_zero_counts() -> None:
    metadata = _metadata(chunk_index=0, char_count=0, token_count=0)

    assert metadata.chunk_index == 0
    assert metadata.char_count == 0
    assert metadata.token_count == 0
    assert metadata.headers == ["Title", "Section"]


def test_chunk_metadata_defaults_headers() -> None:
    metadata = ChunkMetadata(document_id="doc-1", chunk_index=1, char_count=3, token_count=1)

    assert metadata.headers == []


@pytest.mark.parametrize("field", ["chunk_index", "char_count", "token_count"])
def test_chunk_metadata_rejects_negative_counts(field: str) -> None:
    payload: dict[str, object] = {
        "document_id": "doc-1",
        "chunk_index": 0,
        "char_count": 1,
        "token_count": 1,
        field: -1,
    }

    with pytest.raises(ValidationError):
        ChunkMetadata.model_validate(payload)


def test_chunk_embedding_defaults_to_none() -> None:
    chunk = _chunk()

    assert chunk.embedding is None
    assert chunk.metadata.document_id == "doc-1"


def test_chunk_stores_runtime_embedding() -> None:
    chunk = _chunk(embedding=[0.1, 0.2, 0.3])

    assert chunk.embedding == [0.1, 0.2, 0.3]


@pytest.mark.parametrize("stage", ["dense", "sparse", "rrf", "reranked"])
def test_search_result_accepts_pipeline_stages(stage: RetrievalStage) -> None:
    result = SearchResult(chunk=_chunk(), score=0.5, retrieval_stage=stage)

    assert result.retrieval_stage == stage
    assert result.chunk.id == "chunk-1"


def test_search_result_rejects_unknown_stage() -> None:
    with pytest.raises(ValidationError):
        SearchResult.model_validate(
            {
                "chunk": _chunk().model_dump(),
                "score": 1.0,
                "retrieval_stage": "hybrid",
            }
        )


def test_domain_models_are_frozen() -> None:
    document = Document(id="doc-1", content="raw")
    metadata = _metadata()
    chunk = _chunk()
    result = SearchResult(chunk=chunk, score=0.2, retrieval_stage="dense")

    with pytest.raises(ValidationError):
        document.id = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        metadata.chunk_index = 2  # type: ignore[misc]
    with pytest.raises(ValidationError):
        chunk.content = "mutated"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        result.score = 1.0  # type: ignore[misc]
