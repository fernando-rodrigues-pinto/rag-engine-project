"""Domain models and data contracts for the hybrid RAG engine."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """Raw source document ingested into the retrieval pipeline."""

    model_config = ConfigDict(frozen=True)

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    """Structural and relational context bound to a document chunk."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    chunk_index: int = Field(..., ge=0)
    headers: list[str] = Field(default_factory=list)
    char_count: int = Field(..., ge=0)
    token_count: int = Field(..., ge=0)


class Chunk(BaseModel):
    """Semantic fragment ready for dense inference and lexical search."""

    model_config = ConfigDict(frozen=True)

    id: str
    content: str
    metadata: ChunkMetadata
    embedding: list[float] | None = Field(default=None)


class SearchResult(BaseModel):
    """Candidate emitted by a stage of the hybrid retrieval pipeline."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    retrieval_stage: Literal["dense", "sparse", "rrf", "reranked"]
