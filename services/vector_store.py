"""Pinecone index holding one vector (plus metadata) per table.

It's a standard (dense) serverless index: we compute the embeddings ourselves through
the LiteLLM proxy and send the vectors, rather than letting Pinecone embed text.
"""

# Some Pinecone SDK signatures contain unannotated types (e.g. Index's async
# results), so strict mode would flag every call to them.
# pyright: reportUnknownMemberType=false

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from pinecone import Pinecone, QueryResponse, ServerlessSpec, Vector
from pinecone.db_data import Index

from services.config import get_settings

# Pinecone metadata values, exactly as the SDK types them: no None, no booleans in
# its type hints, no nested objects.
MetadataValue = str | int | float | list[str] | list[int] | list[float]

# Pinecone accepts at most 2 MB per upsert request; with up to ~40 KB of metadata per
# record, 40 records per request stays under it.
UPSERT_BATCH_SIZE = 40


@dataclass
class Match:
    """One search result: a table's id, its similarity to the query, and its metadata."""

    id: str
    score: float  # cosine similarity: higher is closer
    metadata: dict[str, Any]


class VectorStore:
    def __init__(self, index: Index):
        self.index = index

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, MetadataValue]],
    ) -> int:
        """Insert or overwrite records by id; returns how many were written. Writing
        the same id again replaces the record, which keeps re-ingestion idempotent."""
        records = [
            Vector(id=i, values=list(v), metadata=dict(m))
            for i, v, m in zip(ids, vectors, metadatas, strict=True)  # strict: lengths must match
        ]
        if not records:
            return 0
        self.index.upsert(vectors=records, batch_size=UPSERT_BATCH_SIZE, show_progress=False)
        return len(records)

    def delete(self, ids: Sequence[str]) -> None:
        if ids:
            self.index.delete(ids=list(ids))

    def query(self, vector: Sequence[float], top_k: int) -> list[Match]:
        """The top_k records most similar to vector, best first."""
        # query() is typed as also returning an async result whose type is unknown.
        # Treat the result as a plain object, then let isinstance narrow it; that also
        # fails loudly if an async result ever comes back.
        response = cast(object, self.index.query(vector=list(vector), top_k=top_k, include_metadata=True))
        if not isinstance(response, QueryResponse):
            raise TypeError(f"Unexpected Pinecone response: {type(response).__name__}")
        matches = cast(list[Any], response.matches)  # loosely typed SDK objects
        return [Match(id=m.id, score=float(m.score), metadata=dict(m.metadata or {})) for m in matches]


@lru_cache  # one client and one index check per process
def vector_store() -> VectorStore:
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    name = settings.pinecone_index_name

    if not pc.has_index(name):
        # Blocks until the index is ready to accept writes
        pc.create_index(
            name=name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    else:
        dimension = pc.describe_index(name).dimension
        if dimension != settings.embedding_dimension:
            raise RuntimeError(
                f"Pinecone index {name!r} has dimension {dimension}, but "
                f"{settings.embedding_model} produces {settings.embedding_dimension}. "
                "Use another PINECONE_INDEX_NAME or delete the index."
            )
    return VectorStore(pc.Index(name))
