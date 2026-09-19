"""Base vector store ABC."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    def create_collection(self, dimension: int) -> None:
        """Create or recreate the collection."""
        ...

    @abstractmethod
    def upsert(self, ids: list[str], vectors: list[list[float]], texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        """Insert or update vectors."""
        ...

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 10, filters: dict[str, Any] | None = None) -> list[SearchResult]:
        """Search for similar vectors."""
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID."""
        ...

    @abstractmethod
    def delete_by_filter(self, filters: dict[str, Any]) -> None:
        """Delete vectors matching filters."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors in the collection."""
        ...

    @abstractmethod
    def collection_info(self) -> dict[str, Any]:
        """Return collection metadata."""
        ...
