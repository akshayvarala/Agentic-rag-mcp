"""Base reranker ABC."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RankedChunk:
    id: str
    text: str
    score: float
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[dict[str, Any]]) -> list[RankedChunk]:
        """Rerank chunks based on relevance to query."""
        ...
