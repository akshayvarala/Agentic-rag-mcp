"""Retrieval logic with optional reranking."""

from dataclasses import dataclass, field
from typing import Any

from rag.embeddings.base import EmbeddingProvider
from rag.errors import RetrievalError
from rag.logging import get_logger
from rag.retrieval.reranker import Reranker, RankedChunk
from rag.vectorstore.base import VectorStore

log = get_logger("retrieval")


@dataclass
class RetrievalResult:
    chunks: list[RankedChunk]
    query: str
    total_found: int = 0


class Retriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
    ):
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        if not query.strip():
            raise RetrievalError("Query cannot be empty")

        log.info("Retrieving for query: %s (top_k=%d)", query[:80], top_k)

        query_embedding = self._embedding_provider.embed_query(query)

        search_results = self._vector_store.search(
            query_vector=query_embedding,
            top_k=top_k * 3 if self._reranker else top_k,
            filters=filters,
        )

        chunks_for_rerank = [
            {"id": r.id, "text": r.text, "score": r.score, **r.metadata}
            for r in search_results
        ]

        if self._reranker and chunks_for_rerank:
            ranked = self._reranker.rerank(query, chunks_for_rerank)
            ranked = ranked[:top_k]
        else:
            ranked = [
                RankedChunk(
                    id=c["id"],
                    text=c["text"],
                    score=c["score"],
                    rank=i,
                    metadata={k: v for k, v in c.items() if k not in ("id", "text", "score")},
                )
                for i, c in enumerate(chunks_for_rerank[:top_k])
            ]

        log.info("Retrieved %d chunks (total found: %d)", len(ranked), len(search_results))

        return RetrievalResult(
            chunks=ranked,
            query=query,
            total_found=len(search_results),
        )
