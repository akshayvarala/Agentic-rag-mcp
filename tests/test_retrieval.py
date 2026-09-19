"""Tests for retrieval and reranking."""

import pytest

from rag.retrieval.retriever import Retriever, RetrievalResult
from rag.retrieval.reranker import Reranker, RankedChunk
from rag.errors import RetrievalError


class TestRerankerABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Reranker()


class TestNoopReranker:
    def test_passthrough(self):
        from rag.retrieval.reranker import Reranker

        class NoopReranker(Reranker):
            def rerank(self, query, chunks):
                return [
                    RankedChunk(
                        id=c["id"],
                        text=c["text"],
                        score=c["score"],
                        rank=i,
                    )
                    for i, c in enumerate(chunks)
                ]

        reranker = NoopReranker()
        chunks = [
            {"id": "1", "text": "a", "score": 0.9},
            {"id": "2", "text": "b", "score": 0.8},
        ]
        result = reranker.rerank("query", chunks)
        assert len(result) == 2
        assert result[0].rank == 0
        assert result[1].rank == 1


class TestRetriever:
    def test_empty_query(self):
        from rag.embeddings.base import EmbeddingProvider
        from rag.vectorstore.base import VectorStore

        class MockEmbedding(EmbeddingProvider):
            def embed_documents(self, texts):
                return [[0.0] * 3 for _ in texts]
            def embed_query(self, query):
                return [0.0] * 3
            def dimension(self):
                return 3
            def model_info(self):
                return {}

        class MockStore(VectorStore):
            def create_collection(self, dimension):
                pass
            def upsert(self, ids, vectors, texts, metadatas):
                pass
            def search(self, query_vector, top_k=10, filters=None):
                return []
            def delete(self, ids):
                pass
            def delete_by_filter(self, filters):
                pass
            def count(self):
                return 0
            def collection_info(self):
                return {}

        retriever = Retriever(MockEmbedding(), MockStore())
        with pytest.raises(RetrievalError, match="Query cannot be empty"):
            retriever.retrieve("")
