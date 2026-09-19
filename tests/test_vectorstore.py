"""Tests for vector store."""

import pytest

from rag.vectorstore.base import VectorStore, SearchResult
from rag.errors import VectorStoreError


class TestVectorStoreABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            VectorStore()


class TestQdrantLocalStore:
    def test_create_collection(self, tmp_path):
        from rag.vectorstore.qdrant_local import QdrantLocalStore

        store = QdrantLocalStore(path=tmp_path / "qdrant", collection="test")
        store.create_collection(dimension=128)
        assert store.count() == 0

    def test_upsert_and_search(self, tmp_path):
        from rag.vectorstore.qdrant_local import QdrantLocalStore

        store = QdrantLocalStore(path=tmp_path / "qdrant", collection="test")
        store.create_collection(dimension=3)

        store.upsert(
            ids=["1", "2", "3"],
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            texts=["doc1", "doc2", "doc3"],
            metadatas=[{"source": "a"}, {"source": "b"}, {"source": "c"}],
        )

        assert store.count() == 3

        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].text == "doc1"

    def test_delete(self, tmp_path):
        from rag.vectorstore.qdrant_local import QdrantLocalStore

        store = QdrantLocalStore(path=tmp_path / "qdrant", collection="test")
        store.create_collection(dimension=3)

        store.upsert(
            ids=["1", "2"],
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            texts=["doc1", "doc2"],
            metadatas=[{"source": "a"}, {"source": "b"}],
        )

        store.delete(["1"])
        assert store.count() == 1

    def test_search_with_filters(self, tmp_path):
        from rag.vectorstore.qdrant_local import QdrantLocalStore

        store = QdrantLocalStore(path=tmp_path / "qdrant", collection="test")
        store.create_collection(dimension=3)

        store.upsert(
            ids=["1", "2"],
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            texts=["doc1", "doc2"],
            metadatas=[{"source": "a"}, {"source": "b"}],
        )

        results = store.search([1.0, 0.0, 0.0], top_k=10, filters={"source": "b"})
        assert len(results) == 1
        assert results[0].text == "doc2"

    def test_collection_info(self, tmp_path):
        from rag.vectorstore.qdrant_local import QdrantLocalStore

        store = QdrantLocalStore(path=tmp_path / "qdrant", collection="test")
        store.create_collection(dimension=3)

        info = store.collection_info()
        assert info["name"] == "test"
        assert info["points_count"] == 0
