"""Tests for the RAG pipeline."""

import pytest
from pathlib import Path
from typing import Any

from rag.pipeline import RAGPipeline
from rag.converters.base import DocumentConverter, ConvertedDocument
from rag.chunking.chunker import TextChunker
from rag.embeddings.base import EmbeddingProvider
from rag.vectorstore.base import VectorStore, SearchResult
from rag.errors import RAGError


class MockConverter(DocumentConverter):
    def convert(self, input_path, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / "test.md"
        md_path.write_text("# Test\n\nThis is test content for the pipeline.")
        return ConvertedDocument(
            original_path=input_path,
            markdown_path=md_path,
            metadata_path=output_dir / "test.json",
            assets_dir=output_dir / "assets",
            metadata={"source_file": input_path.name, "file_type": input_path.suffix},
        )


class MockEmbedding(EmbeddingProvider):
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, query):
        return [1.0, 0.0, 0.0]

    def dimension(self):
        return 3

    def model_info(self):
        return {"model": "mock"}


class MockStore(VectorStore):
    def __init__(self):
        self.data: dict[str, dict] = {}
        self._collection_created = False

    def create_collection(self, dimension):
        self._collection_created = True

    def upsert(self, ids, vectors, texts, metadatas):
        for i, id in enumerate(ids):
            self.data[id] = {"text": texts[i], "vector": vectors[i], **metadatas[i]}

    def search(self, query_vector, top_k=10, filters=None):
        results = []
        for id, item in self.data.items():
            if filters:
                match = all(item.get(k) == v for k, v in filters.items())
                if not match:
                    continue
            results.append(SearchResult(
                id=id,
                text=item["text"],
                score=0.95,
                metadata={k: v for k, v in item.items() if k not in ("text", "vector")},
            ))
        return results[:top_k]

    def delete(self, ids):
        for id in ids:
            self.data.pop(id, None)

    def delete_by_filter(self, filters):
        to_delete = []
        for id, item in self.data.items():
            if all(item.get(k) == v for k, v in filters.items()):
                to_delete.append(id)
        for id in to_delete:
            del self.data[id]

    def count(self):
        return len(self.data)

    def collection_info(self):
        return {"points_count": len(self.data)}


@pytest.fixture
def pipeline(tmp_path):
    return RAGPipeline(
        converter=MockConverter(),
        chunker=TextChunker(chunk_size=1024, chunk_overlap=128),
        embedding_provider=MockEmbedding(),
        vector_store=MockStore(),
    )


class TestRAGPipeline:
    def test_initialize(self, pipeline):
        pipeline.initialize()
        assert pipeline._initialized

    def test_initialize_idempotent(self, pipeline):
        pipeline.initialize()
        pipeline.initialize()
        assert pipeline._initialized

    def test_ingest(self, pipeline, tmp_path):
        input_file = tmp_path / "input" / "test.txt"
        input_file.parent.mkdir()
        input_file.write_text("Test content")
        output_dir = tmp_path / "output"

        result = pipeline.ingest(input_file, output_dir)
        assert result["status"] == "success"
        assert result["chunks"] > 0

    def test_search(self, pipeline, tmp_path):
        input_file = tmp_path / "input" / "test.txt"
        input_file.parent.mkdir()
        input_file.write_text("Test content for search")

        pipeline.ingest(input_file, tmp_path / "output")

        result = pipeline.search("test query")
        assert "chunks" in result
        assert "query" in result

    def test_status(self, pipeline):
        status = pipeline.status()
        assert "initialized" in status
        assert "embedding_model" in status
        assert "chunk_size" in status
