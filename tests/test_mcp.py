"""Tests for MCP server tools."""

import pytest
from pathlib import Path
from typing import Any

from rag.mcp.tools import RAGTools
from rag.pipeline import RAGPipeline
from rag.mcp.security import validate_path, validate_query


class TestRAGTools:
    def test_tools_init(self, tmp_path):
        from rag.chunking.chunker import TextChunker
        from rag.embeddings.base import EmbeddingProvider
        from rag.vectorstore.base import VectorStore
        from rag.converters.base import DocumentConverter, ConvertedDocument

        class MockConverter(DocumentConverter):
            def convert(self, input_path, output_dir):
                return ConvertedDocument(
                    original_path=input_path,
                    markdown_path=output_dir / "test.md",
                    metadata_path=output_dir / "test.json",
                    assets_dir=output_dir / "assets",
                )

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
            def __init__(self):
                self.data = {}
            def create_collection(self, dimension):
                pass
            def upsert(self, ids, vectors, texts, metadatas):
                for i, id in enumerate(ids):
                    self.data[id] = {"text": texts[i], **metadatas[i]}
            def search(self, query_vector, top_k=10, filters=None):
                return []
            def delete(self, ids):
                for id in ids:
                    self.data.pop(id, None)
            def delete_by_filter(self, filters):
                pass
            def count(self):
                return len(self.data)
            def collection_info(self):
                return {"points_count": len(self.data)}

        converter = MockConverter()
        from rag.chunking.chunker import TextChunker
        chunker = TextChunker()
        embedding = MockEmbedding()
        store = MockStore()

        pipeline = RAGPipeline(converter, chunker, embedding, store)
        tools = RAGTools(pipeline, tmp_path)

        assert tools._pipeline == pipeline
        assert tools._allowed_input_dir == tmp_path
