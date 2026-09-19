"""Tests for text chunking."""

import pytest

from rag.chunking.chunker import Chunk, TextChunker
from rag.errors import ChunkingError


class TestTextChunker:
    def test_invalid_chunk_size(self):
        with pytest.raises(ChunkingError, match="chunk_size must be positive"):
            TextChunker(chunk_size=0)

    def test_invalid_chunk_overlap(self):
        with pytest.raises(ChunkingError, match="chunk_overlap must be non-negative"):
            TextChunker(chunk_size=100, chunk_overlap=-1)

    def test_overlap_greater_than_size(self):
        with pytest.raises(ChunkingError, match="chunk_overlap.*must be less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_empty_text(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk("")
        assert chunks == []

    def test_whitespace_only(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk("   \n\t  ")
        assert chunks == []

    def test_single_chunk(self):
        chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
        chunks = chunker.chunk("Hello World", source_document="test.txt")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello World"
        assert chunks[0].source_document == "test.txt"
        assert chunks[0].index == 0

    def test_multiple_chunks(self):
        chunker = TextChunker(chunk_size=20, chunk_overlap=5)
        text = "A" * 50
        chunks = chunker.chunk(text, source_document="test.txt")
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert chunk.source_document == "test.txt"

    def test_chunk_metadata(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk("Test content", metadata={"key": "value"})
        assert chunks[0].metadata == {"key": "value"}

    def test_unique_ids(self):
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = "A" * 50
        chunks = chunker.chunk(text)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))
