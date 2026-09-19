"""Tests for reranker."""

import pytest

from rag.retrieval.reranker import Reranker, RankedChunk


class TestRerankerABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Reranker()


class TestRankedChunk:
    def test_ranked_chunk_defaults(self):
        chunk = RankedChunk(id="1", text="hello", score=0.9)
        assert chunk.rank == 0
        assert chunk.metadata == {}

    def test_ranked_chunk_with_metadata(self):
        chunk = RankedChunk(
            id="1",
            text="hello",
            score=0.9,
            rank=5,
            metadata={"source": "test"},
        )
        assert chunk.rank == 5
        assert chunk.metadata["source"] == "test"
